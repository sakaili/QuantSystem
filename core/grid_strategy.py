"""
网格策略执行器模块
Grid Strategy Module

实现网格交易策略逻辑
"""

import math
import re
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from .exchange_connector import ExchangeConnector, Order
from .position_manager import PositionManager, SymbolPosition
from .config_manager import ConfigManager
from utils.exceptions import OrderError
from utils.logger import get_logger

logger = get_logger("grid")


@dataclass
class GridPrices:
    """网格价格（动态网格）"""
    entry_price: float
    grid_levels: Dict[int, float]  # level -> price，level可以是任意整数（正数=上方，负数=下方）
    stop_loss_price: float         # 止损价格
    spacing: float                 # 网格间距，用于动态计算

    def get_upper_levels(self) -> List[int]:
        """获取所有上方网格层级（正数）"""
        return sorted([level for level in self.grid_levels.keys() if level > 0])

    def get_lower_levels(self) -> List[int]:
        """获取所有下方网格层级（负数）"""
        return sorted([level for level in self.grid_levels.keys() if level < 0], reverse=True)

    def add_level_above(self, max_level: int) -> int:
        """在最上方添加新网格"""
        new_level = max_level + 1
        new_price = self.entry_price * ((1 + self.spacing) ** new_level)
        self.grid_levels[new_level] = new_price
        return new_level

    def add_level_below(self, min_level: int) -> int:
        """在最下方添加新网格"""
        new_level = min_level - 1
        new_price = self.entry_price * ((1 - self.spacing) ** abs(new_level))
        self.grid_levels[new_level] = new_price
        return new_level

    def add_level(self, level: int, price: float) -> None:
        """
        添加指定层级的网格

        Args:
            level: 网格层级
            price: 价格
        """
        self.grid_levels[level] = price
        logger.debug(f"添加网格层级 Grid{level:+d} @ {price:.6f}")

    def remove_level(self, level: int) -> None:
        """移除指定层级的网格"""
        if level in self.grid_levels:
            price = self.grid_levels[level]
            del self.grid_levels[level]
            logger.debug(f"移除网格层级 Grid{level:+d} @ {price:.6f}")


@dataclass
class UpperGridFill:
    """上方网格成交信息"""
    price: float          # 开仓价格
    amount: float         # 开仓数量
    fill_time: datetime   # 成交时间
    order_id: str         # 订单ID
    matched_lower_price: Optional[float] = None  # 匹配的下方止盈价格


@dataclass
class GridState:
    """网格状态"""
    symbol: str
    entry_price: float
    grid_prices: GridPrices
    upper_orders: Dict[float, List[str]] = field(default_factory=dict)  # price -> [order_id]
    lower_orders: Dict[float, List[str]] = field(default_factory=dict)  # price -> [order_id]
    filled_upper_grids: Dict[str, UpperGridFill] = field(default_factory=dict)  # order_id -> fill_info（记录开仓信息）
    tp_to_upper: Dict[str, str] = field(default_factory=dict)  # tp_order_id -> upper_order_id
    peak_short_size: float = 0.0          # 历史峰值空头仓位（ratchet 只增不减）
    core_target_size: float = 0.0         # 需要锁定的核心仓位
    core_ratio: float = 0.0               # 核心仓位比例（默认取配置 min_base_position_ratio）
    ratchet_initialized: bool = False     # 是否已用当前仓位初始化 ratchet
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rebase_deviation_since: Optional[datetime] = None
    last_rebase_time: Optional[datetime] = None
    rebase_frozen: bool = False
    shorts_paused: bool = False
    shorts_pause_reason: Optional[str] = None

    # 网格完整性追踪（简化，移除 failures 字典）
    last_repair_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    grid_integrity_validated: bool = False  # 是否通过初始验证
    upper_success_rate: float = 0.0         # 上方网格创建成功率
    lower_success_rate: float = 0.0         # 下方网格创建成功率
    needs_cleanup: bool = False             # 是否需要清理（仓位完全平仓时标记）


class GridStrategy:
    """
    网格策略执行器

    管理网格订单的创建、监控和调整
    """

    def __init__(
        self,
        config: ConfigManager,
        connector: ExchangeConnector,
        position_mgr: PositionManager,
        db: Optional[object] = None
    ):
        """
        Args:
            config: 配置管理器
            connector: 交易所连接器
            position_mgr: 仓位管理器
        """
        self.config = config
        self.connector = connector
        self.position_mgr = position_mgr
        self.db = db

        # 网格状态字典: symbol -> GridState
        self.grid_states: Dict[str, GridState] = {}

        # 仓位缓存，减少API调用频率: symbol -> (Position, timestamp)
        from .exchange_connector import Position
        self._position_cache: Dict[str, tuple] = {}
        self._cache_ttl = 5  # 缓存TTL (秒)

        # Tick size cache: symbol -> (tick_size, timestamp)
        self._tick_size_cache: Dict[str, tuple] = {}
        self._tick_size_cache_ttl = 60  # seconds

        # 对账时间戳: symbol -> datetime
        self._last_reconciliation: Dict[str, datetime] = {}
        self._reconciliation_interval = 60  # 对账间隔 (秒)

        # Lower-grid/base-TP capacity logs can be noisy; throttle per symbol.
        self._capacity_log_last: Dict[tuple, float] = {}
        self._capacity_log_interval = 60  # seconds
        self._client_order_seq = 0

        logger.info("网格策略执行器初始化完成")

    def _log_capacity_event(
        self,
        symbol: str,
        key: str,
        message: str,
        level: str = "warning",
        interval: Optional[int] = None
    ) -> None:
        """Throttle repetitive capacity-related logs per symbol."""
        now = time.time()
        interval = self._capacity_log_interval if interval is None else interval
        cache_key = (symbol, key)
        last_ts = self._capacity_log_last.get(cache_key)
        if last_ts is not None and (now - last_ts) < interval:
            return
        self._capacity_log_last[cache_key] = now
        log_fn = getattr(logger, level, logger.warning)
        log_fn(message)

    def _add_order_id(self, orders: Dict[float, List[str]], price: float, order_id: str) -> bool:
        """Add order_id under price; return True if newly added."""
        order_list = orders.get(price)
        if order_list is None:
            orders[price] = [order_id]
            return True
        if order_id in order_list:
            return False
        order_list.append(order_id)
        return True

    def _remove_order_id(self, orders: Dict[float, List[str]], price: float, order_id: str) -> None:
        """Remove order_id under price."""
        order_list = orders.get(price)
        if not order_list:
            return
        try:
            order_list.remove(order_id)
        except ValueError:
            return
        if not order_list:
            del orders[price]

    def _iter_order_items(self, orders: Dict[float, List[str]]):
        """Yield (price, order_id) for all orders."""
        for price, order_ids in list(orders.items()):
            for order_id in list(order_ids):
                yield price, order_id

    def _count_orders(self, orders: Dict[float, List[str]]) -> int:
        """Count total orders in price->list map."""
        return sum(len(order_ids) for order_ids in orders.values())

    def _is_price_too_close(self, target_price: float, existing_prices: List[float], min_gap_ratio: float) -> bool:
        """Return True if target_price is within min_gap_ratio of any existing price."""
        if target_price <= 0 or min_gap_ratio <= 0:
            return False
        for price in existing_prices:
            if price <= 0:
                continue
            if abs(target_price - price) / target_price < min_gap_ratio:
                return True
        return False

    def pause_shorts(self, symbol: str, reason: str = "pause") -> None:
        grid_state = self.grid_states.get(symbol)
        if not grid_state or grid_state.shorts_paused:
            return
        grid_state.shorts_paused = True
        grid_state.shorts_pause_reason = reason
        logger.warning(f"{symbol} shorts paused: {reason}")

    def resume_shorts(self, symbol: str, reason: str = "resume") -> None:
        grid_state = self.grid_states.get(symbol)
        if not grid_state or not grid_state.shorts_paused:
            return
        grid_state.shorts_paused = False
        grid_state.shorts_pause_reason = None
        logger.info(f"{symbol} shorts resumed: {reason}")

    def _shorts_paused(self, grid_state: GridState) -> bool:
        return bool(grid_state.shorts_paused)

    def _maybe_soft_rebase(self, symbol: str, grid_state: GridState) -> bool:
        """Soft rebase grids when price deviates too far on the downside."""
        if not self.config.grid.rebase_enabled:
            return False

        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"{symbol} get price failed, skip rebase check: {e}")
            return False

        center_price = grid_state.entry_price
        if not center_price or center_price <= 0:
            return False

        down_pct = self.config.grid.rebase_down_pct
        if down_pct <= 0:
            down_pct = self.config.grid.rebase_distance_k * self.config.grid.spacing
        if down_pct <= 0:
            return False

        up_freeze_pct = self.config.grid.rebase_up_freeze_pct
        if up_freeze_pct < 0:
            up_freeze_pct = 0.0

        now = datetime.now(timezone.utc)

        if grid_state.rebase_frozen:
            avg_entry = center_price
            position = self.position_mgr.get_symbol_position(symbol)
            if position:
                avg_entry = position.get_average_entry_price() or avg_entry
            if current_price <= avg_entry:
                grid_state.rebase_frozen = False
                grid_state.rebase_deviation_since = None
                logger.info(f"{symbol} rebase unfrozen (price {current_price:.6f} <= avg {avg_entry:.6f})")
            else:
                return False

        if up_freeze_pct > 0 and current_price >= center_price * (1 + up_freeze_pct):
            grid_state.rebase_frozen = True
            grid_state.rebase_deviation_since = None
            logger.warning(
                f"{symbol} rebase frozen (price {current_price:.6f} >= "
                f"center {center_price:.6f} * (1+{up_freeze_pct:.3f}))"
            )
            return False

        if current_price >= center_price:
            return False

        deviation = (center_price - current_price) / center_price
        if deviation < down_pct:
            grid_state.rebase_deviation_since = None
            return False

        self._soft_rebase_grid(symbol, grid_state, current_price)
        grid_state.last_rebase_time = now
        grid_state.rebase_deviation_since = None
        return True

    def _soft_rebase_grid(self, symbol: str, grid_state: GridState, new_center: float) -> None:
        """Cancel existing orders and rebuild grids around new_center without closing position."""
        logger.info(f"{symbol} soft rebase, new center {new_center:.6f}")

        # cancel existing orders
        all_order_ids = []
        for _, order_id in self._iter_order_items(grid_state.upper_orders):
            all_order_ids.append(order_id)
        for _, order_id in self._iter_order_items(grid_state.lower_orders):
            all_order_ids.append(order_id)

        for order_id in all_order_ids:
            try:
                self.connector.cancel_order(order_id, symbol)
            except Exception as e:
                logger.warning(f"{symbol} rebase cancel failed: {e}")

        # reset grid state
        grid_state.upper_orders.clear()
        grid_state.lower_orders.clear()
        grid_state.filled_upper_grids.clear()
        grid_state.tp_to_upper.clear()

        grid_state.entry_price = new_center
        grid_state.grid_prices = self.calculate_grid_prices(new_center)
        grid_state.last_update = datetime.now(timezone.utc)
        grid_state.last_repair_check = datetime.now(timezone.utc)
        grid_state.grid_integrity_validated = False

        # rebuild orders
        self._place_base_position_take_profit(symbol, grid_state)
        self._place_upper_grid_orders(symbol, grid_state)

        validation_passed, validation_msg = self._validate_grid_creation(symbol, grid_state)
        if not validation_passed:
            logger.warning(f"{symbol} rebase grid validation failed: {validation_msg}")
    def _count_base_tp_orders(self, grid_state: GridState) -> int:
        """Count base TP orders (exclude upper->TP cycles)."""
        count = 0
        for price, order_ids in grid_state.lower_orders.items():
            for order_id in order_ids:
                if order_id not in grid_state.tp_to_upper:
                    count += 1
        return count

    def _compute_base_tp_allowed_levels(self, total_levels: int) -> int:
        """Compute allowed base TP levels based on min base ratio and margin."""
        base_margin = self.config.position.base_margin
        grid_margin = self.config.position.grid_margin
        min_ratio = self.config.position.min_base_position_ratio
        closeable_ratio = 1.0 - min_ratio
        closeable_margin = base_margin * closeable_ratio
        allowed_levels_by_ratio = int(total_levels * closeable_ratio)
        allowed_levels_by_margin = int(closeable_margin // grid_margin) if grid_margin > 0 else 0
        return min(allowed_levels_by_ratio, allowed_levels_by_margin)

    def _get_core_inventory_ratio(self, grid_state: GridState) -> float:
        """Return configured core inventory ratio for a symbol."""
        ratio = grid_state.core_ratio or self.config.position.min_base_position_ratio
        ratio = max(0.0, min(float(ratio), 1.0))
        return ratio

    def _refresh_inventory_ratchet(
        self,
        symbol: str,
        grid_state: GridState,
        *,
        current_short_size: Optional[float] = None,
        force_refresh: bool = False
    ) -> float:
        """Refresh ratchet state from live short position.

        Ratchet rules:
        - peak_short_size only increases
        - core_target_size = peak_short_size * core_ratio
        - on first adoption, current short size seeds the peak
        """
        if current_short_size is None:
            short_position = self._get_cached_short_position(symbol, force_refresh=force_refresh)
            current_short_size = short_position.size if short_position else 0.0

        ratio = self._get_core_inventory_ratio(grid_state)
        grid_state.core_ratio = ratio

        old_peak = grid_state.peak_short_size
        if not grid_state.ratchet_initialized and current_short_size > 0:
            grid_state.peak_short_size = max(grid_state.peak_short_size, current_short_size)
            grid_state.ratchet_initialized = True
            logger.info(
                f"{symbol} 启用核心仓 ratchet: current={current_short_size:.2f}, "
                f"peak={grid_state.peak_short_size:.2f}, core_ratio={ratio*100:.0f}%"
            )
        elif current_short_size > grid_state.peak_short_size:
            grid_state.peak_short_size = current_short_size
            logger.info(
                f"{symbol} ratchet 峰值更新: {old_peak:.2f} → {grid_state.peak_short_size:.2f}"
            )

        grid_state.core_target_size = grid_state.peak_short_size * ratio if grid_state.peak_short_size > 0 else 0.0
        return current_short_size

    def _get_pending_lower_order_amount(
        self,
        symbol: str,
        grid_state: GridState,
        open_orders: Optional[List[Order]] = None
    ) -> float:
        """Return total open BUY reduce-only exposure tracked in lower_orders."""
        if open_orders is None:
            open_orders = self._get_open_orders_safe(symbol)

        open_order_map = {order.order_id: order for order in open_orders}
        pending_lower_total = 0.0
        for _, order_ids in grid_state.lower_orders.items():
            for order_id in order_ids:
                order = open_order_map.get(order_id)
                if order and order.side == 'buy':
                    pending_lower_total += order.amount
        return pending_lower_total

    def _get_tradable_short_size(
        self,
        symbol: str,
        grid_state: GridState,
        *,
        current_short_size: Optional[float] = None,
        force_refresh: bool = False
    ) -> float:
        """Return short inventory that is allowed to participate in ordinary TP."""
        current_short_size = self._refresh_inventory_ratchet(
            symbol,
            grid_state,
            current_short_size=current_short_size,
            force_refresh=force_refresh
        )
        return max(current_short_size - grid_state.core_target_size, 0.0)

    def _has_tp_for_upper(self, grid_state: GridState, upper_order_id: str) -> bool:
        """Return True if any TP is mapped to upper_order_id."""
        return upper_order_id in grid_state.tp_to_upper.values()

    def _get_lower_levels_by_proximity(self, grid_state: GridState, current_price: float) -> List[int]:
        """Return lower levels ordered by proximity to current price (nearest first)."""
        lower_levels = grid_state.grid_prices.get_lower_levels()
        if not lower_levels:
            return []

        below_levels = []
        all_levels = []
        for level in lower_levels:
            price = grid_state.grid_prices.grid_levels[level]
            distance = abs(current_price - price)
            all_levels.append((distance, level))
            if price <= current_price:
                below_levels.append((distance, level))

        ordered = below_levels if below_levels else all_levels
        ordered.sort(key=lambda item: item[0])
        return [level for _, level in ordered]

    def _complete_cycle_from_tp(
        self,
        symbol: str,
        grid_state: GridState,
        upper_order_id: str,
        tp_order_id: str,
        fill_info: UpperGridFill
    ) -> None:
        """Handle a completed upper->TP cycle and restore grids."""
        tp_price = fill_info.matched_lower_price if fill_info.matched_lower_price is not None else 0.0
        profit_pct = 0.0
        if fill_info.price > 0:
            profit_pct = (fill_info.price - tp_price) / fill_info.price * 100
        logger.info(
            f"完整循环: 开仓 @ {fill_info.price:.6f}, 平仓 @ {tp_price:.6f}, 盈利 {profit_pct:.2f}%"
        )

        # 恢复上方网格
        self._place_single_upper_grid_by_price(symbol, grid_state, fill_info.price)

        # 恢复下方基础止盈单
        if fill_info.matched_lower_price is not None:
            self._place_single_lower_grid_by_price(symbol, grid_state, fill_info.matched_lower_price)

        # 清理映射与成交记录
        if upper_order_id in grid_state.filled_upper_grids:
            del grid_state.filled_upper_grids[upper_order_id]
        grid_state.tp_to_upper.pop(tp_order_id, None)

        if hasattr(self, "db") and self.db:
            self.db.close_grid_cycle_by_tp(tp_order_id)

    def calculate_grid_prices(self, entry_price: float) -> GridPrices:
        """
        计算动态网格价格

        初始化时创建±10个网格，后续可扩展到±15

        Args:
            entry_price: 入场价P0

        Returns:
            GridPrices对象
        """
        spacing = self.config.grid.spacing
        upper_count = self.config.grid.upper_grids
        lower_count = self.config.grid.lower_grids

        # 初始化网格字典
        grid_levels = {}

        # 上方网格：Grid+1 到 Grid+10
        for level in range(1, upper_count + 1):
            price = entry_price * ((1 + spacing) ** level)
            grid_levels[level] = price

        # 下方网格：Grid-1 到 Grid-10
        for level in range(1, lower_count + 1):
            price = entry_price * ((1 - spacing) ** level)
            grid_levels[-level] = price

        # 止损线
        stop_loss_price = entry_price * self.config.stop_loss.ratio

        grid_prices = GridPrices(
            entry_price=entry_price,
            grid_levels=grid_levels,
            stop_loss_price=stop_loss_price,
            spacing=spacing
        )

        # 计算价格范围
        upper_levels = grid_prices.get_upper_levels()
        lower_levels = grid_prices.get_lower_levels()
        min_price = grid_levels[min(lower_levels)] if lower_levels else entry_price
        max_price = grid_levels[max(upper_levels)] if upper_levels else entry_price

        logger.info(
            f"初始化动态网格: P0={entry_price:.4f}, "
            f"上方{len(upper_levels)}个, 下方{len(lower_levels)}个, "
            f"范围={min_price:.4f}~{max_price:.4f}"
        )
        return grid_prices

    def initialize_grid(self, symbol: str, entry_price: float) -> bool:
        """
        初始化网格

        Args:
            symbol: 交易对
            entry_price: 入场价

        Returns:
            是否成功
        """
        try:
            logger.info(f"初始化网格: {symbol} @ {entry_price}")

            # 计算网格价格
            grid_prices = self.calculate_grid_prices(entry_price)

            # 1. 开基础仓位（使用市价单立即成交）
            base_margin = self.config.position.base_margin
            base_amount = self._calculate_amount(symbol, base_margin, entry_price)

            logger.info(f"开基础仓位（市价）: {base_amount}张")
            base_order = self.connector.place_order(
                symbol=symbol,
                side='sell',  # 开空
                amount=base_amount,
                order_type='market'
            )

            base_order_id = base_order.order_id  # 保存用于可能的清理

            # 2. 等待基础仓位成交确认（市价单通常立即成交，短超时即可）
            logger.info(f"等待基础仓位成交确认: order_id={base_order_id}")
            base_filled = self._wait_for_order_fill(symbol, base_order_id, timeout=30)  # 30秒超时

            if not base_filled:
                logger.error(f"基础仓位超时未成交，初始化失败")
                self._cleanup_failed_initialization(symbol, base_order_id)
                return False

            logger.info(f"✅ 基础仓位已成交，开始挂网格")

            # 3. 创建网格状态
            grid_state = GridState(
                symbol=symbol,
                entry_price=entry_price,
                grid_prices=grid_prices,
                core_ratio=self.config.position.min_base_position_ratio
            )

            self.grid_states[symbol] = grid_state
            self._refresh_inventory_ratchet(symbol, grid_state, force_refresh=True)

            # 4. 挂基础仓位的分层止盈单（先挂止盈保护）
            logger.info("挂基础仓位分层止盈单...")
            self._place_base_position_take_profit(symbol, grid_state)

            # 5. 挂上方网格订单(开空)
            logger.info("挂上方网格...")
            self._place_upper_grid_orders(symbol, grid_state)

            # 6. 验证网格创建成功率
            validation_passed, validation_msg = self._validate_grid_creation(symbol, grid_state)

            if not validation_passed:
                logger.warning(f"网格验证失败: {validation_msg}, 但继续运行（已禁用自动平仓）")
                # 不再调用 _cleanup_failed_initialization，允许部分网格运行
                # 后续的网格修复机制会自动补充缺失的网格

            # 7. 添加到仓位管理器
            self.position_mgr.add_position(symbol, entry_price)

            logger.info(f"网格初始化完成: {symbol}")
            return True

        except Exception as e:
            logger.error(f"网格初始化失败: {symbol}: {e}")
            # 尝试清理
            if symbol in self.grid_states:
                self._cleanup_failed_initialization(symbol, None)
            return False

    def _wait_for_order_fill(self, symbol: str, order_id: str, timeout: int = 60) -> bool:
        """
        轮询等待订单成交

        Args:
            symbol: 交易对
            order_id: 订单ID
            timeout: 超时时间(秒)

        Returns:
            是否成交
        """
        import time
        from datetime import datetime, timezone

        start_time = datetime.now(timezone.utc)
        check_interval = 3  # 每3秒检查一次

        logger.info(f"开始轮询订单状态: order_id={order_id}, 超时={timeout}秒")

        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

            if elapsed > timeout:
                logger.warning(f"订单等待超时({timeout}秒): order_id={order_id}")
                return False

            try:
                # 查询订单状态
                open_orders = self.connector.query_open_orders(symbol)
                order_still_open = any(o.order_id == order_id for o in open_orders)

                if not order_still_open:
                    # 订单不在挂单列表中，说明已成交或取消
                    # 确认持仓是否增加
                    positions = self.connector.query_positions()
                    has_position = any(p.symbol == symbol and abs(p.contracts) > 0 for p in positions)

                    if has_position:
                        logger.info(f"✅ 订单已成交: order_id={order_id}, 耗时={elapsed:.1f}秒")
                        return True
                    else:
                        logger.warning(f"订单已取消或失败: order_id={order_id}")
                        return False

                logger.info(f"订单等待中... ({elapsed:.0f}/{timeout}秒)")
                time.sleep(check_interval)

            except Exception as e:
                logger.warning(f"查询订单状态失败: {e}, 继续等待...")
                time.sleep(check_interval)

    def _validate_grid_creation(self, symbol: str, grid_state: GridState) -> tuple:
        """
        验证网格创建成功率

        Args:
            symbol: 交易对
            grid_state: 网格状态

        Returns:
            tuple[bool, str]: (是否通过验证, 详细信息)
        """
        upper_count = len(grid_state.grid_prices.get_upper_levels())
        total_lower_levels = len(grid_state.grid_prices.get_lower_levels())
        lower_count = self._compute_base_tp_allowed_levels(total_lower_levels)
        upper_created = min(self._count_orders(grid_state.upper_orders), upper_count)
        lower_created = min(self._count_orders(grid_state.lower_orders), lower_count) if lower_count > 0 else 0

        upper_success_rate = upper_created / upper_count if upper_count > 0 else 0.0
        lower_success_rate = lower_created / lower_count if lower_count > 0 else 1.0

        grid_state.upper_success_rate = upper_success_rate
        grid_state.lower_success_rate = lower_success_rate

        logger.info(
            f"{symbol} 网格创建统计: "
            f"上方 {upper_created}/{upper_count} ({upper_success_rate*100:.1f}%), "
            f"下方 {lower_created}/{lower_count} ({lower_success_rate*100:.1f}%)"
        )

        # 上方网格严格要求80%（开空单，关键）
        if upper_success_rate < self.config.grid.min_success_rate_upper:
            msg = f"{symbol} 上方网格成功率{upper_success_rate*100:.1f}% < {self.config.grid.min_success_rate_upper*100:.0f}%, 拒绝开仓"
            logger.error(msg)
            return False, msg

        # 下方网格仅告警（止盈单，不关键）
        if lower_count > 0 and lower_success_rate < self.config.grid.min_success_rate_lower:
            logger.warning(
                f"{symbol} 下方网格成功率{lower_success_rate*100:.1f}% < "
                f"{self.config.grid.min_success_rate_lower*100:.0f}%"
            )

        grid_state.grid_integrity_validated = True
        return True, "网格创建成功"

    def _cleanup_failed_initialization(self, symbol: str, base_order_id: Optional[str]) -> None:
        """
        清理初始化失败的订单和状态

        Args:
            symbol: 交易对
            base_order_id: 基础仓位订单ID（如果已创建）
        """
        logger.info(f"清理失败的初始化: {symbol}")

        if symbol not in self.grid_states:
            return

        grid_state = self.grid_states[symbol]

        # 1. 撤销所有上方网格订单
        for price, order_id in list(self._iter_order_items(grid_state.upper_orders)):
            try:
                self.connector.cancel_order(order_id, symbol)
                logger.info(f"已撤销上方网格 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"撤销订单失败: {e}")

        # 2. 撤销所有下方网格订单
        for price, order_id in list(self._iter_order_items(grid_state.lower_orders)):
            try:
                self.connector.cancel_order(order_id, symbol)
                logger.info(f"已撤销下方网格 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"撤销订单失败: {e}")

        # 3. 撤销基础仓位订单
        if base_order_id:
            try:
                self.connector.cancel_order(base_order_id, symbol)
                logger.info(f"已撤销基础仓位订单")
            except Exception as e:
                logger.warning(f"撤销基础仓位订单失败: {e}")

        # 4. 检查并平仓已成交的仓位
        try:
            positions = self.connector.query_positions()
            for pos in positions:
                if pos.symbol == symbol and abs(pos.contracts) > 0:
                    try:
                        self.connector.place_order(
                            symbol=symbol,
                            side='buy',  # 平空
                            amount=abs(pos.contracts),
                            order_type='market',
                            reduce_only=True  # 强制只减仓
                        )
                        logger.info(f"已市价平仓: {abs(pos.contracts)}张")
                    except Exception as e:
                        logger.error(f"平仓失败: {e}")
        except Exception as e:
            logger.warning(f"查询仓位失败: {e}")

        # 5. 移除网格状态
        del self.grid_states[symbol]
        logger.info(f"清理完成: {symbol}")

    def _get_open_orders_safe(self, symbol: str) -> List[Order]:
        """Query open orders safely for dedupe checks."""
        try:
            return self.connector.query_open_orders(symbol)
        except Exception as e:
            logger.warning(f"{symbol} query open orders failed, skip dedupe: {e}")
            return []

    def _match_open_order_by_price(
        self,
        symbol: str,
        open_orders: List[Order],
        side: str,
        target_price: float,
        tolerance: float = 0.001
    ) -> Optional[str]:
        """Find an existing open order within tolerance by price."""
        for order in open_orders:
            if order.side != side or order.price is None:
                continue
            order_price = self._quantize_price(symbol, order.price, side=order.side)
            if abs(order_price - target_price) / target_price < tolerance:
                return order.order_id
        return None

    def _match_open_order_by_client_id(
        self,
        open_orders: List[Order],
        client_order_id: str
    ) -> Optional[str]:
        """Find an existing open order by client_order_id."""
        if not client_order_id:
            return None
        for order in open_orders:
            if order.client_order_id == client_order_id:
                return order.order_id
        return None

    def _get_open_order_by_client_id(
        self,
        open_orders: List[Order],
        client_order_id: str
    ) -> Optional[Order]:
        """Get the open order by client_order_id."""
        if not client_order_id:
            return None
        for order in open_orders:
            if order.client_order_id == client_order_id:
                return order
        return None

    def _query_order_safe(self, symbol: str, order_id: str) -> Optional[Order]:
        """Query a single order status safely."""
        try:
            return self.connector.query_order(order_id, symbol)
        except Exception as e:
            logger.warning(f"{symbol} query order failed {order_id}: {e}")
            return None

    def _sanitize_symbol_for_id(self, symbol: str) -> str:
        """Sanitize symbol for client_order_id usage."""
        return re.sub(r'[^A-Za-z0-9]', '', symbol)

    def _make_client_order_id(
        self,
        symbol: str,
        side: str,
        level: Optional[int] = None,
        price: Optional[float] = None,
        entry_price: Optional[float] = None,
        unique: bool = True
    ) -> str:
        """Create a deterministic client_order_id for grid orders."""
        sym = self._sanitize_symbol_for_id(symbol)
        sym_short = sym if len(sym) <= 10 else sym[-10:]
        side_tag = 'S' if side == 'sell' else 'B'

        lvl = level
        if lvl is None and price is not None and entry_price is not None:
            try:
                lvl = self._calculate_grid_level(price, entry_price, self.config.grid.spacing)
            except Exception:
                lvl = None

        if lvl is not None:
            base_id = f"G{side_tag}L{lvl}_{sym_short}"
        else:
            ptag = int(Decimal(str(price or 0)) * Decimal("1e8"))
            base_id = f"G{side_tag}P{ptag}_{sym_short}"

        if unique:
            self._client_order_seq = (self._client_order_seq + 1) % 1000000
            suffix = f"_{int(time.time() * 1000) % 100000000}{self._client_order_seq:03d}"
            base_id = f"{base_id}{suffix}"

        if len(base_id) > 36:
            sym_short = sym_short[-6:]
            base_id = f"G{side_tag}L{lvl}_{sym_short}" if lvl is not None else f"G{side_tag}P{ptag}_{sym_short}"
            if unique:
                base_id = f"{base_id}{suffix}"
        return base_id[:36]

    def _parse_client_order_id(self, client_order_id: str) -> Optional[tuple]:
        """Parse client_order_id; return (side_tag, level) if matched."""
        if not client_order_id:
            return None
        m = re.match(r'^G([SB])L(-?\\d+)_', client_order_id)
        if not m:
            return None
        side_tag = m.group(1)
        level = int(m.group(2))
        return side_tag, level

    def _place_upper_grid_orders(self, symbol: str, grid_state: GridState) -> None:
        """挂上方网格订单(开空) - 使用价格作为标识"""
        if self._shorts_paused(grid_state):
            logger.info(f"{symbol} shorts paused; skip upper grid placement")
            return
        grid_margin = self.config.position.grid_margin

        for level in grid_state.grid_prices.get_upper_levels():
            try:
                price = self._quantize_price(
                    symbol, grid_state.grid_prices.grid_levels[level], side='sell'
                )
                if price in grid_state.upper_orders:
                    continue
                client_order_id = self._make_client_order_id(
                    symbol, "sell", level=level, price=price, entry_price=grid_state.entry_price, unique=True
                )
                amount = self._calculate_amount(symbol, grid_margin, price)

                logger.debug(f"挂上方网格 @ {price:.6f}: {amount}张")
                order = self.connector.place_order_with_maker_retry(
                    symbol=symbol,
                    side='sell',  # 开空
                    amount=amount,
                    price=price,
                    order_type='limit',
                    post_only=True,
                    client_order_id=client_order_id,
                    max_retries=5
                )

                self._add_order_id(grid_state.upper_orders, price, order.order_id)

            except Exception as e:
                logger.warning(f"挂单失败 @ {price:.6f}: {e}")

    def _place_lower_grid_orders(self, symbol: str, grid_state: GridState) -> None:
        """挂下方网格订单(平空止盈)"""
        grid_margin = self.config.position.grid_margin

        for level in grid_state.grid_prices.get_lower_levels():
            try:
                price = self._quantize_price(
                    symbol, grid_state.grid_prices.grid_levels[level], side='buy'
                )
                if price in grid_state.lower_orders:
                    continue
                amount = self._calculate_amount(symbol, grid_margin, price)

                logger.debug(f"挂下方网格 Grid-{level}: {amount}张 × {price}")
                client_order_id = self._make_client_order_id(
                    symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
                )
                order = self.connector.place_order_with_maker_retry(
                    symbol=symbol,
                    side='buy',  # 平空止盈
                    amount=amount,
                    price=price,
                    order_type='limit',
                    post_only=True,
                    reduce_only=True,  # 强制只减仓
                    client_order_id=client_order_id,
                    max_retries=5
                )

                self._add_order_id(grid_state.lower_orders, price, order.order_id)

            except Exception as e:
                logger.warning(f"挂单失败 Grid-{level}: {e}")

    def _place_base_position_take_profit(self, symbol: str, grid_state: GridState) -> None:
        """Place layered take-profit orders for the tradable tranche only."""
        base_margin = self.config.position.base_margin
        grid_margin = self.config.position.grid_margin
        min_ratio = self.config.position.min_base_position_ratio

        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception:
            current_price = grid_state.entry_price

        lower_levels = self._get_lower_levels_by_proximity(grid_state, current_price)
        total_levels = len(lower_levels)

        allowed_levels = self._compute_base_tp_allowed_levels(total_levels)

        base_amount_per_level = self._calculate_amount(symbol, grid_margin, grid_state.entry_price)

        logger.info(f"Base TP orders: {allowed_levels}/{total_levels} levels, {base_amount_per_level:.1f} each")
        logger.info(f"Ratchet core keep ratio: {min_ratio*100:.0f}%")
        if allowed_levels <= 0:
            logger.info(f"{symbol} base TP levels = 0")
            return

        blocked_due_to_capacity = False
        for i, level in enumerate(sorted(lower_levels, reverse=True)):
            if i >= allowed_levels:
                break
            try:
                price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='buy')
                logger.debug(f"Base TP @ {price:.6f}: {base_amount_per_level:.1f}")
                is_safe, safe_amount, warning = self._validate_total_exposure_before_buy_order(
                    symbol, grid_state, base_amount_per_level
                )
                if not is_safe:
                    blocked_due_to_capacity = True
                    self._log_capacity_event(
                        symbol,
                        "base_tp_blocked",
                        f"{symbol} base TP blocked: {warning}",
                        level="warning"
                    )
                    break

                client_order_id = self._make_client_order_id(
                    symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
                )
                order = self._place_position_aware_buy_order(
                    symbol, price, base_amount_per_level, client_order_id=client_order_id
                )
                if order:
                    self._add_order_id(grid_state.lower_orders, price, order.order_id)

            except Exception as e:
                logger.warning(f"Base TP order failed @ {price:.6f}: {e}")

        success_count = self._count_orders(grid_state.lower_orders)
        logger.info(f"TP orders placed: {success_count}/{allowed_levels}")
        if success_count < allowed_levels:
            if blocked_due_to_capacity:
                self._log_capacity_event(
                    symbol,
                    "base_tp_incomplete",
                    f"{symbol} TP orders incomplete due to capacity: expected {allowed_levels}, actual {success_count}",
                    level="warning"
                )
            else:
                logger.error(f"TP orders incomplete: expected {allowed_levels}, actual {success_count}")
        elif success_count == 0:
            if blocked_due_to_capacity:
                self._log_capacity_event(
                    symbol,
                    "base_tp_zero",
                    f"{symbol} TP orders skipped due to capacity",
                    level="warning"
                )
            else:
                logger.error("No TP orders were placed successfully")

    def _place_lower_grid_order(self, symbol: str, grid_state: GridState, level: int) -> None:
        """挂下方网格订单(平空) - 仅用于重新挂上方成交前的基础止盈单"""
        if level not in grid_state.grid_prices.grid_levels:
            return

        base_price = self._quantize_price(
            symbol, grid_state.grid_prices.grid_levels[level], side='buy'
        )
        price = base_price
        grid_margin = self.config.position.grid_margin
        base_amount_per_level = self._calculate_amount(symbol, grid_margin, grid_state.entry_price)

        try:
            logger.debug(f"重新挂基础止盈单 Grid-{level}: {base_amount_per_level}张 × {price}")
            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',
                amount=base_amount_per_level,
                price=price,
                order_type='limit',
                post_only=True,
                reduce_only=True,  # 强制只减仓
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)

        except Exception as e:
            logger.warning(f"挂单失败 Grid-{level}: {e}")

    def _place_enhanced_lower_grid_order(self, symbol: str, grid_state: GridState, level: int) -> None:
        """挂下方止盈单（与开空单数量一致）"""
        if level not in grid_state.grid_prices.grid_levels:
            return

        base_price = self._quantize_price(
            symbol, grid_state.grid_prices.grid_levels[level], side='buy'
        )
        price = base_price

        # 🔧 FIX: 使用与开空单相同的数量（仅grid_margin）
        grid_margin = self.config.position.grid_margin
        amount = self._calculate_amount(symbol, grid_margin, price)

        try:
            logger.info(f"挂止盈单 Grid-{level}: {amount}张 × {price}")
            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                reduce_only=True,  # 强制只减仓
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)

        except Exception as e:
            logger.warning(f"挂止盈单失败 Grid-{level}: {e}")

    def _place_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        挂单个下方网格订单（用于滚动窗口添加新网格）
        注意：此函数被下方同名函数覆盖，实际不会被调用

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级（负数）
            price: 价格
        """
        try:
            # FIX: 使用与开空单相同的数量（仅grid_margin）
            self._place_enhanced_lower_grid_order(symbol, grid_state, level)
        except Exception as e:
            logger.warning(f"挂下方网格失败 Grid{level}: {e}")

    def _should_check_grid_repair(self, grid_state: GridState) -> bool:
        """
        判断是否应该检查网格修复（正常间隔10秒，恢复模式2秒）

        Args:
            grid_state: 网格状态

        Returns:
            bool: 是否应该检查
        """
        if not self.config.grid.repair_enabled:
            return False

        now = datetime.now(timezone.utc)
        elapsed = (now - grid_state.last_repair_check).total_seconds()

        # 恢复模式：如果完全没有基础止盈单，使用更短的间隔（2秒）
        is_recovery = self._count_base_tp_orders(grid_state) == 0
        repair_interval = 2 if is_recovery else self.config.grid.repair_interval

        return elapsed >= repair_interval

    def _repair_missing_grids(self, symbol: str, grid_state: GridState) -> None:
        """
        检查并补充缺失的网格订单（基于价格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
        """
        if not self._should_check_grid_repair(grid_state):
            return

        grid_state.last_repair_check = datetime.now(timezone.utc)

        # 获取当前市场价格
        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"{symbol} 获取价格失败，跳过修复: {e}")
            return

        max_gap_k = self.config.grid.repair_upper_max_gap_k
        max_gap_ratio = max_gap_k * self.config.grid.spacing if max_gap_k and max_gap_k > 0 else 0.0
        max_repair_upper = self.config.grid.repair_upper_max_count
        repaired_upper = 0

        def _upper_price_too_far(price: float) -> bool:
            if max_gap_ratio <= 0 or current_price <= 0:
                return False
            return (price - current_price) / current_price > max_gap_ratio

        # 查询当前挂单（避免重复）
        try:
            open_orders = self.connector.query_open_orders(symbol)
            open_order_ids = {order.order_id for order in open_orders}
            open_order_prices = {}
            for order in open_orders:
                if order.price is None:
                    continue
                q_price = self._quantize_price(symbol, order.price, side=order.side)
                open_order_prices[q_price] = order.order_id
        except Exception as e:
            logger.warning(f"{symbol} 查询挂单失败，跳过修复: {e}")
            return

        # 先对账：恢复遗失的订单状态（基于价格匹配）
        for order in open_orders:
            if order.price is None:
                continue
            order_price = self._quantize_price(symbol, order.price, side=order.side)

            # 检查是否应该在upper_orders中
            if order.side == 'sell':
                if _upper_price_too_far(order_price):
                    self._log_capacity_event(
                        symbol,
                        "repair_upper_far_cancel",
                        (
                            f"{symbol} cancel far upper order @ {order_price:.6f} "
                            f"(price {current_price:.6f}, gap>{max_gap_ratio*100:.1f}%)"
                        ),
                        level="warning",
                        interval=60
                    )
                    try:
                        self.connector.cancel_order(order.order_id, symbol)
                    except Exception as e:
                        logger.warning(f"{symbol} cancel far upper order failed @ {order_price:.6f}: {e}")
                    self._remove_order_id(grid_state.upper_orders, order_price, order.order_id)
                    continue
                # 检查价格是否接近任何预期的上方网格价格
                for level in grid_state.grid_prices.get_upper_levels():
                    target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='sell')
                    if abs(order_price - target_price) / target_price < 0.001:  # 0.1%容差
                        added = self._add_order_id(grid_state.upper_orders, order_price, order.order_id)
                        if added:
                            logger.info(f"{symbol} 恢复遗失的上方网格 @ {order_price:.6f}")
                        break

            # 检查是否应该在lower_orders中
            elif order.side == 'buy':
                # 检查价格是否接近任何预期的下方网格价格
                for level in grid_state.grid_prices.get_lower_levels():
                    target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='buy')
                    if abs(order_price - target_price) / target_price < 0.001:
                        added = self._add_order_id(grid_state.lower_orders, order_price, order.order_id)
                        if added:
                            logger.info(f"{symbol} 恢复遗失的下方网格 @ {order_price:.6f}")
                        break

        # 清理state中已失效的订单ID
        for price, order_id in list(self._iter_order_items(grid_state.upper_orders)):
            if order_id not in open_order_ids:
                self._remove_order_id(grid_state.upper_orders, price, order_id)
                logger.warning(f"{symbol} 检测到异常消失的上方订单 @ {price:.6f}")

        for price, order_id in list(self._iter_order_items(grid_state.lower_orders)):
            if order_id not in open_order_ids:
                self._remove_order_id(grid_state.lower_orders, price, order_id)
                logger.warning(f"{symbol} 检测到异常消失的下方订单 @ {price:.6f}")

        # 修复上方网格（检查所有预期的网格价格）
        pending_upper_prices = {
            self._quantize_price(symbol, fill.price, side='sell')
            for fill in grid_state.filled_upper_grids.values()
        }

        if self._shorts_paused(grid_state):
            logger.info(f"{symbol} shorts paused; skip repairing upper grids")
        else:
            for level in grid_state.grid_prices.get_upper_levels():
                target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='sell')

                # 检查是否缺失
                if target_price not in grid_state.upper_orders and target_price not in pending_upper_prices:
                    # 只有当市价低于目标价时才补充开空单
                    if current_price < target_price:
                        if _upper_price_too_far(target_price):
                            self._log_capacity_event(
                                symbol,
                                "repair_upper_far_skip",
                                (
                                    f"{symbol} skip far upper repair @ {target_price:.6f} "
                                    f"(price {current_price:.6f}, gap>{max_gap_ratio*100:.1f}%)"
                                ),
                                level="info",
                                interval=60
                            )
                            continue
                        if max_repair_upper > 0 and repaired_upper >= max_repair_upper:
                            self._log_capacity_event(
                                symbol,
                                "repair_upper_limit",
                                f"{symbol} upper repair limit reached ({max_repair_upper})",
                                level="info",
                                interval=60
                            )
                            break
                        logger.info(f"{symbol} 补充缺失的上方网格 @ {target_price:.6f}")
                        self._place_single_upper_grid_by_price(symbol, grid_state, target_price)
                        repaired_upper += 1

        # 修复下方网格（仅补足允许的基础止盈数量）
        base_tp_count = self._count_base_tp_orders(grid_state)
        lower_levels = self._get_lower_levels_by_proximity(grid_state, current_price)
        total_levels = len(lower_levels)
        allowed_levels = self._compute_base_tp_allowed_levels(total_levels)
        is_recovery = base_tp_count == 0

        if allowed_levels > 0:
            for level in lower_levels:
                if base_tp_count >= allowed_levels:
                    break
                target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='buy')
                order_ids = grid_state.lower_orders.get(target_price, [])
                has_base = any(order_id not in grid_state.tp_to_upper for order_id in order_ids)
                if has_base:
                    continue

                if not (is_recovery or current_price > target_price):
                    continue

                if is_recovery:
                    logger.info(f"{symbol} [恢复模式] 补充缺失的下方网格 @ {target_price:.6f}")
                else:
                    logger.info(f"{symbol} 补充缺失的下方网格 @ {target_price:.6f}")

                before_count = self._count_orders(grid_state.lower_orders)
                self._place_single_lower_grid_by_price(symbol, grid_state, target_price)
                after_count = self._count_orders(grid_state.lower_orders)
                if after_count > before_count:
                    base_tp_count += (after_count - before_count)
                else:
                    break

        self._repair_missing_tps(symbol, grid_state, open_order_ids)

    def _repair_missing_tps(
        self,
        symbol: str,
        grid_state: GridState,
        open_order_ids: Set[str]
    ) -> None:
        """Ensure every filled upper grid has a live TP order."""
        if not grid_state.filled_upper_grids:
            return

        for upper_order_id, fill_info in list(grid_state.filled_upper_grids.items()):
            if fill_info.matched_lower_price is None:
                continue
            tp_ids = [tp_id for tp_id, u_id in grid_state.tp_to_upper.items() if u_id == upper_order_id]
            if not tp_ids:
                tp_order = self._place_enhanced_lower_grid_by_price(
                    symbol, grid_state, fill_info.matched_lower_price, fill_info
                )
                if tp_order:
                    grid_state.tp_to_upper[tp_order.order_id] = upper_order_id
                    if hasattr(self, "db") and self.db:
                        self.db.save_grid_cycle({
                            "upper_order_id": upper_order_id,
                            "symbol": symbol,
                            "upper_price": fill_info.price,
                            "upper_amount": fill_info.amount,
                            "tp_order_id": tp_order.order_id,
                            "tp_price": fill_info.matched_lower_price,
                            "tp_amount": tp_order.amount,
                            "status": "open"
                        })

    def _restore_cycles_from_db(
        self,
        symbol: str,
        grid_state: GridState,
        open_orders: List[Order]
    ) -> None:
        """Restore pending upper->TP cycles from database."""
        if not hasattr(self, "db") or not self.db:
            return

        open_order_ids = {order.order_id for order in open_orders}
        open_order_map = {order.order_id: order for order in open_orders}
        cycles = self.db.load_open_grid_cycles(symbol)
        if not cycles:
            return

        for cycle in cycles:
            upper_order_id = cycle.get("upper_order_id")
            tp_order_id = cycle.get("tp_order_id")
            upper_price = cycle.get("upper_price") or 0.0
            upper_amount = cycle.get("upper_amount") or 0.0
            tp_price = cycle.get("tp_price")
            tp_amount = cycle.get("tp_amount") or 0.0

            if not upper_order_id:
                continue

            fill_info = grid_state.filled_upper_grids.get(upper_order_id)
            if not fill_info:
                fill_info = UpperGridFill(
                    price=upper_price,
                    amount=upper_amount,
                    fill_time=datetime.now(timezone.utc),
                    order_id=upper_order_id,
                    matched_lower_price=tp_price
                )
                grid_state.filled_upper_grids[upper_order_id] = fill_info

            if tp_order_id and tp_order_id in open_order_ids:
                tp_order = open_order_map.get(tp_order_id)
                if tp_order and tp_order.price is not None:
                    tp_price = self._quantize_price(symbol, tp_order.price, side=tp_order.side)
                elif tp_price is not None:
                    tp_price = self._quantize_price(symbol, tp_price, side='buy')
                if tp_price is not None:
                    self._add_order_id(grid_state.lower_orders, tp_price, tp_order_id)
                    grid_state.tp_to_upper[tp_order_id] = upper_order_id
                continue

            # TP order not open: check status or replace
            if tp_order_id:
                fetched = self._query_order_safe(symbol, tp_order_id)
                if fetched:
                    status = (fetched.status or "").lower()
                    if status in ("closed", "filled"):
                        self._complete_cycle_from_tp(
                            symbol, grid_state, upper_order_id, tp_order_id, fill_info
                        )
                        continue
                    if status in ("canceled", "cancelled", "rejected", "expired"):
                        grid_state.tp_to_upper.pop(tp_order_id, None)

            # Replace missing TP
            if fill_info.matched_lower_price is None:
                continue
            tp_order = self._place_enhanced_lower_grid_by_price(
                symbol, grid_state, fill_info.matched_lower_price, fill_info
            )
            if tp_order:
                grid_state.tp_to_upper[tp_order.order_id] = upper_order_id
                self.db.save_grid_cycle({
                    "upper_order_id": upper_order_id,
                    "symbol": symbol,
                    "upper_price": fill_info.price,
                    "upper_amount": fill_info.amount,
                    "tp_order_id": tp_order.order_id,
                    "tp_price": fill_info.matched_lower_price,
                    "tp_amount": tp_order.amount,
                    "status": "open"
                })
                continue

            for tp_order_id in list(tp_ids):
                if tp_order_id in open_order_ids:
                    continue

                fetched = self._query_order_safe(symbol, tp_order_id)
                if fetched:
                    status = (fetched.status or "").lower()
                    if status in ("closed", "filled"):
                        self._complete_cycle_from_tp(
                            symbol, grid_state, upper_order_id, tp_order_id, fill_info
                        )
                        continue
                    if status in ("canceled", "cancelled", "rejected", "expired"):
                        grid_state.tp_to_upper.pop(tp_order_id, None)

                # Replace missing/canceled TP
                tp_order = self._place_enhanced_lower_grid_by_price(
                    symbol, grid_state, fill_info.matched_lower_price, fill_info
                )
                if tp_order:
                    grid_state.tp_to_upper[tp_order.order_id] = upper_order_id
                    if hasattr(self, "db") and self.db:
                        self.db.save_grid_cycle({
                            "upper_order_id": upper_order_id,
                            "symbol": symbol,
                            "upper_price": fill_info.price,
                            "upper_amount": fill_info.amount,
                            "tp_order_id": tp_order.order_id,
                            "tp_price": fill_info.matched_lower_price,
                            "tp_amount": tp_order.amount,
                            "status": "open"
                        })

    def _repair_single_upper_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        修复单个上方网格

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级
            price: 目标价格
        """
        if self._shorts_paused(grid_state):
            logger.info(f"{symbol} shorts paused; skip repair upper Grid+{level} @ {price:.6f}")
            return
        try:
            price = self._quantize_price(symbol, price, side='sell')
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "sell", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 开空
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.upper_orders, price, order.order_id)
            logger.info(f"{symbol} 成功补充上方网格 Grid+{level}")

        except Exception as e:
            logger.warning(f"{symbol} 补充上方网格失败 Grid+{level}: {e}")

    def _repair_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        修复单个下方网格

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级
            price: 目标价格
        """
        try:
            # FIX: 使用与开空单相同的数量（仅grid_margin）
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',  # 平空止盈
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                reduce_only=True,  # 强制只减仓
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)
            logger.info(f"{symbol} 成功补充下方网格 Grid-{level}")

        except Exception as e:
            logger.warning(f"{symbol} 补充下方网格失败 Grid-{level}: {e}")

    def _try_extend_grid(self, symbol: str, grid_state: GridState, filled_price: float, is_upper: bool) -> None:
        """
        滚动窗口网格扩展（保持平衡）

        上方成交：在上方添加新网格，并添加对应止盈单
        下方成交：始终滚动窗口（重开空、移除最远上方、补下方保护）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            filled_price: 成交的网格价格
            is_upper: 是否为上方网格
        """
        # 检查是否启用动态扩展
        if not self.config.grid.dynamic_expansion:
            return

        # 获取当前价格
        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"{symbol} 获取当前价格失败，跳过网格扩展: {e}")
            return

        spacing = self.config.grid.spacing  # 0.015
        max_total_grids = self.config.grid.max_total_grids  # 30

        # 获取当前所有网格价格
        upper_prices = sorted(grid_state.upper_orders.keys())
        lower_prices = sorted(grid_state.lower_orders.keys(), reverse=True)
        total_grids = len(upper_prices) + len(lower_prices)

        if is_upper:  # 上方网格成交（价格上涨）
            if self._shorts_paused(grid_state):
                logger.info(f"{symbol} shorts paused; skip upper expansion")
                return
            # 🔧 FIX: 移除边界检查，每个上方网格成交都扩展

            # 检查是否达到数量限制（软限制，仅用于防止异常情况）
            if total_grids >= max_total_grids:
                logger.warning(f"{symbol} 网格数已达软限制 {max_total_grids}（当前{total_grids}），跳过扩展")
                return

            # 1. 在最高价格之上添加新的上方网格
            max_upper_price = max(upper_prices) if upper_prices else current_price
            new_upper_price = self._quantize_price(
                symbol, max_upper_price * (1 + spacing), side='sell'
            )
            self._place_single_upper_grid_by_price(symbol, grid_state, new_upper_price)
            logger.info(f"{symbol} 扩展：添加上方网格 @ {new_upper_price:.6f}")
            # NET: +1 short capacity (EXPANSION)

        else:  # 下方网格成交（价格下跌）
            # 方案A：任何下方成交都滚动窗口
            # 1) 重新开空以保持空头敞口
            if self._shorts_paused(grid_state):
                logger.info(f"{symbol} shorts paused; skip reopen on lower fill")
                # still extend lower protection
                reopen_placed = False
            else:
                reopen_price = self._quantize_price(
                    symbol, current_price * (1 + spacing), side='sell'
                )
                min_gap_ratio = max(0.0, self.config.grid.reopen_min_gap_ratio) * spacing
                if self._is_price_too_close(reopen_price, upper_prices, min_gap_ratio):
                    reopen_placed = False
                    logger.info(
                        f"{symbol} 滚动窗口：重新开空过近 @ {reopen_price:.6f} "
                        f"(min_gap={min_gap_ratio:.4f})，跳过"
                    )
                else:
                    reopen_placed = self._place_single_upper_grid_by_price(symbol, grid_state, reopen_price)
                    if reopen_placed:
                        logger.info(
                            f"{symbol} 滚动窗口：重新开空 @ {reopen_price:.6f} "
                            f"(成交价={filled_price:.6f})"
                        )

            # 2) 移除最远的上方网格（保持窗口大小）
            if reopen_placed and upper_prices:
                max_upper_price = max(upper_prices)
                self._remove_grid_by_price(symbol, grid_state, max_upper_price, is_upper=True)
                logger.info(f"{symbol} 滚动窗口：移除最远上方网格 @ {max_upper_price:.6f}")

            # 3) 在下方添加新网格（更低价格 - 保持下方保护）
            new_lower_price = self._quantize_price(
                symbol, current_price * (1 - spacing), side='buy'
            )
            self._place_single_lower_grid_by_price(symbol, grid_state, new_lower_price)
            logger.info(f"{symbol} 滚动窗口：添加下方保护 @ {new_lower_price:.6f}")

                # NET: +1 short (reopen), -1 short (remove), +1 lower → MAINTAINS SHORT EXPOSURE ✅

    def _place_single_upper_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        挂单个上方网格订单

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级（正数）
            price: 价格
        """
        if self._shorts_paused(grid_state):
            logger.info(f"{symbol} shorts paused; skip upper grid Grid+{level} @ {price:.6f}")
            return
        try:
            price = self._quantize_price(symbol, price, side='sell')
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "sell", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 开空
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.upper_orders, price, order.order_id)
            logger.info(f"{symbol} 成功挂上方网格 Grid+{level} @ {price:.6f}, {amount}张")

        except Exception as e:
            logger.warning(f"{symbol} 挂上方网格失败 Grid+{level}: {e}")

    def _place_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        挂单个下方网格订单（止盈单）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级（负数）
            price: 价格
        """
        try:
            price = self._quantize_price(symbol, price, side='buy')
            # 🔧 FIX: 使用与开空单相同的数量（仅grid_margin）
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',  # 平空止盈
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                reduce_only=True,  # 强制只减仓
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)
            logger.info(f"{symbol} 成功挂下方网格 Grid{level} @ {price:.6f}, {amount}张")

        except Exception as e:
            logger.warning(f"{symbol} 挂下方网格失败 Grid{level}: {e}")

    def _remove_grid_level(self, symbol: str, grid_state: GridState, level: int) -> None:
        """
        移除指定层级的网格（撤单+删除价格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 要移除的网格层级
        """
        # 如果有挂单，先撤销
        level_price = grid_state.grid_prices.grid_levels.get(level)
        if level_price is None:
            return

        if level > 0 and level_price in grid_state.upper_orders:
            order_ids = list(grid_state.upper_orders.get(level_price, []))
            for order_id in order_ids:
                try:
                    self.connector.cancel_order(order_id, symbol)
                    self._remove_order_id(grid_state.upper_orders, level_price, order_id)
                    logger.info(f"{symbol} 已撤销上方网格 Grid+{level} @ {level_price:.6f}")
                except Exception as e:
                    logger.warning(f"{symbol} 撤销上方网格失败 Grid+{level}: {e}")

        elif level < 0 and level_price in grid_state.lower_orders:
            order_ids = list(grid_state.lower_orders.get(level_price, []))
            for order_id in order_ids:
                try:
                    self.connector.cancel_order(order_id, symbol)
                    self._remove_order_id(grid_state.lower_orders, level_price, order_id)
                    logger.info(f"{symbol} 已撤销下方网格 Grid{level} @ {level_price:.6f}")
                except Exception as e:
                    logger.warning(f"{symbol} 撤销下方网格失败 Grid{level}: {e}")

        # 从价格字典中移除
        grid_state.grid_prices.remove_level(level)

    # ==================== 新增：基于价格的网格操作函数 ====================

    def _place_single_upper_grid_by_price(self, symbol: str, grid_state: GridState, price: float) -> bool:
        """
        Place a single upper grid order by price.
        """
        if self._shorts_paused(grid_state):
            logger.info(f"{symbol} shorts paused; skip upper grid @ {price:.6f}")
            return False
        try:
            base_price = self._quantize_price(symbol, price, side='sell')
            if base_price in grid_state.upper_orders:
                logger.info(f"{symbol} upper grid already exists @ {base_price:.6f}, skip")
                return False
            level = self._calculate_grid_level(base_price, grid_state.entry_price, self.config.grid.spacing)
            if level not in grid_state.grid_prices.grid_levels:
                grid_state.grid_prices.add_level(level, base_price)

            client_order_id = self._make_client_order_id(
                symbol, "sell", level=level, price=base_price, entry_price=grid_state.entry_price, unique=True
            )
            price = base_price
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.upper_orders, price, order.order_id)
            logger.info(f"{symbol} upper grid order placed @ {price:.6f}, {amount} contracts")
            return True

        except Exception as e:
            logger.warning(f"{symbol} upper grid order failed @ {price:.6f}: {e}")
            return False

    def _place_single_lower_grid_by_price(self, symbol: str, grid_state: GridState, price: float) -> None:
        """
        挂单个下方网格订单（基础止盈，基于价格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            price: 价格
        """
        try:
            base_price = self._quantize_price(symbol, price, side='buy')  # tick size
            level = self._calculate_grid_level(base_price, grid_state.entry_price, self.config.grid.spacing)
            if level not in grid_state.grid_prices.grid_levels:
                grid_state.grid_prices.add_level(level, base_price)

            # 仅基础止盈（基础仓位的1/total_levels）
            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=base_price, entry_price=grid_state.entry_price, unique=True
            )
            price = base_price
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, grid_state.entry_price)

            # 验证总仓位不会超限
            is_safe, safe_amount, warning = self._validate_total_exposure_before_buy_order(
                symbol, grid_state, amount
            )

            if not is_safe:
                self._log_capacity_event(
                    symbol,
                    "lower_grid_blocked",
                    f"{symbol} 拒绝挂下方网格 @ {price:.6f}: {warning}",
                    level="warning"
                )
                return

            if safe_amount < amount:
                logger.info(f"{symbol} 调整下方网格数量: {amount:.2f} → {safe_amount:.2f}张")
                amount = safe_amount

            # 使用仓位感知买单
            order = self._place_position_aware_buy_order(
                symbol, price, amount, client_order_id=client_order_id
            )

            if order:
                self._add_order_id(grid_state.lower_orders, price, order.order_id)
                logger.info(f"{symbol} 成功挂下方网格（基础） @ {price:.6f}, {amount}张")

        except Exception as e:
            logger.warning(f"{symbol} 挂下方网格失败 @ {price:.6f}: {e}")

    def _place_enhanced_lower_grid_by_price(
        self,
        symbol: str,
        grid_state: GridState,
        price: float,
        upper_fill: UpperGridFill
    ) -> Optional[Order]:
        """
        挂止盈单（与开空单数量一致，基于价格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            price: 价格
            upper_fill: 对应的上方开仓信息
        """
        try:
            base_price = self._quantize_price(symbol, price, side='buy')  # tick size
            level = self._calculate_grid_level(base_price, grid_state.entry_price, self.config.grid.spacing)
            if level not in grid_state.grid_prices.grid_levels:
                grid_state.grid_prices.add_level(level, base_price)

            # FIX: 使用与开空单相同的数量（仅grid_margin）
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, base_price)

            logger.debug(f"{symbol} 止盈单: {amount}张")

            # 验证总仓位不会超限
            is_safe, safe_amount, warning = self._validate_total_exposure_before_buy_order(
                symbol, grid_state, amount
            )

            if not is_safe:
                logger.error(f"{symbol} 拒绝挂止盈单 @ {base_price:.6f}: {warning}")
                return None

            if safe_amount < amount:
                logger.warning(f"{symbol} 调整止盈数量: {amount:.2f} → {safe_amount:.2f}张")
                amount = safe_amount

            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=base_price, entry_price=grid_state.entry_price, unique=True
            )
            price = base_price

            order = self._place_position_aware_buy_order(
                symbol, price, amount, client_order_id=client_order_id
            )

            if order:
                self._add_order_id(grid_state.lower_orders, price, order.order_id)
                logger.info(f"{symbol} 成功挂止盈单 @ {price:.6f}, {amount}张")
                return order

        except Exception as e:
            logger.warning(f"{symbol} 挂止盈单失败 @ {price:.6f}: {e}")
        return None

    def _remove_grid_by_price(self, symbol: str, grid_state: GridState, price: float, is_upper: bool) -> None:
        """
        移除指定价格的网格（撤单）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            price: 要移除的网格价格
            is_upper: 是否为上方网格
        """
        side = 'sell' if is_upper else 'buy'
        price = self._quantize_price(symbol, price, side=side)

        if is_upper and price in grid_state.upper_orders:
            try:
                order_ids = grid_state.upper_orders.get(price, [])
                if not order_ids:
                    return
                order_id = order_ids[0]
                self.connector.cancel_order(order_id, symbol)
                self._remove_order_id(grid_state.upper_orders, price, order_id)
                logger.info(f"{symbol} 已撤销上方网格 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"{symbol} 撤销上方网格失败 @ {price:.6f}: {e}")

        elif not is_upper and price in grid_state.lower_orders:
            try:
                order_ids = grid_state.lower_orders.get(price, [])
                if not order_ids:
                    return
                order_id = order_ids[0]
                self.connector.cancel_order(order_id, symbol)
                self._remove_order_id(grid_state.lower_orders, price, order_id)
                logger.info(f"{symbol} 已撤销下方网格 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"{symbol} 撤销下方网格失败 @ {price:.6f}: {e}")

    # ==================== 结束：基于价格的网格操作函数 ====================

    def _check_base_position_health(self, symbol: str, grid_state: GridState) -> None:
        """
        检查基础仓位健康度

        Args:
            symbol: 交易对
            grid_state: 网格状态
        """
        try:
            # 查询当前持仓
            positions = self.connector.query_positions()
            short_position = next((p for p in positions if p.symbol == symbol and p.side == 'short'), None)

            if not short_position:
                logger.error(f"{symbol} 基础仓位已完全平仓！触发紧急清理")
                # 取消所有订单
                try:
                    self.connector.cancel_all_orders(symbol)
                    logger.info(f"{symbol} 已取消所有订单")
                except Exception as e:
                    logger.error(f"{symbol} 取消订单失败: {e}")
                # 标记需要清理（在trading_bot中处理）
                grid_state.needs_cleanup = True
                return

            current_amount = abs(short_position.contracts)

            # 计算预期的基础仓位
            base_margin = self.config.position.base_margin
            expected_base = self._calculate_amount(symbol, base_margin, grid_state.entry_price)

            # 计算最小仓位
            min_ratio = self.config.position.min_base_position_ratio
            min_base = expected_base * min_ratio

            # 计算当前比例
            current_ratio = current_amount / expected_base

            if current_amount < min_base:
                logger.error(
                    f"{symbol} 基础仓位过低！"
                    f"当前: {current_amount:.1f}张 ({current_ratio*100:.1f}%), "
                    f"最小: {min_base:.1f}张 ({min_ratio*100:.0f}%)"
                )
            elif current_ratio < 0.5:
                logger.warning(
                    f"{symbol} 基础仓位偏低: {current_amount:.1f}张 ({current_ratio*100:.1f}%)"
                )
            else:
                logger.debug(
                    f"{symbol} 基础仓位健康: {current_amount:.1f}张 ({current_ratio*100:.1f}%)"
                )

        except Exception as e:
            logger.error(f"{symbol} 检查基础仓位失败: {e}")

    def update_grid_states(self) -> None:
        """更新所有网格状态"""
        for symbol, grid_state in self.grid_states.items():
            try:
                # 0. 断言检查：不允许多头仓位（每次都检查）
                self._assert_no_long_positions(symbol)

                # 1. 原有逻辑：检查订单成交
                self._update_single_grid(symbol, grid_state)

                # 2. 新增：检查并修复缺失的网格
                if grid_state.grid_integrity_validated:
                    self._repair_missing_grids(symbol, grid_state)

                # 3. 新增：定期对账（60秒间隔）
                self._reconcile_position_with_grids(symbol, grid_state)

                # 4. 新增：检查基础仓位健康度
                self._check_base_position_health(symbol, grid_state)

            except Exception as e:
                logger.error(f"更新网格状态失败 {symbol}: {e}")

        # 🔧 FIX: 添加运行时资金监控（每次更新后检查一次）
        self._validate_total_capital_usage()

    def _validate_total_capital_usage(self) -> None:
        """验证总资金使用不超过90%限制"""
        try:
            total_margin = 0.0
            for symbol, grid_state in self.grid_states.items():
                try:
                    position = self.position_mgr.get_symbol_position(symbol)
                    if position and position.total_margin_used:
                        total_margin += abs(position.total_margin_used)
                except Exception as e:
                    logger.warning(f"获取{symbol}保证金失败: {e}")

            # 获取资金分配器（通过position_manager）
            if hasattr(self.position_mgr, 'capital_allocator'):
                capital_allocator = self.position_mgr.capital_allocator
            else:
                # 如果没有capital_allocator，跳过验证
                return

            available_capital = capital_allocator.available_capital
            total_balance = capital_allocator.total_balance
            usage_pct = (total_margin / total_balance) * 100 if total_balance > 0 else 0

            if total_margin > available_capital:
                logger.error(
                    f"⚠️ 资金超限：使用 {total_margin:.2f} USDT ({usage_pct:.1f}%)，"
                    f"限制 {available_capital:.2f} USDT (90%)"
                )
            elif usage_pct > 85:
                logger.warning(
                    f"⚠️ 资金使用接近限制：{total_margin:.2f} USDT ({usage_pct:.1f}%)，"
                    f"限制 {available_capital:.2f} USDT (90%)"
                )

        except Exception as e:
            logger.warning(f"资金验证失败: {e}")

    def _update_single_grid(self, symbol: str, grid_state: GridState) -> None:
        """更新单个网格状态（基于价格）"""
        if self._maybe_soft_rebase(symbol, grid_state):
            return
        current_short_size = self._refresh_inventory_ratchet(symbol, grid_state, force_refresh=True)

        # 查询所有订单
        orders = {order.order_id: order for order in self.connector.query_open_orders(symbol)}

        # 检查是否需要初始化基础仓位的止盈单
        if not grid_state.lower_orders:
            # 查询实际持仓，判断基础仓位是否已成交
            positions = self.connector.query_positions()
            has_position = any(p.symbol == symbol and abs(p.contracts) > 0 for p in positions)

            if has_position:
                logger.info(f"检测到基础仓位已成交，挂分层止盈单: {symbol}")
                self._place_base_position_take_profit(symbol, grid_state)

        # 检查上方网格订单（基于价格）
        for price, order_id in list(self._iter_order_items(grid_state.upper_orders)):
            order = orders.get(order_id)

            if not order:
                fetched = self._query_order_safe(symbol, order_id)
                if fetched:
                    status = (fetched.status or "").lower()
                    if status in ("closed", "filled"):
                        order = fetched
                    elif status in ("canceled", "cancelled", "rejected", "expired"):
                        self._remove_order_id(grid_state.upper_orders, price, order_id)
                        logger.info(f"{symbol} 上方网格已取消 @ {price:.6f}, 等待补单")
                        continue
                    else:
                        continue
                else:
                    continue

            status = (order.status or "").lower()
            if status in ("closed", "filled"):
                # 订单成交
                logger.info(f"上方网格成交: {symbol} @ {price:.6f}")

                # 记录成交信息
                fill_info = UpperGridFill(
                    price=price,
                    amount=order.amount if order else 0,
                    fill_time=datetime.now(timezone.utc),
                    order_id=order_id,
                    matched_lower_price=self._quantize_price(
                        symbol, price * (1 - self.config.grid.spacing), side='buy'
                    )  # 预期的止盈价格(1x spacing)
                )
                grid_state.filled_upper_grids[order_id] = fill_info
                self._remove_order_id(grid_state.upper_orders, price, order_id)

                # 挂新的止盈单
                matched_lower_price = fill_info.matched_lower_price
                tp_order = self._place_enhanced_lower_grid_by_price(symbol, grid_state, matched_lower_price, fill_info)
                if tp_order:
                    grid_state.tp_to_upper[tp_order.order_id] = order_id
                    if hasattr(self, "db") and self.db:
                        self.db.save_grid_cycle({
                            "upper_order_id": order_id,
                            "symbol": symbol,
                            "upper_price": fill_info.price,
                            "upper_amount": fill_info.amount,
                            "tp_order_id": tp_order.order_id,
                            "tp_price": matched_lower_price,
                            "tp_amount": tp_order.amount,
                            "status": "open"
                        })

                # 尝试扩展网格
                self._try_extend_grid(symbol, grid_state, price, is_upper=True)

        # 检查下方网格订单（基于价格）
        for price, order_id in list(self._iter_order_items(grid_state.lower_orders)):
            order = orders.get(order_id)

            if not order:
                fetched = self._query_order_safe(symbol, order_id)
                if fetched:
                    status = (fetched.status or "").lower()
                    if status in ("closed", "filled"):
                        order = fetched
                    elif status in ("canceled", "cancelled", "rejected", "expired"):
                        self._remove_order_id(grid_state.lower_orders, price, order_id)
                        logger.info(f"{symbol} 下方网格已取消 @ {price:.6f}, 等待补单")
                        continue
                    else:
                        continue
                else:
                    continue

            status = (order.status or "").lower()
            if status in ("closed", "filled"):
                # 订单成交（止盈）
                logger.info(f"下方网格成交: {symbol} @ {price:.6f}")

                # 查找匹配的上方开仓（优先使用映射）
                matched_fill = None
                upper_order_id = grid_state.tp_to_upper.get(order_id)
                if upper_order_id:
                    matched_fill = grid_state.filled_upper_grids.get(upper_order_id)
                if not matched_fill:
                    matched_fill = self._find_matched_upper_fill(grid_state, price)

                if matched_fill:
                    self._complete_cycle_from_tp(
                        symbol,
                        grid_state,
                        upper_order_id or matched_fill.order_id,
                        order_id,
                        matched_fill
                    )

                self._remove_order_id(grid_state.lower_orders, price, order_id)

                # 尝试扩展网格
                self._try_extend_grid(symbol, grid_state, price, is_upper=False)

        grid_state.last_update = datetime.now(timezone.utc)

    def close_grid(self, symbol: str, reason: str = "manual") -> None:
        """
        关闭网格

        Args:
            symbol: 交易对
            reason: 关闭原因
        """
        if symbol not in self.grid_states:
            return

        logger.info(f"关闭网格: {symbol}, 原因: {reason}")

        grid_state = self.grid_states[symbol]

        # 撤销所有挂单
        all_order_ids = []
        for _, order_id in self._iter_order_items(grid_state.upper_orders):
            all_order_ids.append(order_id)
        for _, order_id in self._iter_order_items(grid_state.lower_orders):
            all_order_ids.append(order_id)
        for order_id in all_order_ids:
            try:
                self.connector.cancel_order(order_id, symbol)
            except Exception as e:
                logger.warning(f"撤单失败: {e}")

        # 市价平掉所有持仓
        position = self.position_mgr.get_symbol_position(symbol)
        if position and position.base_position:
            try:
                size = position.base_position.size
                self.connector.place_order(
                    symbol=symbol,
                    side='buy',  # 平空
                    amount=size,
                    order_type='market',
                    reduce_only=True  # 强制只减仓
                )
                logger.info(f"市价平仓: {symbol}, 数量={size}")
            except Exception as e:
                logger.error(f"平仓失败: {e}")

        # 移除网格状态
        del self.grid_states[symbol]

        # 移除仓位
        self.position_mgr.remove_position(symbol)

    def recover_grid_from_position(self, symbol: str, entry_price: float) -> bool:
        """
        从现有持仓恢复网格状态（使用持仓成本价重建网格）

        Args:
            symbol: 交易对
            entry_price: 数据库中保存的入场价（将被忽略）

        Returns:
            是否成功
        """
        try:
            # 🔧 NEW: 查询当前持仓的实际成本价
            positions = self.connector.query_positions()
            short_position = next((p for p in positions if p.symbol == symbol and p.side == 'short'), None)

            if not short_position:
                logger.error(f"恢复网格失败: {symbol} 未找到空头持仓")
                return False

            # 使用持仓的实际成本价作为entry_price
            actual_entry_price = short_position.entry_price
            logger.info(
                f"恢复网格状态: {symbol}\n"
                f"  数据库entry_price: {entry_price:.6f}\n"
                f"  持仓成本价: {actual_entry_price:.6f}\n"
                f"  使用持仓成本价重建网格"
            )

            # 如果已经有grid_state，跳过
            if symbol in self.grid_states:
                logger.info(f"网格状态已存在: {symbol}")
                return True

            # 🔧 使用持仓成本价计算网格价格
            grid_prices = self.calculate_grid_prices(actual_entry_price)

            # 创建网格状态
            grid_state = GridState(
                symbol=symbol,
                entry_price=actual_entry_price,  # 使用持仓成本价
                grid_prices=grid_prices,
                core_ratio=self.config.position.min_base_position_ratio
            )
            self._refresh_inventory_ratchet(
                symbol,
                grid_state,
                current_short_size=short_position.size,
                force_refresh=False
            )

            # 查询现有挂单
            open_orders = self.connector.query_open_orders(symbol)

            # 如果没有挂单，重新挂网格单
            if not open_orders:
                logger.info(f"未发现挂单，重新挂上方网格: {symbol}")
                self.grid_states[symbol] = grid_state

                # 挂上方开空单
                self._place_upper_grid_orders(symbol, grid_state)

                # 挂基础仓位的分层止盈单（恢复时持仓已存在）
                logger.info(f"挂基础仓位分层止盈单: {symbol}")
                self._place_base_position_take_profit(symbol, grid_state)

                # 标记网格为已验证
                grid_state.grid_integrity_validated = True
            else:
                logger.info(f"发现{len(open_orders)}个挂单，恢复网格状态")

                # 解析现有订单，恢复upper_orders/lower_orders
                for order in open_orders:
                    if order.price is None:
                        continue

                    parsed = self._parse_client_order_id(order.client_order_id)
                    order_price = self._quantize_price(symbol, order.price, side=order.side)
                    if parsed:
                        side_tag, level = parsed
                        if level not in grid_state.grid_prices.grid_levels:
                            grid_state.grid_prices.add_level(level, order_price)
                        if side_tag == 'S':
                            self._add_order_id(grid_state.upper_orders, order_price, order.order_id)
                            logger.info(f"  恢复上方网格订单 @ {order_price:.6f} (Grid{level})")
                        else:
                            self._add_order_id(grid_state.lower_orders, order_price, order.order_id)
                            logger.info(f"  恢复下方网格订单 @ {order_price:.6f} (Grid{level})")
                        continue

                    # fallback: price matching
                    if order.side == 'sell':
                        for level in grid_state.grid_prices.get_upper_levels():
                            target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='sell')
                            if abs(order_price - target_price) / target_price < 0.001:  # 0.1%容差
                                self._add_order_id(grid_state.upper_orders, order_price, order.order_id)
                                logger.info(f"  恢复上方网格订单 @ {order_price:.6f} (Grid{level})")
                                break
                    elif order.side == 'buy':
                        for level in grid_state.grid_prices.get_lower_levels():
                            target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='buy')
                            if abs(order_price - target_price) / target_price < 0.001:
                                self._add_order_id(grid_state.lower_orders, order_price, order.order_id)
                                logger.info(f"  恢复下方网格订单 @ {order_price:.6f} (Grid{level})")
                                break

                upper_order_count = self._count_orders(grid_state.upper_orders)
                lower_order_count = self._count_orders(grid_state.lower_orders)
                logger.info(f"订单恢复完成: {upper_order_count}个上方网格, {lower_order_count}个下方网格")
                self.grid_states[symbol] = grid_state

                # 恢复未完成的上方->止盈循环
                self._restore_cycles_from_db(symbol, grid_state, open_orders)

                # 补充缺失的订单
                missing_upper = max(len(grid_state.grid_prices.get_upper_levels()) - len(grid_state.upper_orders), 0)

                min_ratio = self.config.position.min_base_position_ratio
                closeable_ratio = 1.0 - min_ratio
                total_lower_levels = len(grid_state.grid_prices.get_lower_levels())
                allowed_lower_levels = int(total_lower_levels * closeable_ratio)
                missing_lower = max(allowed_lower_levels - len(grid_state.lower_orders), 0)

                if missing_upper > 0:
                    logger.info(f"检测到{missing_upper}个缺失的上方网格订单，开始补充...")
                    self._place_upper_grid_orders(symbol, grid_state)

                if missing_lower > 0:
                    logger.info(f"检测到{missing_lower}个缺失的下方网格订单，开始补充...")
                    self._place_base_position_take_profit(symbol, grid_state)
                    logger.info(f"恢复后止盈单数量: {self._count_orders(grid_state.lower_orders)}/{allowed_lower_levels}")

                # 标记网格为已验证（允许后续修复机制运行）
                grid_state.grid_integrity_validated = True

            logger.info(f"网格恢复完成: {symbol}")
            return True

        except Exception as e:
            logger.error(f"网格恢复失败: {symbol}: {e}")
            return False

    def _calculate_amount(self, symbol: str, margin: float, price: float) -> float:
        """
        计算下单数量

        Args:
            symbol: 交易对
            margin: 保证金
            price: 价格

        Returns:
            合约数量
        """
        leverage = self.config.account.leverage
        # 名义价值 = 保证金 × 杠杆
        notional = margin * leverage
        # 合约数量 = 名义价值 / 价格
        amount = notional / price

        # 获取精度
        try:
            market_info = self.connector.get_market_info(symbol)
            precision = market_info['amount_precision']
            amount = round(amount, precision)
        except:
            amount = round(amount, 3)

        return amount

    def _get_tick_size(self, symbol: str) -> float:
        """Get cached tick size for a symbol."""
        now = time.time()
        cached = self._tick_size_cache.get(symbol)
        if cached:
            tick_size, ts = cached
            if now - ts < self._tick_size_cache_ttl:
                return tick_size

        tick_size = None
        try:
            market_info = self.connector.get_market_info(symbol)
            price_precision = market_info.get('price_precision', 8)
            if isinstance(price_precision, int):
                tick_size = 10 ** (-price_precision)
            else:
                tick_size = float(price_precision)
        except Exception:
            tick_size = 1e-8

        if not tick_size or tick_size <= 0:
            tick_size = 1e-8

        self._tick_size_cache[symbol] = (tick_size, now)
        return tick_size

    def _quantize_price(self, symbol: str, price: float, side: Optional[str] = None) -> float:
        """
        Quantize price to the symbol's tick size.
        side='buy' -> round down, side='sell' -> round up, else nearest.
        """
        tick_size = self._get_tick_size(symbol)
        if tick_size <= 0:
            return round(price, 8)

        d_price = Decimal(str(price))
        d_tick = Decimal(str(tick_size))

        if side == 'sell':
            steps = (d_price / d_tick).to_integral_value(rounding=ROUND_UP)
        elif side == 'buy':
            steps = (d_price / d_tick).to_integral_value(rounding=ROUND_DOWN)
        else:
            steps = (d_price / d_tick).to_integral_value()

        quant = steps * d_tick
        return float(quant)

    def _shift_price_by_tick(self, symbol: str, price: float, side: Optional[str]) -> float:
        """Shift price by one tick to avoid collisions."""
        tick_size = self._get_tick_size(symbol)
        if tick_size <= 0:
            return price

        d_price = Decimal(str(price))
        d_tick = Decimal(str(tick_size))

        if side == 'sell':
            return float(d_price + d_tick)
        if side == 'buy':
            shifted = d_price - d_tick
            return float(shifted) if shifted > 0 else price
        return float(d_price + d_tick)

    def _resolve_price_collision(
        self,
        symbol: str,
        price: float,
        side: str,
        grid_state: GridState,
        open_orders: Optional[List[Order]] = None,
        max_steps: int = 10
    ) -> float:
        """Resolve price collisions by shifting one tick at a time."""
        candidate = self._quantize_price(symbol, price, side=side)
        open_orders = open_orders if open_orders is not None else self._get_open_orders_safe(symbol)
        open_prices: Set[float] = set()
        for order in open_orders:
            if order.price is None:
                continue
            open_prices.add(self._quantize_price(symbol, order.price, side=order.side))

        for _ in range(max_steps + 1):
            if (
                candidate not in grid_state.upper_orders
                and candidate not in grid_state.lower_orders
                and candidate not in open_prices
            ):
                return candidate
            candidate = self._shift_price_by_tick(symbol, candidate, side)

        logger.warning(f"{symbol} price collision unresolved after {max_steps} ticks, use {candidate:.6f}")
        return candidate

    def _get_cached_short_position(self, symbol: str, force_refresh: bool = False):
        """
        获取缓存的空头仓位（减少API调用）

        Args:
            symbol: 交易对
            force_refresh: 是否强制刷新（忽略缓存）

        Returns:
            Position对象，如果找不到则返回None
        """
        now = datetime.now(timezone.utc)

        # 检查缓存
        if not force_refresh and symbol in self._position_cache:
            cached_pos, timestamp = self._position_cache[symbol]
            age = (now - timestamp).total_seconds()

            if age < self._cache_ttl:
                logger.debug(f"{symbol} 使用缓存仓位 (缓存年龄: {age:.1f}秒)")
                return cached_pos

        # 缓存失效或不存在，查询新数据
        try:
            positions = self.connector.query_positions()

            # 🔍 调试日志：打印所有仓位信息
            logger.info(f"{symbol} 查询到 {len(positions)} 个仓位:")
            for idx, p in enumerate(positions):
                logger.info(
                    f"  [{idx}] symbol={p.symbol}, side={p.side}, size={p.size}, "
                    f"contracts={p.contracts}, entry_price={p.entry_price}"
                )

            # 查找空头仓位（使用side字段，更可靠）
            short_pos = next((p for p in positions if p.symbol == symbol and p.side == 'short'), None)

            if short_pos:
                # 更新缓存
                self._position_cache[symbol] = (short_pos, now)
                logger.debug(f"{symbol} 刷新仓位缓存: {short_pos.size}张 @ {short_pos.entry_price}")
                return short_pos
            else:
                logger.warning(f"{symbol} ⚠️ 未找到空头仓位！")
                logger.warning(f"  查询条件: symbol={symbol}, side='short'")

                # 尝试放宽条件：只匹配symbol
                any_pos = next((p for p in positions if p.symbol == symbol), None)
                if any_pos:
                    logger.warning(
                        f"  ⚠️ 找到匹配symbol的仓位，但side不是'short': "
                        f"side={any_pos.side}, size={any_pos.size}"
                    )
                else:
                    logger.warning(f"  ⚠️ 完全没有匹配symbol的仓位")

                return None

        except Exception as e:
            logger.error(f"{symbol} 查询仓位失败: {e}")

            # 如果查询失败，尝试返回过期缓存（总比没有好）
            if symbol in self._position_cache:
                cached_pos, timestamp = self._position_cache[symbol]
                age = (now - timestamp).total_seconds()
                logger.warning(f"{symbol} 使用过期缓存 (缓存年龄: {age:.1f}秒)")
                return cached_pos

            return None


    def _validate_total_exposure_before_buy_order(
        self,
        symbol: str,
        grid_state: GridState,
        new_order_amount: float
    ) -> tuple:
        """Validate that new buy orders won't eat into ratcheted core inventory."""
        short_position = self._get_cached_short_position(symbol)
        if not short_position:
            return False, 0.0, "no short position"

        current_short_size = self._refresh_inventory_ratchet(
            symbol,
            grid_state,
            current_short_size=short_position.size,
            force_refresh=False
        )

        try:
            open_orders = self.connector.query_open_orders(symbol)
            pending_lower_total = self._get_pending_lower_order_amount(symbol, grid_state, open_orders=open_orders)
        except Exception:
            pending_lower_total = 0.0

        max_closeable = max(current_short_size - grid_state.core_target_size, 0.0)
        available_capacity = max(max_closeable - pending_lower_total, 0.0)

        safe_amount = min(new_order_amount, available_capacity)
        if max_closeable <= 0:
            warning = (
                f"core inventory locked: current={current_short_size:.2f}, "
                f"core={grid_state.core_target_size:.2f}"
            )
            return False, 0.0, warning
        if available_capacity <= 0:
            warning = (
                f"pending buy orders fully consume tradable inventory: "
                f"tradable={max_closeable:.2f}, pending={pending_lower_total:.2f}"
            )
            return False, 0.0, warning

        ratio = (pending_lower_total + safe_amount) / max_closeable if max_closeable > 0 else 0
        if safe_amount < new_order_amount:
            warning = (
                f"buy capacity limited by ratchet: tradable={available_capacity:.2f}/{max_closeable:.2f} "
                f"target={new_order_amount:.2f} safe={safe_amount:.2f} ({ratio*100:.1f}%)"
            )
            return True, safe_amount, warning
        elif ratio > 0.90:
            warning = f"lower grid near saturation: {ratio*100:.1f}%"
            return True, safe_amount, warning
        else:
            return True, safe_amount, ""
    def _reconcile_position_with_grids(self, symbol: str, grid_state: GridState) -> None:
        """
        定期对账：验证持仓与网格状态一致

        每60秒运行一次，检查：
        1. 当前空头仓位大小
        2. 所有pending lower order总额
        3. 如果lower总额 > 空头仓位 * 0.95: 记录警报（不强制撤单）

        Args:
            symbol: 交易对
            grid_state: 网格状态
        """
        now = datetime.now(timezone.utc)

        # 检查是否需要对账（60秒间隔）
        if symbol in self._last_reconciliation:
            elapsed = (now - self._last_reconciliation[symbol]).total_seconds()
            if elapsed < self._reconciliation_interval:
                return

        self._last_reconciliation[symbol] = now

        # 1. 获取当前空头仓位
        short_position = self._get_cached_short_position(symbol, force_refresh=True)

        if not short_position:
            logger.critical(f"{symbol} ⚠️ CRITICAL: 对账失败 - 无法找到空头仓位！")
            return

        short_size = self._refresh_inventory_ratchet(
            symbol,
            grid_state,
            current_short_size=short_position.size,
            force_refresh=False
        )
        tradable_size = max(short_size - grid_state.core_target_size, 0.0)

        # 2. 统计所有下方买单的总数量
        total_lower_amount = 0.0
        lower_order_count = 0

        try:
            open_orders = self.connector.query_open_orders(symbol)
            open_order_map = {order.order_id: order for order in open_orders}

            buy_orders = []
            for price, order_ids in grid_state.lower_orders.items():
                for order_id in order_ids:
                    order = open_order_map.get(order_id)
                    if order and order.side == 'buy':
                        total_lower_amount += order.amount
                        lower_order_count += 1
                        effective_price = self._quantize_price(symbol, order.price, side=order.side) if order.price is not None else price
                        buy_orders.append((effective_price, order))

        except Exception as e:
            logger.error(f"{symbol} 对账时查询挂单失败: {e}")
            return

        # 3. 如果 pending buy 超过可交易仓，优先取消更远的低价买单（保留更接近现价的订单）
        if total_lower_amount > tradable_size + 1e-9:
            excess = total_lower_amount - tradable_size
            cancelled_amount = 0.0
            cancelled_count = 0
            buy_orders.sort(key=lambda item: item[0])

            for effective_price, order in buy_orders:
                if excess - cancelled_amount <= 1e-9:
                    break
                try:
                    if self.connector.cancel_order(order.order_id, symbol):
                        self._remove_order_id(grid_state.lower_orders, effective_price, order.order_id)
                        grid_state.tp_to_upper.pop(order.order_id, None)
                        cancelled_amount += order.amount
                        cancelled_count += 1
                except Exception as e:
                    logger.warning(f"{symbol} 取消超额下方买单失败 @{effective_price:.6f}: {e}")

            if cancelled_count > 0:
                total_lower_amount = max(total_lower_amount - cancelled_amount, 0.0)
                lower_order_count = max(lower_order_count - cancelled_count, 0)
                logger.warning(
                    f"{symbol} ratchet 对账取消超额下方买单: {cancelled_count}个, "
                    f"{cancelled_amount:.2f}张; core={grid_state.core_target_size:.2f}, tradable={tradable_size:.2f}"
                )

        # 4. 计算平衡比例
        ratio = total_lower_amount / tradable_size if tradable_size > 0 else (1.0 if total_lower_amount > 0 else 0.0)

        # 5. 记录平衡状态
        if ratio > 0.95:
            logger.warning(
                f"{symbol} 下方买单过高: "
                f"{total_lower_amount:.2f}张 ({lower_order_count}个订单), "
                f"空头仓位={short_size:.2f}张, 核心仓={grid_state.core_target_size:.2f}张, "
                f"比例={ratio*100:.1f}%"
            )
        elif ratio > 0.85:
            logger.warning(
                f"{symbol} 下方买单接近交易仓上限: "
                f"{total_lower_amount:.2f}/{tradable_size:.2f}张 ({ratio*100:.1f}%)"
            )
        else:
            logger.info(
                f"{symbol} 仓位平衡健康: "
                f"下方买单={total_lower_amount:.2f}张 ({lower_order_count}个), "
                f"空头仓位={short_size:.2f}张, 核心仓={grid_state.core_target_size:.2f}张, "
                f"比例={ratio*100:.1f}%"
            )


    def _assert_no_long_positions(self, symbol: str) -> bool:
        """
        断言检查：绝对不允许多头仓位存在

        如果检测到多头：
        1. 记录CRITICAL日志
        2. 立即撤销所有下方买单
        3. 触发告警通知

        Args:
            symbol: 交易对

        Returns:
            bool: 是否检测到多头仓位（True = 检测到）
        """
        try:
            positions = self.connector.query_positions()
            long_position = next((p for p in positions if p.symbol == symbol and p.side == 'long'), None)

            if long_position:
                logger.critical(
                    f"{symbol} ⚠️⚠️⚠️ FORBIDDEN LONG POSITION DETECTED ⚠️⚠️⚠️\n"
                    f"  仓位大小: {long_position.size}张\n"
                    f"  开仓价格: {long_position.entry_price}\n"
                    f"  未实现盈亏: {long_position.unrealized_pnl}\n"
                    f"  这是严重错误！立即采取应急措施..."
                )

                # 应急措施：撤销所有下方买单
                if symbol in self.grid_states:
                    grid_state = self.grid_states[symbol]

                    cancelled_count = 0
                    for price, order_id in list(self._iter_order_items(grid_state.lower_orders)):
                        try:
                            self.connector.cancel_order(order_id, symbol)
                            cancelled_count += 1
                        except Exception as e:
                            logger.error(f"撤单失败 @ {price}: {e}")

                    grid_state.lower_orders.clear()
                    logger.critical(f"{symbol} 已撤销 {cancelled_count} 个下方买单")

                # TODO: 添加通知机制（email/webhook/telegram）
                return True

            return False

        except Exception as e:
            logger.error(f"{symbol} 检查多头仓位失败: {e}")
            return False


    def _calculate_grid_level(self, price: float, entry_price: float, spacing: float) -> int:
        """
        根据价格计算网格层级

        Args:
            price: 目标价格
            entry_price: 入场价格
            spacing: 网格间距

        Returns:
            网格层级（正数=上方，负数=下方，0=入场价）
        """
        if price >= entry_price:
            # 上方网格
            level = round(math.log(price / entry_price) / math.log(1 + spacing))
            return max(1, level)  # 至少为1
        else:
            # 下方网格
            level = round(math.log(price / entry_price) / math.log(1 - spacing))
            return min(-1, level)  # 至少为-1

    def _place_position_aware_buy_order(
        self,
        symbol: str,
        price: float,
        desired_amount: float,
        max_retries: int = 5,
        client_order_id: Optional[str] = None
    ) -> Optional[Order]:
        """Place buy order (reduce-only) with basic short position check."""
        if desired_amount <= 0:
            logger.warning(f"{symbol} invalid desired_amount={desired_amount}, skip buy order")
            return None

        short_position = self._get_cached_short_position(symbol)
        if not short_position:
            logger.critical(f"{symbol} no short position, skip buy order")
            return None

        short_amount = short_position.size
        safe_amount = min(desired_amount, short_amount)
        if safe_amount <= 0:
            return None

        try:
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',
                amount=safe_amount,
                price=price,
                order_type='limit',
                post_only=True,
                reduce_only=True,
                client_order_id=client_order_id,
                max_retries=max_retries
            )
            return order
        except Exception as e:
            logger.error(f"{symbol} buy order failed: {e}")
            return None
    def _find_matched_upper_fill(
        self,
        grid_state: GridState,
        lower_price: float
    ) -> Optional[UpperGridFill]:
        """
        查找匹配的上方开仓

        Args:
            grid_state: 网格状态
            lower_price: 下方成交价格

        Returns:
            匹配的上方开仓信息，如果没有则返回 None
        """
        if not grid_state.filled_upper_grids:
            return None

        # 查找 matched_lower_price 最接近 lower_price 的上方开仓
        best_match = None
        min_diff = float('inf')

        for fill_info in grid_state.filled_upper_grids.values():
            if fill_info.matched_lower_price is None:
                continue

            diff = abs(fill_info.matched_lower_price - lower_price)
            if diff < min_diff:
                min_diff = diff
                best_match = fill_info

        # 如果差异小于 0.5%，认为是匹配的
        if best_match and min_diff / lower_price < 0.005:
            return best_match

        return None
