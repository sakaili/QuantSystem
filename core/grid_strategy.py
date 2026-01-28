"""
网格策略执行器模块
Grid Strategy Module

实现网格交易策略逻辑
"""

import math
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
    upper_orders: Dict[float, str] = field(default_factory=dict)  # price -> order_id（改为基于价格）
    lower_orders: Dict[float, str] = field(default_factory=dict)  # price -> order_id（改为基于价格）
    filled_upper_grids: Dict[str, UpperGridFill] = field(default_factory=dict)  # order_id -> fill_info（记录开仓信息）
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # 网格完整性追踪（简化，移除 failures 字典）
    last_repair_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    grid_integrity_validated: bool = False  # 是否通过初始验证
    upper_success_rate: float = 0.0         # 上方网格创建成功率
    lower_success_rate: float = 0.0         # 下方网格创建成功率


class GridStrategy:
    """
    网格策略执行器

    管理网格订单的创建、监控和调整
    """

    def __init__(
        self,
        config: ConfigManager,
        connector: ExchangeConnector,
        position_mgr: PositionManager
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

        # 网格状态字典: symbol -> GridState
        self.grid_states: Dict[str, GridState] = {}

        logger.info("网格策略执行器初始化完成")

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

            # 1. 开基础仓位
            base_margin = self.config.position.base_margin
            base_amount = self._calculate_amount(symbol, base_margin, entry_price)

            logger.info(f"开基础仓位: {base_amount}张 × {entry_price}")
            base_order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 开空
                amount=base_amount,
                price=entry_price,  # 该参数会被内部根据订单簿调整
                order_type='limit',
                post_only=True,
                max_retries=5
            )

            base_order_id = base_order.order_id  # 保存用于可能的清理

            # 2. 轮询等待基础仓位成交（关键：确保持有空仓后再挂网格）
            logger.info(f"等待基础仓位成交: order_id={base_order_id}")
            base_filled = self._wait_for_order_fill(symbol, base_order_id, timeout=3600)  # 1小时超时

            if not base_filled:
                logger.error(f"基础仓位超时未成交，初始化失败")
                self._cleanup_failed_initialization(symbol, base_order_id)
                return False

            logger.info(f"✅ 基础仓位已成交，开始挂网格")

            # 3. 创建网格状态
            grid_state = GridState(
                symbol=symbol,
                entry_price=entry_price,
                grid_prices=grid_prices
            )

            self.grid_states[symbol] = grid_state

            # 4. 挂基础仓位的分层止盈单（先挂止盈保护）
            logger.info("挂基础仓位分层止盈单...")
            self._place_base_position_take_profit(symbol, grid_state)

            # 5. 挂上方网格订单(开空)
            logger.info("挂上方网格...")
            self._place_upper_grid_orders(symbol, grid_state)

            # 6. 验证网格创建成功率
            validation_passed, validation_msg = self._validate_grid_creation(symbol, grid_state)

            if not validation_passed:
                logger.error(f"网格验证失败: {validation_msg}, 开始清理...")
                self._cleanup_failed_initialization(symbol, base_order_id)
                return False

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
        lower_count = len(grid_state.grid_prices.get_lower_levels())
        upper_created = len(grid_state.upper_orders)
        lower_created = len(grid_state.lower_orders)

        upper_success_rate = upper_created / upper_count if upper_count > 0 else 0.0
        lower_success_rate = lower_created / lower_count if lower_count > 0 else 0.0

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
        if lower_success_rate < self.config.grid.min_success_rate_lower:
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
        for level, order_id in list(grid_state.upper_orders.items()):
            try:
                self.connector.cancel_order(order_id, symbol)
                logger.info(f"已撤销上方网格 Grid+{level}")
            except Exception as e:
                logger.warning(f"撤销订单失败: {e}")

        # 2. 撤销所有下方网格订单
        for level, order_id in list(grid_state.lower_orders.items()):
            try:
                self.connector.cancel_order(order_id, symbol)
                logger.info(f"已撤销下方网格 Grid-{level}")
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
                            reduce_only=False
                        )
                        logger.info(f"已市价平仓: {abs(pos.contracts)}张")
                    except Exception as e:
                        logger.error(f"平仓失败: {e}")
        except Exception as e:
            logger.warning(f"查询仓位失败: {e}")

        # 5. 移除网格状态
        del self.grid_states[symbol]
        logger.info(f"清理完成: {symbol}")

    def _place_upper_grid_orders(self, symbol: str, grid_state: GridState) -> None:
        """挂上方网格订单(开空) - 使用价格作为标识"""
        grid_margin = self.config.position.grid_margin

        for level in grid_state.grid_prices.get_upper_levels():
            try:
                price = round(grid_state.grid_prices.grid_levels[level], 8)
                amount = self._calculate_amount(symbol, grid_margin, price)

                logger.debug(f"挂上方网格 @ {price:.6f}: {amount}张")
                order = self.connector.place_order_with_maker_retry(
                    symbol=symbol,
                    side='sell',  # 开空
                    amount=amount,
                    price=price,
                    order_type='limit',
                    post_only=True,
                    max_retries=5
                )

                grid_state.upper_orders[price] = order.order_id  # 使用价格作为key

            except Exception as e:
                logger.warning(f"挂单失败 @ {price:.6f}: {e}")

    def _place_lower_grid_orders(self, symbol: str, grid_state: GridState) -> None:
        """挂下方网格订单(平空止盈)"""
        grid_margin = self.config.position.grid_margin

        for level in grid_state.grid_prices.get_lower_levels():
            try:
                price = grid_state.grid_prices.grid_levels[level]
                amount = self._calculate_amount(symbol, grid_margin, price)

                logger.debug(f"挂下方网格 Grid-{level}: {amount}张 × {price}")
                order = self.connector.place_order_with_maker_retry(
                    symbol=symbol,
                    side='buy',  # 平空止盈
                    amount=amount,
                    price=price,
                    order_type='limit',
                    post_only=True,
                    reduce_only=False,  # 不使用reduceOnly
                    max_retries=5
                )

                grid_state.lower_orders[level] = order.order_id

            except Exception as e:
                logger.warning(f"挂单失败 Grid-{level}: {e}")

    def _place_base_position_take_profit(self, symbol: str, grid_state: GridState) -> None:
        """挂基础仓位的分层止盈单（限制数量以保留最小仓位）"""
        base_margin = self.config.position.base_margin
        min_ratio = self.config.position.min_base_position_ratio

        # 计算可平仓的比例
        closeable_ratio = 1.0 - min_ratio  # 例如：1.0 - 0.3 = 0.7

        # 获取所有下方网格层级
        lower_levels = grid_state.grid_prices.get_lower_levels()
        total_levels = len(lower_levels)

        # 计算允许挂止盈单的层数（向下取整）
        allowed_levels = int(total_levels * closeable_ratio)  # 例如：10 × 0.7 = 7

        # 每层数量（仍然按总层数计算，保持每层数量一致）
        base_amount_per_level = self._calculate_amount(symbol, base_margin / total_levels, grid_state.entry_price)

        logger.info(f"挂基础仓位止盈单: {allowed_levels}/{total_levels}层, 每层{base_amount_per_level:.1f}张")
        logger.info(f"保留最小仓位: {min_ratio*100:.0f}% = {base_amount_per_level * (total_levels - allowed_levels):.1f}张")

        # 只挂允许的层数（从最近的开始，即从Grid-1开始）
        for i, level in enumerate(sorted(lower_levels, reverse=True)):
            if i >= allowed_levels:
                logger.debug(f"跳过 Grid{level}（保留最小仓位）")
                break

            try:
                price = round(grid_state.grid_prices.grid_levels[level], 8)
                logger.debug(f"挂基础止盈 @ {price:.6f}: {base_amount_per_level:.1f}张")

                # 使用仓位感知的买单
                order = self._place_position_aware_buy_order(symbol, price, base_amount_per_level)

                if order:
                    grid_state.lower_orders[price] = order.order_id  # 使用价格作为key

            except Exception as e:
                logger.warning(f"挂基础止盈单失败 @ {price:.6f}: {e}")

    def _place_lower_grid_order(self, symbol: str, grid_state: GridState, level: int) -> None:
        """挂下方网格订单(平空) - 仅用于重新挂上方成交前的基础止盈单"""
        if level not in grid_state.grid_prices.grid_levels:
            return

        price = grid_state.grid_prices.grid_levels[level]
        base_margin = self.config.position.base_margin
        base_amount_per_level = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)

        try:
            logger.debug(f"重新挂基础止盈单 Grid-{level}: {base_amount_per_level}张 × {price}")
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',
                amount=base_amount_per_level,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
                # 移除reduce_only参数
            )

            grid_state.lower_orders[level] = order.order_id

        except Exception as e:
            logger.warning(f"挂单失败 Grid-{level}: {e}")

    def _place_enhanced_lower_grid_order(self, symbol: str, grid_state: GridState, level: int) -> None:
        """挂增强的下方止盈单（基础仓位1/10 + 网格仓位）"""
        if level not in grid_state.grid_prices.grid_levels:
            return

        price = grid_state.grid_prices.grid_levels[level]

        # 计算总数量：基础仓位的1/10 + 网格仓位
        base_margin = self.config.position.base_margin
        grid_margin = self.config.position.grid_margin

        base_amount = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)
        grid_amount = self._calculate_amount(symbol, grid_margin, price)
        total_amount = base_amount + grid_amount

        try:
            logger.info(f"挂增强止盈单 Grid-{level}: {total_amount}张 × {price} (基础{base_amount}+网格{grid_amount})")
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',
                amount=total_amount,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
                # 移除reduce_only参数
            )

            grid_state.lower_orders[level] = order.order_id

        except Exception as e:
            logger.warning(f"挂增强止盈单失败 Grid-{level}: {e}")

    def _place_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        挂单个下方网格订单（用于滚动窗口添加新网格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级（负数）
            price: 价格
        """
        try:
            # 判断是否需要增强止盈单（如果对应的上方网格已成交）
            opposite_level = abs(level)
            if opposite_level in grid_state.filled_grids:
                # 使用增强止盈单
                self._place_enhanced_lower_grid_order(symbol, grid_state, level)
            else:
                # 使用基础止盈单
                self._place_lower_grid_order(symbol, grid_state, level)
        except Exception as e:
            logger.warning(f"挂下方网格失败 Grid{level}: {e}")

    def _should_check_grid_repair(self, grid_state: GridState) -> bool:
        """
        判断是否应该检查网格修复（至少间隔10秒）

        Args:
            grid_state: 网格状态

        Returns:
            bool: 是否应该检查
        """
        if not self.config.grid.repair_enabled:
            return False

        now = datetime.now(timezone.utc)
        elapsed = (now - grid_state.last_repair_check).total_seconds()
        return elapsed >= self.config.grid.repair_interval

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

        # 查询当前挂单（避免重复）
        try:
            open_orders = self.connector.query_open_orders(symbol)
            open_order_ids = {order.order_id for order in open_orders}
            open_order_prices = {round(order.price, 8): order.order_id for order in open_orders}
        except Exception as e:
            logger.warning(f"{symbol} 查询挂单失败，跳过修复: {e}")
            return

        # 先对账：恢复遗失的订单状态（基于价格匹配）
        for order in open_orders:
            order_price = round(order.price, 8)

            # 检查是否应该在upper_orders中
            if order.side == 'sell' and order_price not in grid_state.upper_orders:
                # 检查价格是否接近任何预期的上方网格价格
                for level in grid_state.grid_prices.get_upper_levels():
                    target_price = round(grid_state.grid_prices.grid_levels[level], 8)
                    if abs(order_price - target_price) / target_price < 0.001:  # 0.1%容差
                        grid_state.upper_orders[order_price] = order.order_id
                        logger.info(f"{symbol} 恢复遗失的上方网格 @ {order_price:.6f}")
                        break

            # 检查是否应该在lower_orders中
            elif order.side == 'buy' and order_price not in grid_state.lower_orders:
                # 检查价格是否接近任何预期的下方网格价格
                for level in grid_state.grid_prices.get_lower_levels():
                    target_price = round(grid_state.grid_prices.grid_levels[level], 8)
                    if abs(order_price - target_price) / target_price < 0.001:
                        grid_state.lower_orders[order_price] = order.order_id
                        logger.info(f"{symbol} 恢复遗失的下方网格 @ {order_price:.6f}")
                        break

        # 清理state中已失效的订单ID
        for price, order_id in list(grid_state.upper_orders.items()):
            if order_id not in open_order_ids:
                del grid_state.upper_orders[price]
                logger.warning(f"{symbol} 检测到异常消失的上方订单 @ {price:.6f}")

        for price, order_id in list(grid_state.lower_orders.items()):
            if order_id not in open_order_ids:
                del grid_state.lower_orders[price]
                logger.warning(f"{symbol} 检测到异常消失的下方订单 @ {price:.6f}")

        # 修复上方网格（检查所有预期的网格价格）
        for level in grid_state.grid_prices.get_upper_levels():
            target_price = round(grid_state.grid_prices.grid_levels[level], 8)

            # 检查是否缺失
            if target_price not in grid_state.upper_orders:
                # 只有当市价低于目标价时才补充开空单
                if current_price < target_price:
                    logger.info(f"{symbol} 补充缺失的上方网格 @ {target_price:.6f}")
                    self._place_single_upper_grid_by_price(symbol, grid_state, target_price)

        # 修复下方网格（检查所有预期的网格价格）
        for level in grid_state.grid_prices.get_lower_levels():
            target_price = round(grid_state.grid_prices.grid_levels[level], 8)

            # 检查是否缺失
            if target_price not in grid_state.lower_orders:
                # 只有当市价高于目标价时才补充平空止盈单
                if current_price > target_price:
                    logger.info(f"{symbol} 补充缺失的下方网格 @ {target_price:.6f}")
                    self._place_single_lower_grid_by_price(symbol, grid_state, target_price)

    def _repair_single_upper_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        修复单个上方网格

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级
            price: 目标价格
        """
        try:
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 开空
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
            )

            grid_state.upper_orders[level] = order.order_id
            grid_state.upper_grid_failures[level] = 0  # 重置失败计数
            logger.info(f"{symbol} 成功补充上方网格 Grid+{level}")

        except Exception as e:
            grid_state.upper_grid_failures[level] = grid_state.upper_grid_failures.get(level, 0) + 1
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
            # 判断是基础止盈还是增强止盈
            if level in grid_state.filled_grids:
                # 已有对应上方开仓，补充增强止盈
                base_margin = self.config.position.base_margin
                grid_margin = self.config.position.grid_margin
                base_amount = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)
                grid_amount = self._calculate_amount(symbol, grid_margin, price)
                total_amount = base_amount + grid_amount
            else:
                # 仅基础止盈
                base_margin = self.config.position.base_margin
                total_amount = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)

            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',  # 平空止盈
                amount=total_amount,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
                # 移除reduce_only参数
            )

            grid_state.lower_orders[level] = order.order_id
            grid_state.lower_grid_failures[level] = 0  # 重置失败计数
            logger.info(f"{symbol} 成功补充下方网格 Grid-{level}")

        except Exception as e:
            grid_state.lower_grid_failures[level] = grid_state.lower_grid_failures.get(level, 0) + 1
            logger.warning(f"{symbol} 补充下方网格失败 Grid-{level}: {e}")

    def _try_extend_grid(self, symbol: str, grid_state: GridState, filled_price: float, is_upper: bool) -> None:
        """
        滚动窗口网格扩展（基于价格）

        当边界网格成交时，在对侧靠近当前价格的位置添加新网格，同时移除最远端网格

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
            max_upper_price = max(upper_prices) if upper_prices else 0

            # 检查是否为边界网格（价格差异小于0.1%）
            if abs(filled_price - max_upper_price) / max_upper_price < 0.001:
                # 1. 在上方添加新网格（基于当前价格计算）
                new_upper_price = round(current_price * (1 + spacing), 8)
                self._place_single_upper_grid_by_price(symbol, grid_state, new_upper_price)
                logger.info(f"{symbol} 滚动窗口：添加上方网格 @ {new_upper_price:.6f}")

                # 2. 在下方添加新网格（靠近当前价格！）
                new_lower_price = round(current_price * (1 - spacing), 8)
                self._place_single_lower_grid_by_price(symbol, grid_state, new_lower_price)
                logger.info(f"{symbol} 滚动窗口：添加下方网格 @ {new_lower_price:.6f}")

                # 3. 如果总数超过30，移除最远的下方网格
                if total_grids + 2 > max_total_grids and lower_prices:
                    min_lower_price = min(lower_prices)
                    self._remove_grid_by_price(symbol, grid_state, min_lower_price, is_upper=False)
                    logger.info(f"{symbol} 移除最远下方网格 @ {min_lower_price:.6f}")

        else:  # 下方网格成交（价格下跌）
            min_lower_price = min(lower_prices) if lower_prices else float('inf')

            # 检查是否为边界网格（价格差异小于0.1%）
            if abs(filled_price - min_lower_price) / min_lower_price < 0.001:
                # 1. 在下方添加新网格（基于当前价格计算）
                new_lower_price = round(current_price * (1 - spacing), 8)
                self._place_single_lower_grid_by_price(symbol, grid_state, new_lower_price)
                logger.info(f"{symbol} 滚动窗口：添加下方网格 @ {new_lower_price:.6f}")

                # 2. 在上方添加新网格（靠近当前价格！）
                new_upper_price = round(current_price * (1 + spacing), 8)
                self._place_single_upper_grid_by_price(symbol, grid_state, new_upper_price)
                logger.info(f"{symbol} 滚动窗口：添加上方网格 @ {new_upper_price:.6f}")

                # 3. 如果总数超过30，移除最远的上方网格
                if total_grids + 2 > max_total_grids and upper_prices:
                    max_upper_price = max(upper_prices)
                    self._remove_grid_by_price(symbol, grid_state, max_upper_price, is_upper=True)
                    logger.info(f"{symbol} 移除最远上方网格 @ {max_upper_price:.6f}")

    def _place_single_upper_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        挂单个上方网格订单

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级（正数）
            price: 价格
        """
        try:
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 开空
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
            )

            grid_state.upper_orders[level] = order.order_id
            logger.info(f"{symbol} 成功挂上方网格 Grid+{level} @ {price:.6f}, {amount}张")

        except Exception as e:
            logger.warning(f"{symbol} 挂上方网格失败 Grid+{level}: {e}")

    def _place_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        挂单个下方网格订单（止盈单）

        注意：下方网格数量取决于是否有对应的上方仓位已成交

        Args:
            symbol: 交易对
            grid_state: 网格状态
            level: 网格层级（负数）
            price: 价格
        """
        try:
            # 判断是否有对应的上方仓位已成交
            # Grid-5 对应 Grid+5
            opposite_level = abs(level)

            if opposite_level in grid_state.filled_grids:
                # 有对应仓位，使用增强止盈（基础仓位1/10 + 网格仓位）
                base_margin = self.config.position.base_margin
                grid_margin = self.config.position.grid_margin
                base_amount = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)
                grid_amount = self._calculate_amount(symbol, grid_margin, price)
                total_amount = base_amount + grid_amount
                logger.debug(f"{symbol} 增强止盈: 基础{base_amount}张 + 网格{grid_amount}张")
            else:
                # 仅基础止盈
                base_margin = self.config.position.base_margin
                total_amount = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)

            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',  # 平空止盈
                amount=total_amount,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
                # 移除reduce_only参数
            )

            grid_state.lower_orders[level] = order.order_id
            logger.info(f"{symbol} 成功挂下方网格 Grid{level} @ {price:.6f}, {total_amount}张")

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
        if level > 0 and level in grid_state.upper_orders:
            try:
                order_id = grid_state.upper_orders[level]
                self.connector.cancel_order(order_id, symbol)
                del grid_state.upper_orders[level]
                logger.info(f"{symbol} 已撤销上方网格 Grid+{level}")
            except Exception as e:
                logger.warning(f"{symbol} 撤销上方网格失败 Grid+{level}: {e}")

        elif level < 0 and level in grid_state.lower_orders:
            try:
                order_id = grid_state.lower_orders[level]
                self.connector.cancel_order(order_id, symbol)
                del grid_state.lower_orders[level]
                logger.info(f"{symbol} 已撤销下方网格 Grid{level}")
            except Exception as e:
                logger.warning(f"{symbol} 撤销下方网格失败 Grid{level}: {e}")

        # 从价格字典中移除
        grid_state.grid_prices.remove_level(level)

    # ==================== 新增：基于价格的网格操作函数 ====================

    def _place_single_upper_grid_by_price(self, symbol: str, grid_state: GridState, price: float) -> None:
        """
        挂单个上方网格订单（基于价格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            price: 价格
        """
        try:
            price = round(price, 8)  # 统一精度

            # 检查是否已存在
            if price in grid_state.upper_orders:
                logger.debug(f"{symbol} 上方网格已存在 @ {price:.6f}")
                return

            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 开空
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
            )

            grid_state.upper_orders[price] = order.order_id
            logger.info(f"{symbol} 成功挂上方网格 @ {price:.6f}, {amount}张")

        except Exception as e:
            logger.warning(f"{symbol} 挂上方网格失败 @ {price:.6f}: {e}")

    def _place_single_lower_grid_by_price(self, symbol: str, grid_state: GridState, price: float) -> None:
        """
        挂单个下方网格订单（基础止盈，基于价格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            price: 价格
        """
        try:
            price = round(price, 8)  # 统一精度

            # 检查是否已存在
            if price in grid_state.lower_orders:
                logger.debug(f"{symbol} 下方网格已存在 @ {price:.6f}")
                return

            # 仅基础止盈（基础仓位的1/10）
            base_margin = self.config.position.base_margin
            amount = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)

            order = self._place_position_aware_buy_order(symbol, price, amount)

            if order:
                grid_state.lower_orders[price] = order.order_id
                logger.info(f"{symbol} 成功挂下方网格（基础） @ {price:.6f}, {amount}张")

        except Exception as e:
            logger.warning(f"{symbol} 挂下方网格失败 @ {price:.6f}: {e}")

    def _place_enhanced_lower_grid_by_price(
        self,
        symbol: str,
        grid_state: GridState,
        price: float,
        upper_fill: UpperGridFill
    ) -> None:
        """
        挂增强止盈单（基础止盈 + 网格仓位，基于价格）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            price: 价格
            upper_fill: 对应的上方开仓信息
        """
        try:
            price = round(price, 8)  # 统一精度

            # 计算增强止盈数量：基础仓位1/10 + 网格仓位
            base_margin = self.config.position.base_margin
            grid_margin = self.config.position.grid_margin
            base_amount = self._calculate_amount(symbol, base_margin / 10, grid_state.entry_price)
            grid_amount = self._calculate_amount(symbol, grid_margin, price)
            total_amount = base_amount + grid_amount

            logger.debug(f"{symbol} 增强止盈: 基础{base_amount}张 + 网格{grid_amount}张 = {total_amount}张")

            order = self._place_position_aware_buy_order(symbol, price, total_amount)

            if order:
                grid_state.lower_orders[price] = order.order_id
                logger.info(f"{symbol} 成功挂增强止盈单 @ {price:.6f}, {total_amount}张")

        except Exception as e:
            logger.warning(f"{symbol} 挂增强止盈单失败 @ {price:.6f}: {e}")

    def _remove_grid_by_price(self, symbol: str, grid_state: GridState, price: float, is_upper: bool) -> None:
        """
        移除指定价格的网格（撤单）

        Args:
            symbol: 交易对
            grid_state: 网格状态
            price: 要移除的网格价格
            is_upper: 是否为上方网格
        """
        price = round(price, 8)  # 统一精度

        if is_upper and price in grid_state.upper_orders:
            try:
                order_id = grid_state.upper_orders[price]
                self.connector.cancel_order(order_id, symbol)
                del grid_state.upper_orders[price]
                logger.info(f"{symbol} 已撤销上方网格 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"{symbol} 撤销上方网格失败 @ {price:.6f}: {e}")

        elif not is_upper and price in grid_state.lower_orders:
            try:
                order_id = grid_state.lower_orders[price]
                self.connector.cancel_order(order_id, symbol)
                del grid_state.lower_orders[price]
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
            short_position = next((p for p in positions if p.symbol == symbol and p.contracts < 0), None)

            if not short_position:
                logger.warning(f"{symbol} 基础仓位已完全平仓！")
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
                # 原有逻辑：检查订单成交
                self._update_single_grid(symbol, grid_state)

                # 新增：检查并修复缺失的网格
                if grid_state.grid_integrity_validated:
                    self._repair_missing_grids(symbol, grid_state)

                # 新增：检查基础仓位健康度
                self._check_base_position_health(symbol, grid_state)

            except Exception as e:
                logger.error(f"更新网格状态失败 {symbol}: {e}")

    def _update_single_grid(self, symbol: str, grid_state: GridState) -> None:
        """更新单个网格状态（基于价格）"""
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
        for price, order_id in list(grid_state.upper_orders.items()):
            order = orders.get(order_id)

            if not order or order.status == 'filled':
                # 订单成交
                logger.info(f"上方网格成交: {symbol} @ {price:.6f}")

                # 记录成交信息
                fill_info = UpperGridFill(
                    price=price,
                    amount=order.amount if order else 0,
                    fill_time=datetime.now(timezone.utc),
                    order_id=order_id,
                    matched_lower_price=round(price * (1 - 2 * self.config.grid.spacing), 8)  # 预期的止盈价格
                )
                grid_state.filled_upper_grids[order_id] = fill_info
                del grid_state.upper_orders[price]

                # 撤销对应的下方止盈单（如果存在）
                matched_lower_price = fill_info.matched_lower_price
                if matched_lower_price in grid_state.lower_orders:
                    old_order_id = grid_state.lower_orders[matched_lower_price]
                    try:
                        self.connector.cancel_order(old_order_id, symbol)
                        logger.info(f"撤销旧止盈单 @ {matched_lower_price:.6f}")
                    except Exception as e:
                        logger.warning(f"撤单失败: {e}")

                # 挂新的增强止盈单
                self._place_enhanced_lower_grid_by_price(symbol, grid_state, matched_lower_price, fill_info)

                # 尝试扩展网格
                self._try_extend_grid(symbol, grid_state, price, is_upper=True)

        # 检查下方网格订单（基于价格）
        for price, order_id in list(grid_state.lower_orders.items()):
            order = orders.get(order_id)

            if not order or order.status == 'filled':
                # 订单成交（止盈）
                logger.info(f"下方网格成交: {symbol} @ {price:.6f}")

                # 查找匹配的上方开仓
                matched_fill = self._find_matched_upper_fill(grid_state, price)

                if matched_fill:
                    # 完整循环：恢复网格
                    profit_pct = (matched_fill.price - price) / matched_fill.price * 100
                    logger.info(f"完整循环: 开仓 @ {matched_fill.price:.6f}, 平仓 @ {price:.6f}, 盈利 {profit_pct:.2f}%")

                    # 恢复上方网格
                    self._place_single_upper_grid_by_price(symbol, grid_state, matched_fill.price)

                    # 恢复下方基础止盈单
                    self._place_lower_grid_by_price(symbol, grid_state, price)

                    # 移除成交记录
                    del grid_state.filled_upper_grids[matched_fill.order_id]

                del grid_state.lower_orders[price]

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
        for order_id in list(grid_state.upper_orders.values()) + list(grid_state.lower_orders.values()):
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
                    reduce_only=False  # 不使用reduceOnly
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
        从现有持仓恢复网格状态

        Args:
            symbol: 交易对
            entry_price: 入场价

        Returns:
            是否成功
        """
        try:
            logger.info(f"恢复网格状态: {symbol} @ {entry_price}")

            # 如果已经有grid_state，跳过
            if symbol in self.grid_states:
                logger.info(f"网格状态已存在: {symbol}")
                return True

            # 计算网格价格
            grid_prices = self.calculate_grid_prices(entry_price)

            # 创建网格状态
            grid_state = GridState(
                symbol=symbol,
                entry_price=entry_price,
                grid_prices=grid_prices
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
            else:
                logger.info(f"发现{len(open_orders)}个挂单，恢复网格状态")
                # TODO: 解析现有订单，恢复upper_orders/lower_orders
                # 暂时简单处理：只恢复grid_state
                self.grid_states[symbol] = grid_state

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
        desired_amount: float
    ) -> Optional[Order]:
        """
        仓位感知的买单（绕过 reduce_only 限制）

        Args:
            symbol: 交易对
            price: 价格
            desired_amount: 期望数量

        Returns:
            订单对象，如果无法下单则返回 None
        """
        # 1. 查询当前持仓
        try:
            positions = self.connector.query_positions()
            short_position = next((p for p in positions if p.symbol == symbol and p.contracts < 0), None)

            if not short_position:
                logger.warning(f"{symbol} 没有空头仓位，跳过买单")
                return None

            short_amount = abs(short_position.contracts)

            # 2. 计算实际可以平仓的数量
            actual_amount = min(desired_amount, short_amount)

            if actual_amount < desired_amount * 0.9:  # 如果实际数量小于期望的90%
                logger.warning(f"{symbol} 空头仓位不足: 期望 {desired_amount:.2f}, 实际 {short_amount:.2f}, 下单 {actual_amount:.2f}")

            # 3. 下单（不使用 reduce_only）
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',
                amount=actual_amount,
                price=price,
                order_type='limit',
                post_only=True,
                max_retries=5
            )
            return order

        except Exception as e:
            logger.error(f"{symbol} 仓位感知买单失败: {e}")
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
