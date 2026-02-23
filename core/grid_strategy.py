"""
缃戞牸绛栫暐鎵ц鍣ㄦā鍧?
Grid Strategy Module

瀹炵幇缃戞牸浜ゆ槗绛栫暐閫昏緫
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
    """缃戞牸浠锋牸锛堝姩鎬佺綉鏍硷級"""
    entry_price: float
    grid_levels: Dict[int, float]  # level -> price锛宭evel鍙互鏄换鎰忔暣鏁帮紙姝ｆ暟=涓婃柟锛岃礋鏁?涓嬫柟锛?
    stop_loss_price: float         # 姝㈡崯浠锋牸
    spacing: float                 # 缃戞牸闂磋窛锛岀敤浜庡姩鎬佽绠?

    def get_upper_levels(self) -> List[int]:
        """鑾峰彇鎵€鏈変笂鏂圭綉鏍煎眰绾э紙姝ｆ暟锛?""
        return sorted([level for level in self.grid_levels.keys() if level > 0])

    def get_lower_levels(self) -> List[int]:
        """鑾峰彇鎵€鏈変笅鏂圭綉鏍煎眰绾э紙璐熸暟锛?""
        return sorted([level for level in self.grid_levels.keys() if level < 0], reverse=True)

    def add_level_above(self, max_level: int) -> int:
        """鍦ㄦ渶涓婃柟娣诲姞鏂扮綉鏍?""
        new_level = max_level + 1
        new_price = self.entry_price * ((1 + self.spacing) ** new_level)
        self.grid_levels[new_level] = new_price
        return new_level

    def add_level_below(self, min_level: int) -> int:
        """鍦ㄦ渶涓嬫柟娣诲姞鏂扮綉鏍?""
        new_level = min_level - 1
        new_price = self.entry_price * ((1 - self.spacing) ** abs(new_level))
        self.grid_levels[new_level] = new_price
        return new_level

    def add_level(self, level: int, price: float) -> None:
        """
        娣诲姞鎸囧畾灞傜骇鐨勭綉鏍?

        Args:
            level: 缃戞牸灞傜骇
            price: 浠锋牸
        """
        self.grid_levels[level] = price
        logger.debug(f"娣诲姞缃戞牸灞傜骇 Grid{level:+d} @ {price:.6f}")

    def remove_level(self, level: int) -> None:
        """绉婚櫎鎸囧畾灞傜骇鐨勭綉鏍?""
        if level in self.grid_levels:
            price = self.grid_levels[level]
            del self.grid_levels[level]
            logger.debug(f"绉婚櫎缃戞牸灞傜骇 Grid{level:+d} @ {price:.6f}")


@dataclass
class UpperGridFill:
    """涓婃柟缃戞牸鎴愪氦淇℃伅"""
    price: float          # 寮€浠撲环鏍?
    amount: float         # 寮€浠撴暟閲?
    fill_time: datetime   # 鎴愪氦鏃堕棿
    order_id: str         # 璁㈠崟ID
    matched_lower_price: Optional[float] = None  # 鍖归厤鐨勪笅鏂规鐩堜环鏍?


@dataclass
class GridState:
    """缃戞牸鐘舵€?""
    symbol: str
    entry_price: float
    grid_prices: GridPrices
    upper_orders: Dict[float, List[str]] = field(default_factory=dict)  # price -> [order_id]
    lower_orders: Dict[float, List[str]] = field(default_factory=dict)  # price -> [order_id]
    filled_upper_grids: Dict[str, UpperGridFill] = field(default_factory=dict)  # order_id -> fill_info锛堣褰曞紑浠撲俊鎭級
    tp_to_upper: Dict[str, str] = field(default_factory=dict)  # tp_order_id -> upper_order_id
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rebase_deviation_since: Optional[datetime] = None
    last_rebase_time: Optional[datetime] = None

    # 缃戞牸瀹屾暣鎬ц拷韪紙绠€鍖栵紝绉婚櫎 failures 瀛楀吀锛?
    last_repair_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    grid_integrity_validated: bool = False  # 鏄惁閫氳繃鍒濆楠岃瘉
    upper_success_rate: float = 0.0         # 涓婃柟缃戞牸鍒涘缓鎴愬姛鐜?
    lower_success_rate: float = 0.0         # 涓嬫柟缃戞牸鍒涘缓鎴愬姛鐜?
    needs_cleanup: bool = False             # 鏄惁闇€瑕佹竻鐞嗭紙浠撲綅瀹屽叏骞充粨鏃舵爣璁帮級


class GridStrategy:
    """
    缃戞牸绛栫暐鎵ц鍣?

    绠＄悊缃戞牸璁㈠崟鐨勫垱寤恒€佺洃鎺у拰璋冩暣
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
            config: 閰嶇疆绠＄悊鍣?
            connector: 浜ゆ槗鎵€杩炴帴鍣?
            position_mgr: 浠撲綅绠＄悊鍣?
        """
        self.config = config
        self.connector = connector
        self.position_mgr = position_mgr
        self.db = db

        # 缃戞牸鐘舵€佸瓧鍏? symbol -> GridState
        self.grid_states: Dict[str, GridState] = {}

        # 浠撲綅缂撳瓨锛屽噺灏慉PI璋冪敤棰戠巼: symbol -> (Position, timestamp)
        from .exchange_connector import Position
        self._position_cache: Dict[str, tuple] = {}
        self._cache_ttl = 5  # 缂撳瓨TTL (绉?

        # Tick size cache: symbol -> (tick_size, timestamp)
        self._tick_size_cache: Dict[str, tuple] = {}
        self._tick_size_cache_ttl = 60  # seconds

        # 瀵硅处鏃堕棿鎴? symbol -> datetime
        self._last_reconciliation: Dict[str, datetime] = {}
        self._reconciliation_interval = 60  # 瀵硅处闂撮殧 (绉?

        # Lower-grid/base-TP capacity logs can be noisy; throttle per symbol.
        self._capacity_log_last: Dict[tuple, float] = {}
        self._capacity_log_interval = 60  # seconds
        self._client_order_seq = 0

        logger.info("缃戞牸绛栫暐鎵ц鍣ㄥ垵濮嬪寲瀹屾垚")

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

    def _add_order_id(self, orders: Dict[float, List[str]], price: float, order_id: str) -> None:
        """Add order_id under price."""
        order_list = orders.get(price)
        if order_list is None:
            orders[price] = [order_id]
            return
        if order_id not in order_list:
            order_list.append(order_id)

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

    def _maybe_soft_rebase(self, symbol: str, grid_state: GridState) -> bool:
        """Soft rebase grids when price deviates too far for too long."""
        if not self.config.grid.rebase_enabled:
            return False

        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"{symbol} 鑾峰彇褰撳墠浠锋牸澶辫触锛岃烦杩囬噸寤烘鏌? {e}")
            return False

        center_price = grid_state.entry_price
        if not center_price or center_price <= 0:
            return False

        threshold = self.config.grid.rebase_distance_k * self.config.grid.spacing
        if threshold <= 0:
            return False

        deviation = abs(current_price - center_price) / center_price
        now = datetime.now(timezone.utc)

        if deviation <= threshold:
            grid_state.rebase_deviation_since = None
            return False

        if grid_state.rebase_deviation_since is None:
            grid_state.rebase_deviation_since = now
            logger.info(
                f"{symbol} 鍋忕缃戞牸涓績{deviation*100:.2f}%>闃堝€納threshold*100:.2f}%锛屽紑濮嬭鏃?
            )
            return False

        confirm_seconds = self.config.grid.rebase_confirm_hours * 3600
        if (now - grid_state.rebase_deviation_since).total_seconds() < confirm_seconds:
            return False

        cooldown_seconds = self.config.grid.rebase_cooldown_hours * 3600
        if grid_state.last_rebase_time:
            elapsed = (now - grid_state.last_rebase_time).total_seconds()
            if elapsed < cooldown_seconds:
                return False

        self._soft_rebase_grid(symbol, grid_state, current_price)
        grid_state.last_rebase_time = now
        grid_state.rebase_deviation_since = None
        return True

    def _soft_rebase_grid(self, symbol: str, grid_state: GridState, new_center: float) -> None:
        """Cancel existing orders and rebuild grids around new_center without closing position."""
        logger.info(f"{symbol} 瑙﹀彂杞噸寤猴紝涓績浠锋洿鏂颁负 {new_center:.6f}")

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
                logger.warning(f"{symbol} 杞噸寤烘挙鍗曞け璐? {e}")

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
            logger.warning(f"{symbol} 杞噸寤哄悗缃戞牸楠岃瘉澶辫触: {validation_msg}")

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
            f"瀹屾暣寰幆: 寮€浠?@ {fill_info.price:.6f}, 骞充粨 @ {tp_price:.6f}, 鐩堝埄 {profit_pct:.2f}%"
        )

        # 鎭㈠涓婃柟缃戞牸
        self._place_single_upper_grid_by_price(symbol, grid_state, fill_info.price)

        # 鎭㈠涓嬫柟鍩虹姝㈢泩鍗?
        if fill_info.matched_lower_price is not None:
            self._place_single_lower_grid_by_price(symbol, grid_state, fill_info.matched_lower_price)

        # 娓呯悊鏄犲皠涓庢垚浜よ褰?
        if upper_order_id in grid_state.filled_upper_grids:
            del grid_state.filled_upper_grids[upper_order_id]
        grid_state.tp_to_upper.pop(tp_order_id, None)

        if hasattr(self, "db") and self.db:
            self.db.close_grid_cycle_by_tp(tp_order_id)

    def calculate_grid_prices(self, entry_price: float) -> GridPrices:
        """
        璁＄畻鍔ㄦ€佺綉鏍间环鏍?

        鍒濆鍖栨椂鍒涘缓卤10涓綉鏍硷紝鍚庣画鍙墿灞曞埌卤15

        Args:
            entry_price: 鍏ュ満浠稰0

        Returns:
            GridPrices瀵硅薄
        """
        spacing = self.config.grid.spacing
        upper_count = self.config.grid.upper_grids
        lower_count = self.config.grid.lower_grids

        # 鍒濆鍖栫綉鏍煎瓧鍏?
        grid_levels = {}

        # 涓婃柟缃戞牸锛欸rid+1 鍒?Grid+10
        for level in range(1, upper_count + 1):
            price = entry_price * ((1 + spacing) ** level)
            grid_levels[level] = price

        # 涓嬫柟缃戞牸锛欸rid-1 鍒?Grid-10
        for level in range(1, lower_count + 1):
            price = entry_price * ((1 - spacing) ** level)
            grid_levels[-level] = price

        # 姝㈡崯绾?
        stop_loss_price = entry_price * self.config.stop_loss.ratio

        grid_prices = GridPrices(
            entry_price=entry_price,
            grid_levels=grid_levels,
            stop_loss_price=stop_loss_price,
            spacing=spacing
        )

        # 璁＄畻浠锋牸鑼冨洿
        upper_levels = grid_prices.get_upper_levels()
        lower_levels = grid_prices.get_lower_levels()
        min_price = grid_levels[min(lower_levels)] if lower_levels else entry_price
        max_price = grid_levels[max(upper_levels)] if upper_levels else entry_price

        logger.info(
            f"鍒濆鍖栧姩鎬佺綉鏍? P0={entry_price:.4f}, "
            f"涓婃柟{len(upper_levels)}涓? 涓嬫柟{len(lower_levels)}涓? "
            f"鑼冨洿={min_price:.4f}~{max_price:.4f}"
        )
        return grid_prices

    def initialize_grid(self, symbol: str, entry_price: float) -> bool:
        """
        鍒濆鍖栫綉鏍?

        Args:
            symbol: 浜ゆ槗瀵?
            entry_price: 鍏ュ満浠?

        Returns:
            鏄惁鎴愬姛
        """
        try:
            logger.info(f"鍒濆鍖栫綉鏍? {symbol} @ {entry_price}")

            # 璁＄畻缃戞牸浠锋牸
            grid_prices = self.calculate_grid_prices(entry_price)

            # 1. 寮€鍩虹浠撲綅锛堜娇鐢ㄥ競浠峰崟绔嬪嵆鎴愪氦锛?
            base_margin = self.config.position.base_margin
            base_amount = self._calculate_amount(symbol, base_margin, entry_price)

            logger.info(f"寮€鍩虹浠撲綅锛堝競浠凤級: {base_amount}寮?)
            base_order = self.connector.place_order(
                symbol=symbol,
                side='sell',  # 寮€绌?
                amount=base_amount,
                order_type='market'
            )

            base_order_id = base_order.order_id  # 淇濆瓨鐢ㄤ簬鍙兘鐨勬竻鐞?

            # 2. 绛夊緟鍩虹浠撲綅鎴愪氦纭锛堝競浠峰崟閫氬父绔嬪嵆鎴愪氦锛岀煭瓒呮椂鍗冲彲锛?
            logger.info(f"绛夊緟鍩虹浠撲綅鎴愪氦纭: order_id={base_order_id}")
            base_filled = self._wait_for_order_fill(symbol, base_order_id, timeout=30)  # 30绉掕秴鏃?

            if not base_filled:
                logger.error(f"鍩虹浠撲綅瓒呮椂鏈垚浜わ紝鍒濆鍖栧け璐?)
                self._cleanup_failed_initialization(symbol, base_order_id)
                return False

            logger.info(f"鉁?鍩虹浠撲綅宸叉垚浜わ紝寮€濮嬫寕缃戞牸")

            # 3. 鍒涘缓缃戞牸鐘舵€?
            grid_state = GridState(
                symbol=symbol,
                entry_price=entry_price,
                grid_prices=grid_prices
            )

            self.grid_states[symbol] = grid_state

            # 4. 鎸傚熀纭€浠撲綅鐨勫垎灞傛鐩堝崟锛堝厛鎸傛鐩堜繚鎶わ級
            logger.info("鎸傚熀纭€浠撲綅鍒嗗眰姝㈢泩鍗?..")
            self._place_base_position_take_profit(symbol, grid_state)

            # 5. 鎸備笂鏂圭綉鏍艰鍗?寮€绌?
            logger.info("鎸備笂鏂圭綉鏍?..")
            self._place_upper_grid_orders(symbol, grid_state)

            # 6. 楠岃瘉缃戞牸鍒涘缓鎴愬姛鐜?
            validation_passed, validation_msg = self._validate_grid_creation(symbol, grid_state)

            if not validation_passed:
                logger.warning(f"缃戞牸楠岃瘉澶辫触: {validation_msg}, 浣嗙户缁繍琛岋紙宸茬鐢ㄨ嚜鍔ㄥ钩浠擄級")
                # 涓嶅啀璋冪敤 _cleanup_failed_initialization锛屽厑璁搁儴鍒嗙綉鏍艰繍琛?
                # 鍚庣画鐨勭綉鏍间慨澶嶆満鍒朵細鑷姩琛ュ厖缂哄け鐨勭綉鏍?

            # 7. 娣诲姞鍒颁粨浣嶇鐞嗗櫒
            self.position_mgr.add_position(symbol, entry_price)

            logger.info(f"缃戞牸鍒濆鍖栧畬鎴? {symbol}")
            return True

        except Exception as e:
            logger.error(f"缃戞牸鍒濆鍖栧け璐? {symbol}: {e}")
            # 灏濊瘯娓呯悊
            if symbol in self.grid_states:
                self._cleanup_failed_initialization(symbol, None)
            return False

    def _wait_for_order_fill(self, symbol: str, order_id: str, timeout: int = 60) -> bool:
        """
        杞绛夊緟璁㈠崟鎴愪氦

        Args:
            symbol: 浜ゆ槗瀵?
            order_id: 璁㈠崟ID
            timeout: 瓒呮椂鏃堕棿(绉?

        Returns:
            鏄惁鎴愪氦
        """
        import time
        from datetime import datetime, timezone

        start_time = datetime.now(timezone.utc)
        check_interval = 3  # 姣?绉掓鏌ヤ竴娆?

        logger.info(f"寮€濮嬭疆璇㈣鍗曠姸鎬? order_id={order_id}, 瓒呮椂={timeout}绉?)

        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

            if elapsed > timeout:
                logger.warning(f"璁㈠崟绛夊緟瓒呮椂({timeout}绉?: order_id={order_id}")
                return False

            try:
                # 鏌ヨ璁㈠崟鐘舵€?
                open_orders = self.connector.query_open_orders(symbol)
                order_still_open = any(o.order_id == order_id for o in open_orders)

                if not order_still_open:
                    # 璁㈠崟涓嶅湪鎸傚崟鍒楄〃涓紝璇存槑宸叉垚浜ゆ垨鍙栨秷
                    # 纭鎸佷粨鏄惁澧炲姞
                    positions = self.connector.query_positions()
                    has_position = any(p.symbol == symbol and abs(p.contracts) > 0 for p in positions)

                    if has_position:
                        logger.info(f"鉁?璁㈠崟宸叉垚浜? order_id={order_id}, 鑰楁椂={elapsed:.1f}绉?)
                        return True
                    else:
                        logger.warning(f"璁㈠崟宸插彇娑堟垨澶辫触: order_id={order_id}")
                        return False

                logger.info(f"璁㈠崟绛夊緟涓?.. ({elapsed:.0f}/{timeout}绉?")
                time.sleep(check_interval)

            except Exception as e:
                logger.warning(f"鏌ヨ璁㈠崟鐘舵€佸け璐? {e}, 缁х画绛夊緟...")
                time.sleep(check_interval)

    def _validate_grid_creation(self, symbol: str, grid_state: GridState) -> tuple:
        """
        楠岃瘉缃戞牸鍒涘缓鎴愬姛鐜?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?

        Returns:
            tuple[bool, str]: (鏄惁閫氳繃楠岃瘉, 璇︾粏淇℃伅)
        """
        upper_count = len(grid_state.grid_prices.get_upper_levels())
        lower_count = len(grid_state.grid_prices.get_lower_levels())
        upper_created = min(self._count_orders(grid_state.upper_orders), upper_count)
        lower_created = min(self._count_orders(grid_state.lower_orders), lower_count)

        upper_success_rate = upper_created / upper_count if upper_count > 0 else 0.0
        lower_success_rate = lower_created / lower_count if lower_count > 0 else 0.0

        grid_state.upper_success_rate = upper_success_rate
        grid_state.lower_success_rate = lower_success_rate

        logger.info(
            f"{symbol} 缃戞牸鍒涘缓缁熻: "
            f"涓婃柟 {upper_created}/{upper_count} ({upper_success_rate*100:.1f}%), "
            f"涓嬫柟 {lower_created}/{lower_count} ({lower_success_rate*100:.1f}%)"
        )

        # 涓婃柟缃戞牸涓ユ牸瑕佹眰80%锛堝紑绌哄崟锛屽叧閿級
        if upper_success_rate < self.config.grid.min_success_rate_upper:
            msg = f"{symbol} 涓婃柟缃戞牸鎴愬姛鐜噞upper_success_rate*100:.1f}% < {self.config.grid.min_success_rate_upper*100:.0f}%, 鎷掔粷寮€浠?
            logger.error(msg)
            return False, msg

        # 涓嬫柟缃戞牸浠呭憡璀︼紙姝㈢泩鍗曪紝涓嶅叧閿級
        if lower_success_rate < self.config.grid.min_success_rate_lower:
            logger.warning(
                f"{symbol} 涓嬫柟缃戞牸鎴愬姛鐜噞lower_success_rate*100:.1f}% < "
                f"{self.config.grid.min_success_rate_lower*100:.0f}%"
            )

        grid_state.grid_integrity_validated = True
        return True, "缃戞牸鍒涘缓鎴愬姛"

    def _cleanup_failed_initialization(self, symbol: str, base_order_id: Optional[str]) -> None:
        """
        娓呯悊鍒濆鍖栧け璐ョ殑璁㈠崟鍜岀姸鎬?

        Args:
            symbol: 浜ゆ槗瀵?
            base_order_id: 鍩虹浠撲綅璁㈠崟ID锛堝鏋滃凡鍒涘缓锛?
        """
        logger.info(f"娓呯悊澶辫触鐨勫垵濮嬪寲: {symbol}")

        if symbol not in self.grid_states:
            return

        grid_state = self.grid_states[symbol]

        # 1. 鎾ら攢鎵€鏈変笂鏂圭綉鏍艰鍗?
        for price, order_id in list(self._iter_order_items(grid_state.upper_orders)):
            try:
                self.connector.cancel_order(order_id, symbol)
                logger.info(f"宸叉挙閿€涓婃柟缃戞牸 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"鎾ら攢璁㈠崟澶辫触: {e}")

        # 2. 鎾ら攢鎵€鏈変笅鏂圭綉鏍艰鍗?
        for price, order_id in list(self._iter_order_items(grid_state.lower_orders)):
            try:
                self.connector.cancel_order(order_id, symbol)
                logger.info(f"宸叉挙閿€涓嬫柟缃戞牸 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"鎾ら攢璁㈠崟澶辫触: {e}")

        # 3. 鎾ら攢鍩虹浠撲綅璁㈠崟
        if base_order_id:
            try:
                self.connector.cancel_order(base_order_id, symbol)
                logger.info(f"宸叉挙閿€鍩虹浠撲綅璁㈠崟")
            except Exception as e:
                logger.warning(f"鎾ら攢鍩虹浠撲綅璁㈠崟澶辫触: {e}")

        # 4. 妫€鏌ュ苟骞充粨宸叉垚浜ょ殑浠撲綅
        try:
            positions = self.connector.query_positions()
            for pos in positions:
                if pos.symbol == symbol and abs(pos.contracts) > 0:
                    try:
                        self.connector.place_order(
                            symbol=symbol,
                            side='buy',  # 骞崇┖
                            amount=abs(pos.contracts),
                            order_type='market',
                            reduce_only=True  # 寮哄埗鍙噺浠?
                        )
                        logger.info(f"宸插競浠峰钩浠? {abs(pos.contracts)}寮?)
                    except Exception as e:
                        logger.error(f"骞充粨澶辫触: {e}")
        except Exception as e:
            logger.warning(f"鏌ヨ浠撲綅澶辫触: {e}")

        # 5. 绉婚櫎缃戞牸鐘舵€?
        del self.grid_states[symbol]
        logger.info(f"娓呯悊瀹屾垚: {symbol}")

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
        """鎸備笂鏂圭綉鏍艰鍗?寮€绌? - 浣跨敤浠锋牸浣滀负鏍囪瘑"""
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

                logger.debug(f"鎸備笂鏂圭綉鏍?@ {price:.6f}: {amount}寮?)
                order = self.connector.place_order_with_maker_retry(
                    symbol=symbol,
                    side='sell',  # 寮€绌?
                    amount=amount,
                    price=price,
                    order_type='limit',
                    post_only=True,
                    client_order_id=client_order_id,
                    max_retries=5
                )

                self._add_order_id(grid_state.upper_orders, price, order.order_id)

            except Exception as e:
                logger.warning(f"鎸傚崟澶辫触 @ {price:.6f}: {e}")

    def _place_lower_grid_orders(self, symbol: str, grid_state: GridState) -> None:
        """鎸備笅鏂圭綉鏍艰鍗?骞崇┖姝㈢泩)"""
        grid_margin = self.config.position.grid_margin

        for level in grid_state.grid_prices.get_lower_levels():
            try:
                price = self._quantize_price(
                    symbol, grid_state.grid_prices.grid_levels[level], side='buy'
                )
                if price in grid_state.lower_orders:
                    continue
                amount = self._calculate_amount(symbol, grid_margin, price)

                logger.debug(f"鎸備笅鏂圭綉鏍?Grid-{level}: {amount}寮?脳 {price}")
                client_order_id = self._make_client_order_id(
                    symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
                )
                order = self.connector.place_order_with_maker_retry(
                    symbol=symbol,
                    side='buy',  # 骞崇┖姝㈢泩
                    amount=amount,
                    price=price,
                    order_type='limit',
                    post_only=True,
                    reduce_only=True,  # 寮哄埗鍙噺浠?
                    client_order_id=client_order_id,
                    max_retries=5
                )

                self._add_order_id(grid_state.lower_orders, price, order.order_id)

            except Exception as e:
                logger.warning(f"鎸傚崟澶辫触 Grid-{level}: {e}")

    def _place_base_position_take_profit(self, symbol: str, grid_state: GridState) -> None:
        """Place layered take-profit orders for the base position (keep min base ratio)."""
        base_margin = self.config.position.base_margin
        grid_margin = self.config.position.grid_margin
        min_ratio = self.config.position.min_base_position_ratio

        closeable_ratio = 1.0 - min_ratio
        closeable_margin = base_margin * closeable_ratio

        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception:
            current_price = grid_state.entry_price

        lower_levels = self._get_lower_levels_by_proximity(grid_state, current_price)
        total_levels = len(lower_levels)

        allowed_levels_by_ratio = int(total_levels * closeable_ratio)
        allowed_levels_by_margin = int(closeable_margin // grid_margin) if grid_margin > 0 else 0
        allowed_levels = min(allowed_levels_by_ratio, allowed_levels_by_margin)

        base_amount_per_level = self._calculate_amount(symbol, grid_margin, grid_state.entry_price)

        logger.info(f"Base TP orders: {allowed_levels}/{total_levels} levels, {base_amount_per_level:.1f} each")
        logger.info(f"Keep min base: {min_ratio*100:.0f}%")
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
        """鎸備笅鏂圭綉鏍艰鍗?骞崇┖) - 浠呯敤浜庨噸鏂版寕涓婃柟鎴愪氦鍓嶇殑鍩虹姝㈢泩鍗?""
        if level not in grid_state.grid_prices.grid_levels:
            return

        base_price = self._quantize_price(
            symbol, grid_state.grid_prices.grid_levels[level], side='buy'
        )
        price = base_price
        grid_margin = self.config.position.grid_margin
        base_amount_per_level = self._calculate_amount(symbol, grid_margin, grid_state.entry_price)

        try:
            logger.debug(f"閲嶆柊鎸傚熀纭€姝㈢泩鍗?Grid-{level}: {base_amount_per_level}寮?脳 {price}")
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
                reduce_only=True,  # 寮哄埗鍙噺浠?
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)

        except Exception as e:
            logger.warning(f"鎸傚崟澶辫触 Grid-{level}: {e}")

    def _place_enhanced_lower_grid_order(self, symbol: str, grid_state: GridState, level: int) -> None:
        """鎸備笅鏂规鐩堝崟锛堜笌寮€绌哄崟鏁伴噺涓€鑷达級"""
        if level not in grid_state.grid_prices.grid_levels:
            return

        base_price = self._quantize_price(
            symbol, grid_state.grid_prices.grid_levels[level], side='buy'
        )
        price = base_price

        # 馃敡 FIX: 浣跨敤涓庡紑绌哄崟鐩稿悓鐨勬暟閲忥紙浠単rid_margin锛?
        grid_margin = self.config.position.grid_margin
        amount = self._calculate_amount(symbol, grid_margin, price)

        try:
            logger.info(f"鎸傛鐩堝崟 Grid-{level}: {amount}寮?脳 {price}")
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
                reduce_only=True,  # 寮哄埗鍙噺浠?
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)

        except Exception as e:
            logger.warning(f"鎸傛鐩堝崟澶辫触 Grid-{level}: {e}")

    def _place_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        鎸傚崟涓笅鏂圭綉鏍艰鍗曪紙鐢ㄤ簬婊氬姩绐楀彛娣诲姞鏂扮綉鏍硷級
        娉ㄦ剰锛氭鍑芥暟琚笅鏂瑰悓鍚嶅嚱鏁拌鐩栵紝瀹為檯涓嶄細琚皟鐢?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            level: 缃戞牸灞傜骇锛堣礋鏁帮級
            price: 浠锋牸
        """
        try:
            # FIX: 浣跨敤涓庡紑绌哄崟鐩稿悓鐨勬暟閲忥紙浠単rid_margin锛?
            self._place_enhanced_lower_grid_order(symbol, grid_state, level)
        except Exception as e:
            logger.warning(f"鎸備笅鏂圭綉鏍煎け璐?Grid{level}: {e}")

    def _should_check_grid_repair(self, grid_state: GridState) -> bool:
        """
        鍒ゆ柇鏄惁搴旇妫€鏌ョ綉鏍间慨澶嶏紙姝ｅ父闂撮殧10绉掞紝鎭㈠妯″紡2绉掞級

        Args:
            grid_state: 缃戞牸鐘舵€?

        Returns:
            bool: 鏄惁搴旇妫€鏌?
        """
        if not self.config.grid.repair_enabled:
            return False

        now = datetime.now(timezone.utc)
        elapsed = (now - grid_state.last_repair_check).total_seconds()

        # 鎭㈠妯″紡锛氬鏋滃畬鍏ㄦ病鏈夋鐩堝崟锛屼娇鐢ㄦ洿鐭殑闂撮殧锛?绉掞級
        is_recovery = len(grid_state.lower_orders) == 0
        repair_interval = 2 if is_recovery else self.config.grid.repair_interval

        return elapsed >= repair_interval

    def _repair_missing_grids(self, symbol: str, grid_state: GridState) -> None:
        """
        妫€鏌ュ苟琛ュ厖缂哄け鐨勭綉鏍艰鍗曪紙鍩轰簬浠锋牸锛?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
        """
        if not self._should_check_grid_repair(grid_state):
            return

        grid_state.last_repair_check = datetime.now(timezone.utc)

        # 鑾峰彇褰撳墠甯傚満浠锋牸
        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"{symbol} 鑾峰彇浠锋牸澶辫触锛岃烦杩囦慨澶? {e}")
            return

        # 鏌ヨ褰撳墠鎸傚崟锛堥伩鍏嶉噸澶嶏級
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
            logger.warning(f"{symbol} 鏌ヨ鎸傚崟澶辫触锛岃烦杩囦慨澶? {e}")
            return

        # 鍏堝璐︼細鎭㈠閬楀け鐨勮鍗曠姸鎬侊紙鍩轰簬浠锋牸鍖归厤锛?
        for order in open_orders:
            order_price = self._quantize_price(symbol, order.price, side=order.side)

            # 妫€鏌ユ槸鍚﹀簲璇ュ湪upper_orders涓?
            if order.side == 'sell':
                # 妫€鏌ヤ环鏍兼槸鍚︽帴杩戜换浣曢鏈熺殑涓婃柟缃戞牸浠锋牸
                for level in grid_state.grid_prices.get_upper_levels():
                    target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='sell')
                    if abs(order_price - target_price) / target_price < 0.001:  # 0.1%瀹瑰樊
                        self._add_order_id(grid_state.upper_orders, order_price, order.order_id)
                        logger.info(f"{symbol} 鎭㈠閬楀け鐨勪笂鏂圭綉鏍?@ {order_price:.6f}")
                        break

            # 妫€鏌ユ槸鍚﹀簲璇ュ湪lower_orders涓?
            elif order.side == 'buy':
                # 妫€鏌ヤ环鏍兼槸鍚︽帴杩戜换浣曢鏈熺殑涓嬫柟缃戞牸浠锋牸
                for level in grid_state.grid_prices.get_lower_levels():
                    target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='buy')
                    if abs(order_price - target_price) / target_price < 0.001:
                        self._add_order_id(grid_state.lower_orders, order_price, order.order_id)
                        logger.info(f"{symbol} 鎭㈠閬楀け鐨勪笅鏂圭綉鏍?@ {order_price:.6f}")
                        break

        # 娓呯悊state涓凡澶辨晥鐨勮鍗旾D
        for price, order_id in list(self._iter_order_items(grid_state.upper_orders)):
            if order_id not in open_order_ids:
                self._remove_order_id(grid_state.upper_orders, price, order_id)
                logger.warning(f"{symbol} 妫€娴嬪埌寮傚父娑堝け鐨勪笂鏂硅鍗?@ {price:.6f}")

        for price, order_id in list(self._iter_order_items(grid_state.lower_orders)):
            if order_id not in open_order_ids:
                self._remove_order_id(grid_state.lower_orders, price, order_id)
                logger.warning(f"{symbol} 妫€娴嬪埌寮傚父娑堝け鐨勪笅鏂硅鍗?@ {price:.6f}")

        # 淇涓婃柟缃戞牸锛堟鏌ユ墍鏈夐鏈熺殑缃戞牸浠锋牸锛?
        pending_upper_prices = {
            self._quantize_price(symbol, fill.price, side='sell')
            for fill in grid_state.filled_upper_grids.values()
        }

        for level in grid_state.grid_prices.get_upper_levels():
            target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='sell')

            # 妫€鏌ユ槸鍚︾己澶?
            if target_price not in grid_state.upper_orders and target_price not in pending_upper_prices:
                # 鍙湁褰撳競浠蜂綆浜庣洰鏍囦环鏃舵墠琛ュ厖寮€绌哄崟
                if current_price < target_price:
                    logger.info(f"{symbol} 琛ュ厖缂哄け鐨勪笂鏂圭綉鏍?@ {target_price:.6f}")
                    self._place_single_upper_grid_by_price(symbol, grid_state, target_price)

        # 淇涓嬫柟缃戞牸锛堟鏌ユ墍鏈夐鏈熺殑缃戞牸浠锋牸锛?
        # 妫€鏌ユ槸鍚﹀浜庢仮澶嶅満鏅紙瀹屽叏娌℃湁姝㈢泩鍗曪級
        is_recovery = len(grid_state.lower_orders) == 0

        for level in grid_state.grid_prices.get_lower_levels():
            target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='buy')

            # 妫€鏌ユ槸鍚︾己澶?
            if target_price not in grid_state.lower_orders:
                # 鍦ㄦ仮澶嶅満鏅笅锛屾棤璁轰环鏍煎浣曢兘琛ュ厖姝㈢泩鍗?
                # 鍦ㄦ甯稿満鏅笅锛屽彧鏈夊綋甯備环楂樹簬鐩爣浠锋椂鎵嶈ˉ鍏?
                if is_recovery or current_price > target_price:
                    if is_recovery:
                        logger.info(f"{symbol} [鎭㈠妯″紡] 琛ュ厖缂哄け鐨勪笅鏂圭綉鏍?@ {target_price:.6f}")
                    else:
                        logger.info(f"{symbol} 琛ュ厖缂哄け鐨勪笅鏂圭綉鏍?@ {target_price:.6f}")
                    self._place_single_lower_grid_by_price(symbol, grid_state, target_price)

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
        淇鍗曚釜涓婃柟缃戞牸

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            level: 缃戞牸灞傜骇
            price: 鐩爣浠锋牸
        """
        try:
            price = self._quantize_price(symbol, price, side='sell')
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "sell", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 寮€绌?
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.upper_orders, price, order.order_id)
            logger.info(f"{symbol} 鎴愬姛琛ュ厖涓婃柟缃戞牸 Grid+{level}")

        except Exception as e:
            logger.warning(f"{symbol} 琛ュ厖涓婃柟缃戞牸澶辫触 Grid+{level}: {e}")

    def _repair_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        淇鍗曚釜涓嬫柟缃戞牸

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            level: 缃戞牸灞傜骇
            price: 鐩爣浠锋牸
        """
        try:
            # FIX: 浣跨敤涓庡紑绌哄崟鐩稿悓鐨勬暟閲忥紙浠単rid_margin锛?
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',  # 骞崇┖姝㈢泩
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                reduce_only=True,  # 寮哄埗鍙噺浠?
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)
            logger.info(f"{symbol} 鎴愬姛琛ュ厖涓嬫柟缃戞牸 Grid-{level}")

        except Exception as e:
            logger.warning(f"{symbol} 琛ュ厖涓嬫柟缃戞牸澶辫触 Grid-{level}: {e}")

    def _try_extend_grid(self, symbol: str, grid_state: GridState, filled_price: float, is_upper: bool) -> None:
        """
        婊氬姩绐楀彛缃戞牸鎵╁睍锛堜繚鎸佸钩琛★級

        涓婃柟鎴愪氦锛氬湪涓婃柟娣诲姞鏂扮綉鏍硷紝骞舵坊鍔犲搴旀鐩堝崟
        涓嬫柟鎴愪氦锛氬缁堟粴鍔ㄧ獥鍙ｏ紙閲嶅紑绌恒€佺Щ闄ゆ渶杩滀笂鏂广€佽ˉ涓嬫柟淇濇姢锛?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            filled_price: 鎴愪氦鐨勭綉鏍间环鏍?
            is_upper: 鏄惁涓轰笂鏂圭綉鏍?
        """
        # 妫€鏌ユ槸鍚﹀惎鐢ㄥ姩鎬佹墿灞?
        if not self.config.grid.dynamic_expansion:
            return

        # 鑾峰彇褰撳墠浠锋牸
        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"{symbol} 鑾峰彇褰撳墠浠锋牸澶辫触锛岃烦杩囩綉鏍兼墿灞? {e}")
            return

        spacing = self.config.grid.spacing  # 0.015
        max_total_grids = self.config.grid.max_total_grids  # 30

        # 鑾峰彇褰撳墠鎵€鏈夌綉鏍间环鏍?
        upper_prices = sorted(grid_state.upper_orders.keys())
        lower_prices = sorted(grid_state.lower_orders.keys(), reverse=True)
        total_grids = len(upper_prices) + len(lower_prices)

        if is_upper:  # 涓婃柟缃戞牸鎴愪氦锛堜环鏍间笂娑級
            # 馃敡 FIX: 绉婚櫎杈圭晫妫€鏌ワ紝姣忎釜涓婃柟缃戞牸鎴愪氦閮芥墿灞?

            # 妫€鏌ユ槸鍚﹁揪鍒版暟閲忛檺鍒讹紙杞檺鍒讹紝浠呯敤浜庨槻姝㈠紓甯告儏鍐碉級
            if total_grids >= max_total_grids:
                logger.warning(f"{symbol} 缃戞牸鏁板凡杈捐蒋闄愬埗 {max_total_grids}锛堝綋鍓峽total_grids}锛夛紝璺宠繃鎵╁睍")
                return

            # 1. 鍦ㄦ渶楂樹环鏍间箣涓婃坊鍔犳柊鐨勪笂鏂圭綉鏍?
            max_upper_price = max(upper_prices) if upper_prices else current_price
            new_upper_price = self._quantize_price(
                symbol, max_upper_price * (1 + spacing), side='sell'
            )
            self._place_single_upper_grid_by_price(symbol, grid_state, new_upper_price)
            logger.info(f"{symbol} 鎵╁睍锛氭坊鍔犱笂鏂圭綉鏍?@ {new_upper_price:.6f}")
            # NET: +1 short capacity (EXPANSION)

        else:  # 涓嬫柟缃戞牸鎴愪氦锛堜环鏍间笅璺岋級
            # 鏂规A锛氫换浣曚笅鏂规垚浜ら兘婊氬姩绐楀彛
            # 1) 閲嶆柊寮€绌轰互淇濇寔绌哄ご鏁炲彛
            reopen_price = self._quantize_price(
                symbol, current_price * (1 + spacing), side='sell'
            )
            min_gap_ratio = max(0.0, self.config.grid.reopen_min_gap_ratio) * spacing
            if self._is_price_too_close(reopen_price, upper_prices, min_gap_ratio):
                reopen_placed = False
                logger.info(
                    f"{symbol} 婊氬姩绐楀彛锛氶噸鏂板紑绌鸿繃杩?@ {reopen_price:.6f} "
                    f"(min_gap={min_gap_ratio:.4f})锛岃烦杩?
                )
            else:
                reopen_placed = self._place_single_upper_grid_by_price(symbol, grid_state, reopen_price)
                if reopen_placed:
                    logger.info(
                        f"{symbol} 婊氬姩绐楀彛锛氶噸鏂板紑绌?@ {reopen_price:.6f} "
                        f"(鎴愪氦浠?{filled_price:.6f})"
                    )

            # 2) 绉婚櫎鏈€杩滅殑涓婃柟缃戞牸锛堜繚鎸佺獥鍙ｅぇ灏忥級
            if reopen_placed and upper_prices:
                max_upper_price = max(upper_prices)
                self._remove_grid_by_price(symbol, grid_state, max_upper_price, is_upper=True)
                logger.info(f"{symbol} 婊氬姩绐楀彛锛氱Щ闄ゆ渶杩滀笂鏂圭綉鏍?@ {max_upper_price:.6f}")

            # 3) 鍦ㄤ笅鏂规坊鍔犳柊缃戞牸锛堟洿浣庝环鏍?- 淇濇寔涓嬫柟淇濇姢锛?
            new_lower_price = self._quantize_price(
                symbol, current_price * (1 - spacing), side='buy'
            )
            self._place_single_lower_grid_by_price(symbol, grid_state, new_lower_price)
            logger.info(f"{symbol} 婊氬姩绐楀彛锛氭坊鍔犱笅鏂逛繚鎶?@ {new_lower_price:.6f}")

                # NET: +1 short (reopen), -1 short (remove), +1 lower 鈫?MAINTAINS SHORT EXPOSURE 鉁?

    def _place_single_upper_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        鎸傚崟涓笂鏂圭綉鏍艰鍗?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            level: 缃戞牸灞傜骇锛堟鏁帮級
            price: 浠锋牸
        """
        try:
            price = self._quantize_price(symbol, price, side='sell')
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "sell", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='sell',  # 寮€绌?
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.upper_orders, price, order.order_id)
            logger.info(f"{symbol} 鎴愬姛鎸備笂鏂圭綉鏍?Grid+{level} @ {price:.6f}, {amount}寮?)

        except Exception as e:
            logger.warning(f"{symbol} 鎸備笂鏂圭綉鏍煎け璐?Grid+{level}: {e}")

    def _place_single_lower_grid(self, symbol: str, grid_state: GridState, level: int, price: float) -> None:
        """
        鎸傚崟涓笅鏂圭綉鏍艰鍗曪紙姝㈢泩鍗曪級

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            level: 缃戞牸灞傜骇锛堣礋鏁帮級
            price: 浠锋牸
        """
        try:
            price = self._quantize_price(symbol, price, side='buy')
            # 馃敡 FIX: 浣跨敤涓庡紑绌哄崟鐩稿悓鐨勬暟閲忥紙浠単rid_margin锛?
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, price)

            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=price, entry_price=grid_state.entry_price, unique=True
            )
            order = self.connector.place_order_with_maker_retry(
                symbol=symbol,
                side='buy',  # 骞崇┖姝㈢泩
                amount=amount,
                price=price,
                order_type='limit',
                post_only=True,
                reduce_only=True,  # 寮哄埗鍙噺浠?
                client_order_id=client_order_id,
                max_retries=5
            )

            self._add_order_id(grid_state.lower_orders, price, order.order_id)
            logger.info(f"{symbol} 鎴愬姛鎸備笅鏂圭綉鏍?Grid{level} @ {price:.6f}, {amount}寮?)

        except Exception as e:
            logger.warning(f"{symbol} 鎸備笅鏂圭綉鏍煎け璐?Grid{level}: {e}")

    def _remove_grid_level(self, symbol: str, grid_state: GridState, level: int) -> None:
        """
        绉婚櫎鎸囧畾灞傜骇鐨勭綉鏍硷紙鎾ゅ崟+鍒犻櫎浠锋牸锛?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            level: 瑕佺Щ闄ょ殑缃戞牸灞傜骇
        """
        # 濡傛灉鏈夋寕鍗曪紝鍏堟挙閿€
        level_price = grid_state.grid_prices.grid_levels.get(level)
        if level_price is None:
            return

        if level > 0 and level_price in grid_state.upper_orders:
            order_ids = list(grid_state.upper_orders.get(level_price, []))
            for order_id in order_ids:
                try:
                    self.connector.cancel_order(order_id, symbol)
                    self._remove_order_id(grid_state.upper_orders, level_price, order_id)
                    logger.info(f"{symbol} 宸叉挙閿€涓婃柟缃戞牸 Grid+{level} @ {level_price:.6f}")
                except Exception as e:
                    logger.warning(f"{symbol} 鎾ら攢涓婃柟缃戞牸澶辫触 Grid+{level}: {e}")

        elif level < 0 and level_price in grid_state.lower_orders:
            order_ids = list(grid_state.lower_orders.get(level_price, []))
            for order_id in order_ids:
                try:
                    self.connector.cancel_order(order_id, symbol)
                    self._remove_order_id(grid_state.lower_orders, level_price, order_id)
                    logger.info(f"{symbol} 宸叉挙閿€涓嬫柟缃戞牸 Grid{level} @ {level_price:.6f}")
                except Exception as e:
                    logger.warning(f"{symbol} 鎾ら攢涓嬫柟缃戞牸澶辫触 Grid{level}: {e}")

        # 浠庝环鏍煎瓧鍏镐腑绉婚櫎
        grid_state.grid_prices.remove_level(level)

    # ==================== 鏂板锛氬熀浜庝环鏍肩殑缃戞牸鎿嶄綔鍑芥暟 ====================

    def _place_single_upper_grid_by_price(self, symbol: str, grid_state: GridState, price: float) -> bool:
        """
        Place a single upper grid order by price.
        """
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
        鎸傚崟涓笅鏂圭綉鏍艰鍗曪紙鍩虹姝㈢泩锛屽熀浜庝环鏍硷級

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            price: 浠锋牸
        """
        try:
            base_price = self._quantize_price(symbol, price, side='buy')  # tick size
            level = self._calculate_grid_level(base_price, grid_state.entry_price, self.config.grid.spacing)
            if level not in grid_state.grid_prices.grid_levels:
                grid_state.grid_prices.add_level(level, base_price)

            # 浠呭熀纭€姝㈢泩锛堝熀纭€浠撲綅鐨?/total_levels锛?
            client_order_id = self._make_client_order_id(
                symbol, "buy", level=level, price=base_price, entry_price=grid_state.entry_price, unique=True
            )
            price = base_price
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, grid_state.entry_price)

            # 楠岃瘉鎬讳粨浣嶄笉浼氳秴闄?
            is_safe, safe_amount, warning = self._validate_total_exposure_before_buy_order(
                symbol, grid_state, amount
            )

            if not is_safe:
                self._log_capacity_event(
                    symbol,
                    "lower_grid_blocked",
                    f"{symbol} 鎷掔粷鎸備笅鏂圭綉鏍?@ {price:.6f}: {warning}",
                    level="warning"
                )
                return

            if safe_amount < amount:
                logger.info(f"{symbol} 璋冩暣涓嬫柟缃戞牸鏁伴噺: {amount:.2f} 鈫?{safe_amount:.2f}寮?)
                amount = safe_amount

            # 浣跨敤浠撲綅鎰熺煡涔板崟
            order = self._place_position_aware_buy_order(
                symbol, price, amount, client_order_id=client_order_id
            )

            if order:
                self._add_order_id(grid_state.lower_orders, price, order.order_id)
                logger.info(f"{symbol} 鎴愬姛鎸備笅鏂圭綉鏍硷紙鍩虹锛?@ {price:.6f}, {amount}寮?)

        except Exception as e:
            logger.warning(f"{symbol} 鎸備笅鏂圭綉鏍煎け璐?@ {price:.6f}: {e}")

    def _place_enhanced_lower_grid_by_price(
        self,
        symbol: str,
        grid_state: GridState,
        price: float,
        upper_fill: UpperGridFill
    ) -> Optional[Order]:
        """
        鎸傛鐩堝崟锛堜笌寮€绌哄崟鏁伴噺涓€鑷达紝鍩轰簬浠锋牸锛?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            price: 浠锋牸
            upper_fill: 瀵瑰簲鐨勪笂鏂瑰紑浠撲俊鎭?
        """
        try:
            base_price = self._quantize_price(symbol, price, side='buy')  # tick size
            level = self._calculate_grid_level(base_price, grid_state.entry_price, self.config.grid.spacing)
            if level not in grid_state.grid_prices.grid_levels:
                grid_state.grid_prices.add_level(level, base_price)

            # FIX: 浣跨敤涓庡紑绌哄崟鐩稿悓鐨勬暟閲忥紙浠単rid_margin锛?
            grid_margin = self.config.position.grid_margin
            amount = self._calculate_amount(symbol, grid_margin, base_price)

            logger.debug(f"{symbol} 姝㈢泩鍗? {amount}寮?)

            # 楠岃瘉鎬讳粨浣嶄笉浼氳秴闄?
            is_safe, safe_amount, warning = self._validate_total_exposure_before_buy_order(
                symbol, grid_state, amount
            )

            if not is_safe:
                logger.error(f"{symbol} 鎷掔粷鎸傛鐩堝崟 @ {base_price:.6f}: {warning}")
                return None

            if safe_amount < amount:
                logger.warning(f"{symbol} 璋冩暣姝㈢泩鏁伴噺: {amount:.2f} 鈫?{safe_amount:.2f}寮?)
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
                logger.info(f"{symbol} 鎴愬姛鎸傛鐩堝崟 @ {price:.6f}, {amount}寮?)
                return order

        except Exception as e:
            logger.warning(f"{symbol} 鎸傛鐩堝崟澶辫触 @ {price:.6f}: {e}")
        return None

    def _remove_grid_by_price(self, symbol: str, grid_state: GridState, price: float, is_upper: bool) -> None:
        """
        绉婚櫎鎸囧畾浠锋牸鐨勭綉鏍硷紙鎾ゅ崟锛?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
            price: 瑕佺Щ闄ょ殑缃戞牸浠锋牸
            is_upper: 鏄惁涓轰笂鏂圭綉鏍?
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
                logger.info(f"{symbol} 宸叉挙閿€涓婃柟缃戞牸 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"{symbol} 鎾ら攢涓婃柟缃戞牸澶辫触 @ {price:.6f}: {e}")

        elif not is_upper and price in grid_state.lower_orders:
            try:
                order_ids = grid_state.lower_orders.get(price, [])
                if not order_ids:
                    return
                order_id = order_ids[0]
                self.connector.cancel_order(order_id, symbol)
                self._remove_order_id(grid_state.lower_orders, price, order_id)
                logger.info(f"{symbol} 宸叉挙閿€涓嬫柟缃戞牸 @ {price:.6f}")
            except Exception as e:
                logger.warning(f"{symbol} 鎾ら攢涓嬫柟缃戞牸澶辫触 @ {price:.6f}: {e}")

    # ==================== 缁撴潫锛氬熀浜庝环鏍肩殑缃戞牸鎿嶄綔鍑芥暟 ====================

    def _check_base_position_health(self, symbol: str, grid_state: GridState) -> None:
        """
        妫€鏌ュ熀纭€浠撲綅鍋ュ悍搴?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
        """
        try:
            # 鏌ヨ褰撳墠鎸佷粨
            positions = self.connector.query_positions()
            short_position = next((p for p in positions if p.symbol == symbol and p.side == 'short'), None)

            if not short_position:
                logger.error(f"{symbol} 鍩虹浠撲綅宸插畬鍏ㄥ钩浠擄紒瑙﹀彂绱ф€ユ竻鐞?)
                # 鍙栨秷鎵€鏈夎鍗?
                try:
                    self.connector.cancel_all_orders(symbol)
                    logger.info(f"{symbol} 宸插彇娑堟墍鏈夎鍗?)
                except Exception as e:
                    logger.error(f"{symbol} 鍙栨秷璁㈠崟澶辫触: {e}")
                # 鏍囪闇€瑕佹竻鐞嗭紙鍦╰rading_bot涓鐞嗭級
                grid_state.needs_cleanup = True
                return

            current_amount = abs(short_position.contracts)

            # 璁＄畻棰勬湡鐨勫熀纭€浠撲綅
            base_margin = self.config.position.base_margin
            expected_base = self._calculate_amount(symbol, base_margin, grid_state.entry_price)

            # 璁＄畻鏈€灏忎粨浣?
            min_ratio = self.config.position.min_base_position_ratio
            min_base = expected_base * min_ratio

            # 璁＄畻褰撳墠姣斾緥
            current_ratio = current_amount / expected_base

            if current_amount < min_base:
                logger.error(
                    f"{symbol} 鍩虹浠撲綅杩囦綆锛?
                    f"褰撳墠: {current_amount:.1f}寮?({current_ratio*100:.1f}%), "
                    f"鏈€灏? {min_base:.1f}寮?({min_ratio*100:.0f}%)"
                )
            elif current_ratio < 0.5:
                logger.warning(
                    f"{symbol} 鍩虹浠撲綅鍋忎綆: {current_amount:.1f}寮?({current_ratio*100:.1f}%)"
                )
            else:
                logger.debug(
                    f"{symbol} 鍩虹浠撲綅鍋ュ悍: {current_amount:.1f}寮?({current_ratio*100:.1f}%)"
                )

        except Exception as e:
            logger.error(f"{symbol} 妫€鏌ュ熀纭€浠撲綅澶辫触: {e}")

    def update_grid_states(self) -> None:
        """鏇存柊鎵€鏈夌綉鏍肩姸鎬?""
        for symbol, grid_state in self.grid_states.items():
            try:
                # 0. 鏂█妫€鏌ワ細涓嶅厑璁稿澶翠粨浣嶏紙姣忔閮芥鏌ワ級
                self._assert_no_long_positions(symbol)

                # 1. 鍘熸湁閫昏緫锛氭鏌ヨ鍗曟垚浜?
                self._update_single_grid(symbol, grid_state)

                # 2. 鏂板锛氭鏌ュ苟淇缂哄け鐨勭綉鏍?
                if grid_state.grid_integrity_validated:
                    self._repair_missing_grids(symbol, grid_state)

                # 3. 鏂板锛氬畾鏈熷璐︼紙60绉掗棿闅旓級
                self._reconcile_position_with_grids(symbol, grid_state)

                # 4. 鏂板锛氭鏌ュ熀纭€浠撲綅鍋ュ悍搴?
                self._check_base_position_health(symbol, grid_state)

            except Exception as e:
                logger.error(f"鏇存柊缃戞牸鐘舵€佸け璐?{symbol}: {e}")

        # 馃敡 FIX: 娣诲姞杩愯鏃惰祫閲戠洃鎺э紙姣忔鏇存柊鍚庢鏌ヤ竴娆★級
        self._validate_total_capital_usage()

    def _validate_total_capital_usage(self) -> None:
        """楠岃瘉鎬昏祫閲戜娇鐢ㄤ笉瓒呰繃90%闄愬埗"""
        try:
            total_margin = 0.0
            for symbol, grid_state in self.grid_states.items():
                try:
                    position = self.position_mgr.get_symbol_position(symbol)
                    if position and position.total_margin_used:
                        total_margin += abs(position.total_margin_used)
                except Exception as e:
                    logger.warning(f"鑾峰彇{symbol}淇濊瘉閲戝け璐? {e}")

            # 鑾峰彇璧勯噾鍒嗛厤鍣紙閫氳繃position_manager锛?
            if hasattr(self.position_mgr, 'capital_allocator'):
                capital_allocator = self.position_mgr.capital_allocator
            else:
                # 濡傛灉娌℃湁capital_allocator锛岃烦杩囬獙璇?
                return

            available_capital = capital_allocator.available_capital
            total_balance = capital_allocator.total_balance
            usage_pct = (total_margin / total_balance) * 100 if total_balance > 0 else 0

            if total_margin > available_capital:
                logger.error(
                    f"鈿狅笍 璧勯噾瓒呴檺锛氫娇鐢?{total_margin:.2f} USDT ({usage_pct:.1f}%)锛?
                    f"闄愬埗 {available_capital:.2f} USDT (90%)"
                )
            elif usage_pct > 85:
                logger.warning(
                    f"鈿狅笍 璧勯噾浣跨敤鎺ヨ繎闄愬埗锛歿total_margin:.2f} USDT ({usage_pct:.1f}%)锛?
                    f"闄愬埗 {available_capital:.2f} USDT (90%)"
                )

        except Exception as e:
            logger.warning(f"璧勯噾楠岃瘉澶辫触: {e}")

    def _update_single_grid(self, symbol: str, grid_state: GridState) -> None:
        """鏇存柊鍗曚釜缃戞牸鐘舵€侊紙鍩轰簬浠锋牸锛?""
        if self._maybe_soft_rebase(symbol, grid_state):
            return

        # 查询所有订单
        orders = {order.order_id: order for order in self.connector.query_open_orders(symbol)}

        # 妫€鏌ユ槸鍚﹂渶瑕佸垵濮嬪寲鍩虹浠撲綅鐨勬鐩堝崟
        if not grid_state.lower_orders:
            # 鏌ヨ瀹為檯鎸佷粨锛屽垽鏂熀纭€浠撲綅鏄惁宸叉垚浜?
            positions = self.connector.query_positions()
            has_position = any(p.symbol == symbol and abs(p.contracts) > 0 for p in positions)

            if has_position:
                logger.info(f"妫€娴嬪埌鍩虹浠撲綅宸叉垚浜わ紝鎸傚垎灞傛鐩堝崟: {symbol}")
                self._place_base_position_take_profit(symbol, grid_state)

        # 妫€鏌ヤ笂鏂圭綉鏍艰鍗曪紙鍩轰簬浠锋牸锛?
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
                        logger.info(f"{symbol} 涓婃柟缃戞牸宸插彇娑?@ {price:.6f}, 绛夊緟琛ュ崟")
                        continue
                    else:
                        continue
                else:
                    continue

            status = (order.status or "").lower()
            if status in ("closed", "filled"):
                # 璁㈠崟鎴愪氦
                logger.info(f"涓婃柟缃戞牸鎴愪氦: {symbol} @ {price:.6f}")

                # 璁板綍鎴愪氦淇℃伅
                fill_info = UpperGridFill(
                    price=price,
                    amount=order.amount if order else 0,
                    fill_time=datetime.now(timezone.utc),
                    order_id=order_id,
                    matched_lower_price=self._quantize_price(
                        symbol, price * (1 - self.config.grid.spacing), side='buy'
                    )  # 棰勬湡鐨勬鐩堜环鏍?1x spacing)
                )
                grid_state.filled_upper_grids[order_id] = fill_info
                self._remove_order_id(grid_state.upper_orders, price, order_id)

                # 鎸傛柊鐨勬鐩堝崟
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

                # 灏濊瘯鎵╁睍缃戞牸
                self._try_extend_grid(symbol, grid_state, price, is_upper=True)

        # 妫€鏌ヤ笅鏂圭綉鏍艰鍗曪紙鍩轰簬浠锋牸锛?
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
                        logger.info(f"{symbol} 涓嬫柟缃戞牸宸插彇娑?@ {price:.6f}, 绛夊緟琛ュ崟")
                        continue
                    else:
                        continue
                else:
                    continue

            status = (order.status or "").lower()
            if status in ("closed", "filled"):
                # 璁㈠崟鎴愪氦锛堟鐩堬級
                logger.info(f"涓嬫柟缃戞牸鎴愪氦: {symbol} @ {price:.6f}")

                # 鏌ユ壘鍖归厤鐨勪笂鏂瑰紑浠擄紙浼樺厛浣跨敤鏄犲皠锛?
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

                # 灏濊瘯鎵╁睍缃戞牸
                self._try_extend_grid(symbol, grid_state, price, is_upper=False)

        grid_state.last_update = datetime.now(timezone.utc)

    def close_grid(self, symbol: str, reason: str = "manual") -> None:
        """
        鍏抽棴缃戞牸

        Args:
            symbol: 浜ゆ槗瀵?
            reason: 鍏抽棴鍘熷洜
        """
        if symbol not in self.grid_states:
            return

        logger.info(f"鍏抽棴缃戞牸: {symbol}, 鍘熷洜: {reason}")

        grid_state = self.grid_states[symbol]

        # 鎾ら攢鎵€鏈夋寕鍗?
        all_order_ids = []
        for _, order_id in self._iter_order_items(grid_state.upper_orders):
            all_order_ids.append(order_id)
        for _, order_id in self._iter_order_items(grid_state.lower_orders):
            all_order_ids.append(order_id)
        for order_id in all_order_ids:
            try:
                self.connector.cancel_order(order_id, symbol)
            except Exception as e:
                logger.warning(f"鎾ゅ崟澶辫触: {e}")

        # 甯備环骞虫帀鎵€鏈夋寔浠?
        position = self.position_mgr.get_symbol_position(symbol)
        if position and position.base_position:
            try:
                size = position.base_position.size
                self.connector.place_order(
                    symbol=symbol,
                    side='buy',  # 骞崇┖
                    amount=size,
                    order_type='market',
                    reduce_only=True  # 寮哄埗鍙噺浠?
                )
                logger.info(f"甯備环骞充粨: {symbol}, 鏁伴噺={size}")
            except Exception as e:
                logger.error(f"骞充粨澶辫触: {e}")

        # 绉婚櫎缃戞牸鐘舵€?
        del self.grid_states[symbol]

        # 绉婚櫎浠撲綅
        self.position_mgr.remove_position(symbol)

    def recover_grid_from_position(self, symbol: str, entry_price: float) -> bool:
        """
        浠庣幇鏈夋寔浠撴仮澶嶇綉鏍肩姸鎬侊紙浣跨敤鎸佷粨鎴愭湰浠烽噸寤虹綉鏍硷級

        Args:
            symbol: 浜ゆ槗瀵?
            entry_price: 鏁版嵁搴撲腑淇濆瓨鐨勫叆鍦轰环锛堝皢琚拷鐣ワ級

        Returns:
            鏄惁鎴愬姛
        """
        try:
            # 馃敡 NEW: 鏌ヨ褰撳墠鎸佷粨鐨勫疄闄呮垚鏈环
            positions = self.connector.query_positions()
            short_position = next((p for p in positions if p.symbol == symbol and p.side == 'short'), None)

            if not short_position:
                logger.error(f"鎭㈠缃戞牸澶辫触: {symbol} 鏈壘鍒扮┖澶存寔浠?)
                return False

            # 浣跨敤鎸佷粨鐨勫疄闄呮垚鏈环浣滀负entry_price
            actual_entry_price = short_position.entry_price
            logger.info(
                f"鎭㈠缃戞牸鐘舵€? {symbol}\n"
                f"  鏁版嵁搴揺ntry_price: {entry_price:.6f}\n"
                f"  鎸佷粨鎴愭湰浠? {actual_entry_price:.6f}\n"
                f"  浣跨敤鎸佷粨鎴愭湰浠烽噸寤虹綉鏍?
            )

            # 濡傛灉宸茬粡鏈塯rid_state锛岃烦杩?
            if symbol in self.grid_states:
                logger.info(f"缃戞牸鐘舵€佸凡瀛樺湪: {symbol}")
                return True

            # 馃敡 浣跨敤鎸佷粨鎴愭湰浠疯绠楃綉鏍间环鏍?
            grid_prices = self.calculate_grid_prices(actual_entry_price)

            # 鍒涘缓缃戞牸鐘舵€?
            grid_state = GridState(
                symbol=symbol,
                entry_price=actual_entry_price,  # 浣跨敤鎸佷粨鎴愭湰浠?
                grid_prices=grid_prices
            )

            # 鏌ヨ鐜版湁鎸傚崟
            open_orders = self.connector.query_open_orders(symbol)

            # 濡傛灉娌℃湁鎸傚崟锛岄噸鏂版寕缃戞牸鍗?
            if not open_orders:
                logger.info(f"鏈彂鐜版寕鍗曪紝閲嶆柊鎸備笂鏂圭綉鏍? {symbol}")
                self.grid_states[symbol] = grid_state

                # 鎸備笂鏂瑰紑绌哄崟
                self._place_upper_grid_orders(symbol, grid_state)

                # 鎸傚熀纭€浠撲綅鐨勫垎灞傛鐩堝崟锛堟仮澶嶆椂鎸佷粨宸插瓨鍦級
                logger.info(f"鎸傚熀纭€浠撲綅鍒嗗眰姝㈢泩鍗? {symbol}")
                self._place_base_position_take_profit(symbol, grid_state)

                # 鏍囪缃戞牸涓哄凡楠岃瘉
                grid_state.grid_integrity_validated = True
            else:
                logger.info(f"鍙戠幇{len(open_orders)}涓寕鍗曪紝鎭㈠缃戞牸鐘舵€?)

                # 瑙ｆ瀽鐜版湁璁㈠崟锛屾仮澶島pper_orders/lower_orders
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
                            logger.info(f"  鎭㈠涓婃柟缃戞牸璁㈠崟 @ {order_price:.6f} (Grid{level})")
                        else:
                            self._add_order_id(grid_state.lower_orders, order_price, order.order_id)
                            logger.info(f"  鎭㈠涓嬫柟缃戞牸璁㈠崟 @ {order_price:.6f} (Grid{level})")
                        continue

                    # fallback: price matching
                    if order.side == 'sell':
                        for level in grid_state.grid_prices.get_upper_levels():
                            target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='sell')
                            if abs(order_price - target_price) / target_price < 0.001:  # 0.1%瀹瑰樊
                                self._add_order_id(grid_state.upper_orders, order_price, order.order_id)
                                logger.info(f"  鎭㈠涓婃柟缃戞牸璁㈠崟 @ {order_price:.6f} (Grid{level})")
                                break
                    elif order.side == 'buy':
                        for level in grid_state.grid_prices.get_lower_levels():
                            target_price = self._quantize_price(symbol, grid_state.grid_prices.grid_levels[level], side='buy')
                            if abs(order_price - target_price) / target_price < 0.001:
                                self._add_order_id(grid_state.lower_orders, order_price, order.order_id)
                                logger.info(f"  鎭㈠涓嬫柟缃戞牸璁㈠崟 @ {order_price:.6f} (Grid{level})")
                                break

                upper_order_count = self._count_orders(grid_state.upper_orders)
                lower_order_count = self._count_orders(grid_state.lower_orders)
                logger.info(f"璁㈠崟鎭㈠瀹屾垚: {upper_order_count}涓笂鏂圭綉鏍? {lower_order_count}涓笅鏂圭綉鏍?)
                self.grid_states[symbol] = grid_state

                # 鎭㈠鏈畬鎴愮殑涓婃柟->姝㈢泩寰幆
                self._restore_cycles_from_db(symbol, grid_state, open_orders)

                # 琛ュ厖缂哄け鐨勮鍗?
                missing_upper = max(len(grid_state.grid_prices.get_upper_levels()) - len(grid_state.upper_orders), 0)

                min_ratio = self.config.position.min_base_position_ratio
                closeable_ratio = 1.0 - min_ratio
                total_lower_levels = len(grid_state.grid_prices.get_lower_levels())
                allowed_lower_levels = int(total_lower_levels * closeable_ratio)
                missing_lower = max(allowed_lower_levels - len(grid_state.lower_orders), 0)

                if missing_upper > 0:
                    logger.info(f"妫€娴嬪埌{missing_upper}涓己澶辩殑涓婃柟缃戞牸璁㈠崟锛屽紑濮嬭ˉ鍏?..")
                    self._place_upper_grid_orders(symbol, grid_state)

                if missing_lower > 0:
                    logger.info(f"妫€娴嬪埌{missing_lower}涓己澶辩殑涓嬫柟缃戞牸璁㈠崟锛屽紑濮嬭ˉ鍏?..")
                    self._place_base_position_take_profit(symbol, grid_state)
                    logger.info(f"鎭㈠鍚庢鐩堝崟鏁伴噺: {self._count_orders(grid_state.lower_orders)}/{allowed_lower_levels}")

                # 鏍囪缃戞牸涓哄凡楠岃瘉锛堝厑璁稿悗缁慨澶嶆満鍒惰繍琛岋級
                grid_state.grid_integrity_validated = True

            logger.info(f"缃戞牸鎭㈠瀹屾垚: {symbol}")
            return True

        except Exception as e:
            logger.error(f"缃戞牸鎭㈠澶辫触: {symbol}: {e}")
            return False

    def _calculate_amount(self, symbol: str, margin: float, price: float) -> float:
        """
        璁＄畻涓嬪崟鏁伴噺

        Args:
            symbol: 浜ゆ槗瀵?
            margin: 淇濊瘉閲?
            price: 浠锋牸

        Returns:
            鍚堢害鏁伴噺
        """
        leverage = self.config.account.leverage
        # 鍚嶄箟浠峰€?= 淇濊瘉閲?脳 鏉犳潌
        notional = margin * leverage
        # 鍚堢害鏁伴噺 = 鍚嶄箟浠峰€?/ 浠锋牸
        amount = notional / price

        # 鑾峰彇绮惧害
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
        鑾峰彇缂撳瓨鐨勭┖澶翠粨浣嶏紙鍑忓皯API璋冪敤锛?

        Args:
            symbol: 浜ゆ槗瀵?
            force_refresh: 鏄惁寮哄埗鍒锋柊锛堝拷鐣ョ紦瀛橈級

        Returns:
            Position瀵硅薄锛屽鏋滄壘涓嶅埌鍒欒繑鍥濶one
        """
        now = datetime.now(timezone.utc)

        # 妫€鏌ョ紦瀛?
        if not force_refresh and symbol in self._position_cache:
            cached_pos, timestamp = self._position_cache[symbol]
            age = (now - timestamp).total_seconds()

            if age < self._cache_ttl:
                logger.debug(f"{symbol} 浣跨敤缂撳瓨浠撲綅 (缂撳瓨骞撮緞: {age:.1f}绉?")
                return cached_pos

        # 缂撳瓨澶辨晥鎴栦笉瀛樺湪锛屾煡璇㈡柊鏁版嵁
        try:
            positions = self.connector.query_positions()

            # 馃攳 璋冭瘯鏃ュ織锛氭墦鍗版墍鏈変粨浣嶄俊鎭?
            logger.info(f"{symbol} 鏌ヨ鍒?{len(positions)} 涓粨浣?")
            for idx, p in enumerate(positions):
                logger.info(
                    f"  [{idx}] symbol={p.symbol}, side={p.side}, size={p.size}, "
                    f"contracts={p.contracts}, entry_price={p.entry_price}"
                )

            # 鏌ユ壘绌哄ご浠撲綅锛堜娇鐢╯ide瀛楁锛屾洿鍙潬锛?
            short_pos = next((p for p in positions if p.symbol == symbol and p.side == 'short'), None)

            if short_pos:
                # 鏇存柊缂撳瓨
                self._position_cache[symbol] = (short_pos, now)
                logger.debug(f"{symbol} 鍒锋柊浠撲綅缂撳瓨: {short_pos.size}寮?@ {short_pos.entry_price}")
                return short_pos
            else:
                logger.warning(f"{symbol} 鈿狅笍 鏈壘鍒扮┖澶翠粨浣嶏紒")
                logger.warning(f"  鏌ヨ鏉′欢: symbol={symbol}, side='short'")

                # 灏濊瘯鏀惧鏉′欢锛氬彧鍖归厤symbol
                any_pos = next((p for p in positions if p.symbol == symbol), None)
                if any_pos:
                    logger.warning(
                        f"  鈿狅笍 鎵惧埌鍖归厤symbol鐨勪粨浣嶏紝浣唖ide涓嶆槸'short': "
                        f"side={any_pos.side}, size={any_pos.size}"
                    )
                else:
                    logger.warning(f"  鈿狅笍 瀹屽叏娌℃湁鍖归厤symbol鐨勪粨浣?)

                return None

        except Exception as e:
            logger.error(f"{symbol} 鏌ヨ浠撲綅澶辫触: {e}")

            # 濡傛灉鏌ヨ澶辫触锛屽皾璇曡繑鍥炶繃鏈熺紦瀛橈紙鎬绘瘮娌℃湁濂斤級
            if symbol in self._position_cache:
                cached_pos, timestamp = self._position_cache[symbol]
                age = (now - timestamp).total_seconds()
                logger.warning(f"{symbol} 浣跨敤杩囨湡缂撳瓨 (缂撳瓨骞撮緞: {age:.1f}绉?")
                return cached_pos

            return None


    def _validate_total_exposure_before_buy_order(
        self,
        symbol: str,
        grid_state: GridState,
        new_order_amount: float
    ) -> tuple:
        """Validate that new buy orders won't reduce below min base ratio."""
        short_position = self._get_cached_short_position(symbol)
        if not short_position:
            return False, 0.0, "no short position"

        current_short_size = short_position.size
        base_margin = self.config.position.base_margin
        min_ratio = self.config.position.min_base_position_ratio
        expected_base = self._calculate_amount(symbol, base_margin, grid_state.entry_price)
        min_base_amount = expected_base * min_ratio

        pending_lower_total = 0.0
        try:
            open_orders = self.connector.query_open_orders(symbol)
            open_order_map = {order.order_id: order for order in open_orders}
            for price, order_ids in grid_state.lower_orders.items():
                for order_id in order_ids:
                    order = open_order_map.get(order_id)
                    if order and order.side == 'buy':
                        pending_lower_total += order.amount
        except Exception:
            pending_lower_total = 0.0

        max_closeable = max(current_short_size - min_base_amount, 0.0)
        available_capacity = max_closeable - pending_lower_total

        safe_amount = min(new_order_amount, available_capacity)
        if max_closeable <= 0:
            warning = "min base reserved, no closeable capacity"
            return False, 0.0, warning
        if available_capacity <= 0:
            warning = "min base reserved, pending orders exceed closeable"
            return False, 0.0, warning

        ratio = (pending_lower_total + safe_amount) / max_closeable if max_closeable > 0 else 0
        if safe_amount < new_order_amount * 0.9:
            warning = (
                f"lower grid capacity insufficient: closeable={available_capacity:.2f}/{max_closeable:.2f} "
                f"target={new_order_amount:.2f} safe={safe_amount:.2f} ({ratio*100:.1f}%)"
            )
            return False, safe_amount, warning
        elif ratio > 0.90:
            warning = f"lower grid near saturation: {ratio*100:.1f}%"
            return True, safe_amount, warning
        else:
            return True, safe_amount, ""
    def _reconcile_position_with_grids(self, symbol: str, grid_state: GridState) -> None:
        """
        瀹氭湡瀵硅处锛氶獙璇佹寔浠撲笌缃戞牸鐘舵€佷竴鑷?

        姣?0绉掕繍琛屼竴娆★紝妫€鏌ワ細
        1. 褰撳墠绌哄ご浠撲綅澶у皬
        2. 鎵€鏈塸ending lower order鎬婚
        3. 濡傛灉lower鎬婚 > 绌哄ご浠撲綅 * 0.95: 璁板綍璀︽姤锛堜笉寮哄埗鎾ゅ崟锛?

        Args:
            symbol: 浜ゆ槗瀵?
            grid_state: 缃戞牸鐘舵€?
        """
        now = datetime.now(timezone.utc)

        # 妫€鏌ユ槸鍚﹂渶瑕佸璐︼紙60绉掗棿闅旓級
        if symbol in self._last_reconciliation:
            elapsed = (now - self._last_reconciliation[symbol]).total_seconds()
            if elapsed < self._reconciliation_interval:
                return

        self._last_reconciliation[symbol] = now

        # 1. 鑾峰彇褰撳墠绌哄ご浠撲綅
        short_position = self._get_cached_short_position(symbol, force_refresh=True)

        if not short_position:
            logger.critical(f"{symbol} 鈿狅笍 CRITICAL: 瀵硅处澶辫触 - 鏃犳硶鎵惧埌绌哄ご浠撲綅锛?)
            return

        short_size = short_position.size

        # 2. 缁熻鎵€鏈変笅鏂逛拱鍗曠殑鎬绘暟閲?
        total_lower_amount = 0.0
        lower_order_count = 0

        try:
            open_orders = self.connector.query_open_orders(symbol)
            open_order_map = {order.order_id: order for order in open_orders}

            for price, order_ids in grid_state.lower_orders.items():
                for order_id in order_ids:
                    order = open_order_map.get(order_id)
                    if order and order.side == 'buy':
                        total_lower_amount += order.amount
                        lower_order_count += 1

        except Exception as e:
            logger.error(f"{symbol} 瀵硅处鏃舵煡璇㈡寕鍗曞け璐? {e}")
            return

        # 3. 璁＄畻骞宠　姣斾緥
        ratio = total_lower_amount / short_size if short_size > 0 else 0

        # 4. 璁板綍骞宠　鐘舵€侊紙浠呰褰曪紝涓嶅仛鎾ゅ崟鎴栨爣璁帮級
        if ratio > 0.95:
            logger.warning(
                f"{symbol} 涓嬫柟涔板崟杩囬珮: "
                f"{total_lower_amount:.2f}寮?({lower_order_count}涓鍗?, "
                f"绌哄ご浠撲綅={short_size:.2f}寮? "
                f"姣斾緥={ratio*100:.1f}%"
            )
        elif ratio > 0.85:
            logger.warning(
                f"{symbol} 涓嬫柟涔板崟鎺ヨ繎涓婇檺: "
                f"{total_lower_amount:.2f}/{short_size:.2f}寮?({ratio*100:.1f}%)"
            )
        else:
            logger.info(
                f"{symbol} 浠撲綅骞宠　鍋ュ悍: "
                f"涓嬫柟涔板崟={total_lower_amount:.2f}寮?({lower_order_count}涓?, "
                f"绌哄ご浠撲綅={short_size:.2f}寮? "
                f"姣斾緥={ratio*100:.1f}%"
            )


    def _assert_no_long_positions(self, symbol: str) -> bool:
        """
        鏂█妫€鏌ワ細缁濆涓嶅厑璁稿澶翠粨浣嶅瓨鍦?

        濡傛灉妫€娴嬪埌澶氬ご锛?
        1. 璁板綍CRITICAL鏃ュ織
        2. 绔嬪嵆鎾ら攢鎵€鏈変笅鏂逛拱鍗?
        3. 瑙﹀彂鍛婅閫氱煡

        Args:
            symbol: 浜ゆ槗瀵?

        Returns:
            bool: 鏄惁妫€娴嬪埌澶氬ご浠撲綅锛圱rue = 妫€娴嬪埌锛?
        """
        try:
            positions = self.connector.query_positions()
            long_position = next((p for p in positions if p.symbol == symbol and p.side == 'long'), None)

            if long_position:
                logger.critical(
                    f"{symbol} 鈿狅笍鈿狅笍鈿狅笍 FORBIDDEN LONG POSITION DETECTED 鈿狅笍鈿狅笍鈿狅笍\n"
                    f"  浠撲綅澶у皬: {long_position.size}寮燶n"
                    f"  寮€浠撲环鏍? {long_position.entry_price}\n"
                    f"  鏈疄鐜扮泩浜? {long_position.unrealized_pnl}\n"
                    f"  杩欐槸涓ラ噸閿欒锛佺珛鍗抽噰鍙栧簲鎬ユ帾鏂?.."
                )

                # 搴旀€ユ帾鏂斤細鎾ら攢鎵€鏈変笅鏂逛拱鍗?
                if symbol in self.grid_states:
                    grid_state = self.grid_states[symbol]

                    cancelled_count = 0
                    for price, order_id in list(self._iter_order_items(grid_state.lower_orders)):
                        try:
                            self.connector.cancel_order(order_id, symbol)
                            cancelled_count += 1
                        except Exception as e:
                            logger.error(f"鎾ゅ崟澶辫触 @ {price}: {e}")

                    grid_state.lower_orders.clear()
                    logger.critical(f"{symbol} 宸叉挙閿€ {cancelled_count} 涓笅鏂逛拱鍗?)

                # TODO: 娣诲姞閫氱煡鏈哄埗锛坋mail/webhook/telegram锛?
                return True

            return False

        except Exception as e:
            logger.error(f"{symbol} 妫€鏌ュ澶翠粨浣嶅け璐? {e}")
            return False


    def _calculate_grid_level(self, price: float, entry_price: float, spacing: float) -> int:
        """
        鏍规嵁浠锋牸璁＄畻缃戞牸灞傜骇

        Args:
            price: 鐩爣浠锋牸
            entry_price: 鍏ュ満浠锋牸
            spacing: 缃戞牸闂磋窛

        Returns:
            缃戞牸灞傜骇锛堟鏁?涓婃柟锛岃礋鏁?涓嬫柟锛?=鍏ュ満浠凤級
        """
        if price >= entry_price:
            # 涓婃柟缃戞牸
            level = round(math.log(price / entry_price) / math.log(1 + spacing))
            return max(1, level)  # 鑷冲皯涓?
        else:
            # 涓嬫柟缃戞牸
            level = round(math.log(price / entry_price) / math.log(1 - spacing))
            return min(-1, level)  # 鑷冲皯涓?1

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
        鏌ユ壘鍖归厤鐨勪笂鏂瑰紑浠?

        Args:
            grid_state: 缃戞牸鐘舵€?
            lower_price: 涓嬫柟鎴愪氦浠锋牸

        Returns:
            鍖归厤鐨勪笂鏂瑰紑浠撲俊鎭紝濡傛灉娌℃湁鍒欒繑鍥?None
        """
        if not grid_state.filled_upper_grids:
            return None

        # 鏌ユ壘 matched_lower_price 鏈€鎺ヨ繎 lower_price 鐨勪笂鏂瑰紑浠?
        best_match = None
        min_diff = float('inf')

        for fill_info in grid_state.filled_upper_grids.values():
            if fill_info.matched_lower_price is None:
                continue

            diff = abs(fill_info.matched_lower_price - lower_price)
            if diff < min_diff:
                min_diff = diff
                best_match = fill_info

        # 濡傛灉宸紓灏忎簬 0.5%锛岃涓烘槸鍖归厤鐨?
        if best_match and min_diff / lower_price < 0.005:
            return best_match

        return None
