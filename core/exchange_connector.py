"""
交易执行引擎模块
Exchange Connector Module

封装Binance API调用,处理订单执行和查询
"""

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, List, Optional, Any

from data_fetcher import BinanceDataFetcher
from utils.exceptions import (
    OrderError, NetworkError, RateLimitError, ExchangeError
)
from utils.logger import get_logger

logger = get_logger("exchange")


@dataclass
class Order:
    """订单数据结构"""
    order_id: str
    client_order_id: str
    symbol: str
    side: str                  # 'buy' / 'sell'
    order_type: str            # 'limit' / 'market'
    price: Optional[float]
    amount: float
    filled: float
    remaining: float
    status: str                # 'open' / 'filled' / 'cancelled' / 'expired'
    timestamp: datetime
    update_time: datetime
    reduce_only: bool = False
    post_only: bool = False


@dataclass
class Position:
    """持仓数据结构"""
    symbol: str
    side: str                  # 'long' / 'short'
    size: float                # 合约数量(负数表示空头)
    contracts: float           # 合约数量
    entry_price: float         # 开仓价格
    mark_price: float          # 标记价格
    margin: float              # 保证金
    leverage: int              # 杠杆倍数
    unrealized_pnl: float      # 未实现盈亏
    liquidation_price: Optional[float]
    timestamp: datetime


@dataclass
class Balance:
    """账户余额数据结构"""
    total: float               # 总余额
    available: float           # 可用余额
    used: float                # 已用保证金
    timestamp: datetime


class RateLimiter:
    """API速率限制器"""

    def __init__(self, max_calls: int = 1200, window: int = 60):
        """
        Args:
            max_calls: 时间窗口内最大调用次数
            window: 时间窗口(秒)
        """
        self.max_calls = max_calls
        self.window = window
        self.calls = deque()

    def wait_if_needed(self) -> None:
        """如果超过速率限制,则等待"""
        now = time.time()

        # 移除过期的调用记录
        while self.calls and self.calls[0] < now - self.window:
            self.calls.popleft()

        # 如果达到限制,等待
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] + self.window - now
            if sleep_time > 0:
                logger.warning(f"达到速率限制,等待{sleep_time:.2f}秒")
                time.sleep(sleep_time)

        self.calls.append(now)


def retry_on_network_error(max_retries: int = 3, backoff: float = 2.0):
    """装饰器:网络错误时自动重试"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, NetworkError) as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        break
                    wait_time = backoff ** attempt
                    logger.warning(
                        f"{func.__name__}网络错误,重试{attempt + 1}/{max_retries}, "
                        f"等待{wait_time}秒: {e}"
                    )
                    time.sleep(wait_time)
            raise NetworkError(f"{func.__name__}失败: {last_exception}")

        return wrapper
    return decorator


class ExchangeConnector:
    """
    交易所连接器

    封装Binance API调用,处理订单执行、查询持仓等操作
    """

    def __init__(self, fetcher: BinanceDataFetcher, rate_limit_config: Optional[Dict] = None):
        """
        Args:
            fetcher: BinanceDataFetcher实例
            rate_limit_config: 速率限制配置
        """
        self.exchange = fetcher.exchange
        self.fetcher = fetcher

        # 速率限制器
        if rate_limit_config:
            self.rate_limiter = RateLimiter(
                max_calls=rate_limit_config.get('max_calls_per_minute', 1200),
                window=60
            )
        else:
            self.rate_limiter = RateLimiter()

        # 订单缓存
        self.order_cache: Dict[str, Order] = {}

        # 设置单向持仓模式（只能持有一个方向的仓位）
        try:
            self.exchange.set_position_mode(False)  # False = 单向持仓模式
            logger.info("已设置为单向持仓模式(One-Way Mode)")
        except Exception as e:
            logger.warning(f"设置持仓模式失败(可能已是单向模式): {e}")

        logger.info("交易所连接器初始化完成")

    @retry_on_network_error()
    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        reduce_only: bool = False,
        post_only: bool = True,
        client_order_id: Optional[str] = None
    ) -> Order:
        """
        下单

        Args:
            symbol: 交易对(如"BTC/USDT:USDT")
            side: 方向('buy'买入平空, 'sell'卖出开空)
            amount: 数量(合约张数)
            price: 价格(限价单必填)
            order_type: 订单类型('limit'/'market')
            reduce_only: 只减仓
            post_only: Post-Only(仅Maker)
            client_order_id: 自定义订单ID

        Returns:
            Order对象
        """
        self.rate_limiter.wait_if_needed()

        try:
            # 构建订单参数
            params = {}

            # 只在需要reduce_only时才添加参数
            if reduce_only:
                params['reduceOnly'] = True

            if post_only and order_type == 'limit':
                params['timeInForce'] = 'GTX'  # Post-Only

            if client_order_id:
                params['newClientOrderId'] = client_order_id

            # 下单
            logger.info(
                f"下单: {symbol} {side} {amount} @ {price} "
                f"(type={order_type}, post_only={post_only}, reduce_only={reduce_only})"
            )

            result = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )

            # 解析订单
            order = self._parse_order(result)
            self.order_cache[order.order_id] = order

            logger.info(f"下单成功: order_id={order.order_id}, status={order.status}")
            return order

        except Exception as e:
            logger.error(f"下单失败: {symbol} {side} {amount}: {e}")
            raise OrderError(f"Failed to place order: {e}")

    def place_order_with_maker_retry(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        reduce_only: bool = False,
        post_only: bool = True,
        client_order_id: Optional[str] = None,
        max_retries: int = 5
    ) -> Order:
        """
        带价格偏移重试的下单方法（解决-5022错误）

        当Post-Only订单因价格不合适被拒绝时，自动调整价格重试

        Args:
            symbol: 交易对
            side: 方向('buy'/'sell')
            amount: 数量
            price: 初始价格（仅供参考，会被调整）
            order_type: 订单类型
            reduce_only: 只减仓
            post_only: Post-Only
            client_order_id: 自定义订单ID
            max_retries: 最大重试次数

        Returns:
            Order对象

        Raises:
            OrderError: 达到最大重试次数仍失败
        """
        # 非Post-Only订单或非限价单，直接下单
        if not post_only or order_type != 'limit':
            return self.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                order_type=order_type,
                reduce_only=reduce_only,
                post_only=post_only,
                client_order_id=client_order_id
            )

        # 获取订单簿和tick_size
        try:
            book = self.fetch_order_book(symbol)
            tick_size = book['tick_size']
        except Exception as e:
            logger.warning(f"获取订单簿失败，使用原价格直接下单: {e}")
            return self.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                order_type=order_type,
                reduce_only=reduce_only,
                post_only=post_only,
                client_order_id=client_order_id
            )

        # 确定基准价格
        if price is not None:
            # 优先使用用户指定的价格
            base_price = price
        else:
            # 如果用户没有指定价格，使用订单簿当前价格
            if side == 'sell':  # 开空
                base_price = book['best_ask']
            else:  # 平空（buy）
                base_price = book['best_bid']

        logger.info(
            f"Post-Only下单准备: {symbol} {side} {amount}张, "
            f"目标价格={base_price:.8f}, tick={tick_size:.8f}"
        )

        # 重试循环（从0开始，第0次尝试使用原始价格）
        last_error = None
        for attempt in range(0, max_retries):
            try:
                # 计算偏移后的价格
                if attempt == 0:
                    # 第一次尝试：使用原始价格
                    adjusted_price = base_price
                else:
                    # 后续尝试：进行偏移
                    if side == 'sell':
                        # 开空：向上偏移（挂在卖方队列）
                        adjusted_price = base_price + (attempt * tick_size)
                    else:
                        # 平空：向下偏移（挂在买方队列）
                        adjusted_price = base_price - (attempt * tick_size)

                # 确保价格为正
                if adjusted_price <= 0:
                    adjusted_price = tick_size

                if attempt == 0:
                    logger.info(
                        f"尝试Post-Only下单 (第1次): {symbol} {side} @ {adjusted_price:.8f} (原始价格)"
                    )
                else:
                    offset_direction = "+" if side == 'sell' else "-"
                    logger.info(
                        f"尝试Post-Only下单 (第{attempt + 1}次): {symbol} {side} @ {adjusted_price:.8f} "
                        f"(偏移{offset_direction}{attempt}×tick)"
                    )

                # 尝试下单
                order = self.place_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=adjusted_price,
                    order_type=order_type,
                    reduce_only=reduce_only,
                    post_only=post_only,
                    client_order_id=client_order_id
                )

                if attempt == 0:
                    logger.info(f"✅ Post-Only下单成功 (使用原始价格)")
                else:
                    logger.info(f"✅ Post-Only下单成功 (第{attempt + 1}次尝试)")
                return order

            except Exception as e:
                error_msg = str(e)
                last_error = e

                # 判断是否为-5022错误（Post-Only被拒绝）
                if '-5022' in error_msg or 'Post Only order will be rejected' in error_msg:
                    logger.warning(
                        f"⚠️ Post-Only被拒绝 (第{attempt + 1}次尝试): 价格{adjusted_price:.8f}会立即成交"
                    )

                    if attempt == max_retries - 1:
                        logger.error(f"❌ 达到最大重试次数({max_retries}), 放弃下单")
                        raise OrderError(
                            f"Post-Only下单失败，已重试{max_retries}次。"
                            f"建议检查市场深度或调整网格间距。"
                        )

                    # 继续下一次重试
                    logger.info(f"将在 0.5 秒后重试第 {attempt + 2} 次...")
                    time.sleep(0.5)
                    continue
                else:
                    # 其他错误（如余额不足、网络错误等）直接抛出
                    logger.error(f"下单失败（非-5022错误）: {error_msg}")
                    raise

        # 理论上不会到这里，但为了安全
        raise OrderError(f"Post-Only下单失败: {last_error}")

    @retry_on_network_error()
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        撤单

        Args:
            order_id: 订单ID
            symbol: 交易对

        Returns:
            是否成功
        """
        self.rate_limiter.wait_if_needed()

        try:
            logger.info(f"撤单: order_id={order_id}, symbol={symbol}")
            self.exchange.cancel_order(order_id, symbol)

            # 更新缓存
            if order_id in self.order_cache:
                self.order_cache[order_id].status = 'cancelled'

            logger.info(f"撤单成功: order_id={order_id}")
            return True

        except Exception as e:
            logger.error(f"撤单失败: order_id={order_id}: {e}")
            return False

    @retry_on_network_error()
    def query_order(self, order_id: str, symbol: str) -> Order:
        """
        查询订单状态

        Args:
            order_id: 订单ID
            symbol: 交易对

        Returns:
            Order对象
        """
        self.rate_limiter.wait_if_needed()

        try:
            result = self.exchange.fetch_order(order_id, symbol)
            order = self._parse_order(result)

            # 更新缓存
            self.order_cache[order.order_id] = order

            return order

        except Exception as e:
            logger.error(f"查询订单失败: order_id={order_id}: {e}")
            raise OrderError(f"Failed to query order: {e}", order_id=order_id)

    @retry_on_network_error()
    def query_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        查询所有挂单

        Args:
            symbol: 交易对(None表示所有)

        Returns:
            Order列表
        """
        self.rate_limiter.wait_if_needed()

        try:
            results = self.exchange.fetch_open_orders(symbol)
            orders = [self._parse_order(r) for r in results]

            # 更新缓存
            for order in orders:
                self.order_cache[order.order_id] = order

            logger.debug(f"查询挂单: {len(orders)}个")
            return orders

        except Exception as e:
            logger.error(f"查询挂单失败: {e}")
            return []

    @retry_on_network_error()
    def query_positions(self) -> List[Position]:
        """
        查询所有持仓

        Returns:
            Position列表
        """
        self.rate_limiter.wait_if_needed()

        try:
            results = self.exchange.fetch_positions()
            positions = []
            filtered_count = 0

            # 🔍 调试日志：查看原始数据
            logger.debug(f"fetch_positions返回 {len(results)} 条记录")

            for r in results:
                # 只保留有仓位的
                contracts = float(r.get('contracts', 0) or 0)
                symbol = r.get('symbol', 'UNKNOWN')

                if abs(contracts) < 0.001:
                    # 🔍 调试日志：记录被过滤的仓位
                    logger.debug(f"过滤空仓位: {symbol}, contracts={contracts}")
                    filtered_count += 1
                    continue

                position = self._parse_position(r)
                positions.append(position)

            logger.debug(f"查询持仓: {len(positions)}个有效, {filtered_count}个空仓位已过滤")

            # 🔍 如果没有找到任何仓位但有订单刚成交，输出更详细的信息
            if len(positions) == 0 and len(results) > 0:
                logger.warning(f"⚠️ fetch_positions返回了{len(results)}条数据，但全部contracts<0.001被过滤!")
                logger.warning("这可能是交易所API延迟导致的，建议增加重试间隔")

            return positions

        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
            return []

    @retry_on_network_error()
    def query_balance(self) -> Balance:
        """
        查询账户余额（包括所有币种折算的保证金）

        Returns:
            Balance对象
        """
        self.rate_limiter.wait_if_needed()

        try:
            result = self.exchange.fetch_balance()

            # 优先使用总保证金余额（包括所有币种）
            info = result.get('info', {})

            # 币安合约API返回的总保证金信息
            total_margin_balance = float(info.get('totalMarginBalance', 0))
            available_balance = float(info.get('availableBalance', 0))
            total_wallet_balance = float(info.get('totalWalletBalance', 0))

            # 如果API返回了总保证金，使用总保证金；否则回退到USDT余额
            if total_margin_balance > 0 or available_balance > 0:
                balance = Balance(
                    total=total_margin_balance if total_margin_balance > 0 else total_wallet_balance,
                    available=available_balance,
                    used=total_margin_balance - available_balance if total_margin_balance > 0 else 0,
                    timestamp=datetime.now(timezone.utc)
                )
                logger.debug(
                    f"账户保证金余额: total={balance.total:.2f} USDT, "
                    f"available={balance.available:.2f} USDT (包括所有币种)"
                )
            else:
                # 回退到仅USDT余额
                usdt = result.get('USDT', {})
                balance = Balance(
                    total=float(usdt.get('total', 0)),
                    available=float(usdt.get('free', 0)),
                    used=float(usdt.get('used', 0)),
                    timestamp=datetime.now(timezone.utc)
                )
                logger.debug(f"账户USDT余额: total={balance.total}, available={balance.available}")

            return balance

        except Exception as e:
            logger.error(f"查询余额失败: {e}")
            raise ExchangeError(f"Failed to query balance: {e}")

    @retry_on_network_error()
    def get_current_price(self, symbol: str) -> float:
        """
        获取当前价格(标记价格)

        Args:
            symbol: 交易对

        Returns:
            当前价格
        """
        self.rate_limiter.wait_if_needed()

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            mark_price = float(ticker.get('info', {}).get('markPrice', 0))

            if mark_price <= 0:
                mark_price = float(ticker.get('last', 0))

            return mark_price

        except Exception as e:
            logger.error(f"获取价格失败: {symbol}: {e}")
            raise ExchangeError(f"Failed to get price: {e}")

    @retry_on_network_error()
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        获取资金费率

        Args:
            symbol: 交易对

        Returns:
            资金费率(None表示获取失败)
        """
        self.rate_limiter.wait_if_needed()

        try:
            fr = self.exchange.fetch_funding_rate(symbol)
            return fr.get('fundingRate')

        except Exception as e:
            logger.warning(f"获取资金费率失败: {symbol}: {e}")
            return None

    @retry_on_network_error()
    def fetch_order_book(self, symbol: str) -> Dict[str, float]:
        """
        获取订单簿深度数据

        Args:
            symbol: 交易对

        Returns:
            包含 best_bid, best_ask, tick_size 的字典
        """
        self.rate_limiter.wait_if_needed()

        try:
            # 获取订单簿（只需要5档即可）
            order_book = self.exchange.fetch_order_book(symbol, limit=5)

            # 提取最佳买卖价
            best_bid = float(order_book['bids'][0][0]) if order_book['bids'] else 0.0
            best_ask = float(order_book['asks'][0][0]) if order_book['asks'] else 0.0

            # 获取 tick_size（最小价格变动单位）
            market = self.exchange.markets.get(symbol)
            if market and 'precision' in market and 'price' in market['precision']:
                # 从 precision 计算 tick_size
                price_precision = market['precision']['price']
                if isinstance(price_precision, int):
                    # precision 是小数位数
                    tick_size = 10 ** -price_precision
                else:
                    # precision 直接是 tick_size
                    tick_size = float(price_precision)
            else:
                # 备用方案：从价格推断（使用价格的1/100000）
                if best_ask > 0:
                    tick_size = best_ask / 100000
                else:
                    tick_size = 0.00000001  # 默认值

            logger.debug(
                f"订单簿: {symbol} bid={best_bid:.8f}, ask={best_ask:.8f}, tick={tick_size:.8f}"
            )

            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'tick_size': tick_size
            }

        except Exception as e:
            logger.error(f"获取订单簿失败: {symbol}: {e}")
            raise ExchangeError(f"Failed to fetch order book: {e}")

    def _parse_order(self, raw: Dict[str, Any]) -> Order:
        """解析订单数据"""
        return Order(
            order_id=str(raw.get('id')),
            client_order_id=raw.get('clientOrderId', ''),
            symbol=raw.get('symbol'),
            side=raw.get('side'),
            order_type=raw.get('type'),
            price=float(raw.get('price') or 0),
            amount=float(raw.get('amount') or 0),
            filled=float(raw.get('filled') or 0),
            remaining=float(raw.get('remaining') or 0),
            status=raw.get('status'),
            timestamp=datetime.fromtimestamp(raw.get('timestamp', 0) / 1000, tz=timezone.utc),
            update_time=datetime.fromtimestamp(raw.get('lastUpdateTimestamp', 0) / 1000, tz=timezone.utc),
            reduce_only=raw.get('reduceOnly', False),
            post_only=raw.get('postOnly', False)
        )

    def _parse_position(self, raw: Dict[str, Any]) -> Position:
        """解析持仓数据"""
        info = raw.get('info', {})
        contracts = float(raw.get('contracts', 0) or 0)

        # 🔧 BUG FIX: contracts总是正数，需要从其他字段判断方向
        # Binance等交易所使用info.positionSide字段标识方向
        position_side = info.get('positionSide', '').upper()
        position_amt = float(info.get('positionAmt', 0) or 0)
        raw_side = raw.get('side', '').lower()

        # 判断优先级：
        # 1. info.positionSide (Binance使用 'SHORT' 或 'LONG')
        # 2. info.positionAmt的正负 (负数=空头)
        # 3. raw.side
        # 4. contracts的正负（最不可靠）
        if position_side == 'SHORT':
            side = 'short'
        elif position_side == 'LONG':
            side = 'long'
        elif position_amt < 0:
            side = 'short'
        elif position_amt > 0:
            side = 'long'
        elif raw_side in ['short', 'sell']:
            side = 'short'
        elif raw_side in ['long', 'buy']:
            side = 'long'
        elif contracts < 0:
            side = 'short'
        else:
            # 默认根据contracts正负判断，但这可能不准确
            side = 'long'
            logger.warning(
                f"无法从positionSide/positionAmt判断仓位方向: "
                f"symbol={raw.get('symbol')}, contracts={contracts}, "
                f"默认为long"
            )

        return Position(
            symbol=raw.get('symbol'),
            side=side,
            size=abs(contracts),
            contracts=contracts,
            entry_price=float(raw.get('entryPrice', 0) or 0),
            mark_price=float(raw.get('markPrice', 0) or 0),
            margin=float(info.get('isolatedMargin', 0) or 0),
            leverage=int(raw.get('leverage') or 1),
            unrealized_pnl=float(raw.get('unrealizedPnl', 0) or 0),
            liquidation_price=float(raw.get('liquidationPrice') or 0) if raw.get('liquidationPrice') else None,
            timestamp=datetime.now(timezone.utc)
        )

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取市场信息(精度、最小下单量等)"""
        markets = self.exchange.load_markets()
        market = markets.get(symbol)

        if not market:
            raise ExchangeError(f"Market not found: {symbol}")

        return {
            'symbol': symbol,
            'price_precision': market.get('precision', {}).get('price', 8),
            'amount_precision': market.get('precision', {}).get('amount', 3),
            'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0.001),
            'min_cost': market.get('limits', {}).get('cost', {}).get('min', 5.01),
        }
