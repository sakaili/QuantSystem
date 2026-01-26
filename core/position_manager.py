"""
仓位管理器模块
Position Manager Module

追踪和管理所有持仓
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .exchange_connector import ExchangeConnector, Position
from .config_manager import ConfigManager
from utils.exceptions import PositionError, InsufficientMarginError, PositionLimitError
from utils.logger import get_logger

logger = get_logger("position")


@dataclass
class SymbolPosition:
    """单币种持仓信息"""
    symbol: str
    entry_price: float                  # 入场价P0
    base_position: Optional[Position] = None   # 基础仓位
    grid_positions: List[Position] = field(default_factory=list)  # 网格仓位列表
    total_margin_used: float = 0.0
    total_size: float = 0.0
    unrealized_pnl: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # 新增：盈利跟踪字段
    initial_margin: float = 0.0              # 开仓时的保证金
    profit_percentage: float = 0.0           # unrealized_pnl / initial_margin
    peak_profit_percentage: float = 0.0      # 历史最高盈利率

    def get_total_contracts(self) -> float:
        """获取总合约数量"""
        total = 0.0
        if self.base_position:
            total += abs(self.base_position.contracts)
        for pos in self.grid_positions:
            total += abs(pos.contracts)
        return total

    def get_average_entry_price(self) -> float:
        """获取平均开仓价"""
        if not self.base_position and not self.grid_positions:
            return self.entry_price

        total_value = 0.0
        total_size = 0.0

        if self.base_position:
            total_value += self.base_position.entry_price * abs(self.base_position.size)
            total_size += abs(self.base_position.size)

        for pos in self.grid_positions:
            total_value += pos.entry_price * abs(pos.size)
            total_size += abs(pos.size)

        return total_value / total_size if total_size > 0 else self.entry_price


class PositionManager:
    """
    仓位管理器

    追踪所有持仓,计算保证金占用,检查仓位限制
    """

    def __init__(self, config: ConfigManager, connector: ExchangeConnector):
        """
        Args:
            config: 配置管理器
            connector: 交易所连接器
        """
        self.config = config
        self.connector = connector

        # 持仓字典: symbol -> SymbolPosition
        self.positions: Dict[str, SymbolPosition] = {}

        # 账户状态
        self.total_balance = 0.0
        self.available_margin = 0.0
        self.used_margin = 0.0

        logger.info("仓位管理器初始化完成")

    def sync_positions(self) -> None:
        """从交易所同步持仓信息"""
        try:
            # 查询余额
            balance = self.connector.query_balance()
            self.total_balance = balance.total
            self.available_margin = balance.available
            self.used_margin = balance.used

            # 查询持仓
            positions = self.connector.query_positions()

            # 更新持仓字典
            for pos in positions:
                symbol = pos.symbol
                if symbol not in self.positions:
                    # 新持仓(可能是从历史恢复)
                    self.positions[symbol] = SymbolPosition(
                        symbol=symbol,
                        entry_price=pos.entry_price,
                        base_position=pos,
                        total_margin_used=pos.margin,
                        total_size=pos.size,
                        unrealized_pnl=pos.unrealized_pnl
                    )
                else:
                    # 更新现有持仓
                    symbol_pos = self.positions[symbol]
                    symbol_pos.last_update = datetime.now(timezone.utc)
                    symbol_pos.unrealized_pnl = pos.unrealized_pnl

                    # 简化处理:将所有持仓合并到base_position
                    symbol_pos.base_position = pos
                    symbol_pos.total_size = pos.size
                    symbol_pos.total_margin_used = pos.margin

            logger.info(
                f"仓位同步完成: {len(self.positions)}个币种, "
                f"可用保证金={self.available_margin:.2f} USDT"
            )

        except Exception as e:
            logger.error(f"仓位同步失败: {e}")

    def get_symbol_position(self, symbol: str) -> Optional[SymbolPosition]:
        """获取单币种持仓"""
        return self.positions.get(symbol)

    def get_total_margin_used(self) -> float:
        """获取总保证金占用"""
        return sum(pos.total_margin_used for pos in self.positions.values())

    def get_available_margin(self) -> float:
        """获取可用保证金"""
        return self.available_margin

    def can_open_new_position(self, required_margin: float) -> bool:
        """
        检查是否可以开新仓位

        Args:
            required_margin: 所需保证金

        Returns:
            是否可以开仓
        """
        # 检查持仓数量限制
        if len(self.positions) >= self.config.position.max_symbols:
            logger.warning(
                f"已达最大持仓数量: {len(self.positions)}/{self.config.position.max_symbols}"
            )
            return False

        # 检查保证金是否充足
        if required_margin > self.available_margin:
            logger.warning(
                f"保证金不足: 需要{required_margin:.2f}, 可用{self.available_margin:.2f}"
            )
            return False

        # 注意: 总保证金限制现在由capital_allocator动态管理
        # available_margin已经反映了交易所的实际可用余额
        return True

    def add_position(self, symbol: str, entry_price: float) -> SymbolPosition:
        """
        添加新持仓

        Args:
            symbol: 交易对
            entry_price: 入场价

        Returns:
            SymbolPosition对象
        """
        if symbol in self.positions:
            raise PositionError(f"持仓已存在: {symbol}")

        position = SymbolPosition(
            symbol=symbol,
            entry_price=entry_price
        )

        self.positions[symbol] = position
        logger.info(f"添加持仓: {symbol} @ {entry_price}")

        return position

    def remove_position(self, symbol: str) -> None:
        """
        移除持仓

        Args:
            symbol: 交易对
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"移除持仓: {symbol}")

    def get_position_count(self) -> int:
        """获取持仓数量"""
        return len(self.positions)

    def get_all_symbols(self) -> List[str]:
        """获取所有持仓的交易对"""
        return list(self.positions.keys())

    def update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """
        更新未实现盈亏

        Args:
            symbol: 交易对
            current_price: 当前价格
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # 简化计算:空头盈亏 = (入场价 - 当前价) * 合约数量
        if position.base_position:
            size = abs(position.base_position.size)
            pnl = (position.entry_price - current_price) * size
            position.unrealized_pnl = pnl

            # 更新盈利百分比
            if position.initial_margin > 0:
                position.profit_percentage = pnl / position.initial_margin

                # 更新峰值
                if position.profit_percentage > position.peak_profit_percentage:
                    position.peak_profit_percentage = position.profit_percentage

    def calculate_total_account_pnl(self) -> float:
        """
        计算总账户盈亏

        Returns:
            float: 总未实现盈亏
        """
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_total_account_drawdown(self) -> float:
        """
        计算总账户回撤百分比

        Returns:
            float: 回撤百分比(负值表示亏损)
        """
        total_pnl = self.calculate_total_account_pnl()
        if self.total_balance > 0:
            return total_pnl / self.total_balance
        return 0.0

    def __str__(self) -> str:
        """状态摘要"""
        return f"""
PositionManager:
  持仓数量: {len(self.positions)}
  总余额: {self.total_balance:.2f} USDT
  可用保证金: {self.available_margin:.2f} USDT
  已用保证金: {self.used_margin:.2f} USDT
"""
