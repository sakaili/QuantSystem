"""
资金分配管理器模块
Capital Allocator Module

动态管理多品种的资金分配,从交易所API获取账户余额并自动计算分配
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from .exchange_connector import ExchangeConnector
from utils.exceptions import InsufficientMarginError, PositionLimitError
from utils.logger import get_logger

logger = get_logger("capital_allocator")


@dataclass
class SymbolAllocation:
    """单币种资金分配信息"""
    symbol: str
    target_margin: float              # 目标分配额度
    allocated_margin: float = 0.0     # 实际已使用
    available_for_grids: float = 0.0  # 剩余可用（用于网格扩展）
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def allocate(self, amount: float) -> None:
        """分配资金"""
        if amount > self.available_for_grids:
            raise InsufficientMarginError(
                f"Insufficient margin for {self.symbol}: "
                f"requested {amount:.2f}, available {self.available_for_grids:.2f}"
            )
        self.allocated_margin += amount
        self.available_for_grids -= amount
        self.last_update = datetime.now(timezone.utc)

    def free(self, amount: float) -> None:
        """释放资金"""
        self.allocated_margin -= amount
        self.available_for_grids += amount
        self.last_update = datetime.now(timezone.utc)


class CapitalAllocator:
    """
    资金分配管理器

    动态从交易所获取账户余额,计算可用资金并分配给各品种
    """

    def __init__(self, connector: ExchangeConnector, max_symbols: int, usage_ratio: float = 0.9):
        """
        Args:
            connector: 交易所连接器
            max_symbols: 最大同时持仓品种数
            usage_ratio: 资金使用率(默认90%,保留10%缓冲)
        """
        self.connector = connector
        self.max_symbols = max_symbols
        self.usage_ratio = usage_ratio

        # 账户状态
        self.total_balance = 0.0         # 总余额
        self.available_capital = 0.0     # 可用资金(total × usage_ratio)
        self.per_symbol_target = 0.0     # 每品种目标分配

        # 分配字典: symbol -> SymbolAllocation
        self.allocations: Dict[str, SymbolAllocation] = {}

        # 启动时获取账户余额
        self.refresh_available_capital()

        logger.info(f"资金分配器初始化完成:")
        logger.info(f"  账户总余额: {self.total_balance:.2f} USDT")
        logger.info(f"  可用资金: {self.available_capital:.2f} USDT ({self.usage_ratio*100:.0f}%)")
        logger.info(f"  每品种目标: {self.per_symbol_target:.2f} USDT")
        logger.info(f"  最大品种数: {self.max_symbols}")

    def refresh_available_capital(self) -> None:
        """从交易所API获取账户余额并计算可用资金"""
        try:
            balance = self.connector.query_balance()
            # 使用总余额（total）而不是可用保证金（available）
            self.total_balance = balance.total
            self.available_capital = self.total_balance * self.usage_ratio
            self.per_symbol_target = self.available_capital / self.max_symbols

            logger.info(
                f"账户余额已更新: 总余额 {self.total_balance:.2f} USDT, "
                f"策略可用 {self.available_capital:.2f} USDT ({self.usage_ratio*100:.0f}%), "
                f"每品种 {self.per_symbol_target:.2f} USDT"
            )

        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            raise

    def allocate_symbol(self, symbol: str) -> SymbolAllocation:
        """
        为新品种分配资金

        Args:
            symbol: 品种代码

        Returns:
            SymbolAllocation: 分配信息

        Raises:
            PositionLimitError: 超过最大品种数
            InsufficientMarginError: 可用资金不足
        """
        # 检查品种数限制
        if len(self.allocations) >= self.max_symbols:
            raise PositionLimitError(
                f"已达到最大品种数限制: {self.max_symbols}, "
                f"当前持有: {list(self.allocations.keys())}"
            )

        # 检查是否已分配
        if symbol in self.allocations:
            logger.warning(f"{symbol} 已分配资金,返回现有分配")
            return self.allocations[symbol]

        # 检查可用资金
        total_allocated = sum(alloc.target_margin for alloc in self.allocations.values())
        remaining_capital = self.available_capital - total_allocated

        if remaining_capital < self.per_symbol_target:
            raise InsufficientMarginError(
                f"可用资金不足: 需要 {self.per_symbol_target:.2f} USDT, "
                f"剩余 {remaining_capital:.2f} USDT"
            )

        # 创建分配
        allocation = SymbolAllocation(
            symbol=symbol,
            target_margin=self.per_symbol_target,
            allocated_margin=0.0,
            available_for_grids=self.per_symbol_target
        )

        self.allocations[symbol] = allocation

        logger.info(
            f"为 {symbol} 分配资金: {self.per_symbol_target:.2f} USDT, "
            f"当前品种数: {len(self.allocations)}/{self.max_symbols}"
        )

        return allocation

    def free_symbol(self, symbol: str) -> Optional[float]:
        """
        释放品种的全部分配资金

        Args:
            symbol: 品种代码

        Returns:
            float: 释放的资金总额,如果品种不存在则返回None
        """
        if symbol not in self.allocations:
            logger.warning(f"{symbol} 未分配资金,无需释放")
            return None

        allocation = self.allocations.pop(symbol)
        freed_amount = allocation.target_margin

        logger.info(
            f"释放 {symbol} 资金: {freed_amount:.2f} USDT "
            f"(已用 {allocation.allocated_margin:.2f}, "
            f"剩余 {allocation.available_for_grids:.2f}), "
            f"当前品种数: {len(self.allocations)}/{self.max_symbols}"
        )

        return freed_amount

    def get_symbol_allocation(self, symbol: str) -> Optional[SymbolAllocation]:
        """获取品种的分配信息"""
        return self.allocations.get(symbol)

    def can_open_new_position(self, required_margin: float) -> bool:
        """
        检查是否可以开新品种

        Args:
            required_margin: 所需保证金

        Returns:
            bool: 是否可以开仓
        """
        # 检查品种数限制
        if len(self.allocations) >= self.max_symbols:
            logger.debug(f"已达最大品种数 {self.max_symbols}")
            return False

        # 检查可用资金
        total_allocated = sum(alloc.target_margin for alloc in self.allocations.values())
        remaining_capital = self.available_capital - total_allocated

        if remaining_capital < required_margin:
            logger.debug(
                f"可用资金不足: 需要 {required_margin:.2f}, "
                f"剩余 {remaining_capital:.2f}"
            )
            return False

        return True

    def get_total_allocated(self) -> float:
        """获取总分配资金"""
        return sum(alloc.target_margin for alloc in self.allocations.values())

    def get_total_used(self) -> float:
        """获取总实际使用资金"""
        return sum(alloc.allocated_margin for alloc in self.allocations.values())

    def get_remaining_capital(self) -> float:
        """获取剩余可分配资金"""
        return self.available_capital - self.get_total_allocated()

    def get_symbol_count(self) -> int:
        """获取当前品种数"""
        return len(self.allocations)

    def __str__(self) -> str:
        """字符串表示"""
        lines = [
            f"CapitalAllocator:",
            f"  Total Balance: {self.total_balance:.2f} USDT",
            f"  Available Capital: {self.available_capital:.2f} USDT ({self.usage_ratio*100:.0f}%)",
            f"  Per Symbol Target: {self.per_symbol_target:.2f} USDT",
            f"  Max Symbols: {self.max_symbols}",
            f"  Current Symbols: {len(self.allocations)}",
            f"  Total Allocated: {self.get_total_allocated():.2f} USDT",
            f"  Total Used: {self.get_total_used():.2f} USDT",
            f"  Remaining Capital: {self.get_remaining_capital():.2f} USDT",
        ]

        if self.allocations:
            lines.append("  Symbol Allocations:")
            for symbol, alloc in self.allocations.items():
                lines.append(
                    f"    {symbol}: target={alloc.target_margin:.2f}, "
                    f"used={alloc.allocated_margin:.2f}, "
                    f"available={alloc.available_for_grids:.2f}"
                )

        return "\n".join(lines)
