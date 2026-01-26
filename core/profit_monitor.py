"""
盈利监控器模块
Profit Monitor Module

实时跟踪每个品种的盈利率,检测达到盈利目标的品种
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("profit_monitor")


@dataclass
class SymbolProfitState:
    """单币种盈利状态"""
    symbol: str
    entry_margin: float              # 开仓时的保证金
    current_margin: float            # 当前使用保证金
    unrealized_pnl: float            # 未实现盈亏
    profit_percentage: float = 0.0   # 盈利百分比(pnl/entry_margin)
    peak_profit: float = 0.0         # 历史最高盈利百分比
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update(self, current_margin: float, unrealized_pnl: float) -> None:
        """更新盈利状态"""
        self.current_margin = current_margin
        self.unrealized_pnl = unrealized_pnl

        # 计算盈利百分比(基于初始保证金)
        if self.entry_margin > 0:
            self.profit_percentage = unrealized_pnl / self.entry_margin
        else:
            self.profit_percentage = 0.0

        # 更新峰值
        if self.profit_percentage > self.peak_profit:
            self.peak_profit = self.profit_percentage

        self.last_update = datetime.now(timezone.utc)


class ProfitMonitor:
    """
    盈利监控器

    跟踪每个品种的盈利状态,检测达到盈利目标的品种
    """

    def __init__(self, profit_threshold: float = 0.15, max_rebalances_per_cycle: int = 2):
        """
        Args:
            profit_threshold: 盈利目标阈值(默认15%)
            max_rebalances_per_cycle: 每周期最多触发换仓数(防止同时多个品种触发)
        """
        self.profit_threshold = profit_threshold
        self.max_rebalances_per_cycle = max_rebalances_per_cycle

        # 品种盈利状态: symbol -> SymbolProfitState
        self.symbol_states: Dict[str, SymbolProfitState] = {}

        # 待换仓队列(按盈利率排序)
        self.rebalance_queue: List[str] = []

        logger.info(f"盈利监控器初始化完成:")
        logger.info(f"  盈利目标: {self.profit_threshold*100:.1f}%")
        logger.info(f"  每周期最大换仓数: {self.max_rebalances_per_cycle}")

    def add_symbol(self, symbol: str, entry_margin: float) -> None:
        """
        添加品种到监控列表

        Args:
            symbol: 品种代码
            entry_margin: 开仓时的保证金
        """
        if symbol in self.symbol_states:
            logger.warning(f"{symbol} 已在监控列表中")
            return

        state = SymbolProfitState(
            symbol=symbol,
            entry_margin=entry_margin,
            current_margin=entry_margin,
            unrealized_pnl=0.0
        )

        self.symbol_states[symbol] = state
        logger.info(f"添加 {symbol} 到盈利监控,初始保证金: {entry_margin:.2f} USDT")

    def remove_symbol(self, symbol: str) -> Optional[SymbolProfitState]:
        """
        从监控列表移除品种

        Args:
            symbol: 品种代码

        Returns:
            SymbolProfitState: 被移除的状态,如果不存在则返回None
        """
        if symbol not in self.symbol_states:
            logger.warning(f"{symbol} 不在监控列表中")
            return None

        state = self.symbol_states.pop(symbol)

        # 同时从换仓队列移除
        if symbol in self.rebalance_queue:
            self.rebalance_queue.remove(symbol)

        logger.info(
            f"从盈利监控移除 {symbol}, "
            f"最终盈利: {state.profit_percentage*100:.2f}%, "
            f"峰值: {state.peak_profit*100:.2f}%"
        )

        return state

    def update_symbol_profit(
        self,
        symbol: str,
        current_margin: float,
        unrealized_pnl: float
    ) -> None:
        """
        更新品种盈利状态

        Args:
            symbol: 品种代码
            current_margin: 当前保证金
            unrealized_pnl: 未实现盈亏
        """
        if symbol not in self.symbol_states:
            logger.warning(f"{symbol} 不在监控列表,无法更新盈利状态")
            return

        state = self.symbol_states[symbol]
        old_profit = state.profit_percentage

        state.update(current_margin, unrealized_pnl)

        # 日志记录显著变化(>5%变动)
        if abs(state.profit_percentage - old_profit) > 0.05:
            logger.info(
                f"{symbol} 盈利变化: {old_profit*100:.2f}% → {state.profit_percentage*100:.2f}%, "
                f"PnL: {unrealized_pnl:.2f} USDT"
            )

    def check_profit_threshold(self) -> List[str]:
        """
        检查所有品种是否达到盈利目标

        Returns:
            List[str]: 达到目标的品种列表(按盈利率降序排序)
        """
        self.rebalance_queue.clear()

        # 筛选达到阈值的品种
        for symbol, state in self.symbol_states.items():
            if state.profit_percentage >= self.profit_threshold:
                self.rebalance_queue.append(symbol)
                logger.info(
                    f"{symbol} 达到盈利目标: {state.profit_percentage*100:.2f}% "
                    f"(目标: {self.profit_threshold*100:.1f}%), "
                    f"PnL: {state.unrealized_pnl:.2f} USDT"
                )

        # 按盈利率降序排序
        self.rebalance_queue.sort(
            key=lambda s: self.symbol_states[s].profit_percentage,
            reverse=True
        )

        # 限制每周期最多处理数量
        if len(self.rebalance_queue) > self.max_rebalances_per_cycle:
            limited_queue = self.rebalance_queue[:self.max_rebalances_per_cycle]
            logger.warning(
                f"多个品种达到盈利目标: {len(self.rebalance_queue)}, "
                f"本周期仅处理前 {self.max_rebalances_per_cycle} 个: "
                f"{limited_queue}"
            )
            return limited_queue

        return self.rebalance_queue

    def get_symbol_profit(self, symbol: str) -> Optional[float]:
        """获取品种当前盈利率"""
        state = self.symbol_states.get(symbol)
        return state.profit_percentage if state else None

    def get_symbol_state(self, symbol: str) -> Optional[SymbolProfitState]:
        """获取品种盈利状态"""
        return self.symbol_states.get(symbol)

    def verify_profit_before_rebalance(self, symbol: str, min_threshold: float = 0.10) -> bool:
        """
        换仓前二次确认盈利率(防止触发时profit=15%,执行时已跌至10%)

        Args:
            symbol: 品种代码
            min_threshold: 最低确认阈值(默认10%)

        Returns:
            bool: 是否确认通过
        """
        state = self.symbol_states.get(symbol)
        if not state:
            logger.warning(f"{symbol} 不在监控列表,无法确认盈利")
            return False

        if state.profit_percentage < min_threshold:
            logger.warning(
                f"{symbol} 盈利已回撤: {state.profit_percentage*100:.2f}% "
                f"< 最低阈值 {min_threshold*100:.0f}%, 取消换仓"
            )
            return False

        logger.info(
            f"{symbol} 盈利确认通过: {state.profit_percentage*100:.2f}% "
            f"≥ 最低阈值 {min_threshold*100:.0f}%"
        )
        return True

    def get_monitored_symbols(self) -> List[str]:
        """获取所有监控中的品种列表"""
        return list(self.symbol_states.keys())

    def get_summary(self) -> Dict[str, any]:
        """获取监控摘要"""
        if not self.symbol_states:
            return {
                "monitored_symbols": 0,
                "avg_profit": 0.0,
                "max_profit": 0.0,
                "min_profit": 0.0,
                "symbols_over_threshold": 0
            }

        profits = [state.profit_percentage for state in self.symbol_states.values()]

        return {
            "monitored_symbols": len(self.symbol_states),
            "avg_profit": sum(profits) / len(profits) * 100,
            "max_profit": max(profits) * 100,
            "min_profit": min(profits) * 100,
            "symbols_over_threshold": len([p for p in profits if p >= self.profit_threshold])
        }

    def __str__(self) -> str:
        """字符串表示"""
        lines = [
            f"ProfitMonitor:",
            f"  Profit Threshold: {self.profit_threshold*100:.1f}%",
            f"  Monitored Symbols: {len(self.symbol_states)}",
        ]

        if self.symbol_states:
            lines.append("  Symbol States:")
            for symbol, state in self.symbol_states.items():
                lines.append(
                    f"    {symbol}: profit={state.profit_percentage*100:.2f}%, "
                    f"peak={state.peak_profit*100:.2f}%, "
                    f"pnl={state.unrealized_pnl:.2f} USDT"
                )

        if self.rebalance_queue:
            lines.append(f"  Rebalance Queue: {self.rebalance_queue}")

        return "\n".join(lines)
