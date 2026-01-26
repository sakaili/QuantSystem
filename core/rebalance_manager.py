"""
换仓管理器模块
Rebalance Manager Module

管理换仓执行流程和冷却期,防止频繁进出同一币种
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

from utils.logger import get_logger

logger = get_logger("rebalance_manager")


@dataclass
class RebalanceHistory:
    """换仓历史记录"""
    symbol: str
    reason: str                      # "profit_target" 或 "stop_loss"
    closed_at: datetime
    realized_pnl: float              # 实现盈亏
    profit_percentage: float         # 盈利率
    next_eligible: datetime          # 下次可进入时间(冷却期结束)

    def is_in_cooldown(self, current_time: Optional[datetime] = None) -> bool:
        """检查是否在冷却期内"""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        return current_time < self.next_eligible


class RebalanceManager:
    """
    换仓管理器

    管理换仓执行、冷却期追踪和历史记录
    """

    def __init__(self, cooldown_hours: int = 24, max_rebalances_per_day: int = 3):
        """
        Args:
            cooldown_hours: 冷却期(小时),品种平仓后多久可再次进入
            max_rebalances_per_day: 单日最大换仓次数
        """
        self.cooldown_hours = cooldown_hours
        self.max_rebalances_per_day = max_rebalances_per_day

        # 换仓历史记录列表
        self.history: List[RebalanceHistory] = []

        # 冷却期品种集合: symbol -> exit_time
        self.cooldown_symbols: Dict[str, datetime] = {}

        logger.info(f"换仓管理器初始化完成:")
        logger.info(f"  冷却期: {self.cooldown_hours} 小时")
        logger.info(f"  单日最大换仓次数: {self.max_rebalances_per_day}")

    def add_to_cooldown(
        self,
        symbol: str,
        reason: str,
        realized_pnl: float = 0.0,
        profit_percentage: float = 0.0
    ) -> None:
        """
        将品种添加到冷却期

        Args:
            symbol: 品种代码
            reason: 换仓原因
            realized_pnl: 实现盈亏
            profit_percentage: 盈利率
        """
        current_time = datetime.now(timezone.utc)
        next_eligible = current_time + timedelta(hours=self.cooldown_hours)

        # 添加到历史记录
        record = RebalanceHistory(
            symbol=symbol,
            reason=reason,
            closed_at=current_time,
            realized_pnl=realized_pnl,
            profit_percentage=profit_percentage,
            next_eligible=next_eligible
        )
        self.history.append(record)

        # 添加到冷却期集合
        self.cooldown_symbols[symbol] = current_time

        logger.info(
            f"将 {symbol} 添加到冷却期: "
            f"原因={reason}, PnL={realized_pnl:.2f} USDT ({profit_percentage*100:.2f}%), "
            f"可再入时间: {next_eligible.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    def is_in_cooldown(self, symbol: str) -> bool:
        """
        检查品种是否在冷却期内

        Args:
            symbol: 品种代码

        Returns:
            bool: 是否在冷却期
        """
        if symbol not in self.cooldown_symbols:
            return False

        exit_time = self.cooldown_symbols[symbol]
        current_time = datetime.now(timezone.utc)
        elapsed = (current_time - exit_time).total_seconds() / 3600  # 小时

        if elapsed >= self.cooldown_hours:
            # 冷却期已结束,移除
            logger.info(f"{symbol} 冷却期结束,可重新进入")
            self.cooldown_symbols.pop(symbol)
            return False

        remaining_hours = self.cooldown_hours - elapsed
        logger.debug(
            f"{symbol} 仍在冷却期: "
            f"已过 {elapsed:.1f}h, 剩余 {remaining_hours:.1f}h"
        )
        return True

    def filter_cooldown_candidates(self, candidates: List[str]) -> List[str]:
        """
        从候选列表中过滤掉冷却期品种

        Args:
            candidates: 候选品种列表

        Returns:
            List[str]: 过滤后的列表
        """
        filtered = []
        for symbol in candidates:
            if not self.is_in_cooldown(symbol):
                filtered.append(symbol)
            else:
                logger.debug(f"过滤候选 {symbol}: 仍在冷却期")

        logger.info(f"候选过滤: {len(candidates)} 个 → {len(filtered)} 个可用")
        return filtered

    def get_cooldown_symbols(self) -> Set[str]:
        """获取当前所有冷却期品种"""
        # 清理过期的冷却期品种
        current_time = datetime.now(timezone.utc)
        expired = []

        for symbol, exit_time in self.cooldown_symbols.items():
            elapsed = (current_time - exit_time).total_seconds() / 3600
            if elapsed >= self.cooldown_hours:
                expired.append(symbol)

        for symbol in expired:
            logger.debug(f"清理过期冷却期品种: {symbol}")
            self.cooldown_symbols.pop(symbol)

        return set(self.cooldown_symbols.keys())

    def can_rebalance_today(self) -> bool:
        """
        检查今天是否还能进行换仓(是否达到单日上限)

        Returns:
            bool: 是否可以换仓
        """
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        today_rebalances = sum(
            1 for record in self.history
            if record.closed_at >= today_start
        )

        if today_rebalances >= self.max_rebalances_per_day:
            logger.warning(
                f"今日换仓已达上限: {today_rebalances}/{self.max_rebalances_per_day}, "
                f"暂停新换仓"
            )
            return False

        logger.debug(f"今日换仓次数: {today_rebalances}/{self.max_rebalances_per_day}")
        return True

    def get_recent_history(self, hours: int = 24) -> List[RebalanceHistory]:
        """
        获取最近N小时的换仓历史

        Args:
            hours: 小时数

        Returns:
            List[RebalanceHistory]: 历史记录列表
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [record for record in self.history if record.closed_at >= cutoff]

    def get_symbol_history(self, symbol: str) -> List[RebalanceHistory]:
        """
        获取特定品种的换仓历史

        Args:
            symbol: 品种代码

        Returns:
            List[RebalanceHistory]: 历史记录列表
        """
        return [record for record in self.history if record.symbol == symbol]

    def get_total_realized_pnl(self, hours: Optional[int] = None) -> float:
        """
        计算总实现盈亏

        Args:
            hours: 统计最近N小时(None=全部)

        Returns:
            float: 总盈亏
        """
        if hours is None:
            records = self.history
        else:
            records = self.get_recent_history(hours)

        return sum(record.realized_pnl for record in records)

    def get_rebalance_stats(self) -> Dict[str, any]:
        """获取换仓统计信息"""
        if not self.history:
            return {
                "total_rebalances": 0,
                "profit_target_count": 0,
                "stop_loss_count": 0,
                "total_realized_pnl": 0.0,
                "avg_realized_pnl": 0.0,
                "symbols_in_cooldown": 0
            }

        profit_target_count = sum(1 for r in self.history if r.reason == "profit_target")
        stop_loss_count = sum(1 for r in self.history if r.reason == "stop_loss")
        total_pnl = sum(r.realized_pnl for r in self.history)

        return {
            "total_rebalances": len(self.history),
            "profit_target_count": profit_target_count,
            "stop_loss_count": stop_loss_count,
            "total_realized_pnl": total_pnl,
            "avg_realized_pnl": total_pnl / len(self.history) if self.history else 0.0,
            "symbols_in_cooldown": len(self.cooldown_symbols),
            "cooldown_list": list(self.cooldown_symbols.keys())
        }

    def clear_old_history(self, days: int = 30) -> int:
        """
        清理旧的历史记录

        Args:
            days: 保留最近N天

        Returns:
            int: 清理的记录数
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        old_count = len(self.history)

        self.history = [record for record in self.history if record.closed_at >= cutoff]

        removed = old_count - len(self.history)
        if removed > 0:
            logger.info(f"清理 {removed} 条超过 {days} 天的历史记录")

        return removed

    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_rebalance_stats()

        lines = [
            f"RebalanceManager:",
            f"  Cooldown Hours: {self.cooldown_hours}h",
            f"  Max Rebalances/Day: {self.max_rebalances_per_day}",
            f"  Total Rebalances: {stats['total_rebalances']}",
            f"  - Profit Target: {stats['profit_target_count']}",
            f"  - Stop Loss: {stats['stop_loss_count']}",
            f"  Total Realized PnL: {stats['total_realized_pnl']:.2f} USDT",
            f"  Avg Realized PnL: {stats['avg_realized_pnl']:.2f} USDT",
            f"  Symbols in Cooldown: {stats['symbols_in_cooldown']}"
        ]

        if stats['cooldown_list']:
            lines.append(f"  Cooldown List: {stats['cooldown_list']}")

        # 显示最近5条记录
        recent = self.get_recent_history(hours=72)  # 最近3天
        if recent:
            lines.append(f"  Recent History (last 72h):")
            for record in recent[-5:]:  # 最后5条
                lines.append(
                    f"    {record.symbol}: {record.reason}, "
                    f"PnL={record.realized_pnl:.2f} ({record.profit_percentage*100:.2f}%), "
                    f"at {record.closed_at.strftime('%m-%d %H:%M')}"
                )

        return "\n".join(lines)
