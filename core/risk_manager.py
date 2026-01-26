"""
风险控制模块
Risk Manager Module

监控风险并执行止损/止盈
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .exchange_connector import ExchangeConnector
from .position_manager import PositionManager
from .grid_strategy import GridStrategy
from .config_manager import ConfigManager
from utils.exceptions import RiskError
from utils.logger import get_logger

logger = get_logger("risk")


@dataclass
class Alert:
    """风险预警"""
    timestamp: datetime
    level: int              # 1=提示, 2=准备行动, 3=立即行动
    symbol: str
    message: str
    action_required: bool
    current_price: float
    entry_price: float


class RiskManager:
    """
    风险控制模块

    监控所有持仓的风险,触发止损/止盈
    """

    def __init__(
        self,
        config: ConfigManager,
        connector: ExchangeConnector,
        position_mgr: PositionManager,
        grid_strategy: GridStrategy
    ):
        """
        Args:
            config: 配置管理器
            connector: 交易所连接器
            position_mgr: 仓位管理器
            grid_strategy: 网格策略执行器
        """
        self.config = config
        self.connector = connector
        self.position_mgr = position_mgr
        self.grid_strategy = grid_strategy

        # 预警历史
        self.alert_history: List[Alert] = []

        # 资金费率监控
        self.funding_rate_history: Dict[str, List[float]] = {}

        logger.info("风险控制模块初始化完成")

    def monitor_all_positions(self) -> List[Alert]:
        """
        监控所有持仓

        Returns:
            Alert列表
        """
        alerts = []

        for symbol in self.position_mgr.get_all_symbols():
            try:
                # 检查价格止损
                alert = self.check_stop_loss(symbol)
                if alert:
                    alerts.append(alert)

                    # Level 3立即止损
                    if alert.level == 3:
                        self.execute_emergency_stop(symbol, alert.message)

                # 检查资金费率
                alert = self.check_funding_rate_risk(symbol)
                if alert:
                    alerts.append(alert)

            except Exception as e:
                logger.error(f"监控失败 {symbol}: {e}")

        return alerts

    def check_stop_loss(self, symbol: str) -> Optional[Alert]:
        """
        检查止损条件

        Args:
            symbol: 交易对

        Returns:
            Alert对象或None
        """
        position = self.position_mgr.get_symbol_position(symbol)
        if not position:
            return None

        # 获取当前价格
        try:
            current_price = self.connector.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"获取价格失败 {symbol}: {e}")
            return None

        entry_price = position.entry_price

        # 计算价格比例
        price_ratio = current_price / entry_price

        # Level 1: 1.10×P0
        if price_ratio >= self.config.alerts.level_1_price:
            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                level=1,
                symbol=symbol,
                message=f"一级预警: 价格{current_price:.4f}接近止损线(距离{(self.config.stop_loss.ratio - price_ratio)*100:.1f}%)",
                action_required=False,
                current_price=current_price,
                entry_price=entry_price
            )

            # Level 2: 1.13×P0
            if price_ratio >= self.config.alerts.level_2_price:
                alert.level = 2
                alert.message = f"二级预警: 价格{current_price:.4f}逼近止损线(距离{(self.config.stop_loss.ratio - price_ratio)*100:.1f}%)"

            # Level 3: 1.15×P0 止损
            if price_ratio >= self.config.alerts.level_3_price:
                alert.level = 3
                alert.message = f"触发止损: 价格{current_price:.4f}突破止损线{entry_price * self.config.stop_loss.ratio:.4f}"
                alert.action_required = True

            logger.warning(f"{alert.message}")
            self.alert_history.append(alert)
            return alert

        return None

    def check_funding_rate_risk(self, symbol: str) -> Optional[Alert]:
        """
        检查资金费率风险

        Args:
            symbol: 交易对

        Returns:
            Alert对象或None
        """
        try:
            funding_rate = self.connector.get_funding_rate(symbol)

            if funding_rate is None:
                return None

            # 记录历史
            if symbol not in self.funding_rate_history:
                self.funding_rate_history[symbol] = []

            self.funding_rate_history[symbol].append(funding_rate)

            # 只保留最近的记录
            max_history = self.config.funding_rate.negative_days * 3  # 每天3次资金费率
            self.funding_rate_history[symbol] = self.funding_rate_history[symbol][-max_history:]

            # 检查是否持续负费率
            recent = self.funding_rate_history[symbol][-self.config.funding_rate.negative_days * 3:]

            if len(recent) >= self.config.funding_rate.negative_days * 3:
                if all(fr < self.config.funding_rate.floor for fr in recent):
                    alert = Alert(
                        timestamp=datetime.now(timezone.utc),
                        level=2,
                        symbol=symbol,
                        message=f"资金费率持续负值{len(recent)}期,当前{funding_rate:.4f}",
                        action_required=True,
                        current_price=0,
                        entry_price=0
                    )

                    logger.warning(alert.message)
                    self.alert_history.append(alert)
                    return alert

        except Exception as e:
            logger.warning(f"检查资金费率失败 {symbol}: {e}")

        return None

    def check_total_account_risk(self) -> Optional[Alert]:
        """
        检查总账户风险(总账户回撤监控)

        Returns:
            Alert对象或None
        """
        try:
            # 计算总账户回撤
            drawdown = self.position_mgr.get_total_account_drawdown()

            # Level 2: -15% 总回撤预警
            if drawdown <= -self.config.account_risk.max_total_drawdown:
                alert = Alert(
                    timestamp=datetime.now(timezone.utc),
                    level=2,
                    symbol="ACCOUNT",
                    message=f"总账户回撤预警: {drawdown*100:.2f}% (预警线 {-self.config.account_risk.max_total_drawdown*100:.0f}%)",
                    action_required=True,
                    current_price=0,
                    entry_price=0
                )

                # Level 3: -25% 紧急止损
                if drawdown <= -self.config.account_risk.emergency_exit_threshold:
                    alert.level = 3
                    alert.message = f"紧急止损: 总账户回撤 {drawdown*100:.2f}% (紧急线 {-self.config.account_risk.emergency_exit_threshold*100:.0f}%), 建议关闭所有仓位"

                logger.warning(f"{alert.message}")
                self.alert_history.append(alert)
                return alert

        except Exception as e:
            logger.warning(f"检查总账户风险失败: {e}")

        return None

    def execute_emergency_stop(self, symbol: str, reason: str) -> None:
        """
        执行紧急止损

        Args:
            symbol: 交易对
            reason: 止损原因
        """
        logger.critical(f"执行紧急止损: {symbol}, 原因: {reason}")

        try:
            # 关闭网格(会自动撤单和平仓)
            self.grid_strategy.close_grid(symbol, reason=f"emergency_stop: {reason}")

            logger.info(f"紧急止损完成: {symbol}")

        except Exception as e:
            logger.error(f"紧急止损失败: {symbol}: {e}")
            raise RiskError(f"Emergency stop failed: {e}")

    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """获取最近的预警"""
        return self.alert_history[-count:]
