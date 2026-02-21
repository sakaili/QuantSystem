"""
配置管理器模块
Configuration Manager Module

负责加载、验证和管理所有配置参数
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

from utils.exceptions import ConfigurationError
from utils.logger import get_logger

logger = get_logger("config")


@dataclass
class AccountConfig:
    """账户配置"""
    usage_ratio: float          # 资金使用率(如0.9表示使用90%)
    leverage: int               # 杠杆倍数
    order_type: str             # 订单类型
    maker_fee: float            # Maker手续费


@dataclass
class GridConfig:
    """网格配置"""
    count: int                  # 网格总数
    spacing: float              # 网格间距
    upper_grids: int            # 上方网格数
    lower_grids: int            # 下方网格数

    # 网格修复配置
    repair_enabled: bool = True
    repair_interval: int = 10
    max_repair_retries: int = 3
    min_success_rate_upper: float = 0.8
    min_success_rate_lower: float = 0.8

    # 移动网格配置（无限网格策略）
    dynamic_expansion: bool = True
    max_total_grids: int = 30
    max_side_grids: int = 15
    reopen_min_gap_ratio: float = 0.5


@dataclass
class PositionConfig:
    """仓位配置"""
    max_symbols: int            # 最多持仓币种数
    base_margin: float          # 基础仓位保证金
    grid_margin: float          # 单网格保证金
    manual_symbols: List[str] = field(default_factory=list)  # 手动指定必开币种
    min_base_position_ratio: float = 0.3  # 最小保留基础仓位比例（默认30%）
    # single_symbol_max: 已移除,改为动态计算
    # total_margin_limit: 已移除,改为动态计算


@dataclass
class ScreeningConfig:
    """币种筛选配置"""
    bottom_n: int               # 流动性排名倒数N
    min_listing_days: int       # 最小上市天数
    funding_rate_floor: float   # 资金费率下限
    atr_spike_multiplier: float # ATR暴涨倍数
    funding_rate_sort: bool = False  # 是否按历史资金费率排序
    funding_rate_lookback_days: int = 365  # 资金费率统计窗口(0=尽可能全历史)
    funding_rate_min_sum: float = 0.0  # 最小累计资金费率(筛选用)
    eth_deviation_filter: bool = False  # 是否启用ETH偏离筛选
    eth_deviation_window: int = 60      # 滚动窗口(天)
    eth_deviation_cooldown_days: int = 30  # 最近N天发生偏离直接剔除
    eth_deviation_rate_window_days: int = 180  # 统计偏离频率窗口
    eth_deviation_ever: bool = False     # 豁免模式：只看最近N天是否偏离
    eth_corr_drop_threshold: float = 0.2       # corr_drop阈值
    eth_corr_drop_rate_limit: float = 0.05     # corr_drop频率上限
    eth_residual_z: float = 2.5                # 残差z阈值
    eth_residual_rate_limit: float = 0.01      # 残差偏离频率上限
    binance_component_max_weight: float = 0.8  # Binance成分占比上限(<=80%)
    binance_component_weight_strict: bool = True  # 无法获取占比时是否剔除
    air_mean_deviation_filter: bool = False    # 是否启用垃圾币均值偏离筛选
    air_mean_use_median: bool = True           # 使用中位数(更抗极端值)
    air_mean_deviation_window: int = 60        # 滚动窗口(天)
    air_mean_deviation_cooldown_days: int = 365  # 最近N天发生偏离直接剔除
    air_mean_deviation_rate_window_days: int = 180  # 统计偏离频率窗口
    air_mean_deviation_ever: bool = False      # 豁免模式：只看最近N天是否偏离
    air_mean_corr_drop_threshold: float = 0.2  # corr_drop阈值
    air_mean_corr_drop_rate_limit: float = 0.05  # corr_drop频率上限
    air_mean_residual_z: float = 2.5           # 残差z阈值
    air_mean_residual_rate_limit: float = 0.01 # 残差偏离频率上限


@dataclass
class ScheduleConfig:
    """交易时间配置"""
    scan_hour: int              # 每日筛选时间(UTC小时)
    scan_interval: int          # 筛选间隔(秒)
    monitor_interval: int       # 监控循环间隔(秒)


@dataclass
class RebalancingConfig:
    """换仓配置"""
    enabled: bool               # 是否启用盈利换仓
    profit_threshold: float     # 盈利目标阈值
    cooldown_hours: int         # 冷却期(小时)
    max_rebalances_per_day: int # 单日最大换仓次数


@dataclass
class StopLossConfig:
    """止损配置"""
    ratio: float                # 止损线比例
    order_type: str             # 止损订单类型


@dataclass
class ProfitTakingConfig:
    """盈利止盈配置"""
    enabled: bool               # 是否启用盈利止盈
    threshold: float            # 盈利目标阈值
    partial_exit: bool          # 是否部分平仓


@dataclass
class TakeProfitConfig:
    """原有止盈配置(保留向后兼容)"""
    partial_close_ratio: float  # 部分平仓比例
    trigger_level: int          # 触发层级


@dataclass
class FundingRateConfig:
    """资金费率风险配置"""
    floor: float                # 资金费率下限
    negative_days: int          # 负费率持续天数阈值


@dataclass
class AlertConfig:
    """预警配置"""
    level_1_price: float        # 一级预警价格比例
    level_2_price: float        # 二级预警价格比例
    level_3_price: float        # 三级预警价格比例


@dataclass
class PositionRiskConfig:
    """仓位风险配置"""
    max_margin_usage: float     # 最大保证金占用比例
    max_leverage_total: int     # 总杠杆上限


@dataclass
class LiquidityConfig:
    """流动性风险配置"""
    volume_spike_multiplier: float  # 成交额暴涨倍数
    min_24h_volume: float           # 最小24h成交额


@dataclass
class AccountRiskConfig:
    """账户级风险配置"""
    max_total_drawdown: float       # 总账户最大回撤比例
    emergency_exit_threshold: float # 紧急止损阈值


@dataclass
class BinanceAPIConfig:
    """Binance API配置"""
    api_key: str
    api_secret: str
    testnet: bool


@dataclass
class ProxyConfig:
    """代理配置"""
    enabled: bool
    http_proxy: str
    https_proxy: str


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    max_calls_per_minute: int
    cooldown: float


@dataclass
class TimeoutConfig:
    """超时配置"""
    connect: int
    read: int


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int
    backoff_factor: float


class ConfigManager:
    """
    配置管理器 - 单例模式

    负责加载和管理所有配置文件
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir: Optional[Path] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config_dir = config_dir or Path("config")

        # 配置对象
        self.account: Optional[AccountConfig] = None
        self.grid: Optional[GridConfig] = None
        self.position: Optional[PositionConfig] = None
        self.screening: Optional[ScreeningConfig] = None
        self.schedule: Optional[ScheduleConfig] = None
        self.rebalancing: Optional[RebalancingConfig] = None  # 新增

        self.stop_loss: Optional[StopLossConfig] = None
        self.take_profit: Optional[TakeProfitConfig] = None
        self.profit_taking: Optional[ProfitTakingConfig] = None  # 新增
        self.funding_rate: Optional[FundingRateConfig] = None
        self.alerts: Optional[AlertConfig] = None
        self.position_risk: Optional[PositionRiskConfig] = None
        self.liquidity: Optional[LiquidityConfig] = None
        self.account_risk: Optional[AccountRiskConfig] = None  # 新增

        self.binance: Optional[BinanceAPIConfig] = None
        self.proxy: Optional[ProxyConfig] = None
        self.rate_limit: Optional[RateLimitConfig] = None
        self.timeout: Optional[TimeoutConfig] = None
        self.retry: Optional[RetryConfig] = None

        self._initialized = False

    def load_configs(self) -> None:
        """加载所有配置文件"""
        logger.info("加载配置文件...")

        try:
            # 加载策略配置
            strategy_config = self._load_yaml(self.config_dir / "strategy_config.yaml")
            self._parse_strategy_config(strategy_config)

            # 加载风险配置
            risk_config = self._load_yaml(self.config_dir / "risk_config.yaml")
            self._parse_risk_config(risk_config)

            # 加载API配置
            api_config = self._load_yaml(self.config_dir / "api_config.yaml")
            self._parse_api_config(api_config)

            # 验证配置
            self.validate_configs()

            self._initialized = True
            logger.info("配置加载完成")

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise ConfigurationError(f"Failed to load configs: {e}")

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        if not file_path.exists():
            raise ConfigurationError(f"配置文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _parse_strategy_config(self, config: Dict[str, Any]) -> None:
        """解析策略配置"""
        self.account = AccountConfig(**config['account'])
        self.grid = GridConfig(**config['grid'])
        self.position = PositionConfig(**config['position'])
        self.screening = ScreeningConfig(**config['screening'])
        self.schedule = ScheduleConfig(**config['schedule'])
        self.rebalancing = RebalancingConfig(**config['rebalancing'])

    def _parse_risk_config(self, config: Dict[str, Any]) -> None:
        """解析风险配置"""
        self.stop_loss = StopLossConfig(**config['stop_loss'])
        self.take_profit = TakeProfitConfig(**config['take_profit'])
        self.profit_taking = ProfitTakingConfig(**config['profit_taking'])
        self.funding_rate = FundingRateConfig(**config['funding_rate'])
        self.alerts = AlertConfig(**config['alerts'])
        self.position_risk = PositionRiskConfig(**config['position_risk'])
        self.liquidity = LiquidityConfig(**config['liquidity'])
        self.account_risk = AccountRiskConfig(**config['account_risk'])

    def _parse_api_config(self, config: Dict[str, Any]) -> None:
        """解析API配置"""
        binance_config = config['binance'].copy()

        # 从环境变量读取API密钥
        api_key = binance_config.get('api_key', '')
        api_secret = binance_config.get('api_secret', '')

        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, '')

        if api_secret.startswith('${') and api_secret.endswith('}'):
            env_var = api_secret[2:-1]
            api_secret = os.environ.get(env_var, '')

        binance_config['api_key'] = api_key
        binance_config['api_secret'] = api_secret

        self.binance = BinanceAPIConfig(**binance_config)
        self.proxy = ProxyConfig(**config['proxy'])
        self.rate_limit = RateLimitConfig(**config['rate_limit'])
        self.timeout = TimeoutConfig(**config['timeout'])
        self.retry = RetryConfig(**config['retry'])

    def validate_configs(self) -> None:
        """验证配置合法性"""
        # 验证账户配置
        if self.account.usage_ratio <= 0 or self.account.usage_ratio > 1:
            raise ConfigurationError("资金使用率必须在(0, 1]范围内")

        if self.account.leverage <= 0:
            raise ConfigurationError("杠杆倍数必须大于0")

        # 验证网格配置
        if self.grid.count != self.grid.upper_grids + self.grid.lower_grids:
            raise ConfigurationError("网格总数必须等于上方网格数+下方网格数")

        if self.grid.spacing <= 0 or self.grid.spacing >= 1:
            raise ConfigurationError("网格间距必须在(0, 1)范围内")

        # 验证最小网格数量
        if self.grid.upper_grids < 3:
            raise ConfigurationError("上方网格数量不能少于3个")
        if self.grid.lower_grids < 3:
            raise ConfigurationError("下方网格数量不能少于3个")

        # 验证网格间距合理性
        if self.grid.spacing < 0.005 or self.grid.spacing > 0.05:
            raise ConfigurationError("网格间距应在0.5%-5%之间")

        # 验证网格修复配置
        if self.grid.repair_interval < 5:
            raise ConfigurationError("修复检查间隔不能小于5秒")

        if self.grid.max_repair_retries < 1:
            raise ConfigurationError("最大重试次数必须≥1")

        if not (0 < self.grid.min_success_rate_upper <= 1):
            raise ConfigurationError("上方网格成功率阈值必须在(0, 1]范围内")

        if not (0 < self.grid.min_success_rate_lower <= 1):
            raise ConfigurationError("下方网格成功率阈值必须在(0, 1]范围内")

        # 验证移动网格配置
        if self.grid.max_total_grids < self.grid.count:
            raise ConfigurationError(f"最大总网格数({self.grid.max_total_grids})不能小于初始网格数({self.grid.count})")

        if self.grid.max_side_grids < self.grid.upper_grids or self.grid.max_side_grids < self.grid.lower_grids:
            raise ConfigurationError(f"单侧最大网格数({self.grid.max_side_grids})不能小于初始单侧网格数({max(self.grid.upper_grids, self.grid.lower_grids)})")

        if self.grid.max_total_grids > 50:
            logger.warning(f"最大总网格数({self.grid.max_total_grids})较大，可能增加资金占用和管理复杂度")

        # 验证仓位配置
        if self.position.max_symbols <= 0:
            raise ConfigurationError("最多持仓币种数必须大于0")

        # 验证换仓配置
        if self.rebalancing.profit_threshold <= 0:
            raise ConfigurationError("盈利目标阈值必须大于0")

        if self.rebalancing.cooldown_hours < 0:
            raise ConfigurationError("冷却期不能为负数")

        # 验证止损配置
        if self.stop_loss.ratio <= 1.0:
            raise ConfigurationError("止损比例必须大于1.0")

        # 验证资金费率排序窗口
        if self.screening.funding_rate_lookback_days < 0:
            raise ConfigurationError("资金费率统计窗口不能为负数")

        # 验证盈利止盈配置
        if self.profit_taking.threshold <= 0:
            raise ConfigurationError("盈利止盈阈值必须大于0")

        # 验证账户风险配置
        if self.account_risk.max_total_drawdown <= 0 or self.account_risk.max_total_drawdown >= 1:
            raise ConfigurationError("总账户最大回撤必须在(0, 1)范围内")

        if self.account_risk.emergency_exit_threshold <= 0 or self.account_risk.emergency_exit_threshold >= 1:
            raise ConfigurationError("紧急止损阈值必须在(0, 1)范围内")

        # 验证API配置
        if not self.binance.api_key or not self.binance.api_secret:
            raise ConfigurationError("Binance API密钥未配置,请设置环境变量BINANCE_API_KEY和BINANCE_API_SECRET")

        logger.info("配置验证通过")

    def get_grid_params(self) -> Dict[str, Any]:
        """获取网格参数"""
        return {
            'count': self.grid.count,
            'spacing': self.grid.spacing,
            'upper_grids': self.grid.upper_grids,
            'lower_grids': self.grid.lower_grids,
            'base_margin': self.position.base_margin,
            'grid_margin': self.position.grid_margin,
        }

    def get_position_limits(self) -> Dict[str, Any]:
        """获取仓位限制"""
        return {
            'max_symbols': self.position.max_symbols,
            # single_symbol_max和total_margin_limit改为动态计算
        }

    def get_risk_params(self) -> Dict[str, Any]:
        """获取风险参数"""
        return {
            'stop_loss_ratio': self.stop_loss.ratio,
            'funding_rate_floor': self.funding_rate.floor,
            'funding_rate_days': self.funding_rate.negative_days,
            'max_margin_usage': self.position_risk.max_margin_usage,
        }

    def __str__(self) -> str:
        """配置摘要"""
        return f"""
ConfigManager:
  Account: 动态获取余额, 使用率 {self.account.usage_ratio*100:.0f}%, {self.account.leverage}x leverage
  Position: max {self.position.max_symbols} symbols
  Grid: {self.grid.count} grids, {self.grid.spacing*100}% spacing
  Stop Loss: {self.stop_loss.ratio}x entry price ({(self.stop_loss.ratio-1)*100:.0f}% stop loss)
  Profit Target: {self.profit_taking.threshold*100:.0f}%
  Rebalancing: {'enabled' if self.rebalancing.enabled else 'disabled'}, {self.rebalancing.cooldown_hours}h cooldown
  Testnet: {self.binance.testnet}
"""


# 全局配置管理器实例
config_manager = ConfigManager()
