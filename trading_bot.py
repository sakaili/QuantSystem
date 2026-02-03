"""
量化交易机器人主程序
Trading Bot Main Program

7x24小时运行,协调各模块执行交易策略
"""

import argparse
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

# 确保可以导入模块 - 适配Docker和本地环境
current_dir = Path(__file__).parent
if current_dir.name == 'QuantSystem':
    # 在QuantSystem目录下运行
    sys.path.insert(0, str(current_dir.parent))
    from QuantSystem.data_fetcher import BinanceDataFetcher
    from QuantSystem.daily_candidate_scan import run_scan, is_listed_long_enough
    from QuantSystem.core.config_manager import ConfigManager
    from QuantSystem.core.exchange_connector import ExchangeConnector
    from QuantSystem.core.position_manager import PositionManager
    from QuantSystem.core.grid_strategy import GridStrategy
    from QuantSystem.core.risk_manager import RiskManager
    from QuantSystem.core.database import Database
    from QuantSystem.core.capital_allocator import CapitalAllocator
    from QuantSystem.core.profit_monitor import ProfitMonitor
    from QuantSystem.core.rebalance_manager import RebalanceManager
    from QuantSystem.utils.logger import setup_logger, get_logger
    from QuantSystem.web_api import WebAPI
else:
    # 在Docker容器中或直接运行
    from data_fetcher import BinanceDataFetcher
    from daily_candidate_scan import run_scan, is_listed_long_enough
    from core.config_manager import ConfigManager
    from core.exchange_connector import ExchangeConnector
    from core.position_manager import PositionManager
    from core.grid_strategy import GridStrategy
    from core.risk_manager import RiskManager
    from core.database import Database
    from core.capital_allocator import CapitalAllocator
    from core.profit_monitor import ProfitMonitor
    from core.rebalance_manager import RebalanceManager
    from utils.logger import setup_logger, get_logger
    from web_api import WebAPI

logger = get_logger("bot")



class BotState(Enum):
    """机器人状态"""
    IDLE = "idle"
    SCANNING = "scanning"
    TRADING = "trading"
    MONITORING = "monitoring"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class TradingBot:
    """
    交易机器人主控程序

    协调各模块,实现7x24小时自动交易
    """

    def __init__(self, config_dir: Path):
        """
        Args:
            config_dir: 配置文件目录
        """
        logger.info("=" * 60)
        logger.info("量化交易系统启动")
        logger.info("=" * 60)

        # 加载配置
        self.config_mgr = ConfigManager(config_dir)
        self.config_mgr.load_configs()

        logger.info(str(self.config_mgr))

        # 初始化模块
        self.data_fetcher = BinanceDataFetcher(
            use_testnet=self.config_mgr.binance.testnet
        )

        self.connector = ExchangeConnector(
            self.data_fetcher,
            rate_limit_config={
                'max_calls_per_minute': self.config_mgr.rate_limit.max_calls_per_minute
            }
        )

        self.position_mgr = PositionManager(self.config_mgr, self.connector)
        self.grid_strategy = GridStrategy(self.config_mgr, self.connector, self.position_mgr)
        self.risk_mgr = RiskManager(
            self.config_mgr, self.connector, self.position_mgr, self.grid_strategy
        )

        # 初始化资金分配器(会自动获取账户余额)
        self.capital_allocator = CapitalAllocator(
            connector=self.connector,
            max_symbols=self.config_mgr.position.max_symbols,
            usage_ratio=self.config_mgr.account.usage_ratio
        )

        # 初始化盈利监控器
        self.profit_monitor = ProfitMonitor(
            profit_threshold=self.config_mgr.profit_taking.threshold,
            max_rebalances_per_cycle=2  # 每周期最多处理2个品种
        )

        # 初始化换仓管理器
        self.rebalance_mgr = RebalanceManager(
            cooldown_hours=self.config_mgr.rebalancing.cooldown_hours,
            max_rebalances_per_day=self.config_mgr.rebalancing.max_rebalances_per_day
        )

        logger.info(f"账户余额: {self.capital_allocator.total_balance:.2f} USDT")
        logger.info(f"可用资金: {self.capital_allocator.available_capital:.2f} USDT ({self.capital_allocator.usage_ratio*100:.0f}%)")
        logger.info(f"每品种目标分配: {self.capital_allocator.per_symbol_target:.2f} USDT")

        self.db = Database("data/database.db")

        # 状态
        self.state = BotState.IDLE
        self.running = True
        self.last_scan_date: Optional[date] = None
        self.current_candidates: List[str] = []  # 当前有效候选币种列表

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 初始化 Web API (如果启用)
        self.web_api = None
        self.web_thread = None
        if hasattr(self.config_mgr, 'web_dashboard') and self.config_mgr.web_dashboard.get('enabled', False):
            try:
                self.web_api = WebAPI(
                    trading_bot=self,
                    host=self.config_mgr.web_dashboard.get('host', '0.0.0.0'),
                    port=self.config_mgr.web_dashboard.get('port', 5000),
                    debug=self.config_mgr.web_dashboard.get('debug', False)
                )
                # 在独立线程中启动 Web 服务器
                self.web_thread = threading.Thread(target=self.web_api.run, daemon=True)
                self.web_thread.start()
                logger.info("Web 仪表板已启动")
            except Exception as e:
                logger.error(f"Web 仪表板启动失败: {e}")

        logger.info("所有模块初始化完成")

    def run(self) -> None:
        """主运行循环"""
        logger.info("进入主循环...")

        # 同步初始状态
        self.position_mgr.sync_positions()

        monitor_interval = self.config_mgr.schedule.monitor_interval

        while self.running:
            try:
                # 1. 每日币种筛选
                if self._should_scan_today():
                    self.state = BotState.SCANNING
                    self.daily_scan_and_select()

                # 2. 监控现有持仓
                self.state = BotState.MONITORING
                self.monitor_existing_positions()

                # 3. 更新网格状态
                self.update_all_grids()

                # 3.5. 盈利监控与换仓(如果启用)
                if self.config_mgr.rebalancing.enabled:
                    self.monitor_profit_rebalancing()

                # 3.6. 🔧 NEW: 检查是否需要补充新品种
                self.check_and_fill_positions()

                # 4. 风险检查
                self.handle_risk_alerts()

                # 5. 同步持仓
                self.position_mgr.sync_positions()

                # 6. 日志状态
                self.log_status()

                # 7. 休眠
                self.state = BotState.IDLE
                time.sleep(monitor_interval)

            except KeyboardInterrupt:
                logger.info("接收到中断信号,准备关闭...")
                break

            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                self.state = BotState.EMERGENCY
                time.sleep(60)  # 异常后等待1分钟

        self.shutdown()

    def _should_scan_today(self) -> bool:
        """检查是否应该执行今日筛选"""
        today = date.today()
        current_hour = datetime.now(timezone.utc).hour
        scan_hour = self.config_mgr.schedule.scan_hour

        # 如果今天还没扫描过,且当前时间>=扫描时间
        if self.last_scan_date != today and current_hour >= scan_hour:
            return True

        return False

    def daily_scan_and_select(self) -> None:
        """每日筛选币种"""
        logger.info("开始每日币种筛选...")

        try:
            today = date.today()

            # 运行筛选
            df = run_scan(
                as_of=today,
                bottom_n=self.config_mgr.screening.bottom_n,
                timeframe="1d",
                funding_cooldown=0.2,
                fetcher=self.data_fetcher
            )

            if df.empty:
                logger.warning("筛选结果为空")
                self.last_scan_date = today
                return

            logger.info(f"筛选到{len(df)}个候选币种")

            # 过滤上市时间
            valid_candidates = []
            for symbol in df['symbol'].head(10):  # 只检查前10个
                try:
                    history = self.data_fetcher.fetch_klines(symbol)
                    if is_listed_long_enough(
                        history,
                        as_of_date=today,
                        min_days=self.config_mgr.screening.min_listing_days
                    ):
                        valid_candidates.append(symbol)
                        logger.info(f"有效候选: {symbol}")
                except Exception as e:
                    logger.warning(f"检查{symbol}失败: {e}")

            logger.info(f"有效候选币种: {len(valid_candidates)}个")

            # 保存当前候选币列表（供换仓逻辑使用）
            self.current_candidates = valid_candidates[:10]  # 保存前10个候选
            logger.info(f"更新候选币列表: {self.current_candidates}")

            # 1. 优先处理手动指定币种（仅处理未持仓的）
            manual_symbols = self.config_mgr.position.manual_symbols
            if manual_symbols:
                # 🔧 FIX: 过滤掉已持仓的manual_symbols
                existing_symbols = set(self.position_mgr.get_all_symbols())
                new_manual_symbols = [s for s in manual_symbols if s not in existing_symbols]

                if new_manual_symbols:
                    logger.info(f"检测到未持仓的手动指定币种: {new_manual_symbols}")
                    self.evaluate_new_entries(new_manual_symbols)
                else:
                    logger.info(f"手动指定币种 {manual_symbols} 已全部持仓")

            # 2. 处理筛选出的候选币种
            self.evaluate_new_entries(valid_candidates[:5])

            self.last_scan_date = today

        except Exception as e:
            logger.error(f"每日筛选失败: {e}", exc_info=True)

    def _validate_capital_before_grid_init(self, required_margin: float, pending_count: int) -> bool:
        """
        验证初始化网格前资金是否充足且不超过90%限制

        Args:
            required_margin: 单个品种所需保证金
            pending_count: 已经准备初始化的品种数量

        Returns:
            bool: True表示可以初始化，False表示资金不足或超限
        """
        try:
            # 计算当前已使用的保证金
            current_usage = 0.0
            for symbol in self.grid_strategy.grid_states.keys():
                try:
                    position = self.connector.get_position(symbol)
                    if position and position.margin:
                        current_usage += abs(position.margin)
                except Exception as e:
                    logger.warning(f"获取{symbol}保证金失败: {e}")

            # 计算新增保证金需求
            new_margin = required_margin * (pending_count + 1)
            total_usage = current_usage + new_margin

            # 检查是否超过90%限制
            available_capital = self.capital_allocator.available_capital
            usage_pct = (total_usage / self.capital_allocator.total_balance) * 100

            if total_usage > available_capital:
                logger.error(
                    f"⚠️ 资金超限：当前使用 {current_usage:.2f} USDT，"
                    f"新增 {new_margin:.2f} USDT，"
                    f"总计 {total_usage:.2f} USDT ({usage_pct:.1f}%)，"
                    f"超过90%限制 ({available_capital:.2f} USDT)"
                )
                return False

            logger.info(
                f"资金验证通过：当前 {current_usage:.2f} USDT，"
                f"新增 {new_margin:.2f} USDT，"
                f"总计 {total_usage:.2f} USDT ({usage_pct:.1f}%)，"
                f"限制 {available_capital:.2f} USDT (90%)"
            )
            return True

        except Exception as e:
            logger.error(f"资金验证失败: {e}", exc_info=True)
            return False

    def evaluate_new_entries(self, candidates: List[str]) -> None:
        """
        评估新入场机会

        Args:
            candidates: 候选币种列表
        """
        current_count = self.position_mgr.get_position_count()
        max_count = self.config_mgr.position.max_symbols

        if current_count >= max_count:
            logger.info(f"已达最大持仓数{max_count},不开新仓")
            return

        # 计算所需保证金
        required_margin = (
            self.config_mgr.position.base_margin +
            self.config_mgr.position.grid_margin * self.config_mgr.grid.upper_grids
        )

        logger.info(f"评估{len(candidates)}个候选币种...")

        # 筛选符合条件的币种
        symbols_to_init = []
        for symbol in candidates:
            if current_count + len(symbols_to_init) >= max_count:
                break

            # 检查是否已持仓
            if self.position_mgr.get_symbol_position(symbol):
                logger.info(f"已持仓: {symbol},跳过")
                continue

            # 检查保证金
            # 🔧 FIX: 添加90%资金约束验证
            if not self._validate_capital_before_grid_init(required_margin, len(symbols_to_init)):
                logger.warning("资金不足或超过90%限制,无法开新仓")
                break

            if not self.position_mgr.can_open_new_position(required_margin):
                logger.warning("保证金不足,无法开新仓")
                break

            symbols_to_init.append(symbol)

        if not symbols_to_init:
            logger.info("没有符合条件的候选币种")
            return

        logger.info(f"准备并行初始化{len(symbols_to_init)}个币种: {symbols_to_init}")

        # 并行初始化所有候选币种
        with ThreadPoolExecutor(max_workers=len(symbols_to_init)) as executor:
            # 提交所有初始化任务
            future_to_symbol = {}
            for symbol in symbols_to_init:
                try:
                    entry_price = self.connector.get_current_price(symbol)
                    logger.info(f"提交初始化任务: {symbol} @ {entry_price}")
                    future = executor.submit(self.grid_strategy.initialize_grid, symbol, entry_price)
                    future_to_symbol[future] = symbol
                except Exception as e:
                    logger.error(f"获取价格失败 {symbol}: {e}")

            # 等待所有任务完成（每个最多1小时，所以总超时 = 1小时 + 缓冲）
            success_count = 0
            for future in as_completed(future_to_symbol.keys(), timeout=3900):  # 65分钟总超时
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"✅ 并行初始化成功: {symbol}")
                    else:
                        logger.warning(f"❌ 并行初始化失败: {symbol}")
                except Exception as e:
                    logger.error(f"❌ 并行初始化异常 {symbol}: {e}")

            logger.info(f"并行初始化完成: 成功{success_count}/{len(symbols_to_init)}个币种")

    def monitor_existing_positions(self) -> None:
        """监控现有持仓"""
        symbols = self.position_mgr.get_all_symbols()

        if not symbols:
            return

        logger.debug(f"监控{len(symbols)}个持仓...")

        for symbol in symbols:
            try:
                # 检查并恢复缺失的网格状态
                if symbol not in self.grid_strategy.grid_states:
                    logger.warning(f"发现持仓但无网格状态: {symbol}, 尝试恢复...")
                    sym_pos = self.position_mgr.get_symbol_position(symbol)
                    if sym_pos:
                        entry_price = sym_pos.entry_price
                        self.grid_strategy.recover_grid_from_position(symbol, entry_price)

                # 更新未实现盈亏
                current_price = self.connector.get_current_price(symbol)
                self.position_mgr.update_unrealized_pnl(symbol, current_price)

            except Exception as e:
                logger.warning(f"监控失败 {symbol}: {e}")

    def check_and_fill_positions(self) -> None:
        """
        检查持仓健康度并自动补充新品种

        如果现有持仓的空头头寸不足，且持仓数量未达上限，则开新品种
        """
        # 每10次循环检查一次（避免过于频繁）
        if not hasattr(self, '_fill_check_count'):
            self._fill_check_count = 0

        self._fill_check_count += 1

        if self._fill_check_count % 10 != 0:
            return

        current_count = self.position_mgr.get_position_count()
        max_count = self.config_mgr.position.max_symbols

        # 检查不健康的持仓（两种来源）
        # 1. PositionManager检测的（空头头寸不足40%）
        unhealthy_from_position = self.position_mgr.get_unhealthy_positions(
            min_ratio=self.config_mgr.position.min_base_position_ratio
        )

        # 2. GridStrategy检测的（IMBALANCE）
        unhealthy_from_grid = self.grid_strategy.get_unhealthy_symbols()

        # 合并两个来源
        unhealthy_positions = list(set(unhealthy_from_position) | unhealthy_from_grid)

        if unhealthy_positions:
            logger.info(
                f"检测到{len(unhealthy_positions)}个不健康持仓: {unhealthy_positions}, "
                f"当前持仓{current_count}/{max_count}"
            )

        # 如果持仓数量未达上限，或有不健康持仓，尝试补充新品种
        if current_count < max_count or unhealthy_positions:
            # 使用当前候选币列表
            if self.current_candidates:
                logger.info(f"尝试从候选币列表补充新品种: {self.current_candidates[:3]}")
                self.evaluate_new_entries(self.current_candidates[:5])
            else:
                logger.debug("没有可用的候选币列表，跳过补充")

    def update_all_grids(self) -> None:
        """更新所有网格状态"""
        self.grid_strategy.update_grid_states()

    def handle_risk_alerts(self) -> None:
        """处理风险预警"""
        alerts = self.risk_mgr.monitor_all_positions()

        for alert in alerts:
            # 保存到数据库
            self.db.save_alert({
                'level': alert.level,
                'symbol': alert.symbol,
                'message': alert.message
            })

            # Level 3 的紧急止损已经在risk_mgr中处理

    def log_status(self) -> None:
        """记录状态"""
        # 每10次循环记录一次详细状态
        if not hasattr(self, '_loop_count'):
            self._loop_count = 0

        self._loop_count += 1

        if self._loop_count % 10 == 0:
            logger.info(f"状态: {self.state.value}")
            logger.info(str(self.position_mgr))

    def monitor_profit_rebalancing(self) -> None:
        """监控盈利并触发换仓"""
        # 更新所有品种的盈利状态
        for symbol in self.position_mgr.get_all_symbols():
            position = self.position_mgr.get_symbol_position(symbol)
            if not position:
                continue

            # 更新盈利监控器
            if symbol not in self.profit_monitor.get_monitored_symbols():
                # 添加到监控列表
                self.profit_monitor.add_symbol(symbol, position.initial_margin or position.total_margin_used)

            # 更新盈利状态
            self.profit_monitor.update_symbol_profit(
                symbol=symbol,
                current_margin=position.total_margin_used,
                unrealized_pnl=position.unrealized_pnl
            )

        # 检查是否有品种达到盈利目标
        symbols_to_rebalance = self.profit_monitor.check_profit_threshold()

        # 检查今日换仓次数
        if not self.rebalance_mgr.can_rebalance_today():
            logger.info("今日换仓已达上限,跳过本次检查")
            return

        # 执行换仓
        for symbol in symbols_to_rebalance:
            # 保护手动指定币种（永不换仓）
            if symbol in self.config_mgr.position.manual_symbols:
                logger.info(f"{symbol} 是手动指定币种，跳过换仓")
                continue

            # 检查是否仍在候选币中（如果仍是好标的，继续持有让利润奔跑）
            if symbol in self.current_candidates:
                logger.info(f"{symbol} 盈利达标但仍在候选币中，继续持有")
                continue

            # 二次确认盈利率(防止触发时profit=15%,执行时已跌至10%)
            if not self.profit_monitor.verify_profit_before_rebalance(symbol, min_threshold=0.10):
                logger.warning(f"{symbol} 盈利已回撤,取消换仓")
                continue

            logger.info(f"{symbol} 盈利达标且不在候选币中，准备换仓")

            try:
                success = self.execute_rebalancing(symbol, reason="profit_target")
                if success:
                    logger.info(f"{symbol} 换仓成功")
                else:
                    logger.error(f"{symbol} 换仓失败")
            except Exception as e:
                logger.error(f"{symbol} 换仓异常: {e}", exc_info=True)

    def execute_rebalancing(self, symbol: str, reason: str) -> bool:
        """
        执行换仓流程

        Args:
            symbol: 需要换仓的品种
            reason: 换仓原因(profit_target/stop_loss)

        Returns:
            bool: 是否成功
        """
        logger.info(f"开始执行换仓: {symbol}, 原因: {reason}")

        # 1. 关闭现有品种
        logger.info(f"Step 1/4: 关闭 {symbol} 的所有持仓和订单")
        try:
            # 获取当前盈利信息
            position = self.position_mgr.get_symbol_position(symbol)
            profit_state = self.profit_monitor.get_symbol_state(symbol)

            realized_pnl = position.unrealized_pnl if position else 0.0
            profit_percentage = profit_state.profit_percentage if profit_state else 0.0

            # 关闭网格(取消订单+平仓)
            self.grid_strategy.close_grid(symbol, reason=reason)

            # 释放资金分配
            freed_margin = self.capital_allocator.free_symbol(symbol)

            # 从盈利监控移除
            self.profit_monitor.remove_symbol(symbol)

            logger.info(f"{symbol} 关闭完成, 释放资金: {freed_margin:.2f} USDT")

        except Exception as e:
            logger.error(f"关闭 {symbol} 失败: {e}", exc_info=True)
            return False

        # 2. 标记冷却期
        logger.info(f"Step 2/4: 将 {symbol} 加入冷却期")
        self.rebalance_mgr.add_to_cooldown(
            symbol=symbol,
            reason=reason,
            realized_pnl=realized_pnl,
            profit_percentage=profit_percentage
        )

        # 3. 选择新候选
        logger.info(f"Step 3/4: 选择新候选品种")
        candidates = self._get_available_candidates()

        if not candidates:
            logger.warning("无可用候选品种,换仓终止")
            return False

        new_symbol = candidates[0]
        logger.info(f"选择新品种: {new_symbol}")

        # 4. 初始化新网格
        logger.info(f"Step 4/4: 为 {new_symbol} 初始化网格")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 分配资金
                allocation = self.capital_allocator.allocate_symbol(new_symbol)

                # 获取当前价格
                entry_price = self.connector.get_current_price(new_symbol)

                # 初始化网格
                success = self.grid_strategy.initialize_grid(new_symbol, entry_price)

                if success:
                    # 添加到盈利监控
                    self.profit_monitor.add_symbol(new_symbol, allocation.target_margin)

                    logger.info(f"换仓完成: {symbol} → {new_symbol}")
                    return True

            except Exception as e:
                logger.warning(f"初始化 {new_symbol} 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2)  # 等待2秒后重试

        # 所有重试失败,尝试第2个候选
        if len(candidates) > 1:
            logger.info(f"尝试备选品种: {candidates[1]}")
            # 递归调用(但用第2个候选,避免无限循环)
            # 这里简化处理,直接返回失败
            pass

        logger.error(f"{new_symbol} 初始化失败,换仓未完成")
        return False

    def _get_available_candidates(self) -> List[str]:
        """
        获取可用候选品种

        Returns:
            List[str]: 候选品种列表(已过滤冷却期和当前持仓)
        """
        # 从文件加载候选(假设daily_scan已生成)
        try:
            # 这里需要实现从daily_candidate_scan的结果读取
            # 简化实现:返回一个示例列表(实际应该从文件或数据库读取)
            all_candidates = self._load_daily_candidates()

            # 过滤:
            # 1. 当前已持有的品种
            # 2. 冷却期内的品种
            current_symbols = set(self.position_mgr.get_all_symbols())
            cooldown_symbols = self.rebalance_mgr.get_cooldown_symbols()

            available = [
                s for s in all_candidates
                if s not in current_symbols and s not in cooldown_symbols
            ]

            logger.info(f"候选品种: 总数 {len(all_candidates)}, 可用 {len(available)}")
            return available[:10]  # 返回前10个

        except Exception as e:
            logger.error(f"获取候选品种失败: {e}")
            return []

    def _load_daily_candidates(self) -> List[str]:
        """
        加载每日筛选的候选品种

        Returns:
            List[str]: 候选品种列表
        """
        # TODO: 实现从daily_candidate_scan的输出文件读取
        # 这里暂时返回空列表,需要集成daily_candidate_scan的输出
        candidates_file = Path("data/daily_candidates.txt")
        if candidates_file.exists():
            with open(candidates_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def shutdown(self) -> None:
        """优雅关闭"""
        logger.info("开始关闭系统...")

        self.state = BotState.SHUTDOWN
        self.running = False

        # 关闭数据库
        self.db.close()

        logger.info("系统已关闭")

    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"接收到信号{signum},准备关闭...")
        self.running = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="量化交易机器人")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="配置文件目录"
    )

    parser.add_argument(
        "--testnet",
        action="store_true",
        help="使用测试网"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟模式(不实际下单)"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logger(
        name="QuantSystem",
        log_dir="logs",
        log_level="INFO",
        console_output=True
    )

    try:
        # 创建并运行交易机器人
        bot = TradingBot(config_dir=args.config)
        bot.run()

    except Exception as e:
        logger.critical(f"系统启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
