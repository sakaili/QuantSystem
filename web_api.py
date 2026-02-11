"""
Web API 服务器
Web API Server

提供 RESTful API 接口供前端仪表板访问
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
import traceback
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("web_api")


class WebAPI:
    """
    Web API 服务器

    提供 RESTful API 接口,暴露交易机器人的状态和数据
    """

    def __init__(self, trading_bot, host='0.0.0.0', port=5000, debug=False):
        """
        Args:
            trading_bot: TradingBot 实例
            host: 监听地址
            port: 监听端口
            debug: 调试模式
        """
        self.bot = trading_bot
        self.host = host
        self.port = port
        self.debug = debug

        # 创建 Flask 应用
        self.app = Flask(__name__,
                        static_folder='static',
                        static_url_path='')

        # 启用 CORS
        CORS(self.app)

        # 注册路由
        self._register_routes()

        logger.info(f"Web API 初始化完成: {host}:{port}")

    def _register_routes(self):
        """注册所有 API 路由"""

        # 首页
        @self.app.route('/')
        def index():
            return send_from_directory('static', 'index.html')

        # API 端点
        @self.app.route('/api/account')
        def get_account():
            """获取账户总览"""
            try:
                return jsonify(self._get_account_data())
            except Exception as e:
                logger.error(f"获取账户数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/positions')
        def get_positions():
            """获取持仓列表"""
            try:
                return jsonify(self._get_positions_data())
            except Exception as e:
                logger.error(f"获取持仓数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/positions/<symbol>/grid')
        def get_grid(symbol):
            """获取特定交易对的网格状态"""
            try:
                return jsonify(self._get_grid_data(symbol))
            except Exception as e:
                logger.error(f"获取网格数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/profit')
        def get_profit():
            """获取盈利监控数据"""
            try:
                return jsonify(self._get_profit_data())
            except Exception as e:
                logger.error(f"获取盈利数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/orders')
        def get_orders():
            """获取订单列表"""
            try:
                return jsonify(self._get_orders_data())
            except Exception as e:
                logger.error(f"获取订单数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trades')
        def get_trades():
            """获取交易历史"""
            try:
                return jsonify(self._get_trades_data())
            except Exception as e:
                logger.error(f"获取交易数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/alerts')
        def get_alerts():
            """获取风险告警"""
            try:
                return jsonify(self._get_alerts_data())
            except Exception as e:
                logger.error(f"获取告警数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/system')
        def get_system():
            """获取系统状态"""
            try:
                return jsonify(self._get_system_data())
            except Exception as e:
                logger.error(f"获取系统数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/performance')
        def get_performance():
            """获取性能指标"""
            try:
                return jsonify(self._get_performance_data())
            except Exception as e:
                logger.error(f"获取性能数据失败: {e}\n{traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

    def _get_account_data(self) -> Dict[str, Any]:
        """获取账户总览数据"""
        try:
            balance = self.bot.connector.get_balance()
            
            # 计算总未实现盈亏
            total_unrealized_pnl = 0.0
            for symbol_pos in self.bot.position_mgr.positions.values():
                total_unrealized_pnl += symbol_pos.unrealized_pnl
            
            # 从数据库获取总已实现盈亏
            total_realized_pnl = self._get_total_realized_pnl()
            
            return {
                'total_balance': round(balance.total, 2),
                'available_margin': round(balance.available, 2),
                'used_margin': round(balance.used, 2),
                'usage_percentage': round((balance.used / balance.total * 100) if balance.total > 0 else 0, 2),
                'total_unrealized_pnl': round(total_unrealized_pnl, 2),
                'total_realized_pnl': round(total_realized_pnl, 2),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"获取账户数据失败: {e}")
            return {
                'total_balance': 0.0,
                'available_margin': 0.0,
                'used_margin': 0.0,
                'usage_percentage': 0.0,
                'total_unrealized_pnl': 0.0,
                'total_realized_pnl': 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    def _get_positions_data(self) -> Dict[str, Any]:
        """获取持仓列表数据"""
        positions = []
        
        for symbol, symbol_pos in self.bot.position_mgr.positions.items():
            try:
                # 获取当前价格
                current_price = self.bot.connector.get_ticker_price(symbol)
                
                positions.append({
                    'symbol': symbol,
                    'entry_price': round(symbol_pos.entry_price, 4),
                    'current_price': round(current_price, 4),
                    'size': round(symbol_pos.total_size, 4),
                    'unrealized_pnl': round(symbol_pos.unrealized_pnl, 2),
                    'profit_percentage': round(symbol_pos.profit_percentage * 100, 2),
                    'peak_profit_percentage': round(symbol_pos.peak_profit_percentage * 100, 2),
                    'margin_used': round(symbol_pos.total_margin_used, 2),
                    'leverage': self.bot.config_mgr.account.leverage,
                    'last_update': symbol_pos.last_update.isoformat()
                })
            except Exception as e:
                logger.error(f"获取 {symbol} 持仓数据失败: {e}")
                continue
        
        return {
            'positions': positions,
            'count': len(positions),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _get_grid_data(self, symbol: str) -> Dict[str, Any]:
        """获取网格状态数据"""
        if symbol not in self.bot.grid_strategy.grid_states:
            return {'error': f'Symbol {symbol} not found'}
        
        grid_state = self.bot.grid_strategy.grid_states[symbol]
        
        try:
            current_price = self.bot.connector.get_ticker_price(symbol)
            
            # 统计网格信息
            upper_active = sum(
                len(order_ids)
                for price, order_ids in grid_state.upper_orders.items()
                if price > grid_state.entry_price
            )
            lower_active = sum(
                len(order_ids)
                for price, order_ids in grid_state.lower_orders.items()
                if price < grid_state.entry_price
            )
            filled_upper = len(grid_state.filled_upper_grids)
            
            # 构建网格价格字典
            grid_levels = {}
            for level, price in grid_state.grid_prices.grid_levels.items():
                grid_levels[str(level)] = round(price, 4)
            
            return {
                'symbol': symbol,
                'entry_price': round(grid_state.entry_price, 4),
                'current_price': round(current_price, 4),
                'grid_spacing': grid_state.grid_prices.spacing,
                'stop_loss_price': round(grid_state.grid_prices.stop_loss_price, 4),
                'upper_grids': {
                    'active': upper_active,
                    'filled': filled_upper,
                    'success_rate': round(grid_state.upper_success_rate, 2)
                },
                'lower_grids': {
                    'active': lower_active,
                    'filled': 0,  # 下网格成交后立即平仓,不累计
                    'success_rate': round(grid_state.lower_success_rate, 2)
                },
                'grid_levels': grid_levels,
                'last_update': grid_state.last_update.isoformat()
            }
        except Exception as e:
            logger.error(f"获取 {symbol} 网格数据失败: {e}")
            return {'error': str(e)}
    
    def _get_profit_data(self) -> Dict[str, Any]:
        """获取盈利监控数据"""
        try:
            summary = self.bot.profit_monitor.get_summary()
            
            # 获取各交易对盈利排行
            symbol_profits = []
            for symbol, state in self.bot.profit_monitor.symbol_states.items():
                symbol_profits.append({
                    'symbol': symbol,
                    'profit_percentage': round(state.profit_percentage * 100, 2),
                    'unrealized_pnl': round(state.unrealized_pnl, 2),
                    'peak_profit': round(state.peak_profit * 100, 2)
                })
            
            # 按盈利率排序
            symbol_profits.sort(key=lambda x: x['profit_percentage'], reverse=True)
            
            return {
                'monitored_symbols': summary['monitored_symbols'],
                'avg_profit': round(summary['avg_profit'] * 100, 2),
                'max_profit': round(summary['max_profit'] * 100, 2),
                'min_profit': round(summary['min_profit'] * 100, 2),
                'symbols_over_threshold': summary['symbols_over_threshold'],
                'profit_threshold': round(self.bot.profit_monitor.profit_threshold * 100, 2),
                'symbol_profits': symbol_profits,
                'rebalance_queue': list(self.bot.profit_monitor.rebalance_queue),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"获取盈利数据失败: {e}")
            return {'error': str(e)}

    def _get_orders_data(self) -> Dict[str, Any]:
        """获取订单列表数据"""
        try:
            # 从数据库获取最近的订单
            orders = self.bot.db.get_orders(limit=100)
            
            # 按交易对统计挂单数量
            open_orders_by_symbol = {}
            for symbol in self.bot.position_mgr.positions.keys():
                try:
                    open_orders = self.bot.connector.get_open_orders(symbol)
                    open_orders_by_symbol[symbol] = len(open_orders)
                except:
                    open_orders_by_symbol[symbol] = 0
            
            return {
                'recent_orders': orders[:50],  # 最近50个订单
                'open_orders_by_symbol': open_orders_by_symbol,
                'total_open_orders': sum(open_orders_by_symbol.values()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"获取订单数据失败: {e}")
            return {'error': str(e)}
    
    def _get_trades_data(self) -> Dict[str, Any]:
        """获取交易历史数据"""
        try:
            # 从数据库获取最近24小时的交易
            trades = self.bot.db.get_trades(limit=100)
            
            # 计算统计信息
            total_trades = len(trades)
            profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'recent_trades': trades[:50],  # 最近50笔交易
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': round(win_rate, 2),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"获取交易数据失败: {e}")
            return {'error': str(e)}
    
    def _get_alerts_data(self) -> Dict[str, Any]:
        """获取风险告警数据"""
        try:
            # 从数据库获取最近的告警
            alerts = self.bot.db.get_alerts(limit=50)
            
            # 按级别分类
            alerts_by_level = {
                'info': [],
                'warning': [],
                'critical': []
            }
            
            for alert in alerts:
                level = alert.get('level', 1)
                if level == 1:
                    alerts_by_level['info'].append(alert)
                elif level == 2:
                    alerts_by_level['warning'].append(alert)
                elif level == 3:
                    alerts_by_level['critical'].append(alert)
            
            return {
                'recent_alerts': alerts,
                'alerts_by_level': alerts_by_level,
                'total_alerts': len(alerts),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"获取告警数据失败: {e}")
            return {'error': str(e)}

    def _get_system_data(self) -> Dict[str, Any]:
        """获取系统状态数据"""
        try:
            return {
                'bot_state': self.bot.state.value if hasattr(self.bot, 'state') else 'unknown',
                'running': self.bot.running if hasattr(self.bot, 'running') else False,
                'last_scan_date': self.bot.last_scan_date.isoformat() if hasattr(self.bot, 'last_scan_date') and self.bot.last_scan_date else None,
                'current_candidates': self.bot.current_candidates if hasattr(self.bot, 'current_candidates') else [],
                'active_positions': len(self.bot.position_mgr.positions),
                'max_positions': self.bot.config_mgr.position.max_symbols,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"获取系统数据失败: {e}")
            return {'error': str(e)}
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """获取性能指标数据"""
        try:
            # 从数据库获取交易历史
            trades = self.bot.db.get_trades(limit=1000)
            
            # 计算日/周/月盈亏
            now = datetime.now(timezone.utc)
            daily_pnl = self._calculate_pnl_for_period(trades, now - timedelta(days=1))
            weekly_pnl = self._calculate_pnl_for_period(trades, now - timedelta(days=7))
            monthly_pnl = self._calculate_pnl_for_period(trades, now - timedelta(days=30))
            
            # 计算胜率
            total_trades = len(trades)
            profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'daily_pnl': round(daily_pnl, 2),
                'weekly_pnl': round(weekly_pnl, 2),
                'monthly_pnl': round(monthly_pnl, 2),
                'total_trades': total_trades,
                'win_rate': round(win_rate, 2),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"获取性能数据失败: {e}")
            return {'error': str(e)}
    
    def _get_total_realized_pnl(self) -> float:
        """从数据库计算总已实现盈亏"""
        try:
            trades = self.bot.db.get_trades(limit=10000)
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            return total_pnl
        except Exception as e:
            logger.error(f"计算总盈亏失败: {e}")
            return 0.0
    
    def _calculate_pnl_for_period(self, trades: list, start_time: datetime) -> float:
        """计算指定时间段的盈亏"""
        period_trades = [
            t for t in trades 
            if datetime.fromisoformat(t.get('timestamp', '').replace('Z', '+00:00')) >= start_time
        ]
        return sum(t.get('pnl', 0) for t in period_trades)
    
    def run(self):
        """启动 Web 服务器"""
        logger.info(f"启动 Web API 服务器: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False)
