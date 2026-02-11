"""
数据库模块
Database Module

持久化订单、持仓、交易记录
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from utils.exceptions import DatabaseError
from utils.logger import get_logger

logger = get_logger("database")


class Database:
    """
    数据库管理器

    使用SQLite存储交易数据
    """

    def __init__(self, db_path: str = "data/database.db"):
        """
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.create_tables()

        logger.info(f"数据库初始化完成: {db_path}")

    def create_tables(self) -> None:
        """创建数据表"""
        cursor = self.conn.cursor()

        # 订单表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                price REAL,
                amount REAL,
                filled REAL,
                status TEXT,
                timestamp TIMESTAMP,
                grid_level INTEGER
            )
        """)

        # 网格循环表（上方开仓 -> 下方止盈）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grid_cycles (
                upper_order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                upper_price REAL,
                upper_amount REAL,
                tp_order_id TEXT,
                tp_price REAL,
                tp_amount REAL,
                status TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        # 持仓表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_price REAL,
                size REAL,
                margin REAL,
                unrealized_pnl REAL,
                timestamp TIMESTAMP
            )
        """)

        # 交易记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT,
                price REAL,
                amount REAL,
                pnl REAL,
                timestamp TIMESTAMP
            )
        """)

        # 风险预警表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level INTEGER,
                symbol TEXT,
                message TEXT,
                timestamp TIMESTAMP
            )
        """)

        self.conn.commit()
        logger.info("数据表创建完成")

    def save_order(self, order: Dict[str, Any]) -> None:
        """保存订单"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, symbol, side, order_type, price, amount, filled, status, timestamp, grid_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.get('order_id'),
                order.get('symbol'),
                order.get('side'),
                order.get('order_type'),
                order.get('price'),
                order.get('amount'),
                order.get('filled'),
                order.get('status'),
                order.get('timestamp'),
                order.get('grid_level')
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"保存订单失败: {e}")

    def save_position_snapshot(self, position: Dict[str, Any]) -> None:
        """保存持仓快照"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO positions
                (symbol, entry_price, size, margin, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                position.get('symbol'),
                position.get('entry_price'),
                position.get('size'),
                position.get('margin'),
                position.get('unrealized_pnl'),
                datetime.now()
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"保存持仓失败: {e}")

    def save_trade(self, trade: Dict[str, Any]) -> None:
        """保存交易记录"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades
                (symbol, side, price, amount, pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                trade.get('symbol'),
                trade.get('side'),
                trade.get('price'),
                trade.get('amount'),
                trade.get('pnl'),
                datetime.now()
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"保存交易失败: {e}")

    def save_alert(self, alert: Dict[str, Any]) -> None:
        """保存风险预警"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO alerts
                (level, symbol, message, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                alert.get('level'),
                alert.get('symbol'),
                alert.get('message'),
                datetime.now()
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"保存预警失败: {e}")

    def save_grid_cycle(self, cycle: Dict[str, Any]) -> None:
        """保存或更新网格循环记录"""
        try:
            cursor = self.conn.cursor()
            now = datetime.now()
            cursor.execute("""
                INSERT OR REPLACE INTO grid_cycles
                (upper_order_id, symbol, upper_price, upper_amount, tp_order_id, tp_price, tp_amount, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM grid_cycles WHERE upper_order_id = ?), ?
                ), ?)
            """, (
                cycle.get("upper_order_id"),
                cycle.get("symbol"),
                cycle.get("upper_price"),
                cycle.get("upper_amount"),
                cycle.get("tp_order_id"),
                cycle.get("tp_price"),
                cycle.get("tp_amount"),
                cycle.get("status", "open"),
                cycle.get("upper_order_id"),
                now,
                now
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"保存网格循环失败: {e}")

    def load_open_grid_cycles(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """加载未完成的网格循环记录"""
        try:
            cursor = self.conn.cursor()
            if symbol:
                cursor.execute("""
                    SELECT upper_order_id, symbol, upper_price, upper_amount,
                           tp_order_id, tp_price, tp_amount, status, created_at, updated_at
                    FROM grid_cycles
                    WHERE symbol = ? AND status = 'open'
                """, (symbol,))
            else:
                cursor.execute("""
                    SELECT upper_order_id, symbol, upper_price, upper_amount,
                           tp_order_id, tp_price, tp_amount, status, created_at, updated_at
                    FROM grid_cycles
                    WHERE status = 'open'
                """)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "upper_order_id": row[0],
                    "symbol": row[1],
                    "upper_price": row[2],
                    "upper_amount": row[3],
                    "tp_order_id": row[4],
                    "tp_price": row[5],
                    "tp_amount": row[6],
                    "status": row[7],
                    "created_at": row[8],
                    "updated_at": row[9]
                })
            return results
        except Exception as e:
            logger.error(f"加载网格循环失败: {e}")
            return []

    def close_grid_cycle_by_tp(self, tp_order_id: str) -> None:
        """标记网格循环为已完成（按止盈单）"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE grid_cycles
                SET status = 'closed', updated_at = ?
                WHERE tp_order_id = ?
            """, (datetime.now(), tp_order_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"关闭网格循环失败: {e}")

    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
