"""
自定义异常类
Custom Exception Classes

定义系统使用的所有自定义异常
"""

from typing import Optional


class QuantSystemException(Exception):
    """量化系统基础异常类"""
    pass


class ConfigurationError(QuantSystemException):
    """配置错误"""
    pass


class ExchangeError(QuantSystemException):
    """交易所相关错误"""
    pass


class OrderError(ExchangeError):
    """订单相关错误"""
    def __init__(self, message: str, order_id: Optional[str] = None):
        super().__init__(message)
        self.order_id = order_id


class PositionError(QuantSystemException):
    """仓位相关错误"""
    pass


class RiskError(QuantSystemException):
    """风险控制错误"""
    pass


class InsufficientMarginError(PositionError):
    """保证金不足错误"""
    pass


class PositionLimitError(PositionError):
    """仓位限制错误"""
    pass


class NetworkError(ExchangeError):
    """网络连接错误"""
    pass


class RateLimitError(ExchangeError):
    """API速率限制错误"""
    pass


class DataError(QuantSystemException):
    """数据相关错误"""
    pass


class DatabaseError(DataError):
    """数据库操作错误"""
    pass
