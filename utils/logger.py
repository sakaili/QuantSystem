"""
日志系统模块
Logging System Module

提供结构化日志记录功能,支持文件和控制台输出
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "QuantSystem",
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """
    配置并返回logger实例

    Args:
        name: Logger名称
        log_dir: 日志文件目录
        log_level: 日志级别(DEBUG/INFO/WARNING/ERROR/CRITICAL)
        console_output: 是否输出到控制台

    Returns:
        配置好的Logger实例
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器 - 按日期分割日志文件
    today = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(
        log_path / f"trading_bot_{today}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 错误日志单独记录
    error_handler = logging.FileHandler(
        log_path / f"error_{today}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取已配置的logger实例

    Args:
        name: Logger名称,None则返回根logger

    Returns:
        Logger实例
    """
    if name:
        return logging.getLogger(f"QuantSystem.{name}")
    return logging.getLogger("QuantSystem")


# 全局logger实例
logger = setup_logger()
