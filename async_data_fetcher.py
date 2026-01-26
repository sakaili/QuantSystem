"""
异步数据获取器
Async Data Fetcher

使用异步+并发控制优化API调用，避免速率限制
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

# 确保可以导入项目模块
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data_fetcher import BinanceDataFetcher
from utils.logger import get_logger

logger = get_logger("async_fetcher")


@dataclass
class FundingRateResult:
    """资金费率查询结果"""
    symbol: str
    funding_rate: Optional[float]
    success: bool
    error: Optional[str] = None


class AsyncDataFetcher:
    """
    异步数据获取器

    特性:
    - 异步批量获取资金费率
    - 并发控制（避免速率限制）
    - 自动重试机制
    - 进度显示
    """

    def __init__(
        self,
        fetcher: Optional[BinanceDataFetcher] = None,
        concurrency: int = 20,  # 并发数限制
        delay_per_request: float = 0.05,  # 每个请求间隔50ms
        max_retries: int = 3,
        verbose: bool = True
    ):
        """
        Args:
            fetcher: BinanceDataFetcher实例
            concurrency: 最大并发数（推荐10-30）
            delay_per_request: 每个请求的延迟（秒）
            max_retries: 最大重试次数
            verbose: 是否显示进度
        """
        self.fetcher = fetcher or BinanceDataFetcher()
        self.concurrency = concurrency
        self.delay_per_request = delay_per_request
        self.max_retries = max_retries
        self.verbose = verbose

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'retries': 0
        }

    async def fetch_funding_rate_async(
        self,
        symbol: str,
        semaphore: asyncio.Semaphore,
        retry_count: int = 0
    ) -> FundingRateResult:
        """
        异步获取单个币种的资金费率

        Args:
            symbol: 交易对
            semaphore: 并发控制信号量
            retry_count: 当前重试次数

        Returns:
            FundingRateResult
        """
        async with semaphore:
            try:
                # 使用asyncio.to_thread在线程池中执行同步CCXT调用
                # 这样可以避免阻塞事件循环
                fr = await asyncio.to_thread(
                    self.fetcher.exchange.fetch_funding_rate,
                    symbol
                )

                funding_rate = fr.get("fundingRate")

                # 添加延迟，控制请求速率
                await asyncio.sleep(self.delay_per_request)

                return FundingRateResult(
                    symbol=symbol,
                    funding_rate=funding_rate,
                    success=True
                )

            except Exception as exc:
                error_msg = str(exc)

                # 如果遇到速率限制且还有重试次数
                if "429" in error_msg and retry_count < self.max_retries:
                    self.stats['retries'] += 1

                    # 指数退避
                    wait_time = 2 ** retry_count
                    logger.warning(
                        f"{symbol} 遇到速率限制，等待 {wait_time}s 后重试 "
                        f"({retry_count+1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)

                    # 递归重试（使用新的semaphore以避免死锁）
                    return await self.fetch_funding_rate_async(
                        symbol, semaphore, retry_count + 1
                    )

                # 其他错误或重试次数用完
                if retry_count == 0:  # 只在第一次失败时记录警告
                    logger.warning(f"获取 {symbol} 资金费率失败: {error_msg}")

                return FundingRateResult(
                    symbol=symbol,
                    funding_rate=None,
                    success=False,
                    error=error_msg
                )

    async def fetch_funding_rates_batch(
        self,
        symbols: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        批量异步获取资金费率

        Args:
            symbols: 交易对列表

        Returns:
            Dict[symbol, funding_rate]
        """
        self.stats = {
            'total': len(symbols),
            'success': 0,
            'failed': 0,
            'retries': 0
        }

        if self.verbose:
            logger.info(f"开始批量获取资金费率: {len(symbols)} 个币种")
            logger.info(f"并发数: {self.concurrency}, 请求间隔: {self.delay_per_request}s")

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.concurrency)

        # 创建所有任务
        tasks = [
            self.fetch_funding_rate_async(symbol, semaphore)
            for symbol in symbols
        ]

        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 汇总结果
        funding_rates = {}
        for result in results:
            if isinstance(result, FundingRateResult):
                if result.success:
                    self.stats['success'] += 1
                    funding_rates[result.symbol] = result.funding_rate
                else:
                    self.stats['failed'] += 1
                    funding_rates[result.symbol] = None
            else:
                # 异常情况
                self.stats['failed'] += 1

        if self.verbose:
            logger.info(
                f"资金费率获取完成: "
                f"成功 {self.stats['success']}/{self.stats['total']}, "
                f"失败 {self.stats['failed']}, "
                f"重试 {self.stats['retries']} 次"
            )

        return funding_rates

    def fetch_funding_rates_sync(
        self,
        symbols: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        同步包装器（用于替代原有同步代码）

        Args:
            symbols: 交易对列表

        Returns:
            Dict[symbol, funding_rate]
        """
        return asyncio.run(self.fetch_funding_rates_batch(symbols))


def fetch_funding_rates_optimized(
    fetcher: BinanceDataFetcher,
    symbols: List[str],
    concurrency: int = 20,
    delay_per_request: float = 0.05
) -> Dict[str, Optional[float]]:
    """
    优化的批量资金费率获取函数

    可直接替代daily_candidate_scan中的循环调用

    Args:
        fetcher: BinanceDataFetcher实例
        symbols: 交易对列表
        concurrency: 并发数
        delay_per_request: 请求间隔

    Returns:
        Dict[symbol, funding_rate]

    Example:
        # 旧代码（会触发429）:
        for symbol in symbols:
            funding = fetch_funding_rate(fetcher, symbol)

        # 新代码（避免429）:
        funding_rates = fetch_funding_rates_optimized(fetcher, symbols)
        for symbol in symbols:
            funding = funding_rates.get(symbol)
    """
    async_fetcher = AsyncDataFetcher(
        fetcher=fetcher,
        concurrency=concurrency,
        delay_per_request=delay_per_request,
        verbose=True
    )

    return async_fetcher.fetch_funding_rates_sync(symbols)


# ===== 使用示例 =====

if __name__ == "__main__":
    """测试脚本"""
    import sys
    from pathlib import Path

    # 添加项目路径
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from data_fetcher import BinanceDataFetcher

    # 测试币种列表
    test_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
    ]

    print("=" * 60)
    print("测试异步资金费率获取")
    print("=" * 60)

    fetcher = BinanceDataFetcher()

    # 方式1: 使用AsyncDataFetcher类
    print("\n方式1: AsyncDataFetcher")
    async_fetcher = AsyncDataFetcher(fetcher, concurrency=5, delay_per_request=0.1)
    rates1 = async_fetcher.fetch_funding_rates_sync(test_symbols)

    for symbol, rate in rates1.items():
        if rate is not None:
            print(f"  {symbol}: {rate*100:.4f}%")
        else:
            print(f"  {symbol}: N/A")

    # 方式2: 使用便捷函数
    print("\n方式2: fetch_funding_rates_optimized")
    rates2 = fetch_funding_rates_optimized(fetcher, test_symbols, concurrency=5)

    for symbol, rate in rates2.items():
        if rate is not None:
            print(f"  {symbol}: {rate*100:.4f}%")
        else:
            print(f"  {symbol}: N/A")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
