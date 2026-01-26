"""
快速测试脚本 - 验证Binance API连接和时间同步
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from QuantSystem.data_fetcher import BinanceDataFetcher
from QuantSystem.utils.logger import setup_logger

# 设置日志
logger = setup_logger(name="Test", log_level="INFO", console_output=True)

def test_connection():
    """测试Binance连接"""
    try:
        logger.info("=" * 60)
        logger.info("测试Binance API连接...")
        logger.info("=" * 60)

        # 创建数据获取器
        fetcher = BinanceDataFetcher(use_testnet=False)

        logger.info(f"✓ 交易所连接成功")
        logger.info(f"✓ 时间差: {fetcher.exchange.options.get('timeDifference', 0)}ms")

        # 测试获取市场列表
        logger.info("\n测试获取市场列表...")
        metas = fetcher.fetch_usdt_perp_symbols()
        logger.info(f"✓ 获取到 {len(metas)} 个USDT永续合约")

        # 显示前5个
        logger.info("\n前5个合约:")
        for i, meta in enumerate(metas[:5], 1):
            logger.info(f"  {i}. {meta.symbol} - {meta.base}/USDT")

        # 测试获取ticker
        logger.info("\n测试获取行情数据...")
        test_symbol = metas[0].symbol
        tickers = fetcher.fetch_24h_tickers([test_symbol])
        logger.info(f"✓ 获取到 {test_symbol} 的行情数据")
        logger.info(f"  价格: {tickers['last'].iloc[0]}")
        logger.info(f"  成交额: {tickers['quote_volume'].iloc[0]:,.0f} USDT")

        logger.info("\n" + "=" * 60)
        logger.info("✅ 所有测试通过! 系统可以正常运行")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
