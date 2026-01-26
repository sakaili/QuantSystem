"""
ETH Correlation Analysis
分析ETH与垃圾币的波动关系

目标：
1. 获取候选垃圾币列表
2. 获取ETH和这些币种的历史数据
3. 计算波动率和相关性
4. 确定Beta系数（垃圾币波动 / ETH波动）
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ensure project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_fetcher import BinanceDataFetcher
from daily_candidate_scan import run_scan

logger = logging.getLogger("eth_correlation")

# ================= 配置 =================

ANALYSIS_DAYS = 1095  # 分析最近3年的数据 (365 * 3)
OUTPUT_DIR = Path("QuantSystem/analysis_results")

# ================= 数据获取 =================

def get_candidate_coins(as_of: date, bottom_n: int = 50, fetcher: Optional[BinanceDataFetcher] = None) -> List[str]:
    """
    使用daily_candidate_scan获取候选币种列表
    """
    logger.info("获取候选币种列表...")

    df = run_scan(
        as_of=as_of,
        bottom_n=bottom_n,
        timeframe="1d",
        funding_cooldown=0.2,
        fetcher=fetcher,
    )

    if df.empty:
        logger.warning("未找到候选币种")
        return []

    symbols = df["symbol"].tolist()
    logger.info(f"找到 {len(symbols)} 个候选币种")

    return symbols


def fetch_price_data(
    fetcher: BinanceDataFetcher,
    symbols: List[str],
    start_date: date,
    end_date: date,
    timeframe: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    获取多个币种的价格数据
    """
    logger.info(f"获取 {len(symbols)} 个币种的价格数据...")

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

    histories = fetcher.fetch_bulk_history(
        symbols,
        start=start_dt,
        end=end_dt,
        timeframe=timeframe
    )

    # 过滤掉空数据
    valid_histories = {
        symbol: df for symbol, df in histories.items()
        if not df.empty and len(df) >= 30  # 至少30天数据
    }

    logger.info(f"成功获取 {len(valid_histories)} 个币种的有效数据")

    return valid_histories


# ================= 波动率分析 =================

def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    计算日收益率
    """
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    returns = df["close"].pct_change()
    return returns.dropna()


def calculate_volatility_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    计算波动率指标
    """
    if len(returns) < 2:
        return {
            "mean_return": np.nan,
            "std_return": np.nan,
            "daily_volatility": np.nan,
            "abs_mean_return": np.nan,
        }

    return {
        "mean_return": returns.mean(),
        "std_return": returns.std(),
        "daily_volatility": returns.std(),  # 日波动率
        "abs_mean_return": returns.abs().mean(),  # 平均绝对收益率
    }


# ================= 相关性分析 =================

def calculate_correlation_and_beta(
    eth_returns: pd.Series,
    coin_returns: pd.Series
) -> Dict[str, float]:
    """
    计算相关性和Beta系数

    Beta = Cov(coin, ETH) / Var(ETH)
    """
    # 对齐时间序列
    aligned = pd.DataFrame({
        "eth": eth_returns,
        "coin": coin_returns
    }).dropna()

    if len(aligned) < 10:  # 至少10个数据点
        return {
            "correlation": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "data_points": len(aligned),
        }

    eth = aligned["eth"]
    coin = aligned["coin"]

    # 相关系数
    correlation = eth.corr(coin)

    # Beta系数
    covariance = np.cov(coin, eth)[0, 1]
    eth_variance = np.var(eth)
    beta = covariance / eth_variance if eth_variance > 0 else np.nan

    # R²
    r_squared = correlation ** 2 if not np.isnan(correlation) else np.nan

    return {
        "correlation": correlation,
        "beta": beta,
        "r_squared": r_squared,
        "data_points": len(aligned),
    }


# ================= 主分析流程 =================

def analyze_eth_correlation(
    as_of: date,
    analysis_days: int = ANALYSIS_DAYS,
    bottom_n: int = 50
) -> pd.DataFrame:
    """
    主分析函数：分析ETH与垃圾币的相关性
    """
    # 配置代理
    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890"
    }
    fetcher = BinanceDataFetcher(proxies=proxies)

    # 1. 获取候选币种
    candidate_symbols = get_candidate_coins(as_of, bottom_n, fetcher)
    if not candidate_symbols:
        logger.error("没有候选币种，退出分析")
        return pd.DataFrame()

    # 2. 添加ETH到列表
    all_symbols = ["ETH/USDT:USDT"] + candidate_symbols

    # 3. 获取历史数据
    start_date = as_of - timedelta(days=analysis_days)
    price_data = fetch_price_data(
        fetcher,
        all_symbols,
        start_date,
        as_of,
        timeframe="1d"
    )

    # 4. 检查ETH数据
    eth_symbol = "ETH/USDT:USDT"
    if eth_symbol not in price_data:
        logger.error("无法获取ETH数据")
        return pd.DataFrame()

    # 5. 计算ETH的收益率和波动率
    eth_returns = calculate_daily_returns(price_data[eth_symbol])
    eth_metrics = calculate_volatility_metrics(eth_returns)

    logger.info(f"ETH波动率指标:")
    logger.info(f"  日均波动率: {eth_metrics['daily_volatility']:.4f} ({eth_metrics['daily_volatility']*100:.2f}%)")
    logger.info(f"  平均绝对收益: {eth_metrics['abs_mean_return']:.4f} ({eth_metrics['abs_mean_return']*100:.2f}%)")

    # 6. 分析每个币种
    results = []

    for symbol in candidate_symbols:
        if symbol not in price_data:
            continue

        # 计算收益率
        coin_returns = calculate_daily_returns(price_data[symbol])

        # 计算波动率指标
        coin_metrics = calculate_volatility_metrics(coin_returns)

        # 计算相关性和Beta
        corr_metrics = calculate_correlation_and_beta(eth_returns, coin_returns)

        # 计算波动率放大倍数
        volatility_multiplier = (
            coin_metrics["daily_volatility"] / eth_metrics["daily_volatility"]
            if eth_metrics["daily_volatility"] > 0 else np.nan
        )

        results.append({
            "symbol": symbol,
            "base": symbol.split("/")[0],

            # 币种波动率
            "coin_daily_volatility": coin_metrics["daily_volatility"],
            "coin_abs_mean_return": coin_metrics["abs_mean_return"],

            # ETH相关性
            "correlation": corr_metrics["correlation"],
            "beta": corr_metrics["beta"],
            "r_squared": corr_metrics["r_squared"],

            # 波动率放大倍数
            "volatility_multiplier": volatility_multiplier,

            # 数据质量
            "data_points": corr_metrics["data_points"],
        })

    # 7. 转换为DataFrame并排序
    df = pd.DataFrame(results)

    if df.empty:
        return df

    # 按Beta系数降序排序
    df = df.sort_values("beta", ascending=False)

    return df


# ================= 结果输出 =================

def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    打印汇总统计
    """
    if df.empty:
        print("没有数据")
        return

    print("\n" + "="*80)
    print("ETH与垃圾币相关性分析 - 汇总统计")
    print("="*80)

    # 过滤有效数据
    valid_df = df[df["beta"].notna() & df["correlation"].notna()]

    if valid_df.empty:
        print("没有有效的相关性数据")
        return

    print(f"\n有效币种数量: {len(valid_df)}")

    print(f"\n相关系数统计:")
    print(f"  平均值: {valid_df['correlation'].mean():.3f}")
    print(f"  中位数: {valid_df['correlation'].median():.3f}")
    print(f"  标准差: {valid_df['correlation'].std():.3f}")
    print(f"  范围: [{valid_df['correlation'].min():.3f}, {valid_df['correlation'].max():.3f}]")

    print(f"\nBeta系数统计:")
    print(f"  平均值: {valid_df['beta'].mean():.3f}")
    print(f"  中位数: {valid_df['beta'].median():.3f}")
    print(f"  标准差: {valid_df['beta'].std():.3f}")
    print(f"  范围: [{valid_df['beta'].min():.3f}, {valid_df['beta'].max():.3f}]")

    print(f"\n波动率放大倍数统计:")
    print(f"  平均值: {valid_df['volatility_multiplier'].mean():.3f}x")
    print(f"  中位数: {valid_df['volatility_multiplier'].median():.3f}x")
    print(f"  标准差: {valid_df['volatility_multiplier'].std():.3f}")
    print(f"  范围: [{valid_df['volatility_multiplier'].min():.3f}x, {valid_df['volatility_multiplier'].max():.3f}x]")

    # 分类统计
    high_beta = valid_df[valid_df["beta"] >= 2.0]
    medium_beta = valid_df[(valid_df["beta"] >= 1.0) & (valid_df["beta"] < 2.0)]
    low_beta = valid_df[valid_df["beta"] < 1.0]

    print(f"\nBeta系数分布:")
    print(f"  高Beta (≥2.0): {len(high_beta)} 个币种 ({len(high_beta)/len(valid_df)*100:.1f}%)")
    print(f"  中Beta (1.0-2.0): {len(medium_beta)} 个币种 ({len(medium_beta)/len(valid_df)*100:.1f}%)")
    print(f"  低Beta (<1.0): {len(low_beta)} 个币种 ({len(low_beta)/len(valid_df)*100:.1f}%)")

    # Top 10 高Beta币种
    print(f"\nTop 10 高Beta币种:")
    print("-" * 80)
    top10 = valid_df.head(10)
    for idx, row in top10.iterrows():
        print(f"{row['base']:8s}  Beta: {row['beta']:6.3f}  "
              f"Corr: {row['correlation']:6.3f}  "
              f"Vol Mult: {row['volatility_multiplier']:6.3f}x")

    print("="*80 + "\n")


def save_results(df: pd.DataFrame, as_of: date) -> None:
    """
    保存分析结果
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    output_file = OUTPUT_DIR / f"eth_correlation_{as_of:%Y%m%d}.csv"
    df.to_csv(output_file, index=False, float_format="%.6f")
    print(f"完整结果已保存到: {output_file}")

    # 保存汇总统计
    if not df.empty:
        valid_df = df[df["beta"].notna()]
        if not valid_df.empty:
            summary_file = OUTPUT_DIR / f"eth_correlation_summary_{as_of:%Y%m%d}.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"ETH与垃圾币相关性分析 - {as_of}\n")
                f.write("="*80 + "\n\n")
                f.write(f"有效币种数量: {len(valid_df)}\n\n")
                f.write(f"Beta系数统计:\n")
                f.write(f"  平均值: {valid_df['beta'].mean():.3f}\n")
                f.write(f"  中位数: {valid_df['beta'].median():.3f}\n")
                f.write(f"  标准差: {valid_df['beta'].std():.3f}\n\n")
                f.write(f"波动率放大倍数统计:\n")
                f.write(f"  平均值: {valid_df['volatility_multiplier'].mean():.3f}x\n")
                f.write(f"  中位数: {valid_df['volatility_multiplier'].median():.3f}x\n")
            print(f"汇总统计已保存到: {summary_file}")


# ================= 主函数 =================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 使用今天作为分析日期
    as_of = date.today()

    logger.info(f"开始ETH相关性分析 (截至日期: {as_of})")

    # 执行分析
    df = analyze_eth_correlation(
        as_of=as_of,
        analysis_days=ANALYSIS_DAYS,
        bottom_n=50
    )

    if df.empty:
        logger.error("分析失败，没有结果")
        return

    # 打印汇总统计
    print_summary_statistics(df)

    # 保存结果
    save_results(df, as_of)

    logger.info("分析完成")


if __name__ == "__main__":
    main()
