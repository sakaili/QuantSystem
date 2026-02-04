
"""
Module : Daily scanner that identifies the weakest Binance USDT perpetuals.

筛选步骤（必须按顺序执行）：
1. 过滤主流币
2. 流动性过滤：24h 成交额倒数 bottom-N
3. 趋势条件：EMA5 < EMA10 < EMA20 < EMA30
4. 风控：
   - funding rate >= -1%
   - ATR14 最近未暴涨
5. 输出结果按「价格相对 EMA30 的偏离度」降序排列
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

# ensure project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data_fetcher import BinanceDataFetcher, SymbolMetadata
from async_data_fetcher import fetch_funding_rates_optimized

logger = logging.getLogger("daily_scan")

# ================= 配置 =================

MAJOR_BASES = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TON",
    "TRX", "LINK", "DOT", "AVAX", "ATOM", "MATIC", "LTC",
    "SHIB", "UNI", "XLM", "ETC",
}

FUNDING_RATE_FLOOR = 0.0  # 只做资金费率>0的币种(多头付费给空头)
ATR_SPIKE_LOOKBACK = 3
ATR_SPIKE_MULTIPLIER = 3.0
BOTTOM_N = 50
OUTPUT_DIR = Path("data/daily_scans")

# ================= 数据结构 =================

@dataclass(frozen=True)
class Candidate:
    symbol: str
    base: str
    timestamp: pd.Timestamp
    quote_volume: float
    market_cap: Optional[float]
    funding_rate: Optional[float]
    ema5: float
    ema10: float
    ema20: float
    ema30: float
    atr14: float
    latest_close: float
    price_deviation: float   # (EMA30 - close) / EMA30


# ================= CLI =================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Binance USDT perpetuals for short candidates."
    )
    parser.add_argument("--as-of", type=str,
                        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--bottom-n", type=int, default=BOTTOM_N)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--timeframe", type=str, default="1d")
    parser.add_argument("--cooldown", type=float, default=0.2)
    return parser.parse_args()


# ================= 核心逻辑 =================

def filter_out_majors(
    tickers: pd.DataFrame,
    meta_map: Dict[str, SymbolMetadata]
) -> pd.DataFrame:
    tickers = tickers.copy()
    tickers["base"] = [
        meta_map[s].base if s in meta_map else s.split("/")[0]
        for s in tickers["symbol"]
    ]
    return tickers[~tickers["base"].isin(MAJOR_BASES)].copy()


def pick_air_coin_pool(
    tickers: pd.DataFrame,
    *,
    bottom_n: int
) -> Set[str]:
    symbols = (
        tickers
        .sort_values("quote_volume", ascending=True)
        .head(bottom_n)["symbol"]
    )
    logger.info("空气币池规模：%d (bottom-%d)", len(symbols), bottom_n)
    return set(symbols)


def ema_cross_filter(history: pd.DataFrame) -> bool:
    last = history.iloc[-1]
    ema5, ema10, ema20, ema30 = (
        last["ema5"], last["ema10"], last["ema20"], last["ema30"]
    )
    if any(pd.isna(x) for x in (ema5, ema10, ema20, ema30)):
        return False
    return ema5 < ema10 < ema20 < ema30


def atr_spike_filter(history: pd.DataFrame) -> bool:
    if len(history) < ATR_SPIKE_LOOKBACK + 1:
        return True
    atr = history["atr14"].dropna()
    if len(atr) < ATR_SPIKE_LOOKBACK + 1:
        return True
    recent = atr.iloc[-1]
    prev_mean = atr.iloc[-(ATR_SPIKE_LOOKBACK + 1):-1].mean()
    return prev_mean <= 0 or recent <= ATR_SPIKE_MULTIPLIER * prev_mean


def fetch_funding_rate(
    fetcher: BinanceDataFetcher,
    symbol: str
) -> Optional[float]:
    try:
        fr = fetcher.exchange.fetch_funding_rate(symbol)
        return fr.get("fundingRate")
    except Exception as exc:
        logger.warning("Funding rate failed %s: %s", symbol, exc)
        return None


def build_candidates(
    fetcher: BinanceDataFetcher,
    symbols: Iterable[str],
    meta_map: Dict[str, SymbolMetadata],
    *,
    timeframe: str,
    funding_cooldown: float,
    as_of_date: date,
) -> List[Candidate]:

    end_dt = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=200)

    histories = fetcher.fetch_bulk_history(
        symbols, start=start_dt, end=end_dt, timeframe=timeframe
    )

    # ⭐ 优化：批量异步获取所有币种的资金费率（避免速率限制）
    logger.info(f"批量获取 {len(symbols)} 个币种的资金费率...")
    funding_rates = fetch_funding_rates_optimized(
        fetcher,
        list(symbols),
        concurrency=10,  # 并发数控制在20
        delay_per_request=0.5  # 每个请求间隔50ms
    )
    logger.info(f"资金费率获取完成: {len([r for r in funding_rates.values() if r is not None])}/{len(symbols)} 成功")

    rows: List[Candidate] = []

    for symbol, history in histories.items():
        if history.empty:
            continue

        history = history[history["timestamp"].dt.date <= as_of_date]
        if history.empty:
            continue

        history = ensure_ema5(history)
        if not ema_cross_filter(history):
            continue
        if not atr_spike_filter(history):
            continue

        # ⭐ 从批量结果中获取资金费率（而不是单独调用API）
        funding = funding_rates.get(symbol)
        if funding is not None and funding < FUNDING_RATE_FLOOR:
            continue

        # 🔧 NEW: 指数成分必须包含Binance
        if not fetcher.index_has_binance_component(symbol):
            logger.info("Skip %s: index has no Binance component", symbol)
            continue

        last = history.iloc[-1]
        ema30 = float(last["ema30"])
        close = float(last["close"])

        deviation = (ema30 - close) / ema30 if ema30 > 0 else 0.0

        rows.append(
            Candidate(
                symbol=symbol,
                base=meta_map[symbol].base,
                timestamp=pd.Timestamp(as_of_date, tz=timezone.utc),
                quote_volume=float("nan"),
                market_cap=None,
                funding_rate=funding,
                ema5=float(last["ema5"]),
                ema10=float(last["ema10"]),
                ema20=float(last["ema20"]),
                ema30=ema30,
                atr14=float(last["atr14"]),
                latest_close=close,
                price_deviation=deviation,
            )
        )

    return rows


def run_scan(
    *,
    as_of: date,
    bottom_n: int,
    timeframe: str,
    funding_cooldown: float,
    fetcher: Optional[BinanceDataFetcher] = None,
) -> pd.DataFrame:

    fetcher = fetcher or BinanceDataFetcher()
    metas = fetcher.fetch_usdt_perp_symbols()
    meta_map = {m.symbol: m for m in metas}

    symbols = list(meta_map.keys())
    tickers = fetcher.fetch_24h_tickers(symbols)
    tickers = filter_out_majors(tickers, meta_map)

    air_pool = pick_air_coin_pool(tickers, bottom_n=bottom_n)
    if not air_pool:
        raise RuntimeError("空气币池为空")

    candidates = build_candidates(
        fetcher,
        air_pool,
        meta_map,
        timeframe=timeframe,
        funding_cooldown=funding_cooldown,
        as_of_date=as_of,
    )

    df = pd.DataFrame([c.__dict__ for c in candidates])
    if df.empty:
        return df

    df = df.merge(
        tickers[["symbol", "quote_volume", "market_cap"]],
        on="symbol",
        how="left",
    )

    df["as_of"] = as_of.strftime("%Y-%m-%d")

    # ⭐ 核心排序：按价格偏离度
    df.sort_values("price_deviation", ascending=False, inplace=True)

    return df


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    df = run_scan(
        as_of=as_of,
        bottom_n=args.bottom_n,
        timeframe=args.timeframe,
        funding_cooldown=args.cooldown,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"candidates_{as_of:%Y%m%d}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} candidates -> {out}")

def ensure_ema5(history: pd.DataFrame) -> pd.DataFrame:
    """
    如果 history 中没有 ema5，则基于 close 计算。
    不修改原 DataFrame，返回 copy。
    """
    if "ema5" in history.columns:
        return history

    hist = history.copy()
    hist["ema5"] = hist["close"].ewm(span=5, adjust=False).mean()
    return hist

MIN_LISTING_DAYS = 365


def is_listed_long_enough(
    history: pd.DataFrame,
    *,
    as_of_date: date,
    min_days: int = MIN_LISTING_DAYS,
) -> bool:
    """
    判断该合约是否已上线足够久（基于最早K线）。
    """
    if history.empty:
        return False

    first_ts = history["timestamp"].min()
    if pd.isna(first_ts):
        return False

    listed_days = (as_of_date - first_ts.date()).days
    return listed_days >= min_days


if __name__ == "__main__":
    main()
