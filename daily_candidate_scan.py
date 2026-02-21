
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
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ensure project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

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
SQUEEZE_LEN = 20
DEFAULT_SQUEEZE_TF = "4h"
WEIGHTS = {
    "squeeze_momentum": 0.35,
    "squeeze_momentum_delta": 0.20,
    "funding_rate": 0.25,
    "price_deviation": 0.20,
}

# ================= SQUEEZE =================

def _linreg_last(values: np.ndarray) -> Optional[float]:
    n = len(values)
    if n == 0:
        return None
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_y = values.sum()
    sum_xy = (x * values).sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return None
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return float(intercept + slope * (n - 1))


def compute_squeeze_momentum(
    history: pd.DataFrame,
    *,
    length: int = SQUEEZE_LEN,
) -> Optional[float]:
    if history is None or history.empty:
        return None

    min_bars = length * 2 - 1
    if len(history) < min_bars:
        return None

    rolling_high = history["high"].rolling(window=length).max()
    rolling_low = history["low"].rolling(window=length).min()
    rolling_sma = history["close"].rolling(window=length).mean()
    avg_hl = (rolling_high + rolling_low) / 2.0
    avg_hl_sma = (avg_hl + rolling_sma) / 2.0
    series = history["close"] - avg_hl_sma
    series = series.dropna()

    if len(series) < length:
        return None

    window = series.iloc[-length:].to_numpy(dtype=float, copy=False)
    return _linreg_last(window)


def compute_squeeze_series(
    history: pd.DataFrame,
    *,
    length: int = SQUEEZE_LEN,
) -> Optional[pd.Series]:
    if history is None or history.empty:
        return None

    min_bars = length * 2 - 1
    if len(history) < min_bars:
        return None

    rolling_high = history["high"].rolling(window=length).max()
    rolling_low = history["low"].rolling(window=length).min()
    rolling_sma = history["close"].rolling(window=length).mean()
    avg_hl = (rolling_high + rolling_low) / 2.0
    avg_hl_sma = (avg_hl + rolling_sma) / 2.0
    series = history["close"] - avg_hl_sma
    series = series.dropna()

    if len(series) < length:
        return None

    return series.rolling(window=length).apply(_linreg_last, raw=True)


def rank_score(series: pd.Series, *, higher_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ascending = not higher_better
    return values.rank(pct=True, ascending=ascending, na_option="bottom")


def apply_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["weighted_score"] = pd.Series(dtype=float)
        return df

    funding_series = df.get("funding_rate_sum")
    if funding_series is None or funding_series.notna().sum() == 0:
        funding_series = df["funding_rate"]

    score_squeeze = rank_score(-df["squeeze_momentum"])
    score_squeeze_delta = rank_score(-df["squeeze_momentum_delta"])
    score_funding = rank_score(funding_series)
    score_deviation = rank_score(df["price_deviation"])

    df["weighted_score"] = (
        WEIGHTS["squeeze_momentum"] * score_squeeze
        + WEIGHTS["squeeze_momentum_delta"] * score_squeeze_delta
        + WEIGHTS["funding_rate"] * score_funding
        + WEIGHTS["price_deviation"] * score_deviation
    )
    return df

# ================= 数据结构 =================

@dataclass(frozen=True)
class Candidate:
    symbol: str
    base: str
    timestamp: pd.Timestamp
    quote_volume: float
    market_cap: Optional[float]
    funding_rate: Optional[float]
    funding_rate_sum: Optional[float]
    ema5: float
    ema10: float
    ema20: float
    ema30: float
    atr14: float
    squeeze_momentum: Optional[float]
    squeeze_momentum_delta: Optional[float]
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
    parser.add_argument("--funding-rate-floor", type=float, default=FUNDING_RATE_FLOOR)
    parser.add_argument("--atr-spike-multiplier", type=float, default=ATR_SPIKE_MULTIPLIER)
    parser.add_argument("--funding-rate-sort", action="store_true")
    parser.add_argument("--funding-rate-lookback-days", type=int, default=365)
    parser.add_argument("--funding-rate-min-sum", type=float, default=0.0)
    parser.add_argument("--eth-deviation-filter", action="store_true")
    parser.add_argument("--eth-deviation-window", type=int, default=60)
    parser.add_argument("--eth-deviation-cooldown-days", type=int, default=30)
    parser.add_argument("--eth-deviation-rate-window-days", type=int, default=180)
    parser.add_argument("--eth-deviation-ever", action="store_true")
    parser.add_argument("--eth-corr-drop-threshold", type=float, default=0.2)
    parser.add_argument("--eth-corr-drop-rate-limit", type=float, default=0.05)
    parser.add_argument("--eth-residual-z", type=float, default=2.5)
    parser.add_argument("--eth-residual-rate-limit", type=float, default=0.01)
    parser.add_argument("--binance-component-max-weight", type=float, default=0.8)
    parser.add_argument("--binance-component-weight-strict", action="store_true")
    parser.add_argument("--air-mean-deviation-filter", action="store_true")
    parser.set_defaults(air_mean_use_median=True)
    parser.add_argument("--air-mean-use-median", dest="air_mean_use_median", action="store_true")
    parser.add_argument("--air-mean-use-mean", dest="air_mean_use_median", action="store_false")
    parser.add_argument("--air-mean-deviation-window", type=int, default=60)
    parser.add_argument("--air-mean-deviation-cooldown-days", type=int, default=365)
    parser.add_argument("--air-mean-deviation-rate-window-days", type=int, default=180)
    parser.add_argument("--air-mean-deviation-ever", action="store_true")
    parser.add_argument("--air-mean-corr-drop-threshold", type=float, default=0.2)
    parser.add_argument("--air-mean-corr-drop-rate-limit", type=float, default=0.05)
    parser.add_argument("--air-mean-residual-z", type=float, default=2.5)
    parser.add_argument("--air-mean-residual-rate-limit", type=float, default=0.01)
    parser.add_argument(
        "--use-squeeze-filter",
        action="store_true",
        help="Enable 4H squeeze momentum < 0 filter (default off)."
    )
    parser.add_argument(
        "--squeeze-timeframe",
        type=str,
        default=DEFAULT_SQUEEZE_TF,
        help="Timeframe for squeeze momentum (default: 4h)."
    )
    parser.add_argument(
        "--plot-squeeze",
        action="store_true",
        help="Plot last-year price + squeeze momentum for top candidates."
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=10,
        help="Number of top candidates to plot."
    )
    parser.add_argument(
        "--select-top",
        type=int,
        default=3,
        help="Print top-N symbols for opening positions."
    )
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


def atr_spike_filter(history: pd.DataFrame, *, multiplier: float = ATR_SPIKE_MULTIPLIER) -> bool:
    if len(history) < ATR_SPIKE_LOOKBACK + 1:
        return True
    atr = history["atr14"].dropna()
    if len(atr) < ATR_SPIKE_LOOKBACK + 1:
        return True
    recent = atr.iloc[-1]
    prev_mean = atr.iloc[-(ATR_SPIKE_LOOKBACK + 1):-1].mean()
    return prev_mean <= 0 or recent <= multiplier * prev_mean


def fetch_funding_rate_sums(
    fetcher: BinanceDataFetcher,
    symbols: Iterable[str],
    *,
    as_of_date: date,
    lookback_days: int,
    max_pages: int = 50,
) -> Dict[str, float]:
    """
    Fetch cumulative funding rate sums for symbols.

    lookback_days=0 means as far back as possible.
    """
    results: Dict[str, float] = {}
    if lookback_days <= 0:
        start_dt = datetime(2019, 1, 1, tzinfo=timezone.utc)
    else:
        start_dt = datetime.combine(
            as_of_date - timedelta(days=lookback_days),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
    end_dt = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)

    for symbol in symbols:
        try:
            hist = fetcher.fetch_funding_rate_history(
                symbol,
                start=start_dt,
                end=end_dt,
                max_pages=max_pages,
            )
            if hist.empty:
                results[symbol] = float("nan")
            else:
                results[symbol] = float(hist["funding_rate"].sum())
        except Exception as exc:
            logger.warning("Funding history failed %s: %s", symbol, exc)
            results[symbol] = float("nan")

    return results


def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    return df["close"].pct_change().dropna()


def build_returns_frame(
    histories: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
) -> pd.DataFrame:
    series_list = []
    for symbol in symbols:
        df = histories.get(symbol)
        if df is None or df.empty:
            continue
        s = calculate_daily_returns(df)
        s = s.rename(symbol)
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    frame = pd.concat(series_list, axis=1).dropna(how="all")
    return frame


def residual_events(
    eth: pd.Series,
    coin: pd.Series,
    *,
    window: int,
    z_threshold: float,
) -> pd.Series:
    aligned = pd.DataFrame({"eth": eth, "coin": coin}).dropna()
    if len(aligned) < window + 5:
        return pd.Series(dtype=bool)
    eth_s = aligned["eth"]
    coin_s = aligned["coin"]
    mean_eth = eth_s.rolling(window).mean().shift(1)
    mean_coin = coin_s.rolling(window).mean().shift(1)
    cov = coin_s.rolling(window).cov(eth_s).shift(1)
    var = eth_s.rolling(window).var().shift(1)
    beta = cov / var.replace(0, np.nan)
    alpha = mean_coin - beta * mean_eth
    pred = alpha + beta * eth_s
    residual = coin_s - pred
    resid_std = residual.rolling(window).std().shift(1)
    z = residual / resid_std.replace(0, np.nan)
    events = z.abs() >= z_threshold
    return events.reindex(aligned.index)


def corr_drop_events(
    eth: pd.Series,
    coin: pd.Series,
    *,
    window: int,
    corr_threshold: float,
) -> pd.Series:
    aligned = pd.DataFrame({"eth": eth, "coin": coin}).dropna()
    if len(aligned) < window + 5:
        return pd.Series(dtype=bool)
    corr = aligned["coin"].rolling(window).corr(aligned["eth"]).shift(1)
    events = corr <= corr_threshold
    return events.reindex(aligned.index)


def _event_rate_in_window(events: pd.Series, *, as_of: date, window_days: int) -> float:
    if events.empty:
        return np.nan
    idx = pd.to_datetime(events.index, utc=True, errors="coerce")
    if idx.isna().all():
        return np.nan
    events = events.copy()
    events.index = idx
    cutoff = pd.Timestamp(as_of - timedelta(days=window_days), tz=timezone.utc)
    sub = events[events.index >= cutoff]
    if sub.empty:
        return np.nan
    return float(sub.mean())


def _has_event_in_last(events: pd.Series, *, as_of: date, days: int) -> bool:
    if events.empty:
        return False
    idx = pd.to_datetime(events.index, utc=True, errors="coerce")
    if idx.isna().all():
        return False
    events = events.copy()
    events.index = idx
    cutoff = pd.Timestamp(as_of - timedelta(days=days), tz=timezone.utc)
    sub = events[events.index >= cutoff]
    if sub.empty:
        return False
    return bool(sub.any())


def apply_eth_deviation_filter(
    histories: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    *,
    as_of: date,
    window: int,
    cooldown_days: int,
    rate_window_days: int,
    deviation_ever: bool,
    corr_threshold: float,
    corr_rate_limit: float,
    residual_z: float,
    residual_rate_limit: float,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Return (kept_symbols, diagnostics_map).
    """
    eth_symbol = "ETH/USDT:USDT"
    if eth_symbol not in histories or histories[eth_symbol].empty:
        logger.warning("ETH history missing; skip ETH deviation filter.")
        return list(symbols), {}

    eth_ret = calculate_daily_returns(histories[eth_symbol])
    diagnostics: Dict[str, Dict[str, float]] = {}
    kept: List[str] = []

    for symbol in symbols:
        hist = histories.get(symbol)
        if hist is None or hist.empty:
            continue
        coin_ret = calculate_daily_returns(hist)

        events_corr = corr_drop_events(
            eth_ret,
            coin_ret,
            window=window,
            corr_threshold=corr_threshold,
        )
        events_resid = residual_events(
            eth_ret,
            coin_ret,
            window=window,
            z_threshold=residual_z,
        )

        corr_recent = _has_event_in_last(events_corr, as_of=as_of, days=cooldown_days)
        resid_recent = _has_event_in_last(events_resid, as_of=as_of, days=cooldown_days)
        corr_rate = _event_rate_in_window(events_corr, as_of=as_of, window_days=rate_window_days)
        resid_rate = _event_rate_in_window(events_resid, as_of=as_of, window_days=rate_window_days)
        corr_any = bool(events_corr.any()) if not events_corr.empty else False
        resid_any = bool(events_resid.any()) if not events_resid.empty else False

        if deviation_ever:
            # "豁免"：只要最近N天没有出现，就放行
            kick_corr = corr_recent
            kick_resid = resid_recent
        else:
            kick_corr = corr_recent or (not np.isnan(corr_rate) and corr_rate > corr_rate_limit)
            kick_resid = resid_recent or (not np.isnan(resid_rate) and resid_rate > residual_rate_limit)

        diagnostics[symbol] = {
            "corr_recent": float(corr_recent),
            "resid_recent": float(resid_recent),
            "corr_rate": corr_rate,
            "resid_rate": resid_rate,
            "corr_any": float(corr_any),
            "resid_any": float(resid_any),
            "kick_corr": float(kick_corr),
            "kick_resid": float(kick_resid),
        }

        if not (kick_corr or kick_resid):
            kept.append(symbol)

    return kept, diagnostics


def apply_air_mean_deviation_filter(
    histories: Dict[str, pd.DataFrame],
    pool_symbols: Iterable[str],
    candidate_symbols: Iterable[str],
    *,
    as_of: date,
    window: int,
    cooldown_days: int,
    rate_window_days: int,
    deviation_ever: bool,
    corr_threshold: float,
    corr_rate_limit: float,
    residual_z: float,
    residual_rate_limit: float,
    use_median: bool,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Use air-coin mean/median returns as baseline for deviation detection.
    """
    pool_returns = build_returns_frame(histories, pool_symbols)
    if pool_returns.empty:
        logger.warning("Air mean returns missing; skip air-mean deviation filter.")
        return list(candidate_symbols), {}

    diagnostics: Dict[str, Dict[str, float]] = {}
    kept: List[str] = []

    for symbol in candidate_symbols:
        coin_returns = pool_returns.get(symbol)
        if coin_returns is None or coin_returns.empty:
            continue

        pool_cols = [c for c in pool_returns.columns if c != symbol]
        if not pool_cols:
            continue

        if use_median:
            baseline = pool_returns[pool_cols].median(axis=1)
        else:
            baseline = pool_returns[pool_cols].mean(axis=1)

        events_corr = corr_drop_events(
            baseline,
            coin_returns,
            window=window,
            corr_threshold=corr_threshold,
        )
        events_resid = residual_events(
            baseline,
            coin_returns,
            window=window,
            z_threshold=residual_z,
        )

        corr_recent = _has_event_in_last(events_corr, as_of=as_of, days=cooldown_days)
        resid_recent = _has_event_in_last(events_resid, as_of=as_of, days=cooldown_days)
        corr_rate = _event_rate_in_window(events_corr, as_of=as_of, window_days=rate_window_days)
        resid_rate = _event_rate_in_window(events_resid, as_of=as_of, window_days=rate_window_days)
        corr_any = bool(events_corr.any()) if not events_corr.empty else False
        resid_any = bool(events_resid.any()) if not events_resid.empty else False

        if deviation_ever:
            kick_corr = corr_recent
            kick_resid = resid_recent
        else:
            kick_corr = corr_recent or (not np.isnan(corr_rate) and corr_rate > corr_rate_limit)
            kick_resid = resid_recent or (not np.isnan(resid_rate) and resid_rate > residual_rate_limit)

        diagnostics[symbol] = {
            "air_corr_recent": float(corr_recent),
            "air_resid_recent": float(resid_recent),
            "air_corr_rate": corr_rate,
            "air_resid_rate": resid_rate,
            "air_corr_any": float(corr_any),
            "air_resid_any": float(resid_any),
            "air_kick_corr": float(kick_corr),
            "air_kick_resid": float(kick_resid),
        }

        if not (kick_corr or kick_resid):
            kept.append(symbol)

    return kept, diagnostics


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
    use_squeeze_filter: bool = False,
    squeeze_timeframe: str = DEFAULT_SQUEEZE_TF,
    funding_cooldown: float,
    as_of_date: date,
    funding_rate_floor: float = FUNDING_RATE_FLOOR,
    atr_spike_multiplier: float = ATR_SPIKE_MULTIPLIER,
    binance_component_max_weight: float = 0.8,
    binance_component_weight_strict: bool = True,
    histories: Optional[Dict[str, pd.DataFrame]] = None,
    squeeze_histories: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Candidate]:

    if histories is None:
        end_dt = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=200)
        histories = fetcher.fetch_bulk_history(
            symbols, start=start_dt, end=end_dt, timeframe=timeframe
        )

    if squeeze_histories is None:
        end_dt = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=200)
        squeeze_histories = fetcher.fetch_bulk_history(
            symbols, start=start_dt, end=end_dt, timeframe=squeeze_timeframe
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
        if not atr_spike_filter(history, multiplier=atr_spike_multiplier):
            continue

        # ⭐ 从批量结果中获取资金费率（而不是单独调用API）
        funding = funding_rates.get(symbol)
        if funding is not None and funding < funding_rate_floor:
            continue

        squeeze_val = None
        squeeze_delta = None
        sq_history = squeeze_histories.get(symbol) if squeeze_histories else None
        if sq_history is not None and not sq_history.empty:
            squeeze_val = compute_squeeze_momentum(sq_history, length=SQUEEZE_LEN)
            squeeze_series = compute_squeeze_series(sq_history, length=SQUEEZE_LEN)
            if squeeze_series is not None and len(squeeze_series) >= 2:
                squeeze_delta = float(squeeze_series.iloc[-1] - squeeze_series.iloc[-2])

        if use_squeeze_filter:
            if squeeze_val is None or squeeze_val >= 0:
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
                funding_rate_sum=None,
                ema5=float(last["ema5"]),
                ema10=float(last["ema10"]),
                ema20=float(last["ema20"]),
                ema30=ema30,
                atr14=float(last["atr14"]),
                squeeze_momentum=squeeze_val,
                squeeze_momentum_delta=squeeze_delta,
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
    use_squeeze_filter: bool = False,
    squeeze_timeframe: str = DEFAULT_SQUEEZE_TF,
    funding_cooldown: float,
    funding_rate_floor: float = FUNDING_RATE_FLOOR,
    atr_spike_multiplier: float = ATR_SPIKE_MULTIPLIER,
    funding_rate_sort: bool = False,
    funding_rate_lookback_days: int = 365,
    funding_rate_min_sum: float = 0.0,
    eth_deviation_filter: bool = False,
    eth_deviation_window: int = 60,
    eth_deviation_cooldown_days: int = 30,
    eth_deviation_rate_window_days: int = 180,
    eth_deviation_ever: bool = False,
    eth_corr_drop_threshold: float = 0.2,
    eth_corr_drop_rate_limit: float = 0.05,
    eth_residual_z: float = 2.5,
    eth_residual_rate_limit: float = 0.01,
    binance_component_max_weight: float = 0.8,
    binance_component_weight_strict: bool = True,
    air_mean_deviation_filter: bool = False,
    air_mean_use_median: bool = True,
    air_mean_deviation_window: int = 60,
    air_mean_deviation_cooldown_days: int = 365,
    air_mean_deviation_rate_window_days: int = 180,
    air_mean_deviation_ever: bool = False,
    air_mean_corr_drop_threshold: float = 0.2,
    air_mean_corr_drop_rate_limit: float = 0.05,
    air_mean_residual_z: float = 2.5,
    air_mean_residual_rate_limit: float = 0.01,
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

    history_lookback_days = 200
    if air_mean_deviation_filter:
        history_lookback_days = max(
            history_lookback_days,
            air_mean_deviation_window,
            air_mean_deviation_rate_window_days,
        ) + 5

    end_dt = datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=history_lookback_days)
    pool_histories = fetcher.fetch_bulk_history(
        air_pool,
        start=start_dt,
        end=end_dt,
        timeframe=timeframe,
    )
    squeeze_histories = fetcher.fetch_bulk_history(
        air_pool,
        start=start_dt,
        end=end_dt,
        timeframe=squeeze_timeframe,
    )

    candidates = build_candidates(
        fetcher,
        air_pool,
        meta_map,
        timeframe=timeframe,
        use_squeeze_filter=use_squeeze_filter,
        squeeze_timeframe=squeeze_timeframe,
        funding_cooldown=funding_cooldown,
        as_of_date=as_of,
        funding_rate_floor=funding_rate_floor,
        atr_spike_multiplier=atr_spike_multiplier,
        binance_component_max_weight=binance_component_max_weight,
        binance_component_weight_strict=binance_component_weight_strict,
        histories=pool_histories,
        squeeze_histories=squeeze_histories,
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

    if eth_deviation_filter:
        logger.info("应用ETH偏离筛选...")
        symbols = df["symbol"].tolist()
        lookback_days = max(eth_deviation_rate_window_days, eth_deviation_window) + 5
        histories = fetcher.fetch_bulk_history(
            ["ETH/USDT:USDT"] + symbols,
            start=datetime.combine(as_of - timedelta(days=lookback_days), datetime.min.time(), tzinfo=timezone.utc),
            end=datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc),
            timeframe="1d",
        )
        kept, diagnostics = apply_eth_deviation_filter(
            histories,
            symbols,
            as_of=as_of,
            window=eth_deviation_window,
            cooldown_days=eth_deviation_cooldown_days,
            rate_window_days=eth_deviation_rate_window_days,
            deviation_ever=eth_deviation_ever,
            corr_threshold=eth_corr_drop_threshold,
            corr_rate_limit=eth_corr_drop_rate_limit,
            residual_z=eth_residual_z,
            residual_rate_limit=eth_residual_rate_limit,
        )
        df = df[df["symbol"].isin(set(kept))].copy()
        if diagnostics:
            diag_df = (
                pd.DataFrame.from_dict(diagnostics, orient="index")
                .reset_index()
                .rename(columns={"index": "symbol"})
            )
            df = df.merge(diag_df, on="symbol", how="left")

    if air_mean_deviation_filter:
        logger.info("应用垃圾币均值偏离筛选...")
        pool_symbols = list(air_pool)
        kept, diagnostics = apply_air_mean_deviation_filter(
            pool_histories,
            pool_symbols,
            df["symbol"].tolist(),
            as_of=as_of,
            window=air_mean_deviation_window,
            cooldown_days=air_mean_deviation_cooldown_days,
            rate_window_days=air_mean_deviation_rate_window_days,
            deviation_ever=air_mean_deviation_ever,
            corr_threshold=air_mean_corr_drop_threshold,
            corr_rate_limit=air_mean_corr_drop_rate_limit,
            residual_z=air_mean_residual_z,
            residual_rate_limit=air_mean_residual_rate_limit,
            use_median=air_mean_use_median,
        )
        df = df[df["symbol"].isin(set(kept))].copy()
        if diagnostics:
            diag_df = (
                pd.DataFrame.from_dict(diagnostics, orient="index")
                .reset_index()
                .rename(columns={"index": "symbol"})
            )
            df = df.merge(diag_df, on="symbol", how="left")

    if funding_rate_sort:
        logger.info("按历史累计资金费率排序...")
        sums = fetch_funding_rate_sums(
            fetcher,
            df["symbol"].tolist(),
            as_of_date=as_of,
            lookback_days=funding_rate_lookback_days,
        )
        df["funding_rate_sum"] = df["symbol"].map(sums)

        valid_count = df["funding_rate_sum"].notna().sum()
        if valid_count == 0:
            logger.warning("历史资金费率不可用，回退为价格偏离度排序")
            df.sort_values("price_deviation", ascending=False, inplace=True)
        else:
            if funding_rate_min_sum is not None:
                df = df[df["funding_rate_sum"].notna()]
                df = df[df["funding_rate_sum"] >= funding_rate_min_sum]

            df.sort_values("funding_rate_sum", ascending=False, inplace=True)
    else:
        # ⭐ 核心排序：按价格偏离度
        if "funding_rate_sum" not in df.columns or df["funding_rate_sum"].notna().sum() == 0:
            sums = fetch_funding_rate_sums(
                fetcher,
                df["symbol"].tolist(),
                as_of_date=as_of,
                lookback_days=funding_rate_lookback_days,
            )
            df["funding_rate_sum"] = df["symbol"].map(sums)
        df = apply_weighted_score(df)
        df.sort_values(
            ["weighted_score", "price_deviation"],
            ascending=[False, False],
            inplace=True
        )

    return df


def plot_squeeze_for_symbols(
    fetcher: BinanceDataFetcher,
    symbols: List[str],
    *,
    as_of_date: date,
    timeframe: str = DEFAULT_SQUEEZE_TF,
    length: int = SQUEEZE_LEN,
    output_dir: Path,
) -> None:
    if not symbols:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Please install it.") from exc

    end_dt = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=365)

    histories = fetcher.fetch_bulk_history(
        symbols, start=start_dt, end=end_dt, timeframe=timeframe
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol, history in histories.items():
        if history.empty:
            continue

        history = history.sort_values("timestamp")
        history = history[history["timestamp"] >= start_dt]
        if history.empty:
            continue

        squeeze_series = compute_squeeze_series(history, length=length)
        if squeeze_series is None:
            continue

        fig, (ax_price, ax_sq) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax_price.plot(history["timestamp"], history["close"], color="tab:blue")
        ax_price.set_title(f"{symbol} Price ({timeframe}) - Last 1Y")
        ax_price.set_ylabel("Close")

        sq_ts = history.loc[squeeze_series.index, "timestamp"]
        ax_sq.plot(sq_ts, squeeze_series, color="tab:orange")
        ax_sq.axhline(0, color="gray", linewidth=1)
        ax_sq.set_title(f"Squeeze Momentum (len={length})")
        ax_sq.set_ylabel("Squeeze")

        fig.tight_layout()
        safe_symbol = symbol.replace("/", "").replace(":", "")
        fig.savefig(output_dir / f"{safe_symbol}_squeeze.png", dpi=150)
        plt.close(fig)


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
        use_squeeze_filter=args.use_squeeze_filter,
        squeeze_timeframe=args.squeeze_timeframe,
        funding_cooldown=args.cooldown,
        funding_rate_floor=args.funding_rate_floor,
        atr_spike_multiplier=args.atr_spike_multiplier,
        funding_rate_sort=args.funding_rate_sort,
        funding_rate_lookback_days=args.funding_rate_lookback_days,
        funding_rate_min_sum=args.funding_rate_min_sum,
        eth_deviation_filter=args.eth_deviation_filter,
        eth_deviation_window=args.eth_deviation_window,
        eth_deviation_cooldown_days=args.eth_deviation_cooldown_days,
        eth_deviation_rate_window_days=args.eth_deviation_rate_window_days,
        eth_deviation_ever=args.eth_deviation_ever,
        eth_corr_drop_threshold=args.eth_corr_drop_threshold,
        eth_corr_drop_rate_limit=args.eth_corr_drop_rate_limit,
        eth_residual_z=args.eth_residual_z,
        eth_residual_rate_limit=args.eth_residual_rate_limit,
        binance_component_max_weight=args.binance_component_max_weight,
        binance_component_weight_strict=args.binance_component_weight_strict,
        air_mean_deviation_filter=args.air_mean_deviation_filter,
        air_mean_use_median=args.air_mean_use_median,
        air_mean_deviation_window=args.air_mean_deviation_window,
        air_mean_deviation_cooldown_days=args.air_mean_deviation_cooldown_days,
        air_mean_deviation_rate_window_days=args.air_mean_deviation_rate_window_days,
        air_mean_deviation_ever=args.air_mean_deviation_ever,
        air_mean_corr_drop_threshold=args.air_mean_corr_drop_threshold,
        air_mean_corr_drop_rate_limit=args.air_mean_corr_drop_rate_limit,
        air_mean_residual_z=args.air_mean_residual_z,
        air_mean_residual_rate_limit=args.air_mean_residual_rate_limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"candidates_{as_of:%Y%m%d}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} candidates -> {out}")
    if not df.empty:
        top_symbols = df["symbol"].head(args.select_top).tolist()
        print(f"待开仓候选 Top{args.select_top}: {top_symbols}")

    if args.plot_squeeze and not df.empty:
        plot_symbols = df["symbol"].head(args.plot_top).tolist()
        plot_dir = args.output_dir / f"squeeze_plots_{as_of:%Y%m%d}"
        plot_squeeze_for_symbols(
            fetcher=BinanceDataFetcher(),
            symbols=plot_symbols,
            as_of_date=as_of,
            timeframe=args.squeeze_timeframe,
            length=SQUEEZE_LEN,
            output_dir=plot_dir,
        )
        print(f"Saved squeeze plots -> {plot_dir}")

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
