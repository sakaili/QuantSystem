#!/usr/bin/env python3
"""
Module 1: Binance USDT perpetual futures data + feature engineering utilities.

This module centralises all data collection tasks that the short-only strategy
needs:

1. Enumerating the entire USDT-M futures universe on Binance (perpetual only).
2. Pulling exchange-wide 24h ticker statistics so we can rank symbols by
   liquidity (quote volume) or basic fundamentals (market cap, funding, etc.).
3. Downloading daily OHLCV history with proper pagination handling so we stay
   within Binance/CCXT limits.
4. Decorating every OHLCV frame with the technical indicators required by the
   scanner / backtester (EMA20, EMA30, ATR14, optional RSI14 and 90D returns).

The class is intentionally stateful (holding a ccxt exchange instance) so later
modules can re-use a single network client.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

import os

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, will use system env vars

import ccxt  # type: ignore
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAME = "1d"
DEFAULT_LIMIT = 1000  # Binance maximum per fetch_ohlcv request
DEFAULT_LOOKBACK_DAYS = 540  # ～1.5 years


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float while tolerating blanks."""
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_utc(dt: datetime) -> datetime:
    """Guarantee that a datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_to_datetime(value: Any) -> Optional[pd.Timestamp]:
    """Parse timestamps safely without deprecated errors='ignore'."""
    if value in (None, "", "null"):
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


@dataclass(frozen=True)
class SymbolMetadata:
    """Basic metadata for a Binance USDT perpetual contract."""

    symbol: str
    market_id: str
    base: str
    quote: str
    active: bool
    contract_size: float
    tick_size: Optional[float]
    lot_size: Optional[float]
    info: Mapping[str, Any]


class BinanceDataFetcher:
    """High-level helper around ccxt.binanceusdm for Module 1."""

    def __init__(
        self,
        *,
        request_cooldown: float = 0.1,
        exchange: Optional[ccxt.binanceusdm] = None,
        proxies: Optional[Dict[str, str]] = None,
        use_testnet: bool = False,
    ) -> None:
        if exchange is not None:
            self.exchange = exchange
        else:
            api_key = os.environ.get("BINANCE_API_KEY")
            api_secret = os.environ.get("BINANCE_API_SECRET")
            urls = None
            if use_testnet:
                urls = {
                    "api": {
                        "fapi": "https://testnet.binancefuture.com/fapi/v1",
                        "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                        "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                        "vapi": "https://testnet.binancefuture.com/fapi/v1",
                        "sapi": "https://testnet.binancefuture.com/sapi/v1",
                        "wapi": "https://testnet.binancefuture.com/wapi/v3",
                        "public": "https://testnet.binancefuture.com/fapi/v1",
                    }
                }
            exchange_config = {
                "enableRateLimit": False,  # 禁用速率限制，提升数据抓取速度
                "apiKey": api_key,
                "secret": api_secret,
                "options": {
                    "defaultType": "future",
                    "defaultSubType": "linear",
                    "adjustForTimeDifference": True,
                    "recvWindow": 60000,  # 增加请求有效时间窗口到60秒
                },
            }
            if urls:
                exchange_config["urls"] = urls
            self.exchange = ccxt.binanceusdm(exchange_config)

            # 禁用fetchCurrencies - 期货交易不需要，且会导致时间同步问题
            self.exchange.has["fetchCurrencies"] = False

            # 主动同步时间差
            try:
                self.exchange.load_time_difference()
                logger.info(f"时间差已同步: {self.exchange.options.get('timeDifference', 0)}ms")
            except Exception as e:
                logger.warning(f"时间差同步失败(将自动调整): {e}")
        proxies = proxies or self._detect_proxies()
        if proxies:
            # Ensure ccxt's internal requests session routes through the proxy.
            self.exchange.session.proxies.update(proxies)
        self.request_cooldown = request_cooldown

    @staticmethod
    def _detect_proxies() -> Dict[str, str]:
        """Auto-detect HTTP/HTTPS proxy settings from environment variables."""
        proxies: Dict[str, str] = {}
        for scheme in ("http", "https"):
            env_value = os.environ.get(f"{scheme.upper()}_PROXY") or os.environ.get(
                f"{scheme.lower()}_proxy"
            )
            if env_value:
                proxies[scheme] = env_value
        return proxies

    # ----------------------------------------------------------------------
    # Market + ticker discovery
    # ----------------------------------------------------------------------
    def fetch_usdt_perp_symbols(
        self,
        *,
        include_inactive: bool = False,
        reload: bool = False,
    ) -> List[SymbolMetadata]:
        """Return the full Binance USDT perpetual universe."""
        markets = self.exchange.load_markets(reload=reload)
        symbols: List[SymbolMetadata] = []
        for market in markets.values():
            if not market.get("contract"):
                continue
            if market.get("inverse"):
                continue
            if market.get("quote") != "USDT":
                continue
            contract_type = market.get("info", {}).get("contractType")
            if contract_type and contract_type != "PERPETUAL":
                continue
            if not include_inactive and not market.get("active", False):
                continue
            metadata = SymbolMetadata(
                symbol=market["symbol"],
                market_id=market.get("id", market["symbol"].replace("/", "")),
                base=market["base"],
                quote=market["quote"],
                active=bool(market.get("active", False)),
                contract_size=float(market.get("contractSize", 1.0)),
                tick_size=_safe_float(market.get("limits", {}).get("price", {}).get("min")),
                lot_size=_safe_float(market.get("limits", {}).get("amount", {}).get("min")),
                info=market.get("info", {}),
            )
            symbols.append(metadata)
        symbols.sort(key=lambda item: item.symbol)
        logger.info("Loaded %s USDT perpetual markets", len(symbols))
        return symbols

    def fetch_24h_tickers(
        self,
        symbols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch 24h ticker statistics for the provided symbol list.

        Returns a dataframe with the following columns:
            symbol, market_id, timestamp, last, mark_price, index_price,
            base_volume, quote_volume, funding_rate, open_interest, market_cap
        """
        if symbols is None:
            symbols = [meta.symbol for meta in self.fetch_usdt_perp_symbols()]
        symbol_list = list(symbols)
        if not symbol_list:
            raise ValueError("symbols cannot be empty when fetching tickers")

        raw = self.exchange.fetch_tickers(symbol_list)
        rows: List[Dict[str, Any]] = []
        for symbol, ticker in raw.items():
            info = ticker.get("info", {})
            market_id = info.get("symbol") or symbol.replace("/", "").replace(":USDT", "")
            rows.append(
                {
                    "symbol": symbol,
                    "market_id": market_id,
                    "timestamp": pd.to_datetime(
                        ticker.get("datetime") or info.get("closeTime"),
                        utc=True,
                    ),
                    "last": _safe_float(ticker.get("last") or ticker.get("close")),
                    "mark_price": _safe_float(info.get("markPrice")),
                    "index_price": _safe_float(info.get("indexPrice")),
                    "base_volume": _safe_float(ticker.get("baseVolume") or info.get("volume")),
                    "quote_volume": _safe_float(
                        ticker.get("quoteVolume")
                        or info.get("quoteVolume")
                        or info.get("volumeQuote")
                    ),
                    "funding_rate": _safe_float(info.get("lastFundingRate")),
                    "next_funding_time": _safe_to_datetime(info.get("nextFundingTime")),
                    "open_interest": _safe_float(info.get("openInterest")),
                    # Binance does not publish perp market cap. Keep optional.
                    "market_cap": _safe_float(info.get("marketCap")),
                }
            )
        frame = pd.DataFrame(rows).sort_values("quote_volume", ascending=False)
        frame.reset_index(drop=True, inplace=True)
        return frame

    # ----------------------------------------------------------------------
    # Index info helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _symbol_to_market_id(symbol: str) -> str:
        """Normalize ccxt symbol to Binance market id for index endpoints."""
        return symbol.replace("/", "").replace(":", "")

    def fetch_index_info(self, symbol: str, market_id: Optional[str] = None) -> Optional[Any]:
        """
        Fetch index constituents for a symbol from Binance futures public endpoint.

        Endpoint: GET /fapi/v1/constituents
        Returns raw response (dict) or None if unavailable.
        """
        target_id = market_id or self._symbol_to_market_id(symbol)
        params = {"symbol": target_id}

        try:
            # ccxt binanceusdm exposes fapiPublicGetConstituents in recent versions
            if hasattr(self.exchange, "fapiPublicGetConstituents"):
                return self.exchange.fapiPublicGetConstituents(params)
            # Fallback to generic request if method is missing
            return self.exchange.request("constituents", "fapiPublic", "GET", params)
        except Exception as exc:
            # Binance returns -1121 for invalid symbol; treat as "no index info"
            if "-1121" in str(exc):
                logger.info("Index info not available %s (%s)", symbol, target_id)
                return None
            logger.warning("Index info fetch failed %s (%s): %s", symbol, target_id, exc)
            return None

    def index_has_binance_component(self, symbol: str, market_id: Optional[str] = None) -> bool:
        """
        Check whether a symbol's index composition includes Binance.

        Returns False if index info is unavailable or does not include Binance.
        """
        data = self.fetch_index_info(symbol, market_id=market_id)
        if not data:
            return False

        target_id = market_id or self._symbol_to_market_id(symbol)
        entries = data if isinstance(data, list) else [data]

        for entry in entries:
            entry_symbol = entry.get("symbol") or entry.get("pair") or entry.get("s")
            if entry_symbol and entry_symbol != target_id:
                continue

            components = (
                entry.get("components")
                or entry.get("constituents")
                or entry.get("component")
                or entry.get("indexComponents")
                or []
            )

            for component in components:
                exchange_name = (
                    component.get("exchange")
                    or component.get("exchangeName")
                    or component.get("source")
                    or component.get("name")
                    or ""
                )
                if "binance" in str(exchange_name).lower():
                    return True

        return False

    # ----------------------------------------------------------------------
    # OHLCV history + indicators
    # ----------------------------------------------------------------------
    def fetch_klines(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        *,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = DEFAULT_LIMIT,
    ) -> pd.DataFrame:
        """Download OHLCV rows for a single symbol (handles pagination)."""
        end = end or datetime.now(timezone.utc)
        start = start or (end - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        start = _ensure_utc(start)
        end = _ensure_utc(end)
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        step_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)

        all_rows: List[List[float]] = []
        cursor = since_ms
        while cursor < end_ms:
            batch = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=cursor,
                limit=limit,
            )
            if not batch:
                break
            all_rows.extend(batch)
            last_open = batch[-1][0]
            cursor = last_open + step_ms
            if len(batch) < limit:
                break
            # time.sleep(self.request_cooldown)  # 已禁用：提升数据抓取速度

        if not all_rows:
            raise RuntimeError(f"No kline data returned for {symbol}")

        frame = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        frame.dropna(subset=["close"], inplace=True)
        frame.sort_values("timestamp", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    @staticmethod
    def attach_indicators(
        frame: pd.DataFrame,
        *,
        add_rsi: bool = True,
        add_return_90d: bool = True,
    ) -> pd.DataFrame:
        """Append EMA/ATR (+ optional RSI & rolling 90d return) columns."""
        enriched = frame.copy()
        
        # 计算 EMA10, EMA20, EMA30 使用 Pandas 的 ewm
        for length in [10, 20, 30]:
            col_name = f"ema{length}"
            # 使用指数加权移动平均 (ewm) 计算，adjust=False 与 pandas_ta 一致
            enriched[col_name] = enriched['close'].ewm(span=length, adjust=False).mean()
        
        # 计算 ATR14 (14期真实波幅平均)
        # 计算真实波幅 (TR) = max(high - low, |high - prev_close|, |low - prev_close|)
        high_low = enriched['high'] - enriched['low']
        high_prev = abs(enriched['high'] - enriched['close'].shift(1))
        low_prev = abs(enriched['low'] - enriched['close'].shift(1))
        tr = pd.DataFrame({
            'high_low': high_low,
            'high_prev': high_prev,
            'low_prev': low_prev
        }).max(axis=1)
        
        # 计算 14 期简单移动平均 (SMA) 作为 ATR
        atr = tr.rolling(window=14, min_periods=1).mean()
        enriched['atr14'] = atr
        
        # RSI 未实现（原脚本未使用，且需要额外依赖）
        # if add_rsi:
        #     # RSI 计算逻辑（可选实现）
        #     delta = enriched['close'].diff()
        #     gain = delta.where(delta > 0, 0)
        #     loss = -delta.where(delta < 0, 0)
        #     avg_gain = gain.rolling(window=14, min_periods=1).mean()
        #     avg_loss = loss.rolling(window=14, min_periods=1).mean()
        #     rs = avg_gain / avg_loss
        #     enriched['rsi14'] = 100 - (100 / (1 + rs))
        
        if add_return_90d:
            enriched["ret_90d"] = enriched["close"] / enriched["close"].shift(90) - 1
        
        return enriched

    def fetch_symbol_history_with_indicators(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        indicators_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Convenience helper: fetch klines + append indicators in one call."""
        history = self.fetch_klines(symbol, start=start, end=end, timeframe=timeframe)
        if indicators_kwargs is None:
            indicators_kwargs = {}
        return self.attach_indicators(history, **indicators_kwargs)

    def fetch_bulk_history(
        self,
        symbols: Iterable[str],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        indicators_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Batch helper that returns a dict[symbol] -> enriched dataframe."""
        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_symbol_history_with_indicators(
                    symbol,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    indicators_kwargs=indicators_kwargs,
                )
            except Exception as exc:  # pragma: no cover - diagnostic logging
                logger.warning("Failed to fetch %s (%s)", symbol, exc)
        return results


if __name__ == "__main__":
    # Minimal smoke-test helper so Module 1 can be run ad-hoc.
    logging.basicConfig(level=logging.INFO)
    fetcher = BinanceDataFetcher()
    if fetcher.exchange.session.proxies:
        print(f"Using proxies: {fetcher.exchange.session.proxies}")
    metas = fetcher.fetch_usdt_perp_symbols()
    print(f"Total USDT perpetual markets: {len(metas)}")
    tickers = fetcher.fetch_24h_tickers(symbols=[meta.symbol for meta in metas[:10]])
    print(tickers[["symbol", "quote_volume", "funding_rate"]].head())
    sample = fetcher.fetch_symbol_history_with_indicators(metas[0].symbol)
    print(sample.tail())
