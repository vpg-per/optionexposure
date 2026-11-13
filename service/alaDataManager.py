"""Alpaca (SIP + overnight) bar manager with a Yahoo near-real-time patch.

Works both in the Streamlit app (credentials from st.secrets) and in the
GitHub-Actions runner (credentials from environment / .env).

Credential lookup order for ALA_API_KEY / ALA_SECRET_KEY:
    1. st.secrets[...]   (.streamlit/secrets.toml or Streamlit Cloud secrets)
    2. environment variable / .env file
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, time as dt_time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import config

try:  # optional: only needed for local .env files
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Re-exported so callers (e.g. sectorperformance.py) can keep using dm.EASTERN.
EASTERN = config.EASTERN

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
# Yahoo limits how far back each intraday interval goes; start dates are clamped.
YAHOO_MAX_LOOKBACK_DAYS = {"1m": 7, "5m": 60, "15m": 60, "30m": 60}
# Yahoo interval string -> pandas offset alias
_YAHOO_TO_PANDAS = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}

SESSION_START = dt_time(4, 0)   # extended-hours open (ET)
SESSION_END = dt_time(20, 0)    # extended-hours close (ET), exclusive


# --------------------------------------------------------------------------
# Credentials / client
# --------------------------------------------------------------------------
def _get_secret(name: str) -> str | None:
    value = None
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:  # no streamlit, or no secrets file
        pass
    return value or os.getenv(name)


@lru_cache(maxsize=4)
def _build_client(api_key: str, secret_key: str) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(api_key, secret_key)


def get_client() -> StockHistoricalDataClient:
    """Authenticated Alpaca historical data client (cached per key pair)."""
    api_key = _get_secret("ALA_API_KEY")
    secret_key = _get_secret("ALA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALA_API_KEY / ALA_SECRET_KEY. Add them to Streamlit secrets "
            "(or to the environment / .env file)."
        )
    return _build_client(api_key, secret_key)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _to_unix(value) -> int:
    """Datetime/Timestamp -> unix seconds; ints/floats pass through.
    Naive datetimes are interpreted as US/Eastern."""
    if hasattr(value, "timestamp"):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(EASTERN)
        return int(ts.timestamp())
    return int(value)


def round_ohlc(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round open/high/low/close columns (whichever are present)."""
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if cols:
        df[cols] = df[cols].round(decimals)
    return df


def filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars stamped within 04:00-20:00 US/Eastern."""
    return df.between_time(SESSION_START, SESSION_END, inclusive="left")


def drop_incomplete_last_candle(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """Drop the last row if its candle window hasn't fully elapsed yet
    (still forming, not a closed bar)."""
    if df.empty:
        return df
    freq = pd.Timedelta(rule)
    now = pd.Timestamp.now(tz=df.index.tz)
    if df.index[-1] + freq > now:
        df = df.iloc[:-1]
    return df


# --------------------------------------------------------------------------
# Yahoo Finance chart API
# --------------------------------------------------------------------------
def download_stock_data(
    symbol: str,
    startPeriod,
    endPeriod,
    interval: str = "5m",
    with_volume: bool = False,
) -> pd.DataFrame | None:
    """Fetch OHLC bars from Yahoo Finance's chart API.
    with_volume=True also returns a `volume` column (missing values -> 0).

    startPeriod/endPeriod may be datetimes/Timestamps or raw unix seconds.
    Returns a DataFrame (open/high/low/close) indexed by tz-aware US/Eastern
    timestamps, or None on failure / no data.
    """
    period1 = _to_unix(startPeriod)
    period2 = _to_unix(endPeriod)

    max_days = YAHOO_MAX_LOOKBACK_DAYS.get(interval)
    if max_days is not None:
        earliest = int(time.time() - max_days * 86400 + 3600)  # 1h safety margin
        if period1 < earliest:
            print(f"Yahoo only serves {max_days} days of '{interval}' data; "
                  f"clamping start for {symbol}.")
            period1 = earliest

    if period1 >= period2:
        return None

    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "includePrePost": "true",
    }

    try:
        resp = requests.get(
            YAHOO_URL.format(symbol=symbol),
            params=params,
            headers=YAHOO_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        chart = resp.json()["chart"]
        if chart.get("error"):
            print(f"Yahoo API error for {symbol}: {chart['error']}")
            return None

        result = chart["result"][0]
        quotes = result["indicators"]["quote"][0]
        data = {c: np.asarray(quotes[c], dtype="float64") for c in ("open", "high", "low", "close")}
        if with_volume and quotes.get("volume") is not None:
            data["volume"] = np.asarray(quotes["volume"], dtype="float64")
        df = pd.DataFrame(
            data,
            index=pd.to_datetime(np.asarray(result["timestamp"], dtype="int64"), unit="s", utc=True),
        ).dropna(subset=["open", "high", "low", "close"])
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)
        if df.empty:
            return None

        # Yahoo can occasionally repeat a timestamp; keep the latest print.
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index = df.index.tz_convert(EASTERN)
        df.index.name = "timestamp"
        return round_ohlc(df)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Yahoo data for {symbol}: {e}")
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"Error parsing Yahoo data for {symbol}: {e}")
    return None


def fetch_sector_data_yahoo(
    symbol: str,
    days_back: int | None = None,
    end_et: datetime | None = None,
    interval: str = "5m",
) -> pd.DataFrame | None:
    """Extended-hours bars for `symbol` from Yahoo, ending at `end_et`
    (now by default). `interval` is a Yahoo interval ("5m", "30m", ...);
    pass interval="30m" to get the older, smoother sector-chart look."""
    if days_back is None:
        days_back = config.LOOKBACK_DAYS
    end_et = end_et or datetime.now(EASTERN)

    df = download_stock_data(symbol, end_et - timedelta(days=days_back), end_et, interval=interval)
    if df is None or df.empty:
        return None

    df = filter_session(df)
    if df.empty:
        return None

    df = drop_incomplete_last_candle(df, _YAHOO_TO_PANDAS.get(interval, interval))
    if df.empty:
        return None

    df["feed"] = "yahoo"
    df["symbol"] = symbol
    df["interval"] = _YAHOO_TO_PANDAS.get(interval, interval)
    return round_ohlc(df)


def fetch_recent_yahoo(
    symbol: str = config.SYMBOL,
    minutes: int = 30,
) -> pd.DataFrame | None:
    """Last `minutes` of 1-minute Yahoo bars. The window start is floored to a
    5-minute boundary so resampling to 5min never yields a half-filled first bar."""
    now_et = datetime.now(EASTERN)
    start = pd.Timestamp(now_et - timedelta(minutes=minutes)).floor("5min")

    df = download_stock_data(symbol, start, now_et, interval="1m", with_volume=True)
    if df is None or df.empty:
        return None

    df = filter_session(df)
    if df.empty:
        return None

    df["feed"] = "yahoo"
    return round_ohlc(df)


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC bar data to a larger timeframe (e.g. '15min', '1h', '4h').
    df must have a DatetimeIndex and open/high/low/close columns."""
    ohlc_agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    agg = {col: fn for col, fn in ohlc_agg.items() if col in df.columns}
    if "volume" in df.columns:  # volume adds up across the merged bars
        agg["volume"] = "sum"

    resampled = df.resample(rule).agg(agg).dropna(subset=["open"])
    resampled["interval"] = rule
    if "symbol" in df.columns and not df.empty:
        resampled["symbol"] = df["symbol"].iloc[0]
    return round_ohlc(resampled)


def append_recent_yahoo_data(
    df: pd.DataFrame,
    symbol: str = config.SYMBOL,
    minutes: int = 30,
) -> pd.DataFrame:
    """Patch the tail of an Alpaca-sourced 5-min `df` with fresher 5-min bars
    rebuilt from Yahoo's 1-min data (Alpaca feeds can lag). On overlapping
    timestamps the Yahoo bar wins. No-op if Yahoo has no data."""
    recent = fetch_recent_yahoo(symbol, minutes)
    if recent is None or recent.empty:
        return df

    recent_5m = resample_bars(recent, "5min")
    if recent_5m.empty:
        return df
    recent_5m["symbol"] = symbol

    merged = pd.concat([df, recent_5m])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return round_ohlc(merged)


# --------------------------------------------------------------------------
# Alpaca
# --------------------------------------------------------------------------
def to_eastern(df: pd.DataFrame, symbol: str = config.SYMBOL) -> pd.DataFrame:
    """Flatten Alpaca's (symbol, timestamp) MultiIndex and convert UTC -> Eastern."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol").copy()
    else:
        df = df.copy()
    df.index = df.index.tz_convert(EASTERN)
    return df


def fetch_feed(
    client: StockHistoricalDataClient,
    feed: str,
    start: datetime,
    symbol: str = config.SYMBOL,
    timeframe: TimeFrame = config.BASE_TIMEFRAME,
) -> pd.DataFrame | None:
    """Fetch bars for a single feed. Returns None (instead of raising) if
    the feed/subscription isn't available on this account."""
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=start,
        feed=feed,
    )
    try:
        bars = client.get_stock_bars(request).df
    except Exception as e:  # noqa: BLE001
        print(f"Feed '{feed}' unavailable (needs a matching subscription?): {e}")
        return None

    if bars.empty:
        return None

    df = to_eastern(bars, symbol)
    # volume and the per-bar vwap are kept: biasCalculator.add_vwap builds the session VWAP from them.
    return df.drop(columns=["trade_count", "feed"], errors="ignore")


def fetch_combined(
    client: StockHistoricalDataClient,
    feeds: list[str],
    start: datetime,
    symbol: str = config.SYMBOL,
    timeframe: TimeFrame = config.BASE_TIMEFRAME,
) -> pd.DataFrame:
    """Fetch each feed and merge into one deduplicated timeline. Earlier feeds
    in `feeds` win on overlapping timestamps."""
    frames = [fetch_feed(client, feed, start, symbol, timeframe) for feed in feeds]
    frames = [f for f in frames if f is not None]

    if not frames:
        raise RuntimeError(
            f"No Alpaca feeds returned data for {symbol} -- check your "
            "subscriptions, API keys and the symbol."
        )

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined["interval"] = "5min"
    combined["symbol"] = symbol
    combined = round_ohlc(combined)

    # Alpaca can lag; patch the tail from Yahoo during the 04:00-20:00 ET session.
    # (Real Eastern clock -- datetime.now() alone would use the server's zone.)
    if 4 <= datetime.now(EASTERN).hour < 20:
        combined = append_recent_yahoo_data(combined, symbol)
    # Always drop a still-forming last bar (also covers Alpaca-only data).
    combined = drop_incomplete_last_candle(combined, "5min")
    return combined


def fetch_stock_data(symbol, interval: str = "5min", lookback_days: int | None = None) -> pd.DataFrame:
    """Closed 5-min bars for `symbol` (or resampled to `interval`).
    `lookback_days` defaults to config.LOOKBACK_DAYS."""
    client = get_client()
    days = lookback_days or config.LOOKBACK_DAYS
    start = datetime.now(EASTERN) - timedelta(days=days)

    combined = fetch_combined(client, config.FEEDS, start, symbol)
    if interval == "5min":
        missing = find_missing_periods(combined)
        print(f"Missing count: {missing.size}")
        return combined

    resampled = resample_bars(combined, interval)
    return drop_incomplete_last_candle(resampled, interval)


def find_missing_periods(
    df: pd.DataFrame,
    freq: str = "5min",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Timestamps missing from df's index over [start, end], assuming
    continuous `freq`-spaced bars."""
    if df.empty:
        raise ValueError("df is empty -- nothing to check.")

    start = start or df.index.min()
    end = end or df.index.max()
    expected = pd.date_range(start=start, end=end, freq=freq, tz=df.index.tz)
    return expected.difference(df.index)


def report_missing_periods(
    df: pd.DataFrame,
    freq: str = "5min",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Group consecutive missing timestamps into gap ranges
    (columns: gap_start, gap_end, bars_missing)."""
    missing = find_missing_periods(df, freq, start, end)
    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "bars_missing"])

    step = pd.Timedelta(freq)
    run_id = (missing.to_series().diff() != step).cumsum()
    grouped = missing.to_series().groupby(run_id)
    gaps = grouped.agg(gap_start="min", gap_end="max")
    gaps["bars_missing"] = grouped.size().values
    return gaps.reset_index(drop=True)
