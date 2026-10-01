"""Stock data manager -- Yahoo Finance chart API only (no Alpaca)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, time as dt_time

import pandas as pd
import pytz
import requests

EASTERN = pytz.timezone("America/New_York")

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Yahoo limits how far back intraday data goes; start dates are clamped.
YAHOO_MAX_LOOKBACK_DAYS = {"30m": 5}

SESSION_START = dt_time(4, 0)   # extended-hours open (ET)
SESSION_END = dt_time(20, 0)    # extended-hours close (ET), exclusive

OHLC = ["open", "high", "low", "close"]


def _to_unix(value) -> int:
    """Datetime/Timestamp -> unix seconds; ints/floats pass through.
    Naive datetimes are interpreted as US/Eastern."""
    if hasattr(value, "timestamp"):
        if getattr(value, "tzinfo", None) is None:
            value = pd.Timestamp(value).tz_localize(EASTERN)
        return int(value.timestamp())
    return int(value)


def round_ohlc(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round open/high/low/close columns (whichever are present)."""
    cols = [c for c in OHLC if c in df.columns]
    if cols:
        df[cols] = df[cols].round(decimals)
    return df


def filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars stamped within 04:00-20:00 US/Eastern."""
    return df.between_time(SESSION_START, SESSION_END, inclusive="left")


def download_stock_data(
    symbol: str,
    startPeriod,
    endPeriod,
    interval: str = "5m",
) -> pd.DataFrame | None:
    """Fetch OHLC bars from Yahoo Finance's chart API.

    startPeriod/endPeriod may be datetimes/Timestamps or raw unix seconds.
    Returns a DataFrame (open/high/low/close) indexed by tz-aware
    US/Eastern timestamps, or None on failure / no data.
    """
    period1 = _to_unix(startPeriod)
    period2 = _to_unix(endPeriod)

    max_days = YAHOO_MAX_LOOKBACK_DAYS.get(interval)
    if max_days is not None:
        earliest = int(time.time() - max_days * 86400)
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
        df = pd.DataFrame(
            {c: quotes[c] for c in OHLC},
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True),
        ).astype("float64").dropna()
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
    days_back: int = 4,
    end_et: datetime | None = None,
) -> pd.DataFrame | None:
    """30-minute extended-hours bars for `symbol`, ending at `end_et` (now by default)."""
    end_et = end_et or datetime.now(EASTERN)
    df = download_stock_data(symbol, end_et - timedelta(days=days_back), end_et, interval="30m")
    if df is None:
        return None

    df = filter_session(df)
    if df.empty:
        return None

    df["interval"] = "30min"
    df["symbol"] = symbol
    return df