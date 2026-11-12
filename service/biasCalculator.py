"""MACD / RSI / overall-bias calculations.

Logic is ported unchanged from dataAnalyzer.ServiceManager (calculate_macd,
calculate_rsi, calculate_overall_bias), minus the Bollinger / trade-level code,
the prints, and the dropping of the numeric columns (the page shows them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Enough bars for MACD(26)+signal(9) and RSI(14)+signal/smoothing to settle.
MIN_BARS = 40


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = df["close"].astype("float64")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()

    out = df.copy()
    out["macd"] = macd_line.round(2).astype("float32")
    out["msignal"] = sig_line.round(2).astype("float32")
    out["histogram"] = (macd_line - sig_line).round(2).astype("float32")

    trend = pd.Series("Neutral", index=df.index)
    trend[(macd_line > sig_line) & (macd_line > 0)] = "Bullish"
    trend[(macd_line < sig_line) & (macd_line < 0)] = "Bearish"
    out["MACDTrend"] = trend
    return out


def add_rsi(df: pd.DataFrame, period: int = 14, smooth: int = 3, threshold: float = 0.5) -> pd.DataFrame:
    diff = df["close"].astype("float64").diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss

    rsi_raw = 100 - (100 / (1 + rs))
    rsi = rsi_raw.round(2).astype("float32")
    rsignal = rsi_raw.ewm(span=period, adjust=False).mean().round(2).astype("float32")
    rsi_hist = rsi_raw - rsignal.astype("float64")
    hist_smooth = rsi_hist.rolling(window=smooth).mean()
    hist_slope = hist_smooth.diff()

    # Direction: RSI above / below its own signal line.
    rsi_dir = pd.Series("Neutral", index=df.index)
    rsi_dir[rsi_raw > rsignal.astype("float64")] = "Up"
    rsi_dir[rsi_raw < rsignal.astype("float64")] = "Down"

    # Accelerating: histogram moving further from zero on its own side.
    # Decelerating: moving back toward zero. None: slope inside the dead zone.
    same_direction = np.sign(hist_slope) == np.sign(hist_smooth)
    meaningful = hist_slope.abs() > threshold
    momentum = pd.Series("None", index=df.index)
    momentum[meaningful & same_direction] = "Accelerating"
    momentum[meaningful & ~same_direction] = "Decelerating"

    # Bullish/Bearish only when direction AND accelerating momentum agree.
    rsitrend = pd.Series("Neutral", index=df.index)
    rsitrend[(rsi_dir == "Up") & (momentum == "Accelerating")] = "Bullish"
    rsitrend[(rsi_dir == "Down") & (momentum == "Accelerating")] = "Bearish"

    out = df.copy()
    out["rsi"] = rsi
    out["rsignal"] = rsignal
    out["momentum"] = momentum
    out["RSITrend"] = rsitrend
    return out


def add_overall_bias(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bias = pd.Series("Neutral", index=out.index)
    bias[(out["MACDTrend"] == "Bullish") & (out["RSITrend"] == "Bullish")] = "Bullish"
    bias[(out["MACDTrend"] == "Bearish") & (out["RSITrend"] == "Bearish")] = "Bearish"
    out["OverallBias"] = bias
    return out


def compute_bias(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC frame -> same frame plus MACD/RSI columns and the bias columns."""
    return add_overall_bias(add_rsi(add_macd(df)))


def resample_anchored(df: pd.DataFrame, hours: int = 4, anchor_hour: int = 1) -> pd.DataFrame:
    """Resample OHLC bars into `hours`-long buckets that start at
    anchor_hour + k*hours on the *local wall clock* (default 4h: 01, 05, 09, 13,
    17, 21 ET).

    pandas' resample("4h") cuts fixed-length bins in UTC, so after a daylight
    saving change the bins slide by an hour (e.g. 03, 07, 11, 15, 19, 23 ET).
    Grouping on wall-clock time keeps the buckets at the same local hours all year.
    `df` must be sorted, tz-aware and have open/high/low/close columns.
    """
    tz = df.index.tz
    local = df.index.tz_localize(None)                       # wall-clock times
    shift = pd.Timedelta(hours=anchor_hour)
    bucket = (local - shift).floor(f"{hours}h") + shift      # local bucket start

    out = df.groupby(bucket).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    # The anchor hours (1, 5, 9, 13, 17, 21) never fall in the spring-forward gap;
    # on the fall-back day the repeated 01:00 resolves to its first occurrence.
    out.index = out.index.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
    out.index.name = df.index.name
    return out.sort_index().round(2)


def drop_unfinished_buckets(
    df: pd.DataFrame, rule: str, base_last_end: pd.Timestamp, wall_clock: bool = False
) -> pd.DataFrame:
    """Drop the last resampled bar if it isn't fully covered by closed base
    (5-min) bars. `base_last_end` = end time of the last closed 5-min bar.
    wall_clock=True measures the bucket length on the local clock (use it for
    resample_anchored buckets so DST days don't mis-time the bucket end)."""
    if df.empty:
        return df
    start = df.index[-1]
    if wall_clock:
        end = (start.tz_localize(None) + pd.Timedelta(rule)).tz_localize(
            start.tz, ambiguous=True, nonexistent="shift_forward")
    else:
        end = start + pd.Timedelta(rule)
    return df.iloc[:-1] if end > base_last_end else df


def latest_snapshot(df_with_bias: pd.DataFrame) -> dict:
    """Last row of a compute_bias() frame as a flat dict."""
    last = df_with_bias.iloc[-1]
    return {
        "bar_time": df_with_bias.index[-1],
        "close": round(float(last["close"]), 2),
        "macd": round(float(last["macd"]), 2),
        "macd_signal": round(float(last["msignal"]), 2),
        "macd_hist": round(float(last["histogram"]), 2),
        "macd_trend": last["MACDTrend"],
        "rsi": round(float(last["rsi"]), 2),
        "rsi_signal": round(float(last["rsignal"]), 2),
        "momentum": last["momentum"],
        "rsi_trend": last["RSITrend"],
        "overall_bias": last["OverallBias"],
    }
