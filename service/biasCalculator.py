"""MACD / RSI / overall-bias calculations.

Logic is ported unchanged from dataAnalyzer.ServiceManager (calculate_macd,
calculate_rsi, calculate_overall_bias), minus the Bollinger / trade-level code,
the prints, and the dropping of the numeric columns (the page shows them).
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

# Enough bars for MACD(26)+signal(9) and RSI(14)+signal/smoothing to settle.
MIN_BARS = 40

# --- VWAP / ATR / projection settings --------------------------------------
VWAP_ANCHOR = dt.time(4, 0)   # session VWAP resets here (ET). Use dt.time(9, 30) for regular-session VWAP.
ATR_PERIOD = 14
VWAP_STRETCH = 1.5            # |price - VWAP| in ATRs at which the VWAP pull switches on
SIGMA_PER_ATR = 0.6           # std-dev of one bar's close-to-close move ~ 0.6 x ATR (range ~ 1.6 sigma)
CONE_ATR_MULT = 1.0           # expected-range cone half-width = CONE_ATR_MULT * ATR * sqrt(bars ahead)
LEAN_BAND = 0.05              # P(up) within 50% +/- this is labelled "Neutral"


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


# --------------------------------------------------------------------------
# ATR / session VWAP (5-min frames)
# --------------------------------------------------------------------------
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Wilder ATR (true range smoothed with alpha = 1/period) -> column `atr`."""
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    prev_close = df["close"].astype("float64").shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out = df.copy()
    out["atr"] = tr.ewm(alpha=1.0 / period, adjust=False).mean().round(4)
    return out


def add_vwap(df: pd.DataFrame, anchor: dt.time = VWAP_ANCHOR) -> pd.DataFrame:
    """Session-anchored cumulative VWAP -> column `svwap` (NaN where no volume yet).

    Needs a `volume` column. Per-bar price is Alpaca's bar `vwap` when present,
    otherwise the typical price (H+L+C)/3 (e.g. Yahoo-patched bars have no bar vwap).
    A session runs from `anchor` to the next `anchor` on the local ET wall clock, so
    overnight bars belong to the session that started the previous day at `anchor`.
    Compute on the full frame (before .tail) so the cumulation starts at the session open.
    """
    out = df.copy()
    if "volume" not in out.columns:
        out["svwap"] = np.nan
        return out

    typical = (out["high"].astype("float64") + out["low"].astype("float64")
               + out["close"].astype("float64")) / 3.0
    if "vwap" in out.columns:
        bar_px = out["vwap"].astype("float64")
        bar_px = bar_px.where(bar_px.notna() & (bar_px > 0), typical)
    else:
        bar_px = typical
    vol = out["volume"].astype("float64").fillna(0.0).clip(lower=0.0)

    local = out.index.tz_localize(None) if out.index.tz is not None else out.index
    shift = pd.Timedelta(hours=anchor.hour, minutes=anchor.minute)
    session = (local - shift).floor("D").to_numpy()

    cum_pv = (bar_px * vol).groupby(session).cumsum()
    cum_v = vol.groupby(session).cumsum()
    out["svwap"] = (cum_pv / cum_v.where(cum_v > 0)).round(2)
    return out


# --------------------------------------------------------------------------
# Next-candle projection
# --------------------------------------------------------------------------
def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def project_next_candles(frame5m: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, dict]:
    """Heuristic scenario for the next n 5-min candles. Not a forecast.

    Returns (projection, context). `projection` has one row per candle:
    open/high/low/close, p_up (probability the candle's close is above the last
    closed price), cone_low/cone_high (expected-range cone) and lean.
    `context` holds the inputs shown to the user (price, vwap, vwap_dist_atr, atr).

    RSI is intentionally not used here (it is still computed by compute_bias for the
    bias table and alerts). Step per bar (all in price units, scaled by ATR):
      * drift: mean net move of the last 5 bars (0.5x on the first bar, 0.3x after);
      * VWAP pull: when |price - VWAP| >= VWAP_STRETCH ATRs, price is pulled back
        toward VWAP by 0.10-0.20 ATR per bar (full strength if the MACD histogram is
        contracting, half otherwise). Re-evaluated each projected bar.
    Probability: P(up) = Phi(cumulative move / (SIGMA_PER_ATR * ATR * sqrt(k))).
    Wicks: the cone of excursion leans toward the projected direction. The wick
    on the favoured side gets 50%-75% of the wick budget (more with higher
    conviction), the other side the rest; the budget is ATR minus the body, min 0.4 ATR.
    """
    empty = (pd.DataFrame(), {})
    if frame5m is None or len(frame5m) < 10:
        return empty
    close = frame5m["close"].astype("float64")
    med_rng = float((frame5m["high"] - frame5m["low"]).tail(12).median())
    atr = float(frame5m["atr"].iloc[-1]) if "atr" in frame5m.columns else float("nan")
    scale = atr if atr == atr and atr > 0 else med_rng   # fall back to median range
    if not scale or scale != scale:
        return empty

    px = float(close.iloc[-1])
    drift = (px - float(close.iloc[-6])) / 5.0
    hist = frame5m["histogram"].astype("float64")
    hist_contracting = abs(hist.iloc[-1]) < abs(hist.iloc[-4])

    vwap = float(frame5m["svwap"].iloc[-1]) if "svwap" in frame5m.columns else float("nan")
    have_vwap = vwap == vwap and vwap > 0
    sigma_bar = SIGMA_PER_ATR * scale

    last_close, rows = px, []
    for i in range(n):
        k = i + 1
        step = drift * (0.5 if i == 0 else 0.3)

        z = (px - vwap) / scale if have_vwap else 0.0
        stretched = have_vwap and abs(z) >= VWAP_STRETCH
        if stretched:
            pull = 0.10 + 0.05 * min(abs(z) - VWAP_STRETCH, 2.0)
            pull *= 1.0 if hist_contracting else 0.5
            step -= math.copysign(pull * scale, z)

        o, c = px, px + step
        p_up = _phi((c - last_close) / (sigma_bar * math.sqrt(k)))
        p_bar = _phi(step / sigma_bar)
        conviction = abs(2.0 * p_bar - 1.0)
        fav = 0.5 + 0.25 * conviction
        up_share = fav if step > 0 else (1.0 - fav if step < 0 else 0.5)
        wick_budget = max(scale - abs(step), 0.4 * scale)
        hi = max(o, c) + wick_budget * up_share
        lo = min(o, c) - wick_budget * (1.0 - up_share)
        half = CONE_ATR_MULT * scale * math.sqrt(k)
        lean = "Up" if p_up >= 0.5 + LEAN_BAND else ("Down" if p_up <= 0.5 - LEAN_BAND else "Neutral")
        rows.append({"open": round(o, 2), "high": round(hi, 2), "low": round(lo, 2),
                     "close": round(c, 2), "p_up": p_up,
                     "cone_low": round(c - half, 2), "cone_high": round(c + half, 2),
                     "lean": lean})
        px = c

    ctx = {"price": round(float(close.iloc[-1]), 2),
           "vwap": round(vwap, 2) if have_vwap else None,
           "vwap_dist_atr": round((float(close.iloc[-1]) - vwap) / scale, 2) if have_vwap else None,
           "atr": round(scale, 3)}
    return pd.DataFrame(rows), ctx
