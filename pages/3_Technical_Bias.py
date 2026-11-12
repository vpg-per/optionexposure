"""
pages/3_Technical_Bias.py

MACD / RSI bias for a list of symbols across 5m, 15m, 30m, 1h and 4h.
Bars come from service.alaDataManager (5-min bars, closed bars only); the
larger timeframes are resampled from them (4h bars start at 01/05/09/13/17/21 ET).
Calculation rules live in service/biasCalculator.py (ported from dataAnalyzer.py).

Debug mode: tick "Debug: as-of time" to see the bias exactly as it would have
looked at a past Eastern date/time (only bars closed by then are used).
"""
import datetime as dt
import re

import pandas as pd
import streamlit as st

from service.alaDataManager import EASTERN, fetch_stock_data, resample_bars
from service.biasCalculator import (
    MIN_BARS,
    compute_bias,
    drop_unfinished_buckets,
    latest_snapshot,
    resample_anchored,
)

st.set_page_config(page_title="Technical", layout="wide")
st.title("Technical Bias (MACD / RSI)")

TIMEFRAMES = [("5 min", "5min"), ("15 min", "15min"), ("30 min", "30min"),
              ("1 hr", "1h"), ("4 hr", "4h")]
TF_LABELS = [label for label, _ in TIMEFRAMES]
LOOKBACK_DAYS = 20   # enough history for the 4h series to warm up MACD/RSI
FOUR_HOUR_ANCHOR = 1  # 4h bars start at 01, 05, 09, 13, 17, 21 ET (local wall clock)
CACHE_TTL = 120      # seconds
MAX_SYMBOLS = 10
BARS_KEPT = 300      # 5-min bars (with indicator columns) kept for the debug table

COLORS = {"Bullish": "#1a9850",
          "Bearish": "#d73027",
          "Neutral": "#6b6b6b"}

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
if "bias_refresh_key" not in st.session_state:
    st.session_state.bias_refresh_key = 0

ctl_symbols, ctl_refresh = st.columns([5, 1], vertical_alignment="bottom")
raw_symbols = ctl_symbols.text_input("Symbols (comma-separated)", value="SPY, GLD, QQQ, IWM")
if ctl_refresh.button("Refresh", type="primary", use_container_width=True):
    st.session_state.bias_refresh_key += 1

now_et = dt.datetime.now(EASTERN)
ctl_debug, ctl_date, ctl_time = st.columns([2, 2, 2], vertical_alignment="bottom")
debug_on = ctl_debug.checkbox("Debug: (ET)", value=False,
                              help="Show the bias as it was at a past time instead of the latest candle.")
asof_date = ctl_date.date_input("As-of date", value=now_et.date(), max_value=now_et.date(),
                                key="bias_asof_date", disabled=not debug_on)
asof_time = ctl_time.time_input(
    "As-of time (ET)", value=now_et.time().replace(minute=now_et.minute // 5 * 5, second=0, microsecond=0),
    step=300, key="bias_asof_time", disabled=not debug_on)

asof_et = None  # None = live (latest closed candle)
if debug_on:
    asof_et = pd.Timestamp(dt.datetime.combine(asof_date, asof_time)).tz_localize(
        EASTERN, ambiguous=True, nonexistent="shift_forward")
    if asof_et >= pd.Timestamp(now_et):
        st.warning("As-of time is in the future; showing the latest closed candle instead.")
        asof_et = None

symbols = list(dict.fromkeys(s for s in re.split(r"[\s,]+", raw_symbols.upper()) if s))
if len(symbols) > MAX_SYMBOLS:
    st.warning(f"Showing the first {MAX_SYMBOLS} symbols only.")
    symbols = symbols[:MAX_SYMBOLS]
if not symbols:
    st.info("Enter at least one symbol.")
    st.stop()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_symbol_bias(symbol: str, refresh_key: int, asof_iso: str | None,
                     lookback_days: int) -> tuple[list[dict], list[str], pd.DataFrame]:
    """One 5-min fetch per symbol; every other timeframe is resampled from it.
    refresh_key only exists so the Refresh button yields a new cache entry.
    asof_iso (debug): ISO timestamp with offset; only 5-min bars that had closed
    by then are kept, so the result is the bias as it stood at that moment.
    Returns (rows, skipped-timeframe messages, last BARS_KEPT 5-min bars with indicators)."""
    base = fetch_stock_data(symbol, "5min", lookback_days=lookback_days)
    if asof_iso:
        base = base[base.index + pd.Timedelta("5min") <= pd.Timestamp(asof_iso)]
        if base.empty:
            raise ValueError(f"no bars closed before {asof_iso} in the fetched history")
    base_last_end = base.index[-1] + pd.Timedelta("5min")

    rows, notes = [], []
    frame5m = pd.DataFrame()
    for label, rule in TIMEFRAMES:
        if rule == "5min":
            frame = base
        elif rule == "4h":
            frame = resample_anchored(base, hours=4, anchor_hour=FOUR_HOUR_ANCHOR)
        else:
            frame = resample_bars(base, rule)
        frame = drop_unfinished_buckets(frame, rule, base_last_end, wall_clock=(rule == "4h"))
        if len(frame) < MIN_BARS:
            notes.append(f"{symbol} {label}: only {len(frame)} closed bars (need {MIN_BARS}).")
            continue
        bias_frame = compute_bias(frame)
        if rule == "5min":
            frame5m = bias_frame.tail(BARS_KEPT)
        snap = latest_snapshot(bias_frame)
        rows.append({"symbol": symbol, "timeframe": label,
                     "data_through": base_last_end, **snap})
    return rows, notes, frame5m


# History must reach back LOOKBACK_DAYS before the as-of time, not just before today.
extra_days = (now_et.date() - asof_et.date()).days + 1 if asof_et is not None else 0
lookback_days = LOOKBACK_DAYS + extra_days
asof_iso = asof_et.isoformat() if asof_et is not None else None

all_rows, notes, errors, frames5m = [], [], [], {}
with st.spinner("Fetching bars and calculating bias..."):
    for sym in symbols:
        try:
            rows, skipped, f5m = load_symbol_bias(sym, st.session_state.bias_refresh_key,
                                                  asof_iso, lookback_days)
            frames5m[sym] = f5m
            all_rows.extend(rows)
            notes.extend(skipped)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sym}: {e}")

for msg in errors:
    st.error(msg)
for msg in notes:
    st.warning(msg)
if not all_rows:
    st.stop()

detail = pd.DataFrame(all_rows)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _color(v):
    c = COLORS.get(v)
    return f"background-color: {c}; color: white; font-weight: 600; text-align: center" if c else ""


def _fmt_time(ts) -> str:
    """Bar start time as HH:MM (prefixed with the date when it isn't the reference
    day: today, or the as-of day in debug mode)."""
    ts = pd.Timestamp(ts)
    ref_day = asof_et.date() if asof_et is not None else pd.Timestamp.now(tz=ts.tz).date()
    return ts.strftime("%H:%M") if ts.date() == ref_day else ts.strftime("%m-%d %H:%M")


def _elementwise(frame: pd.DataFrame, fn) -> pd.DataFrame:
    return frame.map(fn) if hasattr(frame, "map") else frame.applymap(fn)


detail["bar_label"] = detail["bar_time"].map(_fmt_time)


def show_pivot(title: str, column: str) -> None:
    """Cell text = start time of the last closed bar for that timeframe;
    cell colour = the trend/bias value in `column` (green Up/Bullish,
    red Down/Bearish, grey Neutral)."""
    idx = [s for s in symbols if s in set(detail["symbol"])]
    cols = [t for t in TF_LABELS if t in set(detail["timeframe"])]

    def pivot(values: str) -> pd.DataFrame:
        p = detail.pivot(index="symbol", columns="timeframe", values=values).reindex(index=idx, columns=cols)
        p.index.name = None
        p.columns.name = None
        return p

    labels = pivot("bar_label").fillna("-")
    css = _elementwise(pivot(column), _color)
    st.subheader(title)
    st.dataframe(labels.style.apply(lambda _: css, axis=None), use_container_width=True)


show_pivot("Overall bias", "overall_bias")
col_macd, col_rsi = st.columns(2)
with col_macd:
    show_pivot("MACD trend", "macd_trend")
with col_rsi:
    show_pivot("RSI trend", "rsi_trend")


# --------------------------------------------------------------------------
# Next 3 candles (5 min) - heuristic projection from the closed 5-min bars
# --------------------------------------------------------------------------
def project_next_candles(frame5m: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Heuristic OHLC projection for the next n 5-min candles.

    Not a forecast -- a scenario built from the same numbers the bias table shows:
      * drift: average per-bar net move over the last 5 bars (half-strength for the
        first projected bar, fading to 0.3x afterwards -- momentum decays);
      * mean reversion: when RSI is stretched (<35 / >65) AND the MACD histogram is
        contracting (|hist| shrinking vs 4 bars ago, i.e. post-climax basing), a
        counter-drift of ~15% of the typical range is added per bar;
      * range: median 5-min range of the last 12 bars, wick ~45% of it on each side.
    Returns a DataFrame with open/high/low/close columns.
    """
    if frame5m is None or len(frame5m) < 10:
        return pd.DataFrame()
    close = frame5m["close"].astype("float64")
    rng = float((frame5m["high"] - frame5m["low"]).tail(12).median())
    if not rng or rng != rng:  # degenerate flat data
        return pd.DataFrame()

    drift = float(close.iloc[-1] - close.iloc[-6]) / 5.0
    rsi = float(frame5m["rsi"].iloc[-1])
    hist = frame5m["histogram"].astype("float64")
    hist_contracting = abs(hist.iloc[-1]) < abs(hist.iloc[-4])
    oversold, overbought = rsi < 35.0, rsi > 65.0

    px = float(close.iloc[-1])
    rows = []
    for i in range(n):
        step = drift * (0.5 if i == 0 else 0.3)
        if hist_contracting and oversold:
            step += rng * 0.15
        elif hist_contracting and overbought:
            step -= rng * 0.15
        o = px
        c = px + step
        hi = max(o, c) + rng * 0.45
        lo = min(o, c) - rng * 0.45
        rows.append({"open": round(o, 2), "high": round(hi, 2),
                     "low": round(lo, 2), "close": round(c, 2)})
        px = c
    return pd.DataFrame(rows)


st.subheader("Next 3 candles (5 min projection)")
pred_rows = []
for sym in [s for s in symbols if s in frames5m and not frames5m[s].empty]:
    proj = project_next_candles(frames5m[sym], n=3)
    if proj.empty:
        continue
    last_bar = frames5m[sym].index[-1]
    for j, row in enumerate(proj.itertuples(index=False), start=1):
        pred_rows.append({"Symbol": sym,
                          "Bar start (ET)": (last_bar + pd.Timedelta(minutes=5 * j)).strftime("%m-%d %H:%M"),
                          "Open": row.open, "High": row.high,
                          "Low": row.low, "Close": row.close})
if pred_rows:
    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
else:
    st.info("Not enough 5-min bars for a projection.")
st.caption("Heuristic scenario, not a forecast: recent drift with mean-reversion when RSI is "
           "stretched and the MACD histogram is contracting. Ranges use the median 5-min range. "
           "In debug (as-of) mode the projection starts from the as-of bar.")

st.caption("Cell text = start time (ET) of the last closed bar. Colour: green = Bullish, "
           "red = Bearish, grey = Neutral. Exact values are in the detail table below.")

with st.expander("Detail (latest closed bar per timeframe)"):
    view = detail.assign(bar_time=detail["bar_time"].dt.strftime("%m-%d %H:%M"))
    view = view[["symbol", "timeframe", "bar_time", "close", "macd", "macd_signal", "macd_hist",
                 "macd_trend", "rsi", "rsi_signal", "momentum", "rsi_trend", "overall_bias"]]
    view.columns = ["Symbol", "Timeframe", "Bar start (ET)", "Close", "MACD", "MACD signal",
                    "MACD hist", "MACD trend", "RSI", "RSI signal", "RSI momentum",
                    "RSI trend", "Overall bias"]
    order = {t: i for i, t in enumerate(TF_LABELS)}
    view = view.sort_values(["Symbol", "Timeframe"], key=lambda s: s.map(order) if s.name == "Timeframe" else s)
    st.dataframe(view, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Debug: 5-minute bar values
# --------------------------------------------------------------------------
with st.expander("5-min bars (debug)"):
    dbg_syms = [sy for sy in symbols if sy in frames5m and not frames5m[sy].empty]
    if not dbg_syms:
        st.info("No 5-min bars available.")
    else:
        c1, c2, c3 = st.columns([2, 2, 2], vertical_alignment="bottom")
        dbg_sym = c1.selectbox("Symbol", dbg_syms, key="bias_dbg_symbol")
        dbg_n = c2.slider("Last N bars", 5, BARS_KEPT, 30, key="bias_dbg_n")
        to_console = c3.checkbox("Also print to console", value=False, key="bias_dbg_print")

        bars = frames5m[dbg_sym].tail(dbg_n)
        if to_console:  # goes to the terminal running `streamlit run`
            print(f"[{dbg_sym} 5m] last {len(bars)} bars (ET):")
            print(bars.to_string())

        view5 = bars.drop(columns=[c for c in ("interval", "symbol") if c in bars.columns]).copy()
        f32 = view5.select_dtypes("float32").columns
        view5[f32] = view5[f32].astype("float64")  # avoids 65.599998-style display noise
        view5.insert(0, "Bar start (ET)", view5.index.strftime("%m-%d %H:%M"))
        st.dataframe(view5.round(2), use_container_width=True, hide_index=True)
        st.caption("Closed 5-min bars, newest last. In debug (as-of) mode the table ends at the as-of time.")