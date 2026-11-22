"""
pages/3_Technical_Bias.py

MACD / RSI bias for a list of symbols across 5m, 15m, 30m, 1h and 4h.
Bars come from service.alaDataManager (5-min bars, closed bars only); the
larger timeframes are resampled from them (4h bars start at 01/05/09/13/17/21 ET).
Calculation rules live in service/biasCalculator.py (ported from dataAnalyzer.py).

Page order: MACD/RSI trend tables -> price levels (PDH/PDL, pre-market, opening range, ...)
-> 0DTE put-call ratio panel -> key option strikes
-> next-3-candles projection -> detail / debug tables.

Debug mode: tick "Debug: as-of time" to see the bias exactly as it would have
looked at a past Eastern date/time (only bars closed by then are used).
"""
import datetime as dt
import html
import re

import pandas as pd
import altair as alt
import streamlit as st

from service.alaDataManager import EASTERN, fetch_stock_data, resample_bars
from service.levels import build_levels, levels_summary
from service.optionLevels import RANK_MODES, TOP_N, load_level_table, top_strikes
from service.pcrAnalysis import ZERO_DTE_SCOPE, load_pcr_snapshots, pcr_change_info, pcr_reversal_info
from service.pcrView import render_change_metrics
from service.biasCalculator import (
    MIN_BARS,
    add_atr,
    add_vwap,
    compute_bias,
    drop_unfinished_buckets,
    latest_snapshot,
    project_next_candles,
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
OPTIONS_SYMBOL = "SPY"   # the put-call-ratio and key-strike panels are shown for this symbol only

COLORS = {"Bullish": "#1a9850",
          "Bearish": "#d73027",
          "Neutral": "#6b6b6b"}

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
if "bias_refresh_key" not in st.session_state:
    st.session_state.bias_refresh_key = 0

ctl_symbols, ctl_refresh = st.columns([5, 1], vertical_alignment="bottom")
raw_symbols = ctl_symbols.text_input("Symbols (comma-separated)", value="SPY, GLD, IWM, QQQ")
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
                     lookback_days: int) -> tuple[list[dict], list[str], pd.DataFrame, dict | None]:
    """One 5-min fetch per symbol; every other timeframe is resampled from it.
    refresh_key only exists so the Refresh button yields a new cache entry.
    asof_iso (debug): ISO timestamp with offset; only 5-min bars that had closed
    by then are kept, so the result is the bias as it stood at that moment.
    Returns (rows, skipped-timeframe messages, last BARS_KEPT 5-min bars with indicators,
    price levels {"sr": SupportResistance, "last_close": float} or None)."""
    base = fetch_stock_data(symbol, "5min", lookback_days=lookback_days)
    if asof_iso:
        base = base[base.index + pd.Timedelta("5min") <= pd.Timestamp(asof_iso)]
        if base.empty:
            raise ValueError(f"no bars closed before {asof_iso} in the fetched history")
    base_last_end = base.index[-1] + pd.Timedelta("5min")

    rows, notes = [], []
    frame5m = pd.DataFrame()

    # Price levels (service/levels.py) from the FULL fetched history (not the 300-bar tail),
    # as of the end of the last closed bar, so debug as-of mode shows the levels of that moment.
    levels_info = None
    try:
        # Trading day = the as-of date in debug mode (not the last bar's date), so a past
        # moment only uses families whose data exists by then; live uses the last bar's date.
        level_day = pd.Timestamp(asof_iso).date() if asof_iso else base.index[-1].date()
        sr = build_levels(base, level_day, base_last_end)
        levels_info = {"sr": sr, "last_close": round(float(base["close"].iloc[-1]), 2)}
    except Exception as e:  # noqa: BLE001
        notes.append(f"{symbol} levels unavailable: {e}")
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
            # VWAP/ATR on the full frame so the VWAP cumulation starts at the session open.
            bias_frame = add_vwap(add_atr(bias_frame))
            frame5m = bias_frame.tail(BARS_KEPT)
        snap = latest_snapshot(bias_frame)
        rows.append({"symbol": symbol, "timeframe": label,
                     "data_through": base_last_end, **snap})
    return rows, notes, frame5m, levels_info


# History must reach back LOOKBACK_DAYS before the as-of time, not just before today.
extra_days = (now_et.date() - asof_et.date()).days + 1 if asof_et is not None else 0
lookback_days = LOOKBACK_DAYS + extra_days
asof_iso = asof_et.isoformat() if asof_et is not None else None

all_rows, notes, errors, frames5m, levels_by_symbol = [], [], [], {}, {}
with st.spinner("Fetching bars and calculating bias..."):
    for sym in symbols:
        try:
            rows, skipped, f5m, lvl_info = load_symbol_bias(sym, st.session_state.bias_refresh_key,
                                                            asof_iso, lookback_days)
            frames5m[sym] = f5m
            levels_by_symbol[sym] = lvl_info
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

col_macd, col_rsi = st.columns(2)
with col_macd:
    show_pivot("MACD trend", "macd_trend")
with col_rsi:
    show_pivot("RSI trend", "rsi_trend")
st.caption("Cell text = start time (ET) of the last closed bar. Colour: green = Bullish, "
           "red = Bearish, grey = Neutral. Exact values are in the detail table below.")


# --------------------------------------------------------------------------
# Price levels (logic: service/levels.py)
# --------------------------------------------------------------------------
def _beyond_note(summ: dict, bias: str) -> str:
    """Hint when price is above/below every named level (extension levels are already in the row)."""
    if not (summ["above_all"] or summ["below_all"]) or not summ["edge"]:
        return ""
    lbl, px = summ["edge"]
    up = summ["above_all"]
    if bias == ("Bullish" if up else "Bearish"):
        return f"{'Above' if up else 'Below'} all levels, 5m {bias.lower()}: continuation; stop ref {lbl} {px:.2f}"
    return (f"{'Above' if up else 'Below'} all levels, 5m {bias.lower()}: stretched; "
            f"watch retest of {lbl} {px:.2f}")


def show_price_levels() -> None:
    st.subheader("Price levels")
    rows = []
    for sym in symbols:
        info = levels_by_symbol.get(sym)
        if not info:
            continue
        summ = levels_summary(info["sr"], info["last_close"])
        b = detail.loc[(detail["symbol"] == sym) & (detail["timeframe"] == "5 min"), "overall_bias"]
        rows.append({"Symbol": sym, "Price": f"{info['last_close']:,.2f}", "Levels": summ["text"],
                     "Note": _beyond_note(summ, b.iloc[0] if not b.empty else "Neutral")})
    if not rows:
        st.info("No price levels available.")
        return
    table = pd.DataFrame(rows)
    if not table["Note"].any():
        table = table.drop(columns="Note")
    # One line per symbol: HTML table with nowrap cells; scrolls sideways instead of wrapping.
    head = "".join(f"<th>{html.escape(c)}</th>" for c in table.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                   for r in table.itertuples(index=False))
    st.markdown(
        "<style>.lvl-wrap{overflow-x:auto}"
        ".lvl{border-collapse:collapse;font-size:0.9rem;width:100%}"
        ".lvl th,.lvl td{white-space:nowrap;padding:4px 12px;text-align:left;"
        "border-bottom:1px solid rgba(128,128,128,.3)}"
        ".lvl th{font-weight:600}</style>"
        f"<div class='lvl-wrap'><table class='lvl'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True)
    if asof_et is not None:   # debug: say exactly which level families the as-of data supports
        avail = list(dict.fromkeys(k for s_ in symbols if levels_by_symbol.get(s_)
                                   for k in levels_by_symbol[s_]["sr"].labeled() if k != "P"))
        st.caption(f"Debug as-of {asof_et:%Y-%m-%d %H:%M} ET: only bars closed by then are used. "
                   f"Levels available: {', '.join(avail) or '-'}.")
    st.caption("Price = last closed 5-min bar. R = resistance above, S = support below, nearest first "
               "(2 per side before 10:00 ET, 3 after), listed in order R1, R2, R3 | S1, S2, S3; levels within 0.03% are merged (e.g. PDL+PML). "
               "PDH/PDL/PDC = previous regular session; PMH/PML = pre-market (from 09:30); "
               "ORH/ORL = 09:30-10:00 range (from 10:00); DH/DL = day high/low incl. pre-market (from 13:00). "
               "Beyond all levels, R/S are topped up with PR2/PS2 (pivot R2/S2), 5DH/5DL (5-day high/low), "
               "ORM+/ORM- (opening-range measured move). Context only, not trade advice.")

show_price_levels()


# --------------------------------------------------------------------------
# 0DTE put-call ratio + key strikes (SPY only; omitted when it isn't in the input field)
# Logic: service/pcrAnalysis.py, service/optionLevels.py
# --------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_pcr_panel(symbol: str, refresh_key: int, day_iso: str, until_iso: str | None) -> dict:
    """Reversal info, change-vs-previous info and latest ratio for one symbol.
    refresh_key only busts the cache."""
    df = load_pcr_snapshots(symbol, dt.date.fromisoformat(day_iso), ZERO_DTE_SCOPE,
                            until=pd.Timestamp(until_iso) if until_iso else None)
    if df.empty:   # no snapshots stored yet: the analysis helpers need the columns to exist
        df = pd.DataFrame(columns=["downloaded_at", "put_call_ratio"])
    valid = df.dropna(subset=["put_call_ratio"])
    return {"reversal": pcr_reversal_info(df),
            "change": pcr_change_info(df),
            "latest": float(valid["put_call_ratio"].iloc[-1]) if not valid.empty else None,
            # stock price stored with the latest ratio snapshot (same moment as "Latest PCR")
            "spot": float(valid["spot"].iloc[-1]) if "spot" in valid.columns and not valid.empty
                    and pd.notna(valid["spot"].iloc[-1]) else None}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_key_levels(symbol: str, refresh_key: int) -> dict:
    """Live 0DTE strike table (volume / OI / GEX). refresh_key only busts the cache."""
    return load_level_table(symbol)


def show_pcr_panel(symbol: str) -> None:
    st.subheader(f"{symbol} 0DTE put-call ratio: reversal & change")
    pcr_day = (asof_et or pd.Timestamp(now_et)).date()
    try:
        p = load_pcr_panel(symbol, st.session_state.bias_refresh_key, pcr_day.isoformat(), asof_iso)
    except Exception as e:  # noqa: BLE001
        st.warning(f"{symbol} PCR: {e}")
        return

    r = p["reversal"]
    fresh = r["on_last"]   # reversal confirmed by the latest snapshot
    pcr_df = pd.DataFrame([{"Symbol": symbol,
                            "Reversal": r["state"] if fresh else "No change",
                            "PCR at turn": r["value"] if fresh else None,
                            "Turn time (ET)": pd.Timestamp(r["time"]).strftime("%H:%M") if fresh else "-",
                            "Latest PCR": p["latest"],
                            f"{symbol} price": p["spot"]}])
    pcr_color = {"Up→Down": COLORS["Bullish"], "Down→Up": COLORS["Bearish"]}
    st.dataframe(
        pcr_df.style.map(lambda v: f"background-color: {pcr_color[v]}; color: white; font-weight: 600"
                         if v in pcr_color else "", subset=["Reversal"])
        .format({"PCR at turn": "{:.4f}", "Latest PCR": "{:.4f}", f"{symbol} price": "${:,.2f}"}, na_rep="-"),
        use_container_width=True, hide_index=True)
    render_change_metrics(p["change"])


def show_key_strikes(symbol: str) -> None:
    st.subheader(f"{symbol} 0DTE key strikes (top {TOP_N} calls / puts)")
    rank_by = st.radio("Rank by", RANK_MODES, horizontal=True, key="bias_levels_rank",
                       help="Combined = 50% volume + 50% absolute GEX, each scaled to the largest strike.")
    try:
        lv = load_key_levels(symbol, st.session_state.bias_refresh_key)
    except Exception as e:  # noqa: BLE001
        st.warning(f"{symbol} option levels: {e}")
        return

    # Compact diverging bar chart: calls up (green), puts down (red), spot as a dashed vertical line.
    metric = "GEX ($mm)" if rank_by == "GEX" else "Volume"
    rows, tables = [], {}
    for side, label in (("call", "Call"), ("put", "Put")):
        top = top_strikes(lv["by_strike"], lv["spot"], side, TOP_N, rank_by)
        tables[label] = top
        for r in top.to_dict("records"):
            rows.append({"Strike": float(r["Strike"]), "Side": label,
                         "Bar": abs(float(r[metric])) * (1 if side == "call" else -1),
                         "Volume": float(r["Volume"]), "OI": float(r["Open int."]),
                         "GEX": float(r["GEX ($mm)"])})
    if not rows:
        st.info("No call / put volume or GEX found.")
    else:
        chart_df, spot = pd.DataFrame(rows), float(lv["spot"])
        strikes = chart_df["Strike"].tolist() + [spot]
        pad = max(0.5, (max(strikes) - min(strikes)) * 0.08)
        xs = alt.X("Strike:Q", title=None, scale=alt.Scale(domain=[min(strikes) - pad, max(strikes) + pad], nice=False),
                   axis=alt.Axis(values=sorted(set(chart_df["Strike"])), format="~g", labelFontSize=10, grid=False))
        base = alt.Chart(chart_df).encode(x=xs)
        bars = base.mark_bar(size=18).encode(
            y=alt.Y("Bar:Q", title=metric.split(" ")[0], axis=alt.Axis(labelExpr="format(abs(datum.value), '~s')",
                                                                        labelFontSize=10, titleFontSize=10, tickCount=3)),
            color=alt.Color("Side:N", legend=None,
                            scale=alt.Scale(domain=["Call", "Put"], range=[COLORS["Bullish"], COLORS["Bearish"]])),
            tooltip=[alt.Tooltip("Strike:Q", format="~g"), "Side:N", alt.Tooltip("Volume:Q", format=",.0f"),
                     alt.Tooltip("OI:Q", title="Open int.", format=",.0f"), alt.Tooltip("GEX:Q", title="GEX ($mm)", format=",.1f")])
        spot_df = pd.DataFrame({"s": [spot]})
        rule = alt.Chart(spot_df).mark_rule(color="#f5a623", strokeDash=[4, 3], strokeWidth=2).encode(x="s:Q")
        tag = alt.Chart(spot_df).mark_text(color="#f5a623", align="left", dx=4, fontSize=11, fontWeight="bold",
                                           y=0).encode(x="s:Q", text=alt.value(f"${spot:,.2f}"))
        zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#888", strokeWidth=1).encode(y="z:Q")
        st.altair_chart((zero + bars + rule + tag).properties(height=140), use_container_width=True)
        with st.expander("Strike table", expanded=False):
            fmt = {"Volume": "{:,.0f}", "Open int.": "{:,.0f}", "GEX ($mm)": "{:,.1f}"}
            col_calls, col_puts = st.columns(2)
            for col, label in ((col_calls, "Call"), (col_puts, "Put")):
                with col:
                    st.markdown(f"**{label}s**")
                    if not tables[label].empty:
                        st.dataframe(tables[label].style.format(fmt), use_container_width=True, hide_index=True)

    as_of_et = pd.Timestamp(lv["as_of"], tz="UTC").tz_convert(EASTERN)
    st.caption(f"{lv['exp_label']} expiration ({', '.join(lv['expirations'])}) · spot ${lv['spot']:,.2f} · "
               f"chain as of {as_of_et:%H:%M} ET. Bars = {metric}; orange line = spot.")


if OPTIONS_SYMBOL in symbols:
    show_pcr_panel(OPTIONS_SYMBOL)
    if not debug_on:   # the option chain is always live, so it is hidden when viewing a past as-of time
        show_key_strikes(OPTIONS_SYMBOL)

# --------------------------------------------------------------------------
# Next 3 candles (5 min) - heuristic scenario (logic: biasCalculator.project_next_candles)
# --------------------------------------------------------------------------
st.subheader("Next 3 candles (5 min projection)")
pred_rows = []
for sym in [s for s in symbols if s in frames5m and not frames5m[s].empty]:
    proj, _ctx = project_next_candles(frames5m[sym], n=3)
    if proj.empty:
        continue
    last_bar = frames5m[sym].index[-1]
    for j, row in enumerate(proj.itertuples(index=False), start=1):
        pred_rows.append({"Symbol": sym,
                          "Bar start (ET)": (last_bar + pd.Timedelta(minutes=5 * j)).strftime("%m-%d %H:%M"),
                          "Open": row.open, "High": row.high, "Low": row.low, "Close": row.close,
                          "P(close > now)": f"{row.p_up:.0%}", "Lean": row.lean,
                          "Cone low": row.cone_low, "Cone high": row.cone_high})
if pred_rows:
    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
    st.caption("Heuristic scenario from recent drift, ATR and the pull toward session VWAP. Not a forecast.")
else:
    st.info("Not enough 5-min bars for a projection.")

with st.expander("Detail (latest closed bar per timeframe)"):
    view = detail.assign(bar_time=detail["bar_time"].dt.strftime("%m-%d %H:%M"))
    view = view[["symbol", "timeframe", "bar_time", "close", "macd", "macd_signal", "macd_hist",
                 "macd_trend"]]
    view.columns = ["Symbol", "Timeframe", "Bar start (ET)", "Close", "MACD", "MACD signal",
                    "MACD hist", "MACD trend"]
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

        # RSI (and the columns derived from it) are left out of this table; MACD, ATR and VWAP stay.
        hide = ("interval", "symbol", "rsi", "rsignal", "momentum", "RSITrend", "OverallBias")
        view5 = bars.drop(columns=[c for c in hide if c in bars.columns]).copy()
        f32 = view5.select_dtypes("float32").columns
        view5[f32] = view5[f32].astype("float64")  # avoids 65.599998-style display noise
        view5.insert(0, "Bar start (ET)", view5.index.strftime("%m-%d %H:%M"))
        st.dataframe(view5.round(2), use_container_width=True, hide_index=True)
        st.caption("Closed 5-min bars, newest last. In debug (as-of) mode the table ends at the as-of time.")
