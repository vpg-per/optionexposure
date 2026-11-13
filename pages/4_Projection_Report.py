"""
pages/4_Projection_Report.py

Accuracy report for the "next 3 candles (5 min)" projection used on
pages/3_Technical_Bias.py, for ONE symbol and a selected date (optionally
pooled with the previous N trading days to get a usable sample size).

How it works
  * One 5-min fetch (service.alaDataManager) and ONE compute_bias() + ATR + session VWAP
    over the whole history. All indicators are causal (EMA / rolling / cumulative from
    the session open), so slicing the full frame at bar i gives exactly what the page-3
    debug "as-of" mode would have shown.
  * For every as-of time in the chosen window (default 09:00-15:45 ET, so bar 3
    ends by 16:00) the projection is rebuilt from bars closed by then and compared
    with the three bars that actually followed.
  * Results are cut by time slot, horizon, confidence, probability calibration, MACD
    alignment, VWAP state, and day.

The projection is imported from service.biasCalculator (the same function page 3 uses),
so the report always scores the live logic: drift + VWAP pull, P(up), ATR cone and
direction-leaning wicks. RSI is not part of the projection any more.
"""
import datetime as dt
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from service.alaDataManager import EASTERN, fetch_stock_data
from service.biasCalculator import (
    CONE_ATR_MULT,
    VWAP_ANCHOR,
    VWAP_STRETCH,
    add_atr,
    add_vwap,
    compute_bias,
    project_next_candles,
)

st.set_page_config(page_title="Projection report", layout="wide")
st.title("Next-3-candle projection report")

WARMUP_DAYS = 20        # history before the earliest report day so MACD/ATR have settled
CACHE_TTL = 300         # seconds
BASE_MIN = 5            # base bar length (minutes)
HORIZONS = (1, 2, 3)
WINDOW_BARS = 30        # bars handed to the projection (it needs >= 12)
CONF_BINS = [0, 0.05, 0.10, 0.20, 0.40, np.inf]
CONF_LABELS = ["0 (<0.05)", "1 (0.05-0.10)", "2 (0.10-0.20)", "3 (0.20-0.40)", "4 (>=0.40)"]
P_BINS = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.0001]
P_LABELS = ["0 (<40%)", "1 (40-45%)", "2 (45-50%)", "3 (50-55%)", "4 (55-60%)", "5 (60-70%)", "6 (>=70%)"]
GOOD, BAD, MID = "#1a9850", "#d73027", "#6b6b6b"


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
if "proj_rep_refresh" not in st.session_state:
    st.session_state.proj_rep_refresh = 0

now_et = dt.datetime.now(EASTERN)

c1, c2, c3, c4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
raw_symbol = c1.text_input("Symbol", value="SPY")
report_date = c2.date_input("Report date (ET)", value=now_et.date(), max_value=now_et.date())
extra_days = c3.number_input("Also pool previous trading days", min_value=0, max_value=30, value=0,
                             help="One day gives only ~6 samples per 30-min slot. Pool more days "
                                  "for statistically meaningful slot results.")
if c4.button("Refresh", type="primary", use_container_width=True):
    st.session_state.proj_rep_refresh += 1

c5, c6, c7, c8 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
start_t = c5.time_input("As-of from (ET)", value=dt.time(9, 0), step=900)
end_t = c6.time_input("Bar 3 must end by (ET)", value=dt.time(16, 0), step=900)
slot_min = c7.selectbox("Slot size (min)", [15, 30, 60], index=1)
min_n = c8.number_input("Min samples to colour a cell", min_value=1, max_value=200, value=8)
exclude_gap = st.checkbox("Exclude samples where the 5-bar drift window spans a data gap "
                          "(session break / missing bars)", value=False)

tokens = [s for s in re.split(r"[\s,]+", raw_symbol.upper()) if s]
if not tokens:
    st.info("Enter a symbol.")
    st.stop()
symbol = tokens[0]
if len(tokens) > 1:
    st.caption(f"One symbol at a time - using {symbol}.")

start_m = start_t.hour * 60 + start_t.minute
end_m = end_t.hour * 60 + end_t.minute
if end_m - start_m < 15 + BASE_MIN:
    st.error("The window must be at least 20 minutes long.")
    st.stop()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_frame(sym: str, refresh_key: int, lookback_days: int) -> pd.DataFrame:
    """5-min closed bars + MACD/RSI/bias, ATR and session-VWAP columns, computed once over
    the whole history (VWAP must cumulate from the session open, so never compute it on a slice).
    refresh_key only exists so the Refresh button yields a new cache entry."""
    base = fetch_stock_data(sym, "5min", lookback_days=lookback_days)
    return add_vwap(add_atr(compute_bias(base)))


lookback_days = WARMUP_DAYS + (now_et.date() - report_date).days + int(extra_days * 1.6) + 5
try:
    with st.spinner(f"Fetching {symbol} bars..."):
        frame = load_frame(symbol, st.session_state.proj_rep_refresh, lookback_days)
except Exception as e:  # noqa: BLE001
    st.error(f"{symbol}: {e}")
    st.stop()
if frame.empty:
    st.error(f"No bars returned for {symbol}.")
    st.stop()


# --------------------------------------------------------------------------
# Sample builder: one projection per as-of time, scored against the next 3 bars
# --------------------------------------------------------------------------
def _sign_hit(a: float, b: float) -> float:
    """1.0 if same sign, 0.0 if opposite, NaN when either side has no direction."""
    if a == 0 or b == 0 or a != a or b != b:
        return np.nan
    return float(np.sign(a) == np.sign(b))


def build_samples(df: pd.DataFrame, day: dt.date, start_m: int, end_m: int) -> list[dict]:
    idx = df.index
    step = pd.Timedelta(minutes=BASE_MIN)
    rows: list[dict] = []
    for i in np.flatnonzero(idx.date == day):
        if i < WINDOW_BARS or i + 3 >= len(df):
            continue
        as_of = idx[i] + step                       # the moment the last bar closed
        m = as_of.hour * 60 + as_of.minute
        if m < start_m or m > end_m - 3 * BASE_MIN:
            continue
        if list(idx[i + 1:i + 4]) != [as_of + step * k for k in range(3)]:
            continue                                # missing/irregular future bars

        window = df.iloc[i - WINDOW_BARS + 1:i + 1]
        proj, ctx = project_next_candles(window, n=3)
        if proj.empty:
            continue

        close = window["close"].astype("float64")
        last_close = float(close.iloc[-1])
        atr = float(ctx["atr"])                     # the scale the projection itself used
        last_dir = float(np.sign(close.iloc[-1] - close.iloc[-2]))
        gap = (idx[i] - idx[i - 5]) != step * 5

        z = ctx["vwap_dist_atr"]
        if z is None:
            vwap_state = "No VWAP"
        elif z >= VWAP_STRETCH:
            vwap_state = "Stretched above VWAP"
        elif z <= -VWAP_STRETCH:
            vwap_state = "Stretched below VWAP"
        else:
            vwap_state = "Near VWAP"
        macd_trend = str(df["MACDTrend"].iloc[i])
        slot_start = (m // slot_min) * slot_min

        for j in HORIZONS:
            p = proj.iloc[j - 1]
            a = df.iloc[i + j]
            pm = float(p["close"]) - last_close     # projected move vs as-of close
            am = float(a["close"]) - last_close     # actual move vs as-of close
            p_up = float(p["p_up"])
            up = np.nan if am == 0 else float(am > 0)   # flat outcomes carry no direction
            if macd_trend == "Neutral" or pm == 0:
                rel = "Neutral MACD"
            elif (macd_trend == "Bullish") == (pm > 0):
                rel = "Aligned with MACD"
            else:
                rel = "Opposed to MACD"
            rows.append({
                "date": str(day),
                "as_of": f"{as_of.hour:02d}:{as_of.minute:02d}",
                "slot": f"{slot_start // 60:02d}:{slot_start % 60:02d}",
                "h": j,
                "last_close": last_close,
                "proj_close": float(p["close"]),
                "act_close": float(a["close"]),
                "proj_move": round(pm, 3),
                "act_move": round(am, 3),
                "atr": atr,
                "hit": _sign_hit(pm, am),
                "base_hit": _sign_hit(last_dir, am),           # "repeat last bar" baseline
                "candle_hit": _sign_hit(float(p["close"]) - float(p["open"]),
                                        float(a["close"]) - float(a["open"])),
                "err": abs(float(p["close"]) - float(a["close"])),
                "hi_ok": float(float(a["high"]) <= float(p["high"])),
                "lo_ok": float(float(a["low"]) >= float(p["low"])),
                "conf": abs(pm) / atr if atr else np.nan,
                "p_up": round(p_up, 3),
                "lean": str(p["lean"]),
                "up": up,
                "brier_sq": (p_up - up) ** 2 if up == up else np.nan,
                "cone_low": float(p["cone_low"]),
                "cone_high": float(p["cone_high"]),
                "cone_ok": float(float(p["cone_low"]) <= float(a["close"]) <= float(p["cone_high"])),
                "vwap": ctx["vwap"],
                "vwap_dist_atr": z,
                "vwap_state": vwap_state,
                "macd_trend": macd_trend,
                "bias_rel": rel,
                "gap_in_drift": bool(gap),
            })
    return rows


# Pick the report day plus the previous N trading days that actually have samples.
dates = sorted({d for d in frame.index.date if d <= report_date}, reverse=True)
all_rows: list[dict] = []
used_days: list[dt.date] = []
for d in dates:
    if len(used_days) > extra_days:
        break
    day_rows = build_samples(frame, d, start_m, end_m)
    if day_rows:
        all_rows.extend(day_rows)
        used_days.append(d)

if not all_rows:
    st.warning("No complete projection samples for that date/window (weekend, holiday, or the "
               "following bars are not available yet).")
    st.stop()
if used_days[0] != report_date:
    st.info(f"No usable bars on {report_date}; the most recent trading day with data is "
            f"{used_days[0]}.")

samples = pd.DataFrame(all_rows)
if exclude_gap:
    samples = samples[~samples["gap_in_drift"]]
    if samples.empty:
        st.warning("Every sample was excluded by the gap filter.")
        st.stop()
samples["conf_band"] = pd.cut(samples["conf"], CONF_BINS, labels=CONF_LABELS, right=False).astype(str)
samples["p_band"] = pd.cut(samples["p_up"], P_BINS, labels=P_LABELS, right=False).astype(str)
if (samples["vwap_state"] == "No VWAP").all():
    st.warning("No VWAP in this data (no volume returned). The projection ran on drift only and the "
               "'By VWAP state' tab is empty of information.")


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half) * 100, (centre + half) * 100


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        valid = g["hit"].dropna()
        n = len(valid)
        lo, hi = wilson(float(valid.sum()), n)
        base = g["base_hit"].dropna()
        mae = g["err"].mean()
        rw = g["act_move"].abs().mean()           # random walk: next close = as-of close
        rec = dict(zip(keys, key))
        rec.update({
            "n": n,
            "hit": valid.mean() * 100 if n else np.nan,
            "ci_lo": lo, "ci_hi": hi,
            "base_hit": base.mean() * 100 if len(base) else np.nan,
            "candle_hit": g["candle_hit"].mean() * 100 if g["candle_hit"].notna().any() else np.nan,
            "mae_r": (g["err"] / g["atr"]).mean(),
            "rw_mae_r": (g["act_move"].abs() / g["atr"]).mean(),
            "skill": (1 - mae / rw) if rw > 0 else np.nan,
            "hi_ok": g["hi_ok"].mean() * 100,
            "lo_ok": g["lo_ok"].mean() * 100,
            "cone_ok": g["cone_ok"].mean() * 100,
            "brier": g["brier_sq"].mean(),
        })
        out.append(rec)
    return pd.DataFrame(out)


def calibration(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Predicted P(up) vs how often the close really finished above the as-of close
    (flat outcomes excluded). Well calibrated = the two columns match."""
    out = []
    for key, g in df.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        v = g.dropna(subset=["up"])
        rec = dict(zip(keys, key))
        avg_p = v["p_up"].mean() * 100 if len(v) else np.nan
        act = v["up"].mean() * 100 if len(v) else np.nan
        rec.update({"n (non-flat)": len(v), "Avg P(up) %": avg_p, "Actual up %": act,
                    "Over(+)/under(-) confidence, pts": avg_p - act})
        out.append(rec)
    return pd.DataFrame(out).round(1)


PRETTY = {"n": "n (directional)", "hit": "Hit %", "ci_lo": "CI low %", "ci_hi": "CI high %",
          "base_hit": "Last-bar-dir hit %", "candle_hit": "Candle-dir hit %",
          "mae_r": "MAE (x ATR)", "rw_mae_r": "Random-walk MAE (x ATR)",
          "skill": "Skill vs random walk", "hi_ok": "High contained %", "lo_ok": "Low contained %",
          "cone_ok": "Cone coverage %", "brier": "Brier (0.25 = coin flip)",
          "h": "Bar #", "slot": "Slot (ET)", "date": "Date"}


def tidy(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.rename(columns=PRETTY).round(
        {"Hit %": 1, "CI low %": 1, "CI high %": 1, "Last-bar-dir hit %": 1, "Candle-dir hit %": 1,
         "MAE (x ATR)": 3, "Random-walk MAE (x ATR)": 3, "Skill vs random walk": 3,
         "High contained %": 1, "Low contained %": 1, "Cone coverage %": 1,
         "Brier (0.25 = coin flip)": 3})


def _elementwise(frame_: pd.DataFrame, fn) -> pd.DataFrame:
    return frame_.map(fn) if hasattr(frame_, "map") else frame_.applymap(fn)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
days_txt = ", ".join(str(d) for d in sorted(used_days))
st.caption(f"**{symbol}** | days: {days_txt} | window {start_t:%H:%M}-{end_t:%H:%M} ET | "
           f"{samples['as_of'].nunique()} distinct as-of times, {len(samples) // 3} projections "
           f"({len(samples)} bar-level scores).")
if len(used_days) == 1:
    st.warning("Single-day report: slot-level results rest on very few samples and are mostly noise. "
               "Use 'Also pool previous trading days' for conclusions.")

by_h = summarize(samples, ["h"])

st.subheader("Overall by horizon")
mcols = st.columns(3)
for col, (_, r) in zip(mcols, by_h.iterrows()):
    delta = r["hit"] - r["base_hit"] if pd.notna(r["base_hit"]) else None
    col.metric(f"Bar {int(r['h'])} direction hit rate",
               f"{r['hit']:.1f}%" if pd.notna(r["hit"]) else "n/a",
               None if delta is None else f"{delta:+.1f} pts vs last-bar-direction baseline",
               help=f"95% CI {r['ci_lo']:.1f}-{r['ci_hi']:.1f}%, n={int(r['n'])}. Skill vs random walk "
                    f"= {r['skill']:.3f} (>0 = smaller close error than 'price stays put'). "
                    f"Cone coverage {r['cone_ok']:.0f}% (about 90% if the cone is right-sized). "
                    f"Brier {r['brier']:.3f} (below 0.25 = better than a coin flip).")
st.dataframe(tidy(by_h), use_container_width=True, hide_index=True)
st.caption("Hit % = projected close is on the same side of the as-of close as the actual close "
           "(flat outcomes excluded). Skill = 1 - MAE / random-walk MAE. If hit % is inside the "
           "CI around 50% and skill is ~0, there is no demonstrable edge. Cone coverage = share of "
           f"actual closes inside projected close +/- {CONE_ATR_MULT:g} x ATR x sqrt(bars ahead). "
           "Brier = mean squared error of P(up) against the actual up/down outcome.")

# ---- slot heatmap -----------------------------------------------------------
st.subheader(f"By {slot_min}-minute slot (as-of time)")
by_slot = summarize(samples, ["slot", "h"])
hit_p = by_slot.pivot(index="slot", columns="h", values="hit")
n_p = by_slot.pivot(index="slot", columns="h", values="n")


def _cell_css(hit_v, n_v):
    if pd.isna(hit_v) or n_v < min_n:
        return "color: #9a9a9a; text-align: center"
    c = GOOD if hit_v >= 55 else BAD if hit_v <= 45 else MID
    return f"background-color: {c}; color: white; font-weight: 600; text-align: center"


labels = hit_p.copy().astype(object)
css = hit_p.copy().astype(object)
for s in hit_p.index:
    for h in hit_p.columns:
        v, n_v = hit_p.loc[s, h], n_p.loc[s, h]
        labels.loc[s, h] = "-" if pd.isna(v) else f"{v:.0f}% (n={int(n_v)})"
        css.loc[s, h] = _cell_css(v, n_v)
labels.columns = [f"Bar {h}" for h in labels.columns]
css.columns = labels.columns
labels.index.name = None
st.dataframe(labels.style.apply(lambda _: css, axis=None), use_container_width=True)
st.caption(f"Green >= 55%, red <= 45%, grey in between; cells with n < {min_n} are left uncoloured. "
           "Numbers are direction hit rates.")

bar_df = hit_p.copy()
bar_df.columns = [f"Bar {h}" for h in bar_df.columns]
st.bar_chart(bar_df, height=260)

with st.expander("Slot detail (hit rate, CI, skill, containment)"):
    st.dataframe(tidy(by_slot), use_container_width=True, hide_index=True)

# ---- other cuts ---------------------------------------------------------------
st.subheader("Where is the projection more reliable?")
t_conf, t_prob, t_macd, t_vwap, t_day = st.tabs(
    ["By confidence", "By probability", "By MACD alignment", "By VWAP state", "By day"])
with t_conf:
    st.caption("Confidence = |projected move| / ATR(14). If hit % does not rise with "
               "confidence, the size of the projected move carries no information.")
    st.dataframe(tidy(summarize(samples, ["conf_band", "h"]).rename(columns={"conf_band": "Confidence band"})),
                 use_container_width=True, hide_index=True)
with t_prob:
    st.caption("Calibration: in each P(up) band, how often did the close actually finish above the "
               "as-of close? Avg P(up) and Actual up % should match; a large positive gap means the "
               "model is over-confident.")
    st.dataframe(calibration(samples, ["p_band", "h"]).rename(columns={"p_band": "P(up) band"}),
                 use_container_width=True, hide_index=True)
    st.caption("By lean (Up / Down = P(up) at least 5 points away from 50%, otherwise Neutral).")
    st.dataframe(tidy(summarize(samples, ["lean", "h"]).rename(columns={"lean": "Lean"})),
                 use_container_width=True, hide_index=True)
with t_macd:
    st.caption("Projected direction vs the 5-min MACD trend at the as-of bar.")
    st.dataframe(tidy(summarize(samples, ["bias_rel", "h"]).rename(columns={"bias_rel": "Direction vs MACD"})),
                 use_container_width=True, hide_index=True)
with t_vwap:
    st.caption(f"Price vs session VWAP (anchored {VWAP_ANCHOR:%H:%M} ET) at the as-of bar. 'Stretched' = "
               f"{VWAP_STRETCH:g} ATR or more away, where the projection adds a pull back toward VWAP.")
    st.dataframe(tidy(summarize(samples, ["vwap_state", "h"]).rename(columns={"vwap_state": "VWAP state"})),
                 use_container_width=True, hide_index=True)
with t_day:
    if samples["date"].nunique() < 2:
        st.info("Pool more than one day to compare days.")
    else:
        st.dataframe(tidy(summarize(samples, ["date", "h"])), use_container_width=True, hide_index=True)

# ---- raw samples ------------------------------------------------------------------
with st.expander("All scored samples"):
    show = samples.drop(columns=["conf_band", "p_band"]).round(3)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", show.to_csv(index=False).encode("utf-8"),
                       file_name=f"projection_report_{symbol}_{max(used_days)}.csv", mime="text/csv")

st.caption("Indicators (MACD, ATR, session VWAP) are computed once over the full history and sliced per "
           "as-of bar; they are causal, so this matches page 3's debug as-of mode. Recent bars may "
           "include the Yahoo patch from alaDataManager (no per-bar vwap, so the typical price is used), "
           "older bars are Alpaca-only.")