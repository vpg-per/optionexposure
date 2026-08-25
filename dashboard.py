"""
dashboard.py

FatTail-style GEX bar chart dashboard.

Run with:
    streamlit run dashboard.py

Shows dealer gamma exposure by strike, with the gamma flip point, call wall,
and put wall marked. Chart is horizontal: the underlying's price scale runs
up the y-axis (like a price chart), gamma exposure runs along the x-axis.
Data comes from CBOE's public delayed-quotes JSON feed -- works for both
cash indices (SPX, NDX, RUT, VIX, DJX, ...) and any equity/ETF with listed
options (SPY, QQQ, AAPL, ...).
"""

import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from gex_calculator import (
    CBOE_INDEX_SYMBOLS,
    compute_gex,
    default_expiration_selection,
    fetch_chain,
    label_expirations,
    list_expirations,
    nearest_strikes,
)

st.set_page_config(page_title="GEX Dashboard", layout="wide")

# --------------------------------------------------------------------------
# Sidebar: symbol
# --------------------------------------------------------------------------
st.sidebar.title("GEX Dashboard")

PRESETS = {
    "SPY": ("SPY", False),
    "QQQ": ("QQQ", False),
    "IWM": ("IWM", False),
    "Custom symbol": None,
}
preset_label = st.sidebar.selectbox("Symbol", list(PRESETS.keys()))
if PRESETS[preset_label] is None:
    ticker = st.sidebar.text_input("Enter symbol (e.g. SPX, SPY, AAPL, TSLA)", value="AAPL").strip().upper()
    is_index = st.sidebar.checkbox(
        "This is a cash index (adds CBOE's leading underscore)",
        value=ticker in CBOE_INDEX_SYMBOLS,
    )
else:
    ticker, is_index = PRESETS[preset_label]

min_oi = st.sidebar.number_input("Minimum open interest per strike", min_value=0, value=10, step=10)
load = st.sidebar.button("Load / refresh chain", type="primary")

st.sidebar.caption(
    "Data: CBOE public delayed-quotes JSON feed "
    "(cdn.cboe.com/api/global/delayed_quotes/options), ~15-20min delayed. "
    "Works for cash indices (underscore-prefixed CBOE symbol) and any "
    "equity/ETF ticker with listed options."
)

# --------------------------------------------------------------------------
# Fetch full chain (only on button press -- cheap to re-slice afterwards)
# --------------------------------------------------------------------------
if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.chain_error = None
    st.session_state.selected_expirations = None

if load:
    with st.spinner(f"Fetching option chain for {ticker}..."):
        try:
            st.session_state.chain = fetch_chain(ticker=ticker, is_index=is_index)
            st.session_state.chain_error = None
            st.session_state.selected_expirations = None  # reset to defaults for the new chain
        except Exception as e:  # noqa: BLE001
            st.session_state.chain = None
            st.session_state.chain_error = str(e)

st.title("Gamma exposure (GEX) by strike")

if st.session_state.chain_error:
    st.error(st.session_state.chain_error)

chain = st.session_state.chain
if chain is None:
    st.info("Choose a symbol in the sidebar and click **Load / refresh chain**.")
    st.stop()

# --------------------------------------------------------------------------
# Expiration picker: 0DTE / 1DTE / this Friday / everything else
# --------------------------------------------------------------------------
all_expirations = list_expirations(chain)
labels = label_expirations(all_expirations, chain.as_of)


def _fmt(e: dt.datetime) -> str:
    lbl = labels[e]
    return f"{e.strftime('%Y-%m-%d')} ({lbl})" if lbl else e.strftime("%Y-%m-%d")


if st.session_state.selected_expirations is None:
    st.session_state.selected_expirations = default_expiration_selection(all_expirations, chain.as_of)

st.subheader("Expirations")
selected = st.multiselect(
    "Include expiration(s)",
    options=all_expirations,
    default=st.session_state.selected_expirations,
    format_func=_fmt,
    label_visibility="collapsed",
)
st.session_state.selected_expirations = selected

quick_cols = st.columns(4)
if quick_cols[0].button("0DTE only"):
    st.session_state.selected_expirations = [e for e, l in labels.items() if l == "0DTE"] or all_expirations[:1]
    st.rerun()
if quick_cols[1].button("0DTE + 1DTE"):
    st.session_state.selected_expirations = [e for e, l in labels.items() if l in ("0DTE", "1DTE")] or all_expirations[:1]
    st.rerun()
if quick_cols[2].button("This Friday"):
    st.session_state.selected_expirations = [e for e, l in labels.items() if l == "This Fri"] or all_expirations[:1]
    st.rerun()
if quick_cols[3].button("All listed"):
    st.session_state.selected_expirations = all_expirations
    st.rerun()

if not selected:
    st.warning("Select at least one expiration above.")
    st.stop()

# --------------------------------------------------------------------------
# Strike window: N strikes above/below spot (text input, not a % window)
# --------------------------------------------------------------------------
strike_count_raw = st.sidebar.text_input("Strikes to show around spot (each side)", value="10")
try:
    strike_count = max(int(strike_count_raw), 0)
except ValueError:
    st.sidebar.warning("Enter a whole number, e.g. 8. Defaulting to 10.")
    strike_count = 10

# --------------------------------------------------------------------------
# Compute + filter
# --------------------------------------------------------------------------
try:
    result = compute_gex(chain, selected, min_open_interest=min_oi)
except Exception as e:  # noqa: BLE001
    st.error(str(e))
    st.stop()

df = nearest_strikes(result.by_strike, result.spot, strike_count)

# --- Summary metrics -------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"${result.spot:,.2f}")
c2.metric("Net GEX", f"${result.total_gex:,.0f}mm", result.regime.capitalize())
c3.metric("Gamma flip", f"${result.gamma_flip:,.0f}" if result.gamma_flip else "n/a")
c4.metric("Call wall", f"${result.call_wall:,.0f}" if result.call_wall else "n/a")
c5.metric("Put wall", f"${result.put_wall:,.0f}" if result.put_wall else "n/a")

regime_note = (
    "Dealers net long gamma: hedging tends to dampen moves (buy dips, sell rallies)."
    if result.regime == "positive"
    else "Dealers net short gamma: hedging tends to amplify moves (sell dips, buy rallies)."
)
st.caption(regime_note)

# --- Horizontal bar chart: price on y-axis, GEX on x-axis ------------------
fig = go.Figure()

fig.add_bar(
    y=df["strike"], x=df["call_gex"],
    orientation="h",
    name="Call GEX",
    marker_color="#1D9E75",
    hovertemplate="Strike %{y}<br>Call GEX $%{x:.1f}mm<extra></extra>",
)
fig.add_bar(
    y=df["strike"], x=df["put_gex"],
    orientation="h",
    name="Put GEX",
    marker_color="#D85A30",
    hovertemplate="Strike %{y}<br>Put GEX $%{x:.1f}mm<extra></extra>",
)

fig.add_hline(
    y=result.spot, line_width=2, line_dash="dot", line_color="#378ADD",
    annotation_text="Spot", annotation_position="right",
)
if result.gamma_flip is not None and not df.empty and df["strike"].min() <= result.gamma_flip <= df["strike"].max():
    fig.add_hline(
        y=result.gamma_flip, line_width=2, line_dash="dash", line_color="#2C2C2A",
        annotation_text="Gamma flip", annotation_position="left",
    )

fig.update_layout(
    barmode="relative",
    yaxis_title=f"{result.ticker} price",
    xaxis_title="Dealer GEX ($mm per 1% move)",
    yaxis=dict(
        range=[df["strike"].min() - 1, df["strike"].max() + 1] if not df.empty else None,
        autorange=False,
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=max(520, 28 * len(df)),
    margin=dict(t=60, b=40, l=60, r=20),
    hovermode="y unified",
)

st.plotly_chart(fig, use_container_width=True)

exp_str = ", ".join(_fmt(e) for e in sorted(selected))
st.caption(f"As of {result.as_of.strftime('%Y-%m-%d %H:%M UTC')} · Expirations included: {exp_str}")

if result.warnings:
    with st.expander("Warnings"):
        for w in result.warnings:
            st.write("- " + w)

with st.expander("Raw strike-level data"):
    st.dataframe(df.style.format({"call_gex": "{:.1f}", "put_gex": "{:.1f}", "net_gex": "{:.1f}"}))