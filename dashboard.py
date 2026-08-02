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

Query-string support:
    ?symbol=SPY          pre-select ticker (auto-detects index vs equity)
    ?strikes=12          change default strike window
    ?min_oi=50           change minimum open-interest filter
"""

import datetime as dt
from io import BytesIO
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
# Timezone helpers
# --------------------------------------------------------------------------
def _get_browser_timezone() -> str:
    """
    Detect browser timezone via a one-shot JS snippet that reloads the page
    with ?tz=<iana>. If JS is disabled or the snippet hasn't fired yet,
    fall back to America/New_York (EDT/EST).
    """
    qp_tz = st.query_params.get("tz")
    if qp_tz:
        return qp_tz

    if "tz_detected" not in st.session_state:
        st.session_state.tz_detected = True
        components.html(
            """
            <script>
            (function(){
                const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
                const url = new URL(window.location.href);
                if (!url.searchParams.has('tz')) {
                    url.searchParams.set('tz', tz);
                    window.location.href = url.toString();
                }
            })();
            </script>
            """,
            height=0,
        )
    return "America/New_York"


def _format_local_time(naive_utc_dt: dt.datetime, tz_name: str) -> str:
    """
    Convert a naive UTC datetime to a formatted string in the given zone.
    Falls back to America/New_York if the zone is unknown.
    """
    utc_dt = naive_utc_dt.replace(tzinfo=dt.timezone.utc)

    for zone in (tz_name, "America/New_York"):
        if not zone:
            continue
        try:
            local_dt = utc_dt.astimezone(ZoneInfo(zone))
            tz_abbr = local_dt.tzname() or ""
            return local_dt.strftime(f"%Y-%m-%d %H:%M {tz_abbr}")
        except Exception:
            continue

    return naive_utc_dt.strftime("%Y-%m-%d %H:%M UTC")


# Resolve timezone once per session (persists after JS reload)
browser_tz = _get_browser_timezone()

# --------------------------------------------------------------------------
# Sidebar: symbol
# --------------------------------------------------------------------------
st.sidebar.title("GEX Dashboard")

PRESETS = {
    "S&P 500 (SPX)": ("SPX", True),
    "Nasdaq 100 (NDX)": ("NDX", True),
    "Russell 2000 (RUT)": ("RUT", True),
    "Dow Jones (DJX)": ("DJX", True),
    "VIX": ("VIX", True),
    "SPY": ("SPY", False),
    "QQQ": ("QQQ", False),
    "IWM": ("IWM", False),
    "Custom symbol": None,
}

# --- Read query parameters -------------------------------------------------
qp = st.query_params
qp_symbol = qp.get("symbol", "").strip().upper()
qp_strikes = qp.get("strikes", "8").strip()
qp_min_oi = qp.get("min_oi", "10").strip()

# Default to SPY when no query-string symbol is provided
DEFAULT_SYMBOL = "SPY"
preset_index = list(PRESETS.keys()).index(DEFAULT_SYMBOL)
custom_ticker_default = DEFAULT_SYMBOL
custom_is_index_default = False

if qp_symbol:
    for i, (label, val) in enumerate(PRESETS.items()):
        if val and val[0] == qp_symbol:
            preset_index = i
            break
    else:
        preset_index = list(PRESETS.keys()).index("Custom symbol")
        custom_ticker_default = qp_symbol
        custom_is_index_default = qp_symbol in CBOE_INDEX_SYMBOLS

preset_label = st.sidebar.selectbox("Symbol", list(PRESETS.keys()), index=preset_index)

if PRESETS[preset_label] is None:
    ticker = st.sidebar.text_input(
        "Enter symbol (e.g. SPX, SPY, AAPL, TSLA)",
        value=custom_ticker_default,
    ).strip().upper()
    is_index = st.sidebar.checkbox(
        "This is a cash index (adds CBOE's leading underscore)",
        value=custom_is_index_default or (ticker in CBOE_INDEX_SYMBOLS),
    )
else:
    ticker, is_index = PRESETS[preset_label]

# Sync effective symbol back to URL query string
if qp_symbol != ticker:
    st.query_params["symbol"] = ticker

min_oi = st.sidebar.number_input(
    "Minimum open interest per strike",
    min_value=0,
    value=int(qp_min_oi) if qp_min_oi.isdigit() else 10,
    step=10,
)
if str(min_oi) != qp.get("min_oi", ""):
    st.query_params["min_oi"] = str(min_oi)

load = st.sidebar.button("Load / refresh chain", type="primary")

st.sidebar.caption(
    "Data: CBOE public delayed-quotes JSON feed ~15-20min delayed. "
)

# --------------------------------------------------------------------------
# Fetch full chain (auto-load on first run; manual refresh via button)
# --------------------------------------------------------------------------
if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.chain_error = None
    st.session_state.selected_expirations = None

# Auto-fetch when no chain has been loaded yet and no prior error exists
should_fetch = load or (st.session_state.chain is None and st.session_state.chain_error is None)

if should_fetch:
    with st.spinner(f"Fetching option chain for {ticker}..."):
        try:
            st.session_state.chain = fetch_chain(ticker=ticker, is_index=is_index)
            st.session_state.chain_error = None
            st.session_state.selected_expirations = None
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
strike_count_raw = st.sidebar.text_input(
    "Strikes to show around spot (each side)",
    value=qp_strikes,
)
try:
    strike_count = max(int(strike_count_raw), 0)
except ValueError:
    st.sidebar.warning("Enter a whole number, e.g. 8. Defaulting to 8.")
    strike_count = 8

if str(strike_count) != qp.get("strikes", ""):
    st.query_params["strikes"] = str(strike_count)

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
    yaxis=dict(autorange=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=max(520, 28 * len(df)),
    margin=dict(t=60, b=40, l=60, r=20),
    hovermode="y unified",
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PNG download helper (matplotlib backend -- no browser / kaleido needed)
# --------------------------------------------------------------------------
def render_png_matplotlib(df_plot, result_obj, dpi=150):
    matplotlib.use("Agg")
    n = len(df_plot)
    fig_h = max(6, 0.35 * n)
    fig_mpl, ax = plt.subplots(figsize=(9, fig_h), dpi=dpi)

    y_pos = np.arange(n)
    bar_h = 0.75

    ax.barh(y_pos, df_plot["call_gex"], height=bar_h, color="#1D9E75", label="Call GEX")
    ax.barh(y_pos, df_plot["put_gex"], height=bar_h, color="#D85A30", label="Put GEX")

    ax.axhline(y_pos[np.argmin(np.abs(df_plot["strike"].to_numpy() - result_obj.spot))],
               color="#378ADD", linestyle=":", linewidth=2, label="Spot")

    if result_obj.gamma_flip is not None:
        flip_idx = np.argmin(np.abs(df_plot["strike"].to_numpy() - result_obj.gamma_flip))
        ax.axhline(y_pos[flip_idx], color="#2C2C2A", linestyle="--", linewidth=2, label="Gamma flip")

    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s:,.0f}" for s in df_plot["strike"]])
    ax.invert_yaxis()
    ax.set_ylabel(f"{result_obj.ticker} price")
    ax.set_xlabel("Dealer GEX ($mm per 1% move)")
    ax.set_title(f"{result_obj.ticker} GEX by strike")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # --- Summary text box overlaid on the chart ----------------------------
    net_sign = "+" if result_obj.total_gex >= 0 else ""
    summary_lines = [
        f"Net GEX: {net_sign}${result_obj.total_gex:,.0f}mm",
        f"Call wall: ${result_obj.call_wall:,.0f}" if result_obj.call_wall else "Call wall: n/a",
        f"Put wall: ${result_obj.put_wall:,.0f}" if result_obj.put_wall else "Put wall: n/a",
    ]
    summary_text = "\n".join(summary_lines)

    props = dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc", alpha=0.95)
    ax.text(
        0.98, 0.98, summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        family="monospace",
    )

    buf = BytesIO()
    fig_mpl.tight_layout()
    fig_mpl.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig_mpl)
    buf.seek(0)
    return buf.getvalue()

# --- Download PNG ----------------------------------------------------------
st.divider()
png_bytes = render_png_matplotlib(df, result)
st.download_button(
    label="📥 Download chart (PNG)",
    data=png_bytes,
    file_name=f"{result.ticker}_GEX_{result.as_of.strftime('%Y%m%d_%H%M')}.png",
    mime="image/png",
    use_container_width=True,
)

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
exp_str = ", ".join(_fmt(e) for e in sorted(selected))
as_of_local = _format_local_time(result.as_of, browser_tz)
st.caption(f"As of {as_of_local} · Expirations included: {exp_str}")

if result.warnings:
    with st.expander("Warnings"):
        for w in result.warnings:
            st.write("- " + w)

with st.expander("Raw strike-level data"):
    st.dataframe(df.style.format({"call_gex": "{:.1f}", "put_gex": "{:.1f}", "net_gex": "{:.1f}"}))