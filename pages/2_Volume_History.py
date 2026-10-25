"""
pages/2_Volume_History.py

Second page of the GEX dashboard app. Reads spot price, call volume and put
volume snapshots for the chosen symbol from the Neon database table
(written by the main dashboard page each time it loads/refreshes a chain)
and plots:

    - Spot price      (blue line,   left y-axis)
    - Put-call ratio  (orange line, right y-axis)  = put volume / call volume

X-axis = snapshot time (Eastern). Defaults to today's market date (Eastern).
"""

import datetime as dt
from io import BytesIO

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import insert_snapshot, load_snapshots
from gex_calculator import (
    CBOE_INDEX_SYMBOLS,
    VOLUME_SCOPES,
    fetch_chain,
    utc_naive_to_eastern_naive,
    volume_by_scope,
)

st.set_page_config(page_title="Volume", layout="wide")

# --------------------------------------------------------------------------
# Controls (main page, not sidebar): these only apply to this page
# --------------------------------------------------------------------------
st.title("Volume History")

PRESETS = {
    "SPY": ("SPY", False),
    "QQQ": ("QQQ", False),
    "IWM": ("IWM", False),
    "Custom symbol": None,
}

utc_now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
eastern_now = utc_naive_to_eastern_naive(utc_now)
today_eastern = eastern_now.date()

ctl_symbol, ctl_date, ctl_live, ctl_refresh = st.columns(
    [2, 2, 1, 1], vertical_alignment="bottom"
)

preset_label = ctl_symbol.selectbox("Symbol", list(PRESETS.keys()), index=0)
DEBUG_DEFAULT_DATE = dt.date(2026, 10, 23)
view_date = ctl_date.date_input("Market date (Eastern)", value=DEBUG_DEFAULT_DATE)
#view_date = ctl_date.date_input("Market date (Eastern)", value=today_eastern)
fetch_live = ctl_live.checkbox("Retrieve data", value=True)
ctl_refresh.button("Refresh", type="primary", use_container_width=True)  # any click reruns the script

if PRESETS[preset_label] is None:
    cust_symbol, cust_index = st.columns([2, 3], vertical_alignment="bottom")
    ticker = cust_symbol.text_input(
        "Enter symbol (e.g. SPX, SPY, AAPL, TSLA)", value="SPY"
    ).strip().upper()
    is_index = cust_index.checkbox(
        "This is a cash index (adds CBOE's leading underscore)",
        value=ticker in CBOE_INDEX_SYMBOLS,
    )
else:
    ticker, is_index = PRESETS[preset_label]

scope_labels = dict(VOLUME_SCOPES)
scope = st.radio(
    "Expirations to count",
    options=[code for code, _ in VOLUME_SCOPES],
    format_func=lambda c: scope_labels[c],
    index=1,   # default to 0DTE (second option)
    horizontal=True,
)
scope_label = scope_labels[scope]

st.subheader(f"{ticker} ({scope_label}) -- Spot price & put-call ratio over time")

# --------------------------------------------------------------------------
# 1) Optionally pull + store a fresh live snapshot
# --------------------------------------------------------------------------

if fetch_live:
    with st.spinner(f"Fetching latest chain for {ticker}..."):
        try:
            chain = fetch_chain(ticker=ticker, is_index=is_index, save_json=False)
            # One row per scope (All / 0DTE / 1DTE / This Friday) so that
            # history builds up for every view, not just the selected one.
            for code, (c_vol, p_vol) in volume_by_scope(chain).items():
                insert_snapshot(
                    symbol=chain.ticker,
                    scope=code,
                    snapshot_at_utc=chain.as_of,
                    spot=chain.spot,
                    call_volume=c_vol,
                    put_volume=p_vol,
                    source="volume_history",
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not fetch/store live data: {e}")

# --------------------------------------------------------------------------
# 2) Read from the database
# --------------------------------------------------------------------------
try:
    df = load_snapshots(ticker, view_date, scope)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read from database: {e}")
    st.stop()

if df.empty:
    st.info(
        f"No {scope_label} snapshots in the database for {ticker} on {view_date}. "
        "Tick **Retrieve data** to store a snapshot now (0DTE / 1DTE / This Friday "
        "history only exists from when this feature was added)."
    )
    st.stop()

st.caption(f"{len(df)} {scope_label} snapshot(s) for {ticker} on {view_date} (read from Neon database).")

# --------------------------------------------------------------------------
# 3) Chart
# --------------------------------------------------------------------------
GREEN, RED = "#2ECC71", "#E74C3C"


def pcr_change_info(df_plot: pd.DataFrame) -> dict:
    """
    Change in put-call ratio between the last two snapshots that have a ratio
    (same rule gexProcessor uses). Single source of truth for the chart
    annotation and the PNG metadata.

    Annotation convention (arrow shows the market read, sign shows the PCR move):
        PCR fell  -> green "▲ -0.0215"   state "decreased"
        PCR rose  -> red   "▼ +0.0200"   state "increased"
        unchanged -> no annotation        state "neutral"
    """
    valid = df_plot.dropna(subset=["put_call_ratio"])
    info = {"state": "unavailable", "diff": None, "label": None, "color": None, "x": None, "y": None}
    if valid.empty:
        return info
    info["x"] = valid["downloaded_at"].iloc[-1]
    info["y"] = float(valid["put_call_ratio"].iloc[-1])
    if len(valid) < 2:
        return info

    diff = round(float(valid["put_call_ratio"].iloc[-1] - valid["put_call_ratio"].iloc[-2]), 4)
    print(f"diff: {diff}, {valid["put_call_ratio"].iloc[-1]}, {valid["put_call_ratio"].iloc[-2] }")
    info["diff"] = diff
    if diff > 0:
        info.update(state="increased", label=f"▼ {diff:+.4f}", color=RED)
    elif diff < 0:
        info.update(state="decreased", label=f"▲ {diff:+.4f}", color=GREEN)
    else:
        info.update(state="neutral")
    return info


pcr_info = pcr_change_info(df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["downloaded_at"], y=df["spot"], name=f"{ticker} spot",
                         mode="lines+markers", line=dict(color="#2E86FF", width=2), yaxis="y1"))
fig.add_trace(go.Scatter(x=df["downloaded_at"], y=df["put_call_ratio"], name=f"Put-call ratio ({scope_label})",
                         mode="lines+markers", line=dict(color="#F39C12", width=2), yaxis="y2",
                         connectgaps=True, hovertemplate="%{y:.2f}"))

if pcr_info["label"]:
    fig.add_annotation(
        x=pcr_info["x"], y=pcr_info["y"], xref="x", yref="y2",
        text=pcr_info["label"], font=dict(color=pcr_info["color"], size=15),
        showarrow=True, arrowhead=0, arrowcolor=pcr_info["color"], ax=0, ay=-35,
    )

fig.update_layout(
    template="plotly_dark",
    xaxis=dict(title="Snapshot time (Eastern time)",     tickformat="%H:%M",    hoverformat="%H:%M",),
    yaxis=dict(title=f"{ticker} spot price", side="left"),
    yaxis2=dict(title=f"Put-call ratio ({scope_label})", side="right", tickformat=".2f", overlaying="y", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=560,
    margin=dict(t=60, b=40, l=60, r=60),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PNG download helper (matplotlib backend -- no browser / kaleido needed)
# --------------------------------------------------------------------------
def render_png_matplotlib(df_plot: pd.DataFrame, ticker_label: str, scope_text: str = "All dates", dpi: int = 150) -> bytes:
    matplotlib.use("Agg")
    fig_mpl, ax_price = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax_vol = ax_price.twinx()

    x = df_plot["downloaded_at"]

    ax_price.plot(x, df_plot["spot"], color="#2E86FF", linewidth=2, marker="o", label=f"{ticker_label} spot")
    pcr = df_plot["put_call_ratio"]
    ax_vol.plot(x[pcr.notna()], pcr.dropna(), color="#F39C12", linewidth=2, marker="o", label=f"Put-call ratio ({scope_text})")

    ax_price.set_xlabel("Snapshot time (Eastern time)")
    ax_price.set_ylabel(f"{ticker_label} spot price", color="#2E86FF")
    ax_vol.set_ylabel(f"Put-call ratio ({scope_text})")
    ax_price.set_title(f"{ticker_label} ({scope_text}) -- spot price & put-call ratio over time")

    info = pcr_change_info(df_plot)
    if info["label"]:
        ax_vol.annotate(
            info["label"], xy=(info["x"], info["y"]), xytext=(0, 20), textcoords="offset points",
            ha="center", color=info["color"], fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=info["color"]),
        )

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig_mpl.autofmt_xdate()

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_vol.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    buf = BytesIO()
    fig_mpl.tight_layout()
    # Machine-readable result embedded as PNG text chunks (read by gexProcessor).
    png_meta = {
        "PCR_STATE": info["state"],
        "PCR_DIFF": "NA" if info["diff"] is None else f"{info['diff']:+.4f}",
    }
    fig_mpl.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", metadata=png_meta)
    plt.close(fig_mpl)
    buf.seek(0)
    return buf.getvalue()


st.divider()
png_bytes = render_png_matplotlib(df, ticker, scope_label)
st.download_button(
    label="📥 Download chart (PNG)",
    data=png_bytes,
    file_name=f"{ticker}_{scope}_PCRHistory_{view_date.strftime('%Y%m%d')}_{eastern_now.strftime('%H%M')}.png",
    mime="image/png",
    use_container_width=True,
)

# Last 5 most recent rows as a plain-text table
recent_text = (
    f"{ticker} ({scope_label}) -- last 5 snapshots on {view_date} (Eastern time)\n\n"
    + df.tail(5)[["downloaded_at", "spot", "call_volume", "put_volume", "put_call_ratio", "source"]]
    .to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    + "\n"
)
st.download_button(
    label="Download last 5 rows",
    data=recent_text,
    file_name=f"{ticker}_{scope}_last5_{view_date.strftime('%Y%m%d')}_{eastern_now.strftime('%H%M')}.txt",
    mime="text/plain",
    use_container_width=True,
)

st.caption(
    f"Showing snapshots from {view_date} (Eastern time). Put-call ratio = total put "
    "volume / total call volume, summed across the selected expirations at each snapshot time."
)

with st.expander("Raw snapshot data"):
    st.dataframe(df, use_container_width=True)