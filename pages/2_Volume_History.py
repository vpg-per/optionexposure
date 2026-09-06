"""
pages/2_Volume_History.py

Second page of the GEX dashboard app. Builds a time series from every
saved CBOE JSON snapshot (written by the main dashboard page when "Save
downloaded JSON to disk" is checked) plus one fresh live pull, and plots:

    - Spot price        (blue line,  left y-axis)
    - Total call volume (green line, right y-axis)
    - Total put volume  (red line,   right y-axis)

X-axis = the time each snapshot was downloaded (Eastern time). Only
snapshots from the current Eastern-time date are shown -- prior days'
saved data is ignored.

Streamlit auto-discovers this file because it lives in a `pages/` folder
next to the main script (dashboard.py) -- no extra wiring needed, it just
shows up in the sidebar nav.
"""

import datetime as dt
import glob
import json
import os
import re
from io import BytesIO

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gex_calculator import (
    CBOE_INDEX_SYMBOLS,
    DEFAULT_JSON_SAVE_DIR,
    fetch_cboe_json,
    parse_occ_symbol,
    resolve_cboe_symbol,
    save_raw_json,
    utc_naive_to_eastern_naive,
)

st.set_page_config(page_title="Volume History", layout="wide")

# --------------------------------------------------------------------------
# Filename timestamp parsing -- matches gex_calculator.save_raw_json's format:
#   {symbol}_{YYYYMMDD}_{HHMMSS}.json
# The digits in the filename are already Eastern time (no UTC or other
# timezone involved) -- save_raw_json converts before naming the file.
# --------------------------------------------------------------------------
_FILENAME_TS_RE = re.compile(r"_(\d{8})_(\d{6})\.json$")


def _parse_filename_timestamp(path: str) -> dt.datetime | None:
    m = _FILENAME_TS_RE.search(os.path.basename(path))
    if not m:
        return None
    date_str, time_str = m.groups()
    try:
        return dt.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def compute_spot_and_volumes(payload: dict) -> tuple[float, int, int]:
    """
    Sums total call volume and put volume across every contract in the
    payload (all strikes, all expirations), plus the underlying spot price.
    """
    data = payload["data"]
    spot = float(data.get("current_price") or data.get("close") or 0.0)

    call_vol = 0
    put_vol = 0
    for opt in data.get("options", []):
        parsed = parse_occ_symbol(opt.get("option", ""))
        if parsed is None:
            continue
        vol = opt.get("volume", 0) or 0
        if parsed["type"] == "call":
            call_vol += vol
        else:
            put_vol += vol

    return spot, int(call_vol), int(put_vol)


def load_saved_snapshots(symbol: str, save_dir: str, only_date: dt.date | None = None) -> list[dict]:
    """
    Finds every saved JSON file for `symbol` in `save_dir`, parses each,
    and returns a list of {downloaded_at, spot, call_volume, put_volume}.
    Files that don't match the expected naming pattern (so we can't recover
    a reliable timestamp) are skipped. If `only_date` is given, snapshots
    from any other calendar date (Eastern time) are skipped too.
    """
    clean_symbol = symbol.lstrip("_") or symbol
    pattern = os.path.join(save_dir, f"{clean_symbol}_*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        downloaded_at = _parse_filename_timestamp(path)
        if downloaded_at is None:
            continue
        if only_date is not None and downloaded_at.date() != only_date:
            continue
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            spot, call_vol, put_vol = compute_spot_and_volumes(payload)
        except Exception:
            continue
        rows.append(
            {
                "downloaded_at": downloaded_at,
                "spot": spot,
                "call_volume": call_vol,
                "put_volume": put_vol,
                "source": os.path.basename(path),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Sidebar: symbol (mirrors the main dashboard page)
# --------------------------------------------------------------------------
st.sidebar.title("Volume History")

PRESETS = {
    "SPY": ("SPY", False),
    "QQQ": ("QQQ", False),
    "IWM": ("IWM", False),
    "Custom symbol": None,
}
preset_label = st.sidebar.selectbox("Symbol", list(PRESETS.keys()), index=0)

if PRESETS[preset_label] is None:
    ticker = st.sidebar.text_input("Enter symbol (e.g. SPX, SPY, AAPL, TSLA)", value="SPY").strip().upper()
    is_index = st.sidebar.checkbox(
        "This is a cash index (adds CBOE's leading underscore)",
        value=ticker in CBOE_INDEX_SYMBOLS,
    )
else:
    ticker, is_index = PRESETS[preset_label]

save_dir = st.sidebar.text_input("Saved-data folder", value=DEFAULT_JSON_SAVE_DIR)
save_latest_too = st.sidebar.checkbox("Also save this live pull to disk", value=True)

refresh = st.sidebar.button("Refresh (fetch live + reload saved)", type="primary")

st.title(f"{ticker} -- Spot price & options volume over time")

symbol = resolve_cboe_symbol(ticker, is_index=is_index)

# "Today" is defined in Eastern time (market time), not UTC or server time.
utc_now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
eastern_now = utc_naive_to_eastern_naive(utc_now)
today_eastern = eastern_now.date()

# --------------------------------------------------------------------------
# 1) Load whatever's already saved on disk -- today's snapshots only
# --------------------------------------------------------------------------
rows = load_saved_snapshots(symbol, save_dir, only_date=today_eastern)

if rows:
    st.caption(f"Found {len(rows)} saved snapshot(s) for {ticker} today ({today_eastern}) in `{save_dir}/`.")
else:
    st.info(
        f"No saved snapshots found for {ticker} today ({today_eastern}) in "
        f"`{save_dir}/` yet. Showing just the live pull below. Turn on "
        "**Save downloaded JSON to disk** on the main dashboard page (or "
        "the checkbox in this sidebar) to start building today's history."
    )

# --------------------------------------------------------------------------
# 2) Always fetch the latest live snapshot too, and fold it in
# --------------------------------------------------------------------------
with st.spinner(f"Fetching latest chain for {ticker}..."):
    try:
        payload = fetch_cboe_json(symbol)[0]
        spot, call_vol, put_vol = compute_spot_and_volumes(payload)

        saved_path = None
        if save_latest_too:
            saved_path = save_raw_json(payload, symbol, save_dir=save_dir, downloaded_at=utc_now)

        rows.append(
            {
                "downloaded_at": eastern_now,
                "spot": spot,
                "call_volume": call_vol,
                "put_volume": put_vol,
                "source": os.path.basename(saved_path) if saved_path else "live (not saved)",
            }
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not fetch live data: {e}")

if not rows:
    st.stop()

df = pd.DataFrame(rows).drop_duplicates(subset="downloaded_at").sort_values("downloaded_at").reset_index(drop=True)

# --------------------------------------------------------------------------
# 3) Chart: spot on left axis (blue -- visible in light and dark themes),
#    call/put volume on right axis
# --------------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["downloaded_at"], y=df["spot"],
        name=f"{ticker} spot",
        mode="lines+markers",
        line=dict(color="#2E86FF", width=2),
        yaxis="y1",
    )
)
fig.add_trace(
    go.Scatter(
        x=df["downloaded_at"], y=df["call_volume"],
        name="Call volume",
        mode="lines+markers",
        line=dict(color="#2ECC71", width=2),
        yaxis="y2",
    )
)
fig.add_trace(
    go.Scatter(
        x=df["downloaded_at"], y=df["put_volume"],
        name="Put volume",
        mode="lines+markers",
        line=dict(color="#E74C3C", width=2),
        yaxis="y2",
    )
)

fig.update_layout(
    template="plotly_dark",
    xaxis=dict(title="Downloaded at (Eastern time)"),
    yaxis=dict(title=f"{ticker} spot price", side="left"),
    yaxis2=dict(title="Total options volume", side="right", overlaying="y", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=560,
    margin=dict(t=60, b=40, l=60, r=60),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PNG download helper (matplotlib backend -- no browser / kaleido needed),
# same approach as the main dashboard page's "Download chart (PNG)" button.
# --------------------------------------------------------------------------
def render_png_matplotlib(df_plot: pd.DataFrame, ticker_label: str, dpi: int = 150) -> bytes:
    matplotlib.use("Agg")
    fig_mpl, ax_price = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax_vol = ax_price.twinx()

    x = df_plot["downloaded_at"]

    ax_price.plot(x, df_plot["spot"], color="#2E86FF", linewidth=2, marker="o", label=f"{ticker_label} spot")
    ax_vol.plot(x, df_plot["call_volume"], color="#2ECC71", linewidth=2, marker="o", label="Call volume")
    ax_vol.plot(x, df_plot["put_volume"], color="#E74C3C", linewidth=2, marker="o", label="Put volume")

    ax_price.set_xlabel("Downloaded at (Eastern time)")
    ax_price.set_ylabel(f"{ticker_label} spot price", color="#2E86FF")
    ax_vol.set_ylabel("Total options volume")
    ax_price.set_title(f"{ticker_label} -- spot price & options volume over time")

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig_mpl.autofmt_xdate()

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_vol.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    buf = BytesIO()
    fig_mpl.tight_layout()
    fig_mpl.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig_mpl)
    buf.seek(0)
    return buf.getvalue()


st.divider()
png_bytes = render_png_matplotlib(df, ticker)
st.download_button(
    label="📥 Download chart (PNG)",
    data=png_bytes,
    file_name=f"{ticker}_VolumeHistory_{eastern_now.strftime('%Y%m%d_%H%M')}.png",
    mime="image/png",
    use_container_width=True,
)

st.caption(
    f"Showing only snapshots from today ({today_eastern}, Eastern time). "
    "Volume totals are summed across every strike and expiration in the "
    "chain at each snapshot time (not just the strikes shown on the main "
    "GEX chart)."
)

with st.expander("Raw snapshot data"):
    st.dataframe(df, use_container_width=True)
