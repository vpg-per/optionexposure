"""Sector performance comparison chart.

Fetches sector ETF data, builds a matplotlib chart comparing normalized
intraday performance plus a risk-on/risk-off side panel, and saves the
result to a PNG. Call generate_sector_chart() for "today" (default), or pass
start_et/end_et (US/Eastern datetimes) for any other trading day/window.
Designed to be invoked periodically (see main.py), not run at import time.
"""
import time
from datetime import datetime, timedelta, time as dt_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from service import alaDataManager as dm

# Order matters: it's the fetch order. XLV is fetched and shown in the side
# panel (risk-off classification) but not drawn on the line chart.
SECTOR_NAMES = {
    "XLB": "Materials", "XLC": "Comm Svcs", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Cons Staples", "XLU": "Utilities", "XLY": "Cons Discret",
    "XLV": "Health Care",
}
TABLE_ONLY_SYMS = {"XLV"}
PLOTTED_SYMS = [s for s in SECTOR_NAMES if s not in TABLE_ONLY_SYMS]

# Fixed color per symbol so a sector always renders the same color.
SECTOR_COLORS = {
    "XLB": "#e6194B", "XLC": "#3cb44b", "XLE": "#4363d8", "XLF": "#f58231",
    "XLI": "#911eb4", "XLK": "#42d4f4", "XLP": "#f032e6", "XLU": "#9A6324",
    "XLV": "#d73027", "XLY": "#000075",
}
BLUE_LBL = "#1f4e9c"
UP_COLOR = "#1a9850"
DOWN_COLOR = "#d73027"
FLAT_COLOR = "#999999"
SHADE_COLOR = "#f2f2f2"

SMOOTH_BARS = 3              # rolling-mean window; 1 disables smoothing
LABEL_GAP_PAD_FACTOR = 1.25  # breathing room on top of the measured label height
LABEL_FONTSIZE = 8.5

RISK_ON = ["XLK", "XLI", "XLY"]
RISK_OFF = ["XLP", "XLV", "XLU"]
OTHER_SYMS = ["XLE", "XLB", "XLF", "XLC"]
PANEL_GROUPS = [
    ("RISK ON", UP_COLOR, RISK_ON),
    ("RISK OFF", DOWN_COLOR, RISK_OFF),
    ("OTHER SECTORS", "#555555", OTHER_SYMS),
]


def _contrast_text_color(hex_color: str) -> str:
    """White or black text, whichever reads better on `hex_color`."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.6 else "#ffffff"


def _pct_change_since_prev_close(sym, df, midnight):
    """Latest price vs. the prior trading day's regular-hours close (last bar
    before 4:00 PM ET). Returns (prev_close, latest, pct_change), or None if
    there isn't enough data."""
    if df is None or df.empty:
        return None

    prior = df[df.index < midnight]
    if prior.empty:
        return None

    regular = prior[prior.index.time < dt_time(16, 0)]
    prev_close = (regular if not regular.empty else prior)["close"].iloc[-1]
    if not prev_close:
        return None

    # Latest bar available, even if it printed before midnight (thin ETFs
    # may have no post-midnight print yet).
    latest = df["close"].iloc[-1]
    pct = (latest - prev_close) / prev_close * 100
    print(f"{sym}, prev_close: {prev_close}, cur_price:{latest}, "
          f"change: {latest - prev_close:.2f}, percent: {pct:.2f}")
    return float(prev_close), float(latest), float(pct)


def _min_label_gap(fig, ax, y_range):
    """Minimum vertical gap (data units) between end-of-line badges, derived
    from the badge's rendered pixel height so it holds for any data range."""
    fig.canvas.draw()  # need finalized axes position
    ax_height_px = ax.get_window_extent(renderer=fig.canvas.get_renderer()).height
    # line height (~1.4x fontsize) + bbox padding (0.3 * fontsize top and bottom)
    label_height_pt = LABEL_FONTSIZE * 1.4 + 0.3 * LABEL_FONTSIZE * 2
    label_height_px = label_height_pt * fig.dpi / 72.0
    return label_height_px / (ax_height_px / y_range) * LABEL_GAP_PAD_FACTOR


def _spread_labels(ys, min_gap):
    """Spread sorted label positions so none are closer than min_gap."""
    ys = list(ys)
    for i in range(1, len(ys)):          # push up
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for i in range(len(ys) - 2, -1, -1):  # pull back down where possible
        ys[i] = min(ys[i], ys[i + 1] - min_gap)
    return ys


# right edges of the side-panel columns (axes fraction)
PREV_X, CUR_X, PCT_X = 0.38, 0.65, 0.99


def _draw_side_panel(ax_table, stats):
    ax_table.axis("off")

    def valid(group):
        return [stats[s][2] for s in group if stats[s] is not None]

    on, off = valid(RISK_ON), valid(RISK_OFF)
    on_up = sum(p > 0 for p in on)
    off_up = sum(p > 0 for p in off)

    if on and on_up > len(on) / 2:
        regime, regime_color, vals = "RISK ON", UP_COLOR, on
        detail = f"{on_up}/{len(on)} risk-on sectors up"
    elif off and off_up > len(off) / 2:
        regime, regime_color, vals = "RISK OFF", DOWN_COLOR, off
        detail = f"{off_up}/{len(off)} risk-off sectors up"
    else:
        regime, regime_color, vals = "N/A", FLAT_COLOR, []
        detail = "No clear sector rotation"
    if vals:
        detail += f"  (avg {sum(vals) / len(vals):+.2f}%)"

    t = ax_table.transAxes
    ax_table.text(0.5, 0.985, regime, transform=t, fontsize=16, fontweight="bold",
                  color=regime_color, ha="center", va="top")
    ax_table.text(0.5, 0.915, detail, transform=t, fontsize=8.5, color="#555555",
                  ha="center", va="top", style="italic")
    ax_table.text(0.5, 0.875, "vs. yesterday's regular-hours close", transform=t,
                  fontsize=7.5, color="#999999", ha="center", va="top")
    ax_table.plot([0.0, 1.0], [0.845, 0.845], transform=t, color="#dddddd", lw=1)

    # column headers (right-aligned over the numeric columns)
    for x, label in ((PREV_X, "Prev Close"), (CUR_X, "Current"), (PCT_X, "Change")):
        ax_table.text(x, 0.838, label, transform=t, fontsize=7.5,
                      fontweight="bold", color="#777777", ha="right", va="top")

    y, line_h, header_gap = 0.79, 0.058, 0.020
    for title, title_color, group in PANEL_GROUPS:
        y -= header_gap
        ax_table.text(0.02, y, title, transform=t, fontsize=10,
                      fontweight="bold", color=title_color, va="top")
        y -= line_h * 0.85
        for sym in group:
            if stats[sym] is None:
                prev_str = cur_str = pct_str = "N/A"
                pct_color = FLAT_COLOR
            else:
                prev_close, latest, pct = stats[sym]
                prev_str, cur_str = f"{prev_close:,.2f}", f"{latest:,.2f}"
                arrow = "\u25B2" if pct > 0 else ("\u25BC" if pct < 0 else "\u25AC")
                pct_color = UP_COLOR if pct > 0 else (DOWN_COLOR if pct < 0 else FLAT_COLOR)
                pct_str = f"{arrow} {pct:+.2f}%"
            ax_table.text(0.04, y, sym, transform=t, fontsize=9.5,
                          fontweight="bold", color=SECTOR_COLORS[sym], va="top")
            ax_table.text(PREV_X, y, prev_str, transform=t, fontsize=9,
                          color="#333333", va="top", ha="right")
            ax_table.text(CUR_X, y, cur_str, transform=t, fontsize=9,
                          color="#333333", va="top", ha="right")
            ax_table.text(PCT_X, y, pct_str, transform=t, fontsize=9.5,
                          fontweight="bold", color=pct_color, va="top", ha="right")
            y -= line_h


def generate_sector_chart(
    start_et: datetime | None = None,
    end_et: datetime | None = None,
    output_path: str = "sectors_5min.png",
) -> str:
    anchor_et = end_et or datetime.now(dm.EASTERN)
    midnight = anchor_et.replace(hour=0, minute=0, second=0, microsecond=0)
    display_start = start_et or midnight.replace(hour=6)
    max_display_end = end_et or midnight.replace(hour=17)
    open_t = midnight.replace(hour=9, minute=30)
    close_t = midnight.replace(hour=16)

    # ---- fetch ----
    fetched, latest_data_ts = {}, None
    for i, sym in enumerate(SECTOR_NAMES):
        if i:
            time.sleep(1.0)  # avoid tripping Yahoo's throttling
        df = fetched[sym] = dm.fetch_sector_data_yahoo(sym, end_et=anchor_et)
        if sym in TABLE_ONLY_SYMS or df is None:
            continue
        in_window = df[(df.index >= display_start) & (df.index <= max_display_end)]
        if not in_window.empty:
            ts = in_window.index[-1]
            latest_data_ts = ts if latest_data_ts is None else max(latest_data_ts, ts)

    if latest_data_ts is not None:
        # small right pad so the last point/labels aren't against the edge
        pad = (latest_data_ts - display_start) * 0.03
        display_end = min(latest_data_ts + pad, max_display_end)
        # floor the window width so a couple of bars doesn't collapse the chart
        if display_end - display_start < timedelta(minutes=30):
            display_end = min(display_start + timedelta(minutes=30), max_display_end)
    else:
        display_end = max_display_end

    # ---- figure ----
    fig = plt.figure(figsize=(15.5, 7.5), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.85, 0.72], wspace=0.17, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # shade pre-market / after-hours; baseline at 100
    ax.axvspan(display_start, open_t, color=SHADE_COLOR, zorder=0)
    if display_end > close_t:
        ax.axvspan(close_t, display_end, color=SHADE_COLOR, zorder=0)
    ax.set_xlim(display_start, display_end)
    ax.axhline(100, color="#bbbbbb", lw=0.8, ls="--", zorder=1)

    # ---- lines + stats ----
    stats = {sym: _pct_change_since_prev_close(sym, fetched[sym], midnight)
             for sym in SECTOR_NAMES}
    series = []  # (sym, last_x, last_y)
    all_values = []
    for sym in PLOTTED_SYMS:
        full_df = fetched[sym]
        if full_df is None:
            continue
        df = full_df[(full_df.index >= display_start) & (full_df.index <= display_end)]
        if df.empty:
            continue
        norm = (df["close"] / df["close"].iloc[0] * 100).rolling(SMOOTH_BARS, min_periods=1).mean()
        ax.plot(df.index, norm, color=SECTOR_COLORS[sym], lw=1.2, zorder=2)
        series.append((sym, df.index[-1], norm.iloc[-1]))
        all_values.extend(norm.dropna().tolist())

    # ---- end-of-line badges, de-overlapped ----
    series.sort(key=lambda t: t[2])
    min_gap = 0
    if series:
        # range covers every plotted value so intraday peaks/troughs aren't clipped
        lo, hi = min(all_values), max(all_values)
        pad = (hi - lo) * 0.15 or 1.0
        y_min, y_max = lo - pad, hi + pad
        ax.set_ylim(y_min, y_max)
        min_gap = _min_label_gap(fig, ax, y_max - y_min)

    label_ys = _spread_labels([y for *_, y in series], min_gap)
    x_right = ax.get_xlim()[1]
    for (sym, x_last, y_last), y_label in zip(series, label_ys):
        color = SECTOR_COLORS[sym]
        pct = stats[sym][2] if stats[sym] else None
        pct_text = f"{pct:+.2f}%" if pct is not None else "N/A"
        ax.annotate(f"{sym} {pct_text}",
                    xy=(1.005, y_label), xycoords=("axes fraction", "data"),
                    color=_contrast_text_color(color), fontsize=LABEL_FONTSIZE,
                    va="center", ha="left", clip_on=False, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.35",
                              facecolor=color, edgecolor="none"))
        # leader line from the true data point to the shifted label
        if abs(y_label - y_last) > min_gap * 0.3:
            ax.plot([mdates.date2num(x_last), x_right], [y_last, y_last],
                    color=color, lw=0.5, alpha=0.5, clip_on=False, zorder=1)

    # ---- styling ----
    ax.grid(True, color="#e2e2e2", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=BLUE_LBL, labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=dm.EASTERN))
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    # y tick labels dropped: the badges carry that info
    ax.yaxis.tick_right()
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for side in ("top", "left"):
        ax.spines[side].set_visible(False)
    for side in ("right", "bottom"):
        ax.spines[side].set_color("#d9d9d9")
    ax.set_title("SECTOR ETFS Comparison chart",
                 color=BLUE_LBL, fontsize=11, fontweight="bold", loc="left")

    _draw_side_panel(ax_table, stats)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.08)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_sector_chart()