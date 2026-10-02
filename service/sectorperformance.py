"""Sector performance comparison chart.

Fetches sector ETF data, builds a matplotlib chart comparing normalized
intraday performance plus a risk-on/risk-off side panel, and saves the
result to a PNG file. Call generate_sector_chart() to (re)build the chart;
it's designed to be invoked periodically (see main.py), not run at import
time, so all "now"-dependent state lives inside the function.
"""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import config
import alaDataManager as dm
from datetime import datetime, timedelta, time as dt_time

SECTORS = {
    "XLB": "Materials", "XLC": "Comm Svcs", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Cons Staples", "XLU": "Utilities", "XLY": "Cons Discret",
}
# "XLRE": "Real Estate",  "XLV": "Health Care",

# Fixed color per symbol (not list position), so a given sector always
# renders the same color run to run, regardless of dict ordering.
# Distinct, colorblind-friendlier palette (tab10 + 1 extra).
SECTOR_COLORS = {
    "XLB":  "#e6194B",
    "XLC":  "#3cb44b",
    "XLE":  "#4363d8",
    "XLF":  "#f58231",
    "XLI":  "#911eb4",
    "XLK":  "#42d4f4",
    "XLP":  "#f032e6",
    "XLRE": "#469990",
    "XLU":  "#9A6324",
    "XLV":  "#800000",
    "XLY":  "#000075",
    "XLV": "#d73027",
}
BLUE_LBL = "#1f4e9c"
UP_COLOR = "#1a9850"
DOWN_COLOR = "#d73027"
FLAT_COLOR = "#999999"


def _contrast_text_color(hex_color: str) -> str:
    """Pick white or black text for readability on top of `hex_color`,
    based on standard relative-luminance thresholding."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.6 else "#ffffff"

SMOOTH_BARS = 3          # rolling-mean window; set to 1 to disable smoothing
LABEL_GAP_PAD_FACTOR = 1.25  # extra breathing room multiplier applied on top
                              # of the measured label height (see min_gap calc
                              # in generate_sector_chart)

# ---- risk-on / risk-off sector groupings for the side panel ----
RISK_ON = ["XLK", "XLI", "XLY"]
RISK_OFF = ["XLP", "XLV", "XLU"]
OTHER_SYMS = ["XLE", "XLB", "XLF", "XLC"]

# Names for symbols referenced in the table but not plotted on the chart
# (XLV was dropped from the line chart to reduce clutter, but is still
# needed here to classify the risk-off group).
TABLE_NAMES = dict(SECTORS)
TABLE_NAMES["XLV"] = "Health Care"


def _pct_change_since_prev_close(sym, df, midnight):
    """Compare the latest available price in df to the prior trading day's
    regular-hours close (the last bar timestamped before 4:00 PM ET on the
    prior day). Returns (dollar_change, pct_change, latest_price), any of
    which may be None if there isn't enough data to compute it.
    """
    if df is None or df.empty:
        return None, None, None

    prior_df = df[df.index < midnight]
    if prior_df.empty:
        return None, None, None

    reg_prior = prior_df[prior_df.index.time < dt_time(16, 0)]
    if reg_prior.empty:
        reg_prior = prior_df  # fallback: best available prior bar

    prev_close = reg_prior["close"].iloc[-1]
    # Use the single most recent bar available, whether or not it printed
    # since midnight -- some ETFs trade thinner overnight/pre-market and
    # may not have a post-midnight print yet even though a perfectly good
    # "current price" (from late yesterday) exists.
    latest_price = df["close"].iloc[-1]
    if prev_close in (0, None):
        return None, None, latest_price

    change = latest_price - prev_close
    pct = change / prev_close * 100
    print(f"{sym}, prev_close: {prev_close}, cur_price:{latest_price}, change: {change:.2f}, percent: {pct:.2f}")
    return change, pct, latest_price


def generate_sector_chart(output_path: str = "sectors_5min.png") -> str:
    """Build the sector comparison chart + risk-on/off panel and save it
    as a PNG. Returns the path to the saved file.
    """
    # ---- data is still downloaded from midnight (unchanged fetch behavior) ----
    now_et = datetime.now(config.EASTERN)
    midnight = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    # `(now_et - midnight).days` is always 0 (midnight is *today's* midnight,
    # so the gap is always < 24h), so this used to always evaluate to 1 --
    # not enough to reach the prior trading session's close across a
    # weekend or holiday (e.g. Monday would only look back to Sunday).
    # Floor it at 5 calendar days so a short holiday weekend can't leave
    # _pct_change_since_prev_close with no prior-close data to compare against.
    MIN_LOOKBACK_DAYS = 5
    config.LOOKBACK_DAYS = max((now_et - midnight).days + 1, MIN_LOOKBACK_DAYS)

    # ---- the chart window is capped to 5am-5pm ET, but only extends as
    # far as data actually exists ----
    # (4-5am and 5-8pm have too little data to be worth showing, and
    # reserving empty space all the way to 5pm when e.g. only data up to
    # 2pm exists just wastes horizontal space / squishes the real lines)
    display_start = midnight.replace(hour=6, minute=0)
    max_display_end = midnight.replace(hour=17, minute=0)

    open_t = midnight.replace(hour=9, minute=30)
    close_t = midnight.replace(hour=16, minute=0)

    # ---- pre-fetch every sector's data up front so we know how far the
    # latest available bar reaches before deciding where the x-axis ends ----
    fetched = {}
    latest_data_ts = None
    for i, sym in enumerate(SECTORS):
        if i > 0:
            time.sleep(1.0)  # avoid tripping Yahoo's throttling with back-to-back requests
        full_df = dm.fetch_sector_data_yahoo(sym)
        fetched[sym] = full_df
        if full_df is not None and not full_df.empty:
            df_in_window = full_df[(full_df.index >= display_start) & (full_df.index <= max_display_end)]
            if not df_in_window.empty:
                ts = df_in_window.index[-1]
                if latest_data_ts is None or ts > latest_data_ts:
                    latest_data_ts = ts

    if latest_data_ts is not None:
        # small right-hand pad so the last point / end-of-line labels
        # aren't jammed right against the axis edge
        pad = (latest_data_ts - display_start) * 0.03
        display_end = min(latest_data_ts + pad, max_display_end)
        # floor the window width so a chart with only a couple of bars
        # (e.g. right at pre-market open) doesn't collapse to near-zero width
        min_window = timedelta(minutes=30)
        if display_end - display_start < min_window:
            display_end = min(display_start + min_window, max_display_end)
    else:
        display_end = max_display_end

    fig = plt.figure(figsize=(15.5, 7.5), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.97, 0.68], wspace=0.15, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax.set_facecolor("white")

    # ---- shade pre-market / regular / after-hours ----
    ax.axvspan(display_start, open_t, color="#f2f2f2", zorder=0)
    if display_end > close_t:
        ax.axvspan(close_t, display_end, color="#f2f2f2", zorder=0)
    ax.set_xlim(display_start, display_end)

    # ---- baseline at 100 ----
    ax.axhline(100, color="#bbbbbb", lw=0.8, ls="--", zorder=1)

    series_data = []  # (sym, color, x, y, last_value)
    table_stats = {}  # sym -> (dollar_change, pct_change, latest_price)
    all_plot_values = []  # every plotted y-value across all series, for ylim

    for sym in SECTORS:
        color = SECTOR_COLORS[sym]
        full_df = fetched[sym]

        table_stats[sym] = _pct_change_since_prev_close(sym, full_df, midnight)

        # data is fetched from midnight as before; only the DISPLAYED slice
        # (and the rebase-to-100 anchor) runs from 5am to the dynamic
        # display_end computed above.
        df = full_df[(full_df.index >= display_start) & (full_df.index <= display_end)] if full_df is not None else full_df
        if df is None or df.empty:
            continue
        norm = df["close"] / df["close"].iloc[0] * 100

        if SMOOTH_BARS > 1:
            plot_norm = norm.rolling(SMOOTH_BARS, min_periods=1).mean()
        else:
            plot_norm = norm

        ax.plot(df.index, plot_norm, color=color, lw=1.2, alpha=1.0, zorder=2)

        series_data.append((sym, color, df.index[-1], plot_norm.iloc[-1], table_stats[sym][1]))
        all_plot_values.extend(plot_norm.dropna().tolist())

    # XLV isn't charted (dropped to reduce clutter) but is still needed to
    # classify the risk-off group in the side panel.
    time.sleep(1.0)

    # ---- de-overlap end-of-line labels ----
    series_data.sort(key=lambda t: t[3])  # ascending by last plotted value
    y_min, y_max = None, None
    if all_plot_values:
        # Base the axis range on the full extent of every plotted line (not
        # just each line's endpoint), so intraday peaks/troughs away from the
        # last value aren't clipped off the top/bottom of the chart.
        data_min, data_max = min(all_plot_values), max(all_plot_values)
        pad = (data_max - data_min) * 0.15 or 1.0
        y_min, y_max = data_min - pad, data_max + pad
        ax.set_ylim(y_min, y_max)

    min_gap = 0
    if series_data:
        # Measure the *actual* rendered label height in pixels (fontsize +
        # bbox padding) instead of using a fixed fraction of the data range.
        # A fixed fraction only happens to be "enough" for some combinations
        # of data range / figure size / dpi -- when the plotted values are
        # tightly clustered (a quiet market) the same fraction maps to far
        # fewer pixels than the label actually needs, so badges still
        # collide. Measuring in pixels and converting back to data units
        # makes the gap correct regardless of how much the sectors moved.
        fig.canvas.draw()  # need a renderer + finalized axes position first
        renderer = fig.canvas.get_renderer()
        ax_bbox_px = ax.get_window_extent(renderer=renderer)
        pixels_per_data_unit = ax_bbox_px.height / (y_max - y_min)

        label_fontsize_pt = 8.5
        # line height (~1.4x fontsize is typical for a single line of bold
        # text) plus the bbox's own padding (pad=0.3 * fontsize, top+bottom)
        label_height_pt = label_fontsize_pt * 1.4 + (0.3 * label_fontsize_pt * 2)
        label_height_px = label_height_pt * fig.dpi / 72.0
        # extra breathing room so badges aren't touching edge-to-edge
        min_gap = (label_height_px / pixels_per_data_unit) * LABEL_GAP_PAD_FACTOR

    label_ys = [y_last for *_, y_last, _ in series_data]
    # Pass 1 (ascending): push each label up if it's too close to the one
    # below it, guaranteeing no overlap looking upward.
    for i in range(1, len(label_ys)):
        if label_ys[i] - label_ys[i - 1] < min_gap:
            label_ys[i] = label_ys[i - 1] + min_gap
    # Pass 2 (descending): pull labels back down where possible, so a dense
    # cluster near the bottom doesn't drag the *entire* stack upward and
    # leave it drifted far from the actual data -- without this, only
    # upward pushes accumulate and labels for sectors near the top of the
    # range can end up needlessly far above their true value.
    for i in range(len(label_ys) - 2, -1, -1):
        if label_ys[i + 1] - label_ys[i] < min_gap:
            label_ys[i] = label_ys[i + 1] - min_gap

    for (sym, color, x_last, y_last, pct_last), y_label in zip(series_data, label_ys):
        pct_text = f"{pct_last:+.2f}%" if pct_last is not None else "N/A"
        text_color = _contrast_text_color(color)
        ax.annotate(f"{sym} {pct_text}",
                    xy=(1.005, y_label),
                    xycoords=("axes fraction", "data"),
                    color=text_color, fontsize=8.5, va="center", ha="left", clip_on=False,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.35",
                              facecolor=color, edgecolor="none"))
        # leader line from the true data point to the (possibly shifted) label
        if abs(y_label - y_last) > min_gap * 0.3:
            x_last_num = mdates.date2num(x_last)   # match get_xlim()'s float date-space
            ax.plot([x_last_num, ax.get_xlim()[1]], [y_last, y_last],
                    color=color, lw=0.5, alpha=0.5, clip_on=False, zorder=1)

    # ---- reference-style look ----
    ax.grid(True, color="#e2e2e2", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=BLUE_LBL, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    # y-axis numeric labels removed: the colored sector badges at the end
    # of each line already carry the info that mattered here, and dropping
    # the tick labels avoids them cluttering/overlapping the badges.
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_color("#d9d9d9")
    ax.spines["bottom"].set_color("#d9d9d9")
    ax.set_title("SECTOR ETFS Comparison chart",
                 color=BLUE_LBL, fontsize=11, fontweight="bold", loc="left")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=config.EASTERN))

    # ============================================================
    # ---- risk-on / risk-off side panel ----
    # ============================================================
    ax_table.axis("off")
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)

    def _up_down(group):
        ups = sum(1 for s in group
                  if table_stats.get(s) and table_stats[s][1] is not None and table_stats[s][1] > 0)
        total = sum(1 for s in group if table_stats.get(s) and table_stats[s][1] is not None)
        return ups, total

    def _avg_pct(group):
        vals = [table_stats[s][1] for s in group
                if table_stats.get(s) and table_stats[s][1] is not None]
        return sum(vals) / len(vals) if vals else None

    on_up, on_total = _up_down(RISK_ON)
    off_up, off_total = _up_down(RISK_OFF)

    on_majority = on_total > 0 and on_up > on_total / 2
    off_majority = off_total > 0 and off_up > off_total / 2

    if on_majority:
        regime, regime_color = "RISK ON", UP_COLOR
        regime_detail = f"{on_up}/{on_total} risk-on sectors up"
        regime_avg = _avg_pct(RISK_ON)
    elif off_majority:
        regime, regime_color = "RISK OFF", DOWN_COLOR
        regime_detail = f"{off_up}/{off_total} risk-off sectors up"
        regime_avg = _avg_pct(RISK_OFF)
    else:
        regime, regime_color = "N/A", FLAT_COLOR
        regime_detail = "No clear sector rotation"
        regime_avg = None

    detail_text = regime_detail
    if regime_avg is not None:
        detail_text += f"  (avg {regime_avg:+.2f}%)"

    ax_table.text(0.5, 0.985, regime, transform=ax_table.transAxes,
                  fontsize=16, fontweight="bold", color=regime_color,
                  ha="center", va="top")
    ax_table.text(0.5, 0.915, detail_text, transform=ax_table.transAxes,
                  fontsize=8.5, color="#555555", ha="center", va="top", style="italic")
    ax_table.text(0.5, 0.875, "vs. yesterday's regular-hours close",
                  transform=ax_table.transAxes, fontsize=7.5, color="#999999",
                  ha="center", va="top")
    ax_table.plot([0.0, 1.0], [0.845, 0.845], transform=ax_table.transAxes,
                  color="#dddddd", lw=1)

    blocks = [("header", "RISK ON", UP_COLOR)]
    blocks += [("row", s) for s in RISK_ON]
    blocks.append(("header", "RISK OFF", DOWN_COLOR))
    blocks += [("row", s) for s in RISK_OFF]
    blocks.append(("header", "OTHER SECTORS", "#555555"))
    blocks += [("row", s) for s in OTHER_SYMS]

    y = 0.79
    line_h = 0.058
    header_gap = 0.020

    for kind, *rest in blocks:
        if kind == "header":
            text, color = rest
            y -= header_gap
            ax_table.text(0.02, y, text, transform=ax_table.transAxes,
                          fontsize=10, fontweight="bold", color=color, va="top")
            y -= line_h * 0.85
        else:
            sym = rest[0]
            name = TABLE_NAMES.get(sym, "")
            color = SECTOR_COLORS.get(sym, "#333333")
            change, pct, price = table_stats.get(sym, (None, None, None))
            if pct is None:
                pct_str, pct_color = "N/A", FLAT_COLOR
            else:
                arrow = "\u25B2" if pct > 0 else ("\u25BC" if pct < 0 else "\u25AC")
                pct_color = UP_COLOR if pct > 0 else (DOWN_COLOR if pct < 0 else FLAT_COLOR)
                pct_str = f"{arrow} {pct:+.2f}%"
            ax_table.text(0.04, y, sym, transform=ax_table.transAxes, fontsize=9.5,
                          fontweight="bold", color=color, va="top")
            ax_table.text(0.20, y, name, transform=ax_table.transAxes, fontsize=9,
                          color="#333333", va="top")
            ax_table.text(0.98, y, pct_str, transform=ax_table.transAxes, fontsize=9.5,
                          fontweight="bold", color=pct_color, va="top", ha="right")
            y -= line_h

    for spine in ax_table.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.08)
    fig.savefig(output_path, dpi=75)
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    # Allows `python sectorperformance.py` to still generate a chart
    # directly, same as the old top-level-script behavior.
    generate_sector_chart()