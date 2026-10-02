"""
pages/2_Sector_Performance.py

Full-size sector ETF comparison chart (dealer-neutral sector rotation view),
as its own page in the multipage app. Streamlit auto-discovers this file from
the `pages/` directory next to Dashboard.py and lists it in the sidebar.
"""
import base64
import os
import tempfile
import threading

import streamlit as st
import streamlit.components.v1 as components

from service.sectorperformance import generate_sector_chart

st.set_page_config(page_title="Sector Performance", layout="wide")

st.title("Sector performance")

# matplotlib's pyplot state isn't thread-safe and Streamlit runs each session
# in its own thread -- serialize chart builds.
_CHART_LOCK = threading.Lock()

# Reference tables shown under the chart. Styles are scoped to .sector-tables
# so they don't restyle the rest of the Streamlit app. (No indentation inside
# the HTML: 4+ leading spaces would make st.markdown treat it as a code block.)
REFERENCE_TABLES_HTML = """
<style>
.sector-tables .tables-row { display: flex; gap: 40px; flex-wrap: wrap; }
.sector-tables table { border-collapse: collapse; }
.sector-tables th, .sector-tables td { border: 1px solid #ddd; padding: 8px; text-align: left; }
.sector-tables th { background-color: #f2f2f2; color: #333; }
</style>
<div class="sector-tables">
<div class="tables-row">
<div>
<table>
<tr><th>Cycle-phases</th><th>Sectors</th></tr>
<tr><td>Early expansion</td><td>XLK, XLY, XLF</td></tr>
<tr><td>Mid expansion</td><td>XLI, XLK, XLV</td></tr>
<tr><td>Late expansion</td><td>XLE, XLB</td></tr>
<tr><td>Contraction</td><td>XLU, XLP, XLV, XLRE</td></tr>
<tr><td>Not-in-phase</td><td>XLC</td></tr>
</table>
</div>
<div>
<table>
<tr><th>Sector symbols</th><th>Mega volume stocks</th></tr>
<tr><td>XLP</td><td>COST, WMT</td></tr>
<tr><td>XLC</td><td>GOOG, META, NFLX</td></tr>
<tr><td>XLY</td><td>AMZN, TSLA</td></tr>
</table>
</div>
</div>
</div>
"""

# Pinch-zoom / drag-pan / double-tap-reset viewer. Plain string (not an
# f-string) so JS braces need no escaping; __B64__ / __H__ are substituted.
ZOOM_VIEWER_HTML = """
<div id="box" style="width:100%;height:__H__px;overflow:hidden;position:relative;
     background:#fff;touch-action:pan-y;">
  <img id="img" src="data:image/png;base64,__B64__"
       style="width:100%;position:absolute;top:0;left:0;transform-origin:0 0;
              user-select:none;-webkit-user-drag:none;">
</div>
<script>
const box = document.getElementById('box');
const img = document.getElementById('img');
const MAX_SCALE = 6, DOUBLE_TAP_MS = 300;
let s = 1, x = 0, y = 0, last = null, lastTap = 0;

function fitHeight() {
  // size the box (and, if allowed, the iframe) to the image's aspect ratio
  if (!img.naturalWidth) return;
  const h = Math.round(box.clientWidth * img.naturalHeight / img.naturalWidth);
  box.style.height = h + 'px';
  try { window.frameElement.style.height = (h + 4) + 'px'; } catch (e) {}
  apply();
}

function apply() {
  const w = box.clientWidth, h = box.clientHeight;
  // clamp so the image can't be dragged out of view
  x = Math.min(0, Math.max(x, w - w * s));
  y = Math.min(0, Math.max(y, h - h * s));
  img.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + s + ')';
  // when not zoomed, let one-finger swipes scroll the page
  box.style.touchAction = s > 1 ? 'none' : 'pan-y';
}

const dist = t => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
const mid = t => ({x: (t[0].clientX + t[1].clientX) / 2, y: (t[0].clientY + t[1].clientY) / 2});

box.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    const now = Date.now();
    if (now - lastTap < DOUBLE_TAP_MS) { s = 1; x = y = 0; apply(); }  // reset
    lastTap = now;
    last = {x: e.touches[0].clientX, y: e.touches[0].clientY};
  } else if (e.touches.length === 2) {
    last = {d: dist(e.touches), m: mid(e.touches)};
  }
}, {passive: true});

box.addEventListener('touchmove', e => {
  if (e.touches.length === 2 && last && last.d) {
    e.preventDefault();
    const d = dist(e.touches), m = mid(e.touches), r = box.getBoundingClientRect();
    const ns = Math.min(MAX_SCALE, Math.max(1, s * d / last.d));
    const px = m.x - r.left, py = m.y - r.top;
    x = px - (px - x) * ns / s;   // zoom around the pinch center
    y = py - (py - y) * ns / s;
    x += m.x - last.m.x;          // follow the fingers while pinching
    y += m.y - last.m.y;
    s = ns; last = {d: d, m: m}; apply();
  } else if (e.touches.length === 1 && s > 1 && last && !last.d) {
    e.preventDefault();
    const t = e.touches[0];
    x += t.clientX - last.x; y += t.clientY - last.y;
    last = {x: t.clientX, y: t.clientY}; apply();
  }
}, {passive: false});

box.addEventListener('touchend', e => {
  // re-seed the pan anchor if one finger is lifted after a pinch
  if (e.touches.length === 1) {
    last = {x: e.touches[0].clientX, y: e.touches[0].clientY};
  }
}, {passive: true});

img.addEventListener('load', fitHeight);
window.addEventListener('resize', fitHeight);
if (img.complete) fitHeight();
</script>
"""


def zoomable_image(png_bytes: bytes, fallback_height: int = 360) -> None:
    """Show a PNG with touch pinch-zoom, drag-pan and double-tap reset.

    The viewer resizes itself to the image's aspect ratio once loaded;
    fallback_height is only the initial iframe height before that happens.
    """
    b64 = base64.b64encode(png_bytes).decode()
    html = (ZOOM_VIEWER_HTML
            .replace("__B64__", b64)
            .replace("__H__", str(fallback_height)))
    components.html(html, height=fallback_height + 4)


@st.cache_data(ttl=300, show_spinner=False)  # reuse for 5 min across visitors
def _build_chart_bytes(refresh_key: int) -> bytes:
    """Run generate_sector_chart() and return the PNG bytes.

    refresh_key only exists so the Refresh button yields a new cache entry.
    """
    with _CHART_LOCK:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            generate_sector_chart(output_path=path)
            with open(path, "rb") as f:
                return f.read()
        finally:
            os.remove(path)


def render_sector_chart(compact: bool = False, key_prefix: str = "sector") -> None:
    """Render the sector performance chart with a refresh control.

    compact=True: smaller heading, used for the landing-page preview.
    compact=False: full heading + caption, used on the dedicated page.
    """
    if "sector_chart_refresh_key" not in st.session_state:
        st.session_state.sector_chart_refresh_key = 0

    header_col, btn_col = st.columns([6, 1])
    with header_col:
        if compact:
            st.subheader("Sector performance")
        else:
            st.caption(
                "Normalized intraday performance across the SPDR sector "
            )

    try:
        with st.spinner("Building sector chart..."):
            png_bytes = _build_chart_bytes(st.session_state.sector_chart_refresh_key)
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't build the sector chart: {e}")
        return

    zoomable_image(png_bytes)
    st.caption("Pinch to zoom, drag to pan, double-tap to reset.")

    # Lets SectorProcessor (Playwright) fetch the PNG by clicking this button.
    st.download_button(
        "Download chart",
        data=png_bytes,
        file_name="sector_performance.png",
        mime="image/png",
        key=f"{key_prefix}_download",
        on_click="ignore",  # don't rerun the script (and rebuild) on click; needs Streamlit >= 1.43
    )

    if compact:
        st.caption("See the **Sector Performance** page (sidebar) for the full-size chart.")


render_sector_chart(compact=False, key_prefix="sector_page")

st.markdown(REFERENCE_TABLES_HTML, unsafe_allow_html=True)
