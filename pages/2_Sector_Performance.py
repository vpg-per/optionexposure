"""
pages/2_Sector_Performance.py

Full-size sector ETF comparison chart (dealer-neutral sector rotation view),
as its own page in the multipage app. Streamlit auto-discovers this file from
the `pages/` directory next to Dashboard.py and lists it in the sidebar.
"""
import os
import tempfile
import threading

import streamlit as st

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

    st.image(png_bytes, use_container_width=True)

    if compact:
        st.caption("See the **Sector Performance** page (sidebar) for the full-size chart.")


render_sector_chart(compact=False, key_prefix="sector_page")

st.markdown(REFERENCE_TABLES_HTML, unsafe_allow_html=True)