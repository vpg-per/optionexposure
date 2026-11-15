"""Streamlit widgets for put-call-ratio (PCR) numbers, shared by the Volume
History and Technical Bias pages. Pure calculations stay in pcrAnalysis.py."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def _hhmm(ts) -> str:
    return f"{pd.Timestamp(ts):%H:%M}"


def render_change_metrics(info: dict, cols=None) -> None:
    """Latest PCR, 'Change vs previous snapshot' and previous PCR.
    `info` is pcrAnalysis.pcr_change_info(df).
    `cols` = containers for the metrics. Three (latest, change, previous) is the compact form;
    a fourth adds 'Calculated at': the snapshot time (ET) the change was calculated for.
    Defaults to a fresh st.columns(4)."""
    cols = cols or st.columns(4)
    m_latest, m_change, m_prev = cols[:3]
    m_time = cols[3] if len(cols) > 3 else None

    if info["y"] is None:
        m_latest.metric("Put-call ratio (latest)", "n/a")
        m_change.metric("Change vs previous snapshot", "n/a")
        m_change.caption("No snapshots with a put-call ratio yet.")
        if m_time is not None:
            m_time.metric("Calculated at (ET)", "n/a")
        return

    m_latest.metric("Put-call ratio (latest)", f"{info['y']:.4f}",
                    help=f"Snapshot at {_hhmm(info['x'])} ET")
    if info["diff"] is None:
        m_change.metric("Change vs previous snapshot", "n/a")
        m_change.caption("Needs at least 2 snapshots with a put-call ratio.")
        if m_time is not None:
            m_time.metric("Calculated at (ET)", "n/a")
        return

    pct = f" ({info['diff'] / info['prev_y']:+.1%})" if info["prev_y"] else ""
    # "inverse": a rising ratio is red, a falling one green (matches the chart). Unchanged = grey.
    m_change.metric("Change vs previous snapshot", info["state"],
                    delta=f"{info['diff']:+.4f}{pct}",
                    delta_color="off" if info["state"] == "No change" else "inverse",
                    help=f"{_hhmm(info['prev_x'])} -> {_hhmm(info['x'])} ET")
    m_prev.metric("Previous put-call ratio", f"{info['prev_y']:.4f}",
                  help=f"Snapshot at {_hhmm(info['prev_x'])} ET")
    if m_time is not None:
        m_time.metric("Calculated at (ET)", _hhmm(info["x"]),
                      help=f"Change = snapshot at {_hhmm(info['x'])} minus snapshot at {_hhmm(info['prev_x'])}")
        m_time.caption(f"vs previous snapshot at {_hhmm(info['prev_x'])}")


def render_reversal_metric(rev: dict, col=None) -> None:
    """PCR reversal metric; shown only when the latest snapshot flipped the direction."""
    col = col or st
    if rev["on_last"]:
        col.metric(f"PCR reversal ({rev['state']})", f"{rev['value']:.4f}",
                   help=f"Turning point at {_hhmm(rev['time'])} ET, confirmed by the latest snapshot")
    else:
        col.metric("PCR reversal", "No change",
                   help="The latest snapshot did not reverse the ratio's direction.")
