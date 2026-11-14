"""Put-call ratio (PCR) analysis shared by the Volume History and Technical Bias pages.

    load_pcr_snapshots  - DB snapshots for a symbol/day/scope (optionally cut off at a time)
    pcr_change_info     - change between the last two snapshots (chart annotation / PNG metadata)
    pcr_reversal_info   - most recent turning point of the PCR (direction flip)

Colour convention (market read): a falling PCR is green, a rising PCR is red.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from db import load_snapshots
from gex_calculator import VOLUME_SCOPES

GREEN, RED = "#2ECC71", "#E74C3C"

# Scope code of the 0DTE option (second entry of VOLUME_SCOPES).
ZERO_DTE_SCOPE = VOLUME_SCOPES[1][0]


def load_pcr_snapshots(symbol: str, day: dt.date, scope: str,
                       until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Snapshots for `symbol` on Eastern market date `day`.
    `until` (naive or tz-aware Eastern) keeps only snapshots taken at or before it."""
    df = load_snapshots(symbol, day, scope)
    if until is not None and not df.empty:
        cutoff = pd.Timestamp(until)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        df = df[pd.to_datetime(df["downloaded_at"]) <= cutoff]
    return df


def pcr_change_info(df_plot: pd.DataFrame) -> dict:
    """
    Change in put-call ratio between the last two snapshots that have a ratio.
    Single source of truth for the chart annotation, the metrics and the PNG metadata.

    Annotation convention (arrow shows the market read, sign shows the PCR move):
        PCR fell  -> green "▲ -0.0215"   state "Down"
        PCR rose  -> red   "▼ +0.0200"   state "Up"
        unchanged -> no annotation        state "No change"
    """
    valid = df_plot.dropna(subset=["put_call_ratio"])
    info = {"state": "unavailable", "diff": None, "label": None, "color": None, "x": None, "y": None,
            "prev_x": None, "prev_y": None}
    if valid.empty:
        return info
    info["x"] = valid["downloaded_at"].iloc[-1]
    info["y"] = float(valid["put_call_ratio"].iloc[-1])
    if len(valid) < 2:
        return info
    info["prev_x"] = valid["downloaded_at"].iloc[-2]
    info["prev_y"] = float(valid["put_call_ratio"].iloc[-2])

    diff = round(float(valid["put_call_ratio"].iloc[-1] - valid["put_call_ratio"].iloc[-2]), 4)
    info["diff"] = diff
    if diff > 0:
        info.update(state="Up", label=f"▼ {diff:+.4f}", color=RED)
    elif diff < 0:
        info.update(state="Down", label=f"▲ {diff:+.4f}", color=GREEN)
    else:
        info.update(state="No change")
    return info


def pcr_reversal_info(df_plot: pd.DataFrame) -> dict:
    """
    Most recent reversal of the put-call ratio: the direction of the move flips
    (unchanged snapshots are ignored). The turning point is the snapshot where the
    new direction starts.

        "Up→Down"  ratio was rising, now falling  (green)
        "Down→Up"  ratio was falling, now rising  (red)
        "None"     no flip found (fewer than two moves, or one direction only)

    Returns state, value (ratio at the turning point), time (its snapshot time),
    ago (snapshots since the turning point), label ("Up→Down @ 0.8123"), color, and
    on_last: True when the flip was confirmed by the latest snapshot itself (the last
    move is the first one in the new direction), so callers can show it only when fresh.
    """
    info = {"state": "None", "value": None, "time": None, "ago": None, "label": None, "color": None,
            "on_last": False}
    valid = df_plot.dropna(subset=["put_call_ratio"])
    if len(valid) < 3:
        return info

    vals = valid["put_call_ratio"].to_numpy(dtype="float64")
    moves = np.round(np.diff(vals), 4)
    nz = np.flatnonzero(moves)                 # positions of real (non-flat) moves
    if len(nz) < 2:
        return info
    sign = np.sign(moves[nz])
    flips = np.flatnonzero(sign[1:] != sign[:-1])
    if flips.size == 0:
        return info

    k = int(nz[flips[-1] + 1])                 # first move in the new direction starts at snapshot k
    rising_now = sign[flips[-1] + 1] > 0
    state = "Down→Up" if rising_now else "Up→Down"
    info.update(state=state, value=float(vals[k]), time=valid["downloaded_at"].iloc[k],
                ago=len(vals) - 1 - k, color=RED if rising_now else GREEN,
                on_last=(k == len(vals) - 2))
    info["label"] = f"{state} @ {info['value']:.4f}"
    return info
