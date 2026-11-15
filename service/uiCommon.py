"""Streamlit helpers shared by Dashboard, Volume History and Technical Bias.

    PRESETS / locate_preset / pick_symbol  - symbol selector logic (was duplicated)
    save_snapshots                         - persist one row per volume scope for a chain
"""
from __future__ import annotations

import streamlit as st

from db import insert_snapshot
from gex_calculator import CBOE_INDEX_SYMBOLS, volume_by_scope

# label -> (ticker, is_index); None = custom symbol typed by the user
PRESETS = {
    "SPY": ("SPY", False),
    "QQQ": ("QQQ", False),
    "IWM": ("IWM", False),
    "Custom symbol": None,
}
CUSTOM_LABEL = "Custom symbol"


def locate_preset(symbol: str, default: str = "SPY") -> tuple[int, str, bool]:
    """Map a (query-string) symbol to (selectbox index, custom ticker default,
    custom 'is index' default)."""
    labels = list(PRESETS)
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return labels.index(default), default, False
    for i, val in enumerate(PRESETS.values()):
        if val and val[0] == symbol:
            return i, default, False
    return labels.index(CUSTOM_LABEL), symbol, symbol in CBOE_INDEX_SYMBOLS


def pick_symbol(preset_label: str, custom_default: str = "SPY",
                custom_is_index_default: bool = False) -> tuple[str, bool]:
    """Return (ticker, is_index) for the chosen preset; for 'Custom symbol'
    renders the extra text input / checkbox row."""
    preset = PRESETS[preset_label]
    if preset is not None:
        return preset
    c_symbol, c_index = st.columns([2, 3], vertical_alignment="bottom")
    ticker = c_symbol.text_input(
        "Enter symbol (e.g. SPX, SPY, AAPL, TSLA)", value=custom_default
    ).strip().upper()
    is_index = c_index.checkbox(
        "This is a cash index (adds CBOE's leading underscore)",
        value=custom_is_index_default or ticker in CBOE_INDEX_SYMBOLS,
    )
    return ticker, is_index


def save_snapshots(chain, source: str) -> None:
    """One DB row per scope (All / 0DTE / 1DTE / This Friday) so history builds
    up for every view. Raises on DB errors; callers decide how to report them."""
    for code, (c_vol, p_vol) in volume_by_scope(chain).items():
        insert_snapshot(
            symbol=chain.ticker,
            scope=code,
            snapshot_at_utc=chain.as_of,
            spot=chain.spot,
            call_volume=c_vol,
            put_volume=p_vol,
            source=source,
        )
