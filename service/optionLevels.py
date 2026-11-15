"""Key option strikes: the calls and puts with the highest volume and/or GEX.

    load_level_table  - fetch the chain and aggregate GEX / volume / OI by strike (0DTE)
    top_strikes       - rank one side (call or put) and return the top N strikes

No Streamlit here; the page wraps load_level_table in st.cache_data.
"""
from __future__ import annotations

import pandas as pd

from gex_calculator import compute_gex, fetch_chain, list_expirations, utc_naive_to_eastern_naive

TOP_N = 3
# Ranking options shown on the page. Combined = 50% volume + 50% |GEX|, each scaled to its max.
RANK_MODES = ("Combined", "Volume", "GEX")


def zero_dte_expirations(chain) -> tuple[list, str]:
    """0DTE expiration(s) of the chain; falls back to the nearest one (same rule as the Dashboard).
    Uses the Eastern market date, like volume_by_scope (which feeds the PCR snapshots), so the
    strikes and the put-call ratio on the Technical page always describe the same expiration.
    (label_expirations compares against the UTC date, which is already 'tomorrow' after 8pm ET.)"""
    today = utc_naive_to_eastern_naive(chain.as_of).date()
    exps = list_expirations(chain)
    zero = [e for e in exps if e.date() == today]
    return (zero, "0DTE") if zero else (exps[:1], "Nearest expiry")


def load_level_table(ticker: str, is_index: bool = False) -> dict:
    """Live chain -> strike-level table for the 0DTE expiration."""
    chain = fetch_chain(ticker=ticker, is_index=is_index, save_json=False)
    exps, exp_label = zero_dte_expirations(chain)
    result = compute_gex(chain, exps, min_open_interest=0)
    return {
        "by_strike": result.by_strike.copy(),
        "spot": float(result.spot),
        "as_of": chain.as_of,                       # naive UTC
        "exp_label": exp_label,
        "expirations": [e.strftime("%Y-%m-%d") for e in exps],
    }


def top_strikes(by_strike: pd.DataFrame, spot: float, side: str,
                n: int = TOP_N, rank_by: str = "Combined") -> pd.DataFrame:
    """Top `n` strikes for side 'call' or 'put'.

    Columns: Strike, Volume, Open int., GEX ($mm).
    Put GEX is negative (dealer convention); ranking uses its absolute size.
    Ties are broken by closeness to spot; strikes with no volume and no GEX are dropped."""
    if side not in ("call", "put"):
        raise ValueError("side must be 'call' or 'put'")
    d = pd.DataFrame({
        "Strike": by_strike["strike"],
        "Volume": by_strike[f"{side}_volume"],
        "Open int.": by_strike[f"{side}_oi"],
        "GEX ($mm)": by_strike[f"{side}_gex"],
    })
    d["_gex"] = d["GEX ($mm)"].abs()
    d = d[(d["Volume"] > 0) | (d["_gex"] > 0)].copy()
    if d.empty:
        return d.drop(columns="_gex")

    v_max, g_max = d["Volume"].max(), d["_gex"].max()
    v_part = d["Volume"] / v_max if v_max > 0 else 0.0
    g_part = d["_gex"] / g_max if g_max > 0 else 0.0
    d["_score"] = {"Volume": v_part, "GEX": g_part}.get(rank_by, 0.5 * v_part + 0.5 * g_part)
    d["_dist"] = (d["Strike"] - spot).abs()   # tie-break only; not shown

    d = d.sort_values(["_score", "_dist"], ascending=[False, True]).head(n)
    return d.drop(columns=["_gex", "_score", "_dist"]).reset_index(drop=True)
