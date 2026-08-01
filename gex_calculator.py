"""
gex_calculator.py

Core gamma exposure (GEX) calculation engine.

Data source: CBOE's public delayed-quotes JSON endpoint --
    https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json
This is the same underlying feed behind cboe.com/delayed_quotes and is what
the old "quote-table-download" .dat file was generated from -- just as JSON,
fetched live, with no manual download step. It returns bid/ask, open
interest, implied vol, and CBOE's own computed greeks (delta, gamma, vega,
theta) for every listed contract across all expirations, plus the current
underlying price.

Index symbols use a leading underscore, e.g.:
    _SPX  -> S&P 500 index options
    _NDX  -> Nasdaq 100 index options
    _RUT  -> Russell 2000 index options
    _VIX  -> VIX index options
    _DJX  -> Dow Jones index options
Equity/ETF tickers (SPY, QQQ, AAPL, ...) use no prefix.

Methodology (standard dealer-gamma convention):
  - Assume dealers are net long calls and net short puts (the common retail
    approximation used by most public GEX dashboards, since real dealer
    positioning isn't published).
  - Gamma exposure per strike, in $ per 1% move in the underlying:
        Call GEX =  OpenInterest_call * Gamma_call * ContractSize * Spot^2 * 0.01
        Put  GEX = -OpenInterest_put  * Gamma_put  * ContractSize * Spot^2 * 0.01
        Net GEX  =  Call GEX + Put GEX
  - Gamma is taken directly from CBOE's own per-contract greek field. If it's
    missing or zero for a contract (can happen for stale/no-trade strikes),
    it's recomputed from Black-Scholes using CBOE's reported IV as a fallback.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

CONTRACT_SIZE = 100
CBOE_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
USER_AGENT = "Mozilla/5.0 (compatible; gex-dashboard/1.0)"

# Common index tickers -> their CBOE JSON symbol (leading underscore).
CBOE_INDEX_SYMBOLS = {
    "SPX": "_SPX",
    "SPXW": "_SPXW",
    "NDX": "_NDX",
    "RUT": "_RUT",
    "VIX": "_VIX",
    "DJX": "_DJX",
    "OEX": "_OEX",
    "XSP": "_XSP",
}

# OCC-style option symbol, e.g. "SPX   250815C05500000" or "SPXW250815C05500000"
_OPT_SYMBOL_RE = re.compile(r"^(?P<root>.+?)(?P<date>\d{6})(?P<right>[CP])(?P<strike>\d+)$")


def resolve_cboe_symbol(ticker: str, is_index: bool | None = None) -> str:
    """
    Map a user-facing ticker to the symbol CBOE's JSON endpoint expects.
    If `is_index` is None, auto-detect using the known index list.
    """
    t = ticker.strip().upper().lstrip("_")
    if is_index is None:
        is_index = t in CBOE_INDEX_SYMBOLS
    if is_index:
        return CBOE_INDEX_SYMBOLS.get(t, f"_{t}")
    return t


# --------------------------------------------------------------------------
# Black-Scholes gamma (fallback only -- CBOE normally supplies gamma directly)
# --------------------------------------------------------------------------
def bs_gamma(spot: float, strike: float, t_years: float, iv: float, r: float = 0.05) -> float:
    """
    Black-Scholes gamma (identical for calls and puts).
    Returns 0 for degenerate inputs (expired/zero iv) instead of raising.
    """
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * np.sqrt(t_years))
    return norm.pdf(d1) / (spot * iv * np.sqrt(t_years))


def parse_occ_symbol(symbol: str) -> dict | None:
    """Parse an OCC-style option symbol into root/expiry/type/strike."""
    m = _OPT_SYMBOL_RE.match(symbol.strip())
    if not m:
        return None
    root = m.group("root").strip()
    date_str = m.group("date")
    right = "call" if m.group("right") == "C" else "put"
    strike = int(m.group("strike")) / 1000.0
    try:
        expiry = dt.datetime.strptime("20" + date_str, "%Y%m%d")
    except ValueError:
        return None
    return {"root": root, "expiry": expiry, "type": right, "strike": strike}


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass
class GexResult:
    ticker: str
    spot: float
    as_of: dt.datetime
    expirations_used: list
    by_strike: pd.DataFrame          # strike, call_gex, put_gex, net_gex
    total_gex: float
    gamma_flip: float | None
    call_wall: float | None
    put_wall: float | None
    regime: str                      # "positive" | "negative"
    warnings: list = field(default_factory=list)


# --------------------------------------------------------------------------
# CBOE fetch
# --------------------------------------------------------------------------
def fetch_cboe_json(symbol: str, session: requests.Session | None = None, timeout: int = 15) -> dict:
    """
    Hits CBOE's public delayed-quotes JSON endpoint for a single symbol.
    `symbol` should already be resolved (e.g. "_SPX", "SPY") -- use
    `resolve_cboe_symbol()` first if you have a bare ticker.
    """
    url = CBOE_BASE_URL.format(symbol=symbol)
    sess = session or requests
    resp = sess.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "data" not in payload or "options" not in payload["data"]:
        raise ValueError(f"Unexpected CBOE response shape for symbol '{symbol}'.")
    return payload


def cboe_json_to_dataframe(payload: dict) -> tuple[pd.DataFrame, float]:
    """
    Flattens the CBOE JSON payload into a per-contract dataframe with
    parsed strike/expiry/type columns, plus returns the current spot price.
    """
    data = payload["data"]
    spot = float(data.get("current_price") or data.get("close") or 0.0)

    records = []
    for opt in data["options"]:
        parsed = parse_occ_symbol(opt.get("option", ""))
        if parsed is None:
            continue
        records.append(
            {
                "strike": parsed["strike"],
                "expiry": parsed["expiry"],
                "type": parsed["type"],
                "open_interest": opt.get("open_interest", 0) or 0,
                "iv": opt.get("iv", 0) or 0,
                "gamma": opt.get("gamma", 0) or 0,
                "bid": opt.get("bid", 0) or 0,
                "ask": opt.get("ask", 0) or 0,
            }
        )
    df = pd.DataFrame.from_records(records)
    return df, spot


# --------------------------------------------------------------------------
# Fetch (raw chain only -- cheap to re-slice afterwards without refetching)
# --------------------------------------------------------------------------
@dataclass
class ChainData:
    ticker: str
    symbol: str
    spot: float
    as_of: dt.datetime
    raw: pd.DataFrame        # strike, expiry, type, open_interest, iv, gamma, gex_mm
    warnings: list = field(default_factory=list)


def fetch_chain(
    ticker: str,
    is_index: bool | None = None,
    risk_free_rate: float = 0.05,
    session: requests.Session | None = None,
) -> ChainData:
    """
    Pulls the FULL option chain (every expiration CBOE lists) for `ticker`
    and computes per-contract GEX once. Expiration and strike-window
    selection happen afterwards on this cached dataframe -- no need to hit
    CBOE again just because the user changed which expiries to include.

    `is_index`: True forces the CBOE index symbol format (leading
    underscore, e.g. "_SPX"); False forces the plain equity/ETF format;
    None auto-detects using the known index list (SPX, NDX, RUT, VIX, ...).
    """
    warnings: list[str] = []
    symbol = resolve_cboe_symbol(ticker, is_index=is_index)

    payload = fetch_cboe_json(symbol, session=session)
    raw, spot = cboe_json_to_dataframe(payload)

    if spot <= 0:
        raise ValueError(f"Could not resolve a spot price for '{ticker}' ({symbol}).")
    if raw.empty:
        raise ValueError(f"No parseable option contracts returned for '{ticker}' ({symbol}).")

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    # Drop anything already expired (keep same-day/0DTE, which is still live
    # for the whole trading day -- compare calendar dates, not exact
    # timestamps, since parsed expiries carry no time-of-day component).
    raw = raw[raw["expiry"].dt.date >= now.date()].copy()
    if raw.empty:
        raise ValueError(f"No upcoming expirations found for '{ticker}' ({symbol}).")

    # CBOE supplies per-contract gamma directly; fall back to Black-Scholes
    # (using CBOE's own IV) only where gamma is missing/zero.
    t_years = (raw["expiry"] - now).dt.total_seconds().clip(lower=0) / (365.0 * 24 * 3600)
    t_years = t_years.clip(lower=1.0 / 252 / 24)  # floor so 0DTE gamma doesn't vanish
    needs_fallback = raw["gamma"].fillna(0) <= 0
    if needs_fallback.any():
        fallback_gamma = [
            bs_gamma(spot, k, t, iv, risk_free_rate)
            for k, t, iv in zip(raw.loc[needs_fallback, "strike"], t_years[needs_fallback], raw.loc[needs_fallback, "iv"])
        ]
        raw.loc[needs_fallback, "gamma"] = fallback_gamma
        warnings.append(f"{int(needs_fallback.sum())} contracts had no CBOE gamma; used Black-Scholes fallback.")

    # $ GEX per 1% move in the underlying, in millions -- computed once here
    # so downstream expiration/strike filtering is a cheap pandas slice.
    raw["gex_raw"] = raw["open_interest"] * raw["gamma"] * CONTRACT_SIZE * spot ** 2 * 0.01
    raw["gex_raw"] = np.where(raw["type"] == "put", -raw["gex_raw"], raw["gex_raw"])
    raw["gex_mm"] = raw["gex_raw"] / 1_000_000

    return ChainData(ticker=ticker.upper(), symbol=symbol, spot=float(spot), as_of=now, raw=raw, warnings=warnings)


# --------------------------------------------------------------------------
# Expiration listing + labeling (0DTE / 1DTE / this Friday)
# --------------------------------------------------------------------------
def list_expirations(chain: ChainData) -> list[dt.datetime]:
    return sorted(chain.raw["expiry"].unique())


def label_expirations(expirations: list[dt.datetime], as_of: dt.datetime) -> dict:
    """
    Tags each expiration with a human label where applicable:
      "0DTE"      -> expires today
      "1DTE"      -> expires the next calendar day
      "This Fri"  -> the nearest upcoming Friday (weekly index expiration),
                     unless it's already tagged 0DTE/1DTE
    Untagged dates get "".
    """
    today = as_of.date()
    tomorrow = today + dt.timedelta(days=1)
    fridays = [e for e in expirations if e.date().weekday() == 4 and e.date() >= today]
    nearest_friday = min(fridays, key=lambda e: e.date()) if fridays else None

    labels = {}
    for e in expirations:
        d = e.date()
        if d == today:
            labels[e] = "0DTE"
        elif d == tomorrow:
            labels[e] = "1DTE"
        elif nearest_friday is not None and d == nearest_friday.date():
            labels[e] = "This Fri"
        else:
            labels[e] = ""
    return labels


def default_expiration_selection(expirations: list[dt.datetime], as_of: dt.datetime) -> list[dt.datetime]:
    """0DTE only if present; otherwise just the nearest expiration."""
    labels = label_expirations(expirations, as_of)
    picked = [e for e, lbl in labels.items() if lbl == "0DTE"]
    return picked if picked else expirations[:1]


# --------------------------------------------------------------------------
# Compute GEX for a chosen subset of expirations
# --------------------------------------------------------------------------
def compute_gex(
    chain: ChainData,
    selected_expirations: list[dt.datetime],
    min_open_interest: int = 10,
) -> GexResult:
    raw = chain.raw[chain.raw["expiry"].isin(selected_expirations)].copy()
    raw = raw[raw["open_interest"].fillna(0) >= min_open_interest]
    if raw.empty:
        raise ValueError(
            f"No option contracts with open interest >= {min_open_interest} "
            f"found for the selected expiration(s)."
        )

    by_strike = (
        raw.pivot_table(index="strike", columns="type", values="gex_mm", aggfunc="sum", fill_value=0.0)
        .reindex(columns=["call", "put"], fill_value=0.0)
        .reset_index()
        .rename(columns={"call": "call_gex", "put": "put_gex"})
    )
    by_strike["net_gex"] = by_strike["call_gex"] + by_strike["put_gex"]
    by_strike = by_strike.sort_values("strike").reset_index(drop=True)

    total_gex = float(by_strike["net_gex"].sum())
    gamma_flip = _find_zero_crossing(by_strike["strike"].values, by_strike["net_gex"].values)

    call_wall = None
    put_wall = None
    if not by_strike.empty:
        call_wall = float(by_strike.loc[by_strike["call_gex"].idxmax(), "strike"])
        put_wall = float(by_strike.loc[by_strike["put_gex"].idxmin(), "strike"])

    regime = "positive" if total_gex >= 0 else "negative"

    return GexResult(
        ticker=chain.ticker,
        spot=chain.spot,
        as_of=chain.as_of,
        expirations_used=[d.strftime("%Y-%m-%d") for d in sorted(selected_expirations)],
        by_strike=by_strike,
        total_gex=total_gex,
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
        regime=regime,
        warnings=list(chain.warnings),
    )


def nearest_strikes(by_strike: pd.DataFrame, spot: float, n: int) -> pd.DataFrame:
    """
    Returns the `n` strikes above and `n` strikes below the strike closest
    to `spot` (i.e. up to 2n+1 rows total) -- a strike-count window rather
    than a price-percentage window.
    """
    if by_strike.empty or n <= 0:
        return by_strike
    strikes = by_strike["strike"].to_numpy()
    atm_idx = int(np.argmin(np.abs(strikes - spot)))
    lo = max(0, atm_idx - n)
    hi = min(len(strikes), atm_idx + n + 1)
    return by_strike.iloc[lo:hi].reset_index(drop=True)


# --------------------------------------------------------------------------
# Convenience one-shot wrapper (fetch + auto-pick expirations + compute)
# --------------------------------------------------------------------------
def fetch_and_compute_gex(
    ticker: str,
    is_index: bool | None = None,
    max_expirations: int = 6,
    min_open_interest: int = 10,
    risk_free_rate: float = 0.05,
    session: requests.Session | None = None,
) -> GexResult:
    """
    One-shot convenience wrapper: fetch the full chain, take the nearest
    `max_expirations` expirations chronologically, and compute GEX. Useful
    for scripts/tests; the interactive dashboard instead calls
    `fetch_chain()` once and `compute_gex()` repeatedly so expiration/strike
    toggles don't refetch from CBOE.
    """
    chain = fetch_chain(ticker, is_index=is_index, risk_free_rate=risk_free_rate, session=session)
    expirations = list_expirations(chain)[:max_expirations]
    return compute_gex(chain, expirations, min_open_interest=min_open_interest)


def _find_zero_crossing(x: np.ndarray, y: np.ndarray) -> float | None:
    """Linear-interpolate the strike where cumulative/curve GEX crosses zero."""
    if len(x) < 2:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    sign_changes = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
