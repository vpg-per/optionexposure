"""Support/resistance levels for day trading, built from bars the project
already fetches, and gated by the time the application RUNS ("as of" time).

Level families:
  * previous day high/low/close (PDH/PDL/PDC) + classic pivot P/R1/S1
    -- always available (last regular session before the day)
  * pre-market high/low (PMH/PML, 04:00-09:30 ET)
    -- ONLY when the run is at/after 09:30; before the open they are
       ignored per strategy spec ("they don't exist at that time")
  * opening range high/low (ORH/ORL, 09:30-10:00 ET)
    -- ONLY once the range is complete (run at/after 10:00)
  * day high/low incl. pre-market (DH/DL, 04:00-16:00 ET so far)
    -- ONLY for runs at/after 13:00

Pure functions on a 5-min OHLCV frame (no network, no config, no state).

    sr = build_levels(df, day, now)   # now = tz-aware "as of" time
    sr.labeled()                      # {"PDH": ..., "ORH": ..., "DH": ...}
    levels_line(sr)                   # "PDH 602.1 PDL 595.0 ..." (available families)
    nearest_line(sr, price)           # "R 602.1 (0.14%) | R 605.3 (0.67%) | S 601.0 (0.04%) ..."
    nearest_levels(sr, price)         # closest resistance above / support below
    distance_to_level(price, lvl)     # signed %-distance (neg = price below it)

Session conventions (US/Eastern, bar-start timestamps):
  * "previous day" = the last date BEFORE `day` with REGULAR-session bars
    (09:30 <= t < 16:00). After-hours (16:00-20:00) and overnight (boats)
    bars are excluded on purpose: a thin after-hours spike must not become
    the "previous day high".
  * Bars with start time >= `now` are dropped, so replaying a past bar
    only sees the levels a live run would have seen then. Callers should
    still pass already-closed bars (the fetch path drops the forming bar).

All levels are floats rounded to 2 decimals, matching the rest of the project.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as dt_date, time as dt_time

import pandas as pd

REGULAR_START = dt_time(9, 30)
REGULAR_END = dt_time(16, 0)
PREMARKET_START = dt_time(4, 0)
PREMARKET_END = dt_time(9, 30)

# When each level family becomes available during the trading day (ET):
PREMARKET_READY = dt_time(9, 30)       # pre-market levels ignored before the open
OPENING_RANGE_END = dt_time(10, 0)     # 09:30-10:00 range complete at 10:00
DAY_EXTREMES_READY = dt_time(13, 0)   # day high/low incl. pre-market from 13:00

# Nearest-level display (nearest_named / levels_summary)
NEAR_BEFORE_OR = 2        # levels per side before the opening range is complete (< 10:00)
NEAR_AFTER_OR = 3         # levels per side from 10:00
MERGE_PCT = 0.03          # levels closer than this % are shown as one (e.g. PDL+PML)
EXT_SESSIONS = 5          # prior sessions for the 5-day high/low extension levels


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DailyLevels:
    """OHLC summary of one regular session."""
    day: dt_date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class SessionExtremes:
    """High/low of one ET session slice on one trading day."""
    day: dt_date
    start: dt_time
    end: dt_time
    high: float
    low: float
    high_time: pd.Timestamp | None   # start of the 5m bar that made the high
    low_time: pd.Timestamp | None
    bars: int


@dataclass(frozen=True)
class SupportResistance:
    """All S/R candidates for one trading day, as of `now`.

    resistance: unique candidates strictly above the reference price (previous
                day close), sorted DESCENDING (highest first).
    support:    unique candidates strictly below it, sorted ASCENDING.
    candidates: every level candidate (PDH, PDL, R1, S1 and any available
                PM/OR/day high-low), ascending. nearest_levels() uses this
                pool and splits it around the CURRENT price, not around the
                reference close.
    The close itself and the pivot P are context, not directional levels.
    Families not yet available at `now` (pre-market before 09:30, opening
    range before 10:00, day extremes before 13:00) are None.
    """
    day: dt_date
    asof: pd.Timestamp | None
    previous_day: DailyLevels
    premarket: SessionExtremes | None
    opening_range: SessionExtremes | None
    day_extremes: SessionExtremes | None
    pivot: float
    resistance: tuple[float, ...]
    support: tuple[float, ...]
    candidates: tuple[float, ...] = ()
    named: tuple[tuple[str, float], ...] = ()       # (label, price): PDH PDL PDC + PM/OR/DH/DL available at asof
    extension: tuple[tuple[str, float], ...] = ()   # fallback levels when price is beyond all `named`

    def labeled(self) -> dict[str, float]:
        """Trader shorthand for every level family available at `asof`."""
        out = {
            "PDH": self.previous_day.high,
            "PDL": self.previous_day.low,
            "PDC": self.previous_day.close,
            "P": self.pivot,
        }
        if self.premarket is not None:
            out["PMH"] = self.premarket.high
            out["PML"] = self.premarket.low
        if self.opening_range is not None:
            out["ORH"] = self.opening_range.high
            out["ORL"] = self.opening_range.low
        if self.day_extremes is not None:
            out["DH"] = self.day_extremes.high
            out["DL"] = self.day_extremes.low
        return out


# --------------------------------------------------------------------------
# Session slicing
# --------------------------------------------------------------------------
def _session(df: pd.DataFrame, day: dt_date, start: dt_time, end: dt_time) -> pd.DataFrame:
    """Rows of `df` on calendar `day` whose bar-start time is in [start, end)."""
    mask = (df.index.date == day) & (df.index.time >= start) & (df.index.time < end)
    return df.loc[mask]


def _session_extremes(df: pd.DataFrame, day: dt_date, start: dt_time,
                      end: dt_time) -> SessionExtremes | None:
    sess = _session(df, day, start, end)
    if sess.empty:
        return None
    return SessionExtremes(
        day=day, start=start, end=end,
        high=round(float(sess["high"].max()), 2),
        low=round(float(sess["low"].min()), 2),
        high_time=sess["high"].idxmax(),
        low_time=sess["low"].idxmin(),
        bars=len(sess),
    )


def previous_day_levels(df: pd.DataFrame, day: dt_date) -> DailyLevels:
    """The last regular session strictly before `day` (09:30-16:00 ET bars).
    Scans past weekends/holidays automatically."""
    regular = df.loc[(df.index.time >= REGULAR_START) & (df.index.time < REGULAR_END)]
    days = sorted({ts.date() for ts in regular.index if ts.date() < day})
    if not days:
        raise ValueError(f"no regular-session bars before {day}")
    prev = days[-1]
    sess = regular.loc[regular.index.date == prev]
    return DailyLevels(
        day=prev,
        open=round(float(sess["open"].iloc[0]), 2),
        high=round(float(sess["high"].max()), 2),
        low=round(float(sess["low"].min()), 2),
        close=round(float(sess["close"].iloc[-1]), 2),
    )


# --------------------------------------------------------------------------
# Level math
# --------------------------------------------------------------------------
def pivot_point(high: float, low: float, close: float) -> dict[str, float]:
    """Classic floor-trader pivot from the previous H/L/C."""
    p = (high + low + close) / 3.0
    return {"P": round(p, 2), "R1": round(2 * p - low, 2), "S1": round(2 * p - high, 2)}


def _prior_sessions_range(df: pd.DataFrame, day: dt_date, sessions: int):
    """(high, low) of the last `sessions` regular sessions before `day`, or None."""
    regular = df.loc[(df.index.time >= REGULAR_START) & (df.index.time < REGULAR_END)]
    days = sorted({ts.date() for ts in regular.index if ts.date() < day})[-sessions:]
    if not days:
        return None
    sel = regular.loc[pd.Index(regular.index.date).isin(days)]
    return round(float(sel["high"].max()), 2), round(float(sel["low"].min()), 2)


def build_levels(df: pd.DataFrame, day: dt_date,
                 now: pd.Timestamp | None = None) -> SupportResistance:
    """All levels for `day` as of `now` (tz-aware US/Eastern).

    `now=None` means "end of day": every family is included. Time gates:
    pre-market levels only at/after 09:30, opening range only at/after
    10:00, day extremes only at/after 13:00.
    """
    df = df.sort_index()
    now_ts = pd.Timestamp(now) if now is not None else None
    if now_ts is not None:
        if now_ts.tzinfo is None:
            raise ValueError("now must be timezone-aware (US/Eastern recommended)")
        df = df.loc[df.index < now_ts]          # only bars that have started
    asof_time = now_ts.time() if now_ts is not None else dt_time(23, 59)

    prev = previous_day_levels(df, day)
    piv = pivot_point(prev.high, prev.low, prev.close)

    premarket = (_session_extremes(df, day, PREMARKET_START, PREMARKET_END)
                 if asof_time >= PREMARKET_READY else None)
    opening_range = (_session_extremes(df, day, REGULAR_START, OPENING_RANGE_END)
                     if asof_time >= OPENING_RANGE_END else None)
    # Day extremes = 04:00-16:00 ET so far (pre-market + regular session;
    # after-hours 16:00-20:00 bars stay excluded).
    day_extremes = (_session_extremes(df, day, PREMARKET_START, REGULAR_END)
                    if asof_time >= DAY_EXTREMES_READY else None)

    ref = prev.close
    res_candidates = [prev.high, piv["R1"]]
    sup_candidates = [prev.low, piv["S1"]]
    for ext in (premarket, opening_range, day_extremes):
        if ext is not None:
            res_candidates.append(ext.high)
            sup_candidates.append(ext.low)

    resistance = tuple(sorted({round(v, 2) for v in res_candidates if v > ref}, reverse=True))
    support = tuple(sorted({round(v, 2) for v in sup_candidates if v < ref}))
    candidates = tuple(sorted({round(v, 2) for v in res_candidates + sup_candidates}))

    # Named levels for the nearest-levels display (no pivot R1/S1, PDC included).
    named = [("PDH", prev.high), ("PDL", prev.low), ("PDC", prev.close)]
    if premarket is not None:
        named += [("PMH", premarket.high), ("PML", premarket.low)]
    if opening_range is not None:
        named += [("ORH", opening_range.high), ("ORL", opening_range.low)]
    if day_extremes is not None:
        named += [("DH", day_extremes.high), ("DL", day_extremes.low)]

    # Fallback levels, used only when price is beyond every named level.
    rng = prev.high - prev.low
    extension = [("PR2", round(piv["P"] + rng, 2)), ("PS2", round(piv["P"] - rng, 2))]
    five = _prior_sessions_range(df, day, EXT_SESSIONS)
    if five is not None:
        extension += [("5DH", five[0]), ("5DL", five[1])]
    if opening_range is not None:
        orr = opening_range.high - opening_range.low
        extension += [("ORM+", round(opening_range.high + orr, 2)),
                      ("ORM-", round(opening_range.low - orr, 2))]

    return SupportResistance(
        day=day, asof=now_ts, previous_day=prev, premarket=premarket,
        opening_range=opening_range, day_extremes=day_extremes,
        pivot=piv["P"], resistance=resistance, support=support,
        candidates=candidates,
        named=tuple((k, round(float(v), 2)) for k, v in named),
        extension=tuple(extension),
    )


def nearest_levels(sr: SupportResistance, price: float, n: int = 1) -> dict[str, list[float]]:
    """The `n` closest resistance levels above and support levels below
    `price`, nearest first: {"resistance": [...], "support": [...]}.

    Splits ALL candidates around `price` itself, so a level that sits between
    the previous close and the current price (e.g. the opening-range high
    after a rally) is correctly offered as support/resistance."""
    pool = sr.candidates or tuple(sorted(set(sr.resistance) | set(sr.support)))
    above = [lvl for lvl in pool if lvl > price][:n]                 # ascending -> nearest first
    below = [lvl for lvl in reversed(pool) if lvl < price][:n]       # descending -> nearest first
    return {"resistance": above, "support": below}


def distance_to_level(price: float, level: float) -> float:
    """Signed %-distance from `price` to `level`
    (negative = price is below the level)."""
    return round((level / price - 1.0) * 100.0, 3)


# --------------------------------------------------------------------------
# One-line summaries (shared by main.py's Telegram message and replay.py's
# CSV/table, so both show levels with identical wording).
# --------------------------------------------------------------------------
def levels_line(sr: SupportResistance) -> str:
    """'PDH 602.1 PDL 595.0 PDC 598.5 P 598.5 PMH 600.9 ...' --
    only the families available at sr.asof (time-gated)."""
    return " ".join(f"{k} {v:g}" for k, v in sr.labeled().items())


def nearest_line(sr: SupportResistance, price: float, n: int = 2) -> str:
    """'R 602.1 (0.14%) | R 605.3 (0.67%) | S 601.0 (0.04%) | S 599.2 (0.34%)' --
    the `n` nearest resistances above and supports below `price` (default 2
    each), each with its distance from `price` in percent. R is always above
    and S always below, so no sign is printed (a '+' in the text would be
    turned into a space by the Telegram URL)."""
    near = nearest_levels(sr, price, n)
    parts = [f"R {lvl:g} ({abs(distance_to_level(price, lvl)):.2f}%)" for lvl in near["resistance"]]
    parts += [f"S {lvl:g} ({abs(distance_to_level(price, lvl)):.2f}%)" for lvl in near["support"]]
    return " | ".join(parts)


# --------------------------------------------------------------------------
# Named nearest levels (Technical page): R3 .. R1 | S1 .. S3 with labels
# --------------------------------------------------------------------------
def _cluster(levels) -> list[tuple[str, float]]:
    """Merge (label, price) pairs within MERGE_PCT of the cluster's lowest price:
    [("PDL", 748.0), ("PML", 748.1)] -> [("PDL+PML", 748.05)]. Ascending."""
    groups: list[list[tuple[str, float]]] = []
    for name, p in sorted(levels, key=lambda x: x[1]):
        if groups and (p / groups[-1][0][1] - 1.0) * 100.0 <= MERGE_PCT:
            groups[-1].append((name, p))
        else:
            groups.append([(name, p)])
    return [("+".join(n for n, _ in g), round(sum(p for _, p in g) / len(g), 2)) for g in groups]


def nearest_named(sr: SupportResistance, price: float, n: int | None = None) -> dict:
    """Nearest labelled levels around `price`.

    n defaults to NEAR_BEFORE_OR (2) per side before the opening range is
    complete, NEAR_AFTER_OR (3) after. Near-coincident levels are merged.
    If a side has fewer than n levels (price beyond them), it is topped up from
    sr.extension (PR2/PS2, 5DH/5DL, ORM+/ORM-).
    Returns {"resistance": [(label, price)] nearest first, "support": [...],
             "above_all": bool, "below_all": bool, "edge": (label, price) | None}
    where edge = highest named level when above_all, lowest when below_all."""
    if n is None:
        n = NEAR_AFTER_OR if sr.opening_range is not None else NEAR_BEFORE_OR
    pool = _cluster(sr.named)
    res = [x for x in pool if x[1] > price]
    sup = [x for x in pool if x[1] <= price]
    above_all = bool(pool) and not res
    below_all = bool(pool) and not sup
    edge = pool[-1] if above_all else (pool[0] if below_all else None)

    def tol_far(x, existing):
        return all(abs(x[1] / e[1] - 1.0) * 100.0 > MERGE_PCT for e in existing)

    ext = _cluster(sr.extension)
    for x in ext:                                   # ascending
        if len(res) < n and x[1] > price and tol_far(x, pool + res):
            res.append(x)
    for x in reversed(ext):                         # descending
        if len(sup) < n and x[1] <= price and tol_far(x, pool + sup):
            sup.append(x)
    return {"resistance": sorted(res, key=lambda x: x[1])[:n],
            "support": sorted(sup, key=lambda x: -x[1])[:n],
            "above_all": above_all, "below_all": below_all, "edge": edge}


def levels_summary(sr: SupportResistance, price: float, n: int | None = None) -> dict:
    """One-line text, nearest first on each side (R1 = closest resistance above, S1 = closest support below): 'R1 752.11 (PDH) · R2 752.97 (PMH) · R3 753.40 (DH)  ‖  S1 748.00 (PDL) · S2 747.00 (PDC) · S3 745.10 (ORL)'
    plus the above_all / below_all / edge flags from nearest_named."""
    near = nearest_named(sr, price, n)
    r = [f"R{i} {p:.2f} ({lbl})" for i, (lbl, p) in enumerate(near["resistance"], 1)]
    s = [f"S{i} {p:.2f} ({lbl})" for i, (lbl, p) in enumerate(near["support"], 1)]
    text = "  ‖  ".join(part for part in (" · ".join(r), " · ".join(s)) if part)
    return {"text": text or "-", "above_all": near["above_all"],
            "below_all": near["below_all"], "edge": near["edge"]}
