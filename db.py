"""
db.py

Neon (serverless Postgres) persistence for options-volume snapshots.

Connection string lookup order:
    1. st.secrets["NEON_DATABASE_URL"]   (.streamlit/secrets.toml)
    2. env var NEON_DATABASE_URL

Example value (copy from the Neon console -> Connection string):
    postgresql://user:password@ep-xxxx.us-east-2.aws.neon.tech/neondb?sslmode=require

Requires:  pip install psycopg2-binary
"""

from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2

EASTERN = ZoneInfo("America/New_York")

TABLE_NAME = "options_volume_history"

def _get_dsn() -> str:
    dsn = None
    try:
        import streamlit as st

        dsn = st.secrets.get("NEON_DATABASE_URL")
    except Exception:
        pass
    dsn = dsn or os.environ.get("NEON_DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "NEON_DATABASE_URL not set. Add it to .streamlit/secrets.toml "
            "or export it as an environment variable."
        )
    return dsn


def get_connection():
    # Short-lived connections suit Neon (compute auto-suspends when idle).
    return psycopg2.connect(_get_dsn(), connect_timeout=15)


def insert_snapshot(
    symbol: str,
    snapshot_at_utc: dt.datetime,
    spot: float | None,
    call_volume: int,
    put_volume: int,
    source: str = "dashboard",
    scope: str = "ALL",
) -> bool:
    """
    Insert one snapshot row. `snapshot_at_utc` is a naive-UTC datetime (same
    convention as ChainData.as_of). `scope` is which expirations the volume
    totals cover: "ALL", "0DTE", "1DTE" or "FRI". Returns True if a row was
    inserted, False if (symbol, scope, snapshot_at) already existed.
    """
    ts = snapshot_at_utc.replace(microsecond=0, tzinfo=dt.timezone.utc)
    trade_date = ts.astimezone(EASTERN).date()
    pcr = (put_volume / call_volume) if call_volume else None

    sql = f"""
        INSERT INTO {TABLE_NAME}
            (symbol, scope, snapshot_at, trade_date, spot, call_volume, put_volume,
             put_call_ratio, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, scope, snapshot_at) DO NOTHING
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            (symbol.upper(), scope, ts, trade_date, spot, int(call_volume), int(put_volume), pcr, source),
        )
        return cur.rowcount == 1


def load_snapshots(symbol: str, trade_date: dt.date, scope: str = "ALL") -> pd.DataFrame:
    """
    Read datetime, spot, calls volume and puts volume for `symbol` on the
    given Eastern-time market date and expiration `scope` ("ALL", "0DTE",
    "1DTE", "FRI"). `downloaded_at` is returned as a naive
    Eastern datetime, sorted ascending.
    """
    sql = f"""
        SELECT snapshot_at, spot, call_volume, put_volume, put_call_ratio, source
        FROM {TABLE_NAME}
        WHERE symbol = %s AND scope = %s AND trade_date = %s
        ORDER BY snapshot_at
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol.upper(), scope, trade_date))
        rows = cur.fetchall()
        cols = [c.name for c in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df.assign(downloaded_at=pd.Series(dtype="datetime64[ns]"))

    df["downloaded_at"] = (
        pd.to_datetime(df["snapshot_at"], utc=True).dt.tz_convert(EASTERN).dt.tz_localize(None)
    )
    for c in ("spot", "put_call_ratio"):
        df[c] = df[c].astype(float)
    return df.drop(columns=["snapshot_at"])
