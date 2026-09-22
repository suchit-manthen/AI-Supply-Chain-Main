"""SQLite database connection + schema for the supermarket dashboard."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "supermarket.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id   TEXT PRIMARY KEY,
    category     TEXT,
    product_name TEXT,
    base_demand  REAL,
    base_price   REAL
);

CREATE TABLE IF NOT EXISTS sales (
    product_id TEXT,
    date       TEXT,
    sales      REAL,
    is_promo   INTEGER,
    PRIMARY KEY (product_id, date)
);

CREATE TABLE IF NOT EXISTS test_actuals (
    product_id TEXT,
    date       TEXT,
    sales      REAL,
    PRIMARY KEY (product_id, date)
);

CREATE TABLE IF NOT EXISTS predictions (
    model      TEXT,
    product_id TEXT,
    date       TEXT,
    yhat       REAL,
    PRIMARY KEY (model, product_id, date)
);

CREATE TABLE IF NOT EXISTS metrics (
    model      TEXT,
    product_id TEXT,
    mae        REAL,
    rmse       REAL,
    mape       REAL,
    PRIMARY KEY (model, product_id)
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id            TEXT PRIMARY KEY,
    avg_daily_demand      REAL,
    demand_std            REAL,
    annual_demand         REAL,
    lead_time_days        INTEGER,
    service_level         REAL,
    z_score               REAL,
    safety_stock          REAL,
    reorder_point         REAL,
    ordering_cost         REAL,
    holding_cost_per_unit REAL,
    eoq                   REAL
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return a list of dict rows."""
    conn = get_conn()
    cur = conn.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None
