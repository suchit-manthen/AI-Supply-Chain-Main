"""Seed the SQLite database from the existing CSV artefacts.

Run once from the backend directory:

    python seed.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from db import get_conn, init_db

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

MODELS = ["sarima", "prophet", "random_forest", "xgboost", "lstm_gru"]

# Inventory-optimization defaults (configurable at query time via the API).
LEAD_TIME_DAYS = 3
SERVICE_LEVEL = 0.95
Z_SCORE = 1.645          # z for 95% service level
ORDERING_COST = 50.0     # $ per order
HOLDING_RATE = 0.20      # 20% of unit price per year


def _load(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _drop_and_seed() -> None:
    init_db()
    conn = get_conn()

    # 1. Product catalogue
    products = _load(RAW / "products.csv")[
        ["product_id", "category", "product_name", "base_demand", "base_price"]
    ]
    products.to_sql("products", conn, if_exists="replace", index=False)

    # 2. Full sales history
    sales = _load(RAW / "sales.csv", usecols=["product_id", "date", "sales", "is_promo"])
    sales.to_sql("sales", conn, if_exists="replace", index=False)

    # 3. Test-set actuals (for actual-vs-forecast charts)
    test = _load(PROC / "test.csv", usecols=["product_id", "date", "sales"])
    test.to_sql("test_actuals", conn, if_exists="replace", index=False)

    # 4. Model predictions + per-SKU metrics
    for model in MODELS:
        preds = _load(PROC / f"predictions_{model}.csv")
        preds["model"] = model
        preds.to_sql("predictions", conn, if_exists="append", index=False)

        metrics = _load(PROC / f"metrics_{model}.csv")
        metrics["model"] = model
        metrics = metrics[["model", "product_id", "mae", "rmse", "mape"]]
        metrics.to_sql("metrics", conn, if_exists="append", index=False)

    # 5. Inventory optimization per SKU
    g = sales.groupby("product_id")["sales"]
    avg = g.mean().rename("avg_daily_demand")
    std = g.std().rename("demand_std")

    inventory = pd.concat([avg, std], axis=1).reset_index()
    inventory = inventory.merge(
        products[["product_id", "base_price"]], on="product_id", how="left"
    )
    inventory["annual_demand"] = inventory["avg_daily_demand"] * 365
    inventory["lead_time_days"] = LEAD_TIME_DAYS
    inventory["service_level"] = SERVICE_LEVEL
    inventory["z_score"] = Z_SCORE
    inventory["safety_stock"] = (
        inventory["z_score"] * inventory["demand_std"] * math.sqrt(LEAD_TIME_DAYS)
    )
    inventory["reorder_point"] = (
        inventory["avg_daily_demand"] * LEAD_TIME_DAYS + inventory["safety_stock"]
    )
    inventory["ordering_cost"] = ORDERING_COST
    inventory["holding_cost_per_unit"] = inventory["base_price"] * HOLDING_RATE
    inventory["eoq"] = (
        (2 * inventory["annual_demand"] * ORDERING_COST)
        / inventory["holding_cost_per_unit"]
    ) ** 0.5

    cols = [
        "product_id",
        "avg_daily_demand",
        "demand_std",
        "annual_demand",
        "lead_time_days",
        "service_level",
        "z_score",
        "safety_stock",
        "reorder_point",
        "ordering_cost",
        "holding_cost_per_unit",
        "eoq",
    ]
    inventory[cols].to_sql("inventory", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    n = {
        "products": len(products),
        "sales": len(sales),
        "test": len(test),
        "predictions": sum(
            len(_load(PROC / f"predictions_{m}.csv")) for m in MODELS
        ),
        "metrics": len(MODELS) * len(products),
        "inventory": len(inventory),
    }
    print("Seeded SQLite database:", n)


if __name__ == "__main__":
    _drop_and_seed()
