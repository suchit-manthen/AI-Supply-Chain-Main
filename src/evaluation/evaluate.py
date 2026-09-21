"""Evaluation helpers: compare model predictions against actual sales.

Predictions are expected in long format (``product_id``, ``date``, ``yhat``).
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import mae, mape, rmse


def evaluate_predictions(
    predictions: pd.DataFrame, actual: pd.DataFrame, target: str = "sales"
) -> dict[str, float]:
    """Compute MAE/RMSE/MAPE over the joined prediction/actual rows."""
    merged = actual[["product_id", "date", target]].merge(
        predictions[["product_id", "date", "yhat"]], on=["product_id", "date"]
    )
    y_true = merged[target].values
    y_pred = merged["yhat"].values
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "n_rows": len(merged),
    }


def evaluate_per_sku(
    predictions: pd.DataFrame, actual: pd.DataFrame, target: str = "sales"
) -> pd.DataFrame:
    """Return one row per SKU with its own MAE/RMSE/MAPE."""
    merged = actual[["product_id", "date", target]].merge(
        predictions[["product_id", "date", "yhat"]], on=["product_id", "date"]
    )
    rows = []
    for pid, g in merged.groupby("product_id"):
        rows.append(
            {
                "product_id": pid,
                "mae": mae(g[target], g["yhat"]),
                "rmse": rmse(g[target], g["yhat"]),
                "mape": mape(g[target], g["yhat"]),
            }
        )
    return pd.DataFrame(rows)
