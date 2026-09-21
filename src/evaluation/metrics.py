"""Forecasting metrics: MAE, RMSE, MAPE.

All functions accept array-like inputs and return floats. MAPE guards against
division by zero by ignoring rows where the actual value is zero.
"""
from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (%), ignoring zero actuals."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
