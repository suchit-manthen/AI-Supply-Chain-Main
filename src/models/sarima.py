"""ARIMA / SARIMA forecaster.

Fits one univariate SARIMAX model per SKU on its chronological demand history.

Model selection: if ``candidates`` are provided in config, each candidate order
is fitted on the training data and scored on the *validation* set (multi-step
forecast over the validation horizon using MAPE); the best order per SKU is
retained. Otherwise a single fixed order is used.
"""
from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import BaseForecaster
from src.evaluation.metrics import mape


class SARIMAForecaster(BaseForecaster):
    name = "sarima"

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config.models.sarima
        self.order = tuple(cfg.order)
        self.seasonal_order = tuple(cfg.seasonal_order)
        # Optional list of {"order": [...], "seasonal_order": [...]} candidates.
        self.candidates = cfg.get("candidates") or []
        self.models_ = {}      # product_id -> fitted SARIMAX
        self.best_order_ = {}  # product_id -> (order, seasonal_order)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _series(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out = {}
        for pid, g in df.groupby("product_id"):
            s = g.set_index("date")["sales"].asfreq("D")
            out[pid] = s
        return out

    def _fit_one(self, series: pd.Series, order, seasonal_order) -> SARIMAX:
        with warnings.catch_warnings():
            # Convergence warnings are expected when trialling complex orders
            # during model selection; a non-converged fit is simply skipped.
            warnings.simplefilter("ignore")
            return SARIMAX(
                series, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)

    def _val_score(self, model: SARIMAX, val: pd.DataFrame, pid: str) -> float:
        val_dates = pd.DatetimeIndex(pd.to_datetime(val["date"].unique()))
        actual = (
            val[val["product_id"] == pid]
            .sort_values("date")["sales"]
            .to_numpy(dtype=float)
        )
        if len(actual) == 0:
            return float("inf")
        fc = model.forecast(steps=len(actual))
        return mape(actual, fc)

    # ------------------------------------------------------------------ #
    # Interface
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "SARIMAForecaster":
        series = self._series(train)
        for pid, s in series.items():
            if self.candidates:
                best_model, best_score, best_order = None, float("inf"), None
                for cand in self.candidates:
                    order = tuple(cand["order"])
                    seasonal = tuple(cand["seasonal_order"])
                    try:
                        model = self._fit_one(s, order, seasonal)
                    except Exception:
                        continue
                    score = self._val_score(model, val, pid) if val is not None else float("inf")
                    if score < best_score:
                        best_model, best_score, best_order = model, score, (order, seasonal)
                self.models_[pid] = best_model
                self.best_order_[pid] = best_order
            else:
                self.models_[pid] = self._fit_one(s, self.order, self.seasonal_order)
                self.best_order_[pid] = (self.order, self.seasonal_order)
        return self

    def predict(self, dates) -> pd.DataFrame:
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        frames = []
        for pid, model in self.models_.items():
            fc = model.forecast(steps=len(dates))
            frames.append(
                pd.DataFrame({"product_id": pid, "date": dates, "yhat": fc.to_numpy()})
            )
        return pd.concat(frames, ignore_index=True)
