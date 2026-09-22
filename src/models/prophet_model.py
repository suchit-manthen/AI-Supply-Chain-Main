"""Prophet forecaster.

Fits one Prophet model per SKU on its chronological demand history (``ds``/``y``).

Model selection: if ``candidates`` are provided in config, each candidate
hyperparameter set is fitted on training data and scored on the *validation*
set (multi-step forecast over the validation horizon using MAPE); the best
configuration per SKU is retained. Otherwise Prophet defaults are used.
"""
from __future__ import annotations

import logging

import cmdstanpy
import pandas as pd
from prophet import Prophet

from src.models.base import BaseForecaster
from src.evaluation.metrics import mape


class ProphetForecaster(BaseForecaster):
    name = "prophet"

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config.models.prophet
        self.yearly = cfg.yearly_seasonality
        self.weekly = cfg.weekly_seasonality
        self.daily = cfg.get("daily_seasonality", False)
        self.candidates = cfg.get("candidates") or []
        self.models_ = {}        # product_id -> fitted Prophet
        self.best_params_ = {}   # product_id -> dict of selected params

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _series(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        out = {}
        for pid, g in df.groupby("product_id"):
            out[pid] = g[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
        return out

    def _fit_one(self, s: pd.DataFrame, params: dict) -> Prophet:
        m = Prophet(
            yearly_seasonality=self.yearly,
            weekly_seasonality=self.weekly,
            daily_seasonality=self.daily,
            changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=params.get("seasonality_prior_scale", 10.0),
        )
        with cmdstanpy.disable_logging():
            m.fit(s)
        return m

    def _val_score(self, model: Prophet, val: pd.DataFrame, pid: str) -> float:
        val_dates = pd.DatetimeIndex(pd.to_datetime(val["date"].unique()))
        actual = (
            val[val["product_id"] == pid]
            .sort_values("date")["sales"]
            .to_numpy(dtype=float)
        )
        if len(actual) == 0:
            return float("inf")
        future = pd.DataFrame({"ds": val_dates})
        fc = model.predict(future)["yhat"].to_numpy()
        return mape(actual, fc)

    # ------------------------------------------------------------------ #
    # Interface
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "ProphetForecaster":
        series = self._series(train)
        for pid, s in series.items():
            if self.candidates:
                best_model, best_score, best_params = None, float("inf"), None
                for cand in self.candidates:
                    try:
                        model = self._fit_one(s, cand)
                    except Exception:
                        continue
                    score = self._val_score(model, val, pid) if val is not None else float("inf")
                    if score < best_score:
                        best_model, best_score, best_params = model, score, cand
                self.models_[pid] = best_model
                self.best_params_[pid] = best_params
            else:
                self.models_[pid] = self._fit_one(s, {})
                self.best_params_[pid] = {}
        return self

    def predict(self, dates) -> pd.DataFrame:
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        future = pd.DataFrame({"ds": dates})
        frames = []
        for pid, model in self.models_.items():
            yhat = model.predict(future)["yhat"].to_numpy()
            frames.append(pd.DataFrame({"product_id": pid, "date": dates, "yhat": yhat}))
        return pd.concat(frames, ignore_index=True)
