"""XGBoost forecaster (feature-based).

Mirrors ``RandomForestForecaster``: one XGBRegressor per SKU, trained on the
Step 2 engineered feature table (lags, rolling statistics, calendar and
price/promo features). Hyperparameters are selected on the validation set via a
small candidate grid scored by MAPE, and the final model is evaluated on the
test set.

Leakage: the feature table is already past-only (lag/rolling features use
information strictly before each prediction date), so no additional shifting is
required here. See ``src/features/make_dataset.py``.
"""
from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.models.base import BaseForecaster
from src.models.random_forest_model import RandomForestForecaster
from src.evaluation.metrics import mape


class XGBoostForecaster(BaseForecaster):
    name = "xgboost"

    # Feature-based model: predict() consumes the full test feature table.
    needs_features = True

    # Reuse the exact feature set as Random Forest so the comparison is fair.
    FEATURE_COLS = RandomForestForecaster.FEATURE_COLS

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config.models.xgboost
        self.candidates = cfg.get("candidates") or []
        self.random_state = cfg.get("random_state", 42)
        self.models_ = {}        # product_id -> fitted XGBRegressor
        self.best_params_ = {}   # product_id -> selected hyperparameter dict

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = df[self.FEATURE_COLS]
        y = df["sales"]
        return X, y

    def _fit_one(self, X, y, params: dict) -> XGBRegressor:
        model = XGBRegressor(
            random_state=self.random_state, verbosity=0, **params
        )
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------ #
    # Interface
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "XGBoostForecaster":
        for pid, g in train.groupby("product_id"):
            X_train, y_train = self._xy(g)
            if self.candidates and val is not None:
                val_g = val[val["product_id"] == pid]
                X_val, y_val = self._xy(val_g)
                best_model, best_score, best_params = None, float("inf"), None
                for cand in self.candidates:
                    model = self._fit_one(X_train, y_train, cand)
                    score = mape(y_val.to_numpy(), model.predict(X_val))
                    if score < best_score:
                        best_model, best_score, best_params = model, score, cand
                self.models_[pid] = best_model
                self.best_params_[pid] = best_params
            else:
                self.models_[pid] = self._fit_one(X_train, y_train, {})
                self.best_params_[pid] = {}
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for pid, g in test.groupby("product_id"):
            model = self.models_[pid]
            yhat = model.predict(g[self.FEATURE_COLS])
            frames.append(
                pd.DataFrame({"product_id": pid, "date": g["date"].values, "yhat": yhat})
            )
        return pd.concat(frames, ignore_index=True)
