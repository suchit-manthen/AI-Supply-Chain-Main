"""Common forecasting model interface.

Each forecaster must implement ``fit`` and ``predict``. The input format is NOT
forced to be identical across models:

- Time-series models (ARIMA/SARIMA, Prophet) consume chronological demand
  history (a ``product_id -> Series`` of daily sales).
- Machine-learning / deep-learning models (Random Forest, XGBoost/LightGBM,
  LSTM/GRU) consume the engineered feature table from Step 2.

To keep evaluation uniform, every ``predict`` returns a long DataFrame with
columns ``product_id``, ``date`` and ``yhat``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    name: str = "base"

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "BaseForecaster":
        """Fit the model on training data (val may be used for tuning/selection)."""

    @abstractmethod
    def predict(self, dates) -> pd.DataFrame:
        """Forecast ``sales`` for the given dates.

        Returns a long DataFrame with columns ``product_id``, ``date``, ``yhat``.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(name={self.name!r})"
