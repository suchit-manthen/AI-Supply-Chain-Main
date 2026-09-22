"""LSTM / GRU sequence forecaster (deep-learning, feature-based).

Input format (per SKU)
----------------------
For every prediction date ``t`` we build a fixed-length sequence of the
previous ``lookback`` days (t-lookback .. t-1). Each timestep is a feature
vector of length ``F``:

    PER_TIMESTEP_FEATURES = [sales, price, is_promo, discount_pct,
                             is_holiday, is_holiday_eve, weekday, is_weekend, month]

The target is ``sales`` on day ``t``. Because the window only ever contains
days strictly before ``t``, no future demand is used (same leakage guarantee as
the other feature-based models; see ``src/features/make_dataset.py``).

Tensor shapes
-------------
- input  X:  (batch, lookback, F)  e.g. (32, 28, 9)
- target y:  (batch,)
- RNN out:   (batch, lookback, hidden_size)
- linear:    hidden_size -> 1

One GRU/LSTM is trained per SKU (consistent with the other models). Features
and the target are standardised per SKU using scalers fitted on training data
only. Candidate architectures are selected on the validation set (MAPE).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseForecaster
from src.evaluation.metrics import mape

# Features used at each timestep of the input sequence. ``sales`` is the target
# history (raw past demand); the remaining columns are calendar/price context.
PER_TIMESTEP_FEATURES = [
    "sales",
    "price",
    "is_promo",
    "discount_pct",
    "is_holiday",
    "is_holiday_eve",
    "weekday",
    "is_weekend",
    "month",
]
F = len(PER_TIMESTEP_FEATURES)


class _RNNRegressor(nn.Module):
    """Single-layer (optionally stacked) GRU/LSTM -> linear head."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 rnn_type: str, dropout: float = 0.0):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, F)
        out, _ = self.rnn(x)          # (B, L, H)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (B,)


class LSTMGRUForecaster(BaseForecaster):
    name = "lstm_gru"

    # Feature-based model: predict() consumes the test feature table.
    needs_features = True

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config.models.lstm_gru
        self.lookback = cfg.lookback
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.rnn_type = cfg.rnn_type
        self.dropout = cfg.get("dropout", 0.0)
        self.batch_size = cfg.batch_size
        self.epochs = cfg.epochs
        self.learning_rate = cfg.learning_rate
        self.patience = cfg.get("patience", 6)
        self.random_state = cfg.get("random_state", 42)
        self.candidates = cfg.get("candidates") or []

        self.models_ = {}        # product_id -> trained nn.Module (best state)
        self.scalers_X_ = {}     # product_id -> StandardScaler for inputs
        self.scalers_y_ = {}     # product_id -> StandardScaler for target
        self.best_params_ = {}   # product_id -> selected hyperparameter dict
        self.history_ = {}       # product_id -> (values, dates) of train+val

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _make_sequences(self, vals: np.ndarray, dates: np.ndarray):
        """Build sliding windows; each window ends at day t-1, target is day t."""
        L = self.lookback
        n = len(vals)
        X = np.empty((n - L, L, vals.shape[1]), dtype=np.float32)
        for i in range(n - L):
            X[i] = vals[i:i + L]
        y = vals[L:, 0].copy()          # sales at day t
        return X, y, dates[L:]

    def _set_seed(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _train_model(self, X_tr, y_tr, X_va, y_va, scaler_X, scaler_y, params) -> nn.Module:
        rnn_type = params.get("rnn_type", self.rnn_type)
        hidden_size = params.get("hidden_size", self.hidden_size)
        num_layers = params.get("num_layers", self.num_layers)
        lr = params.get("learning_rate", self.learning_rate)

        X_tr_s = torch.tensor(
            scaler_X.transform(X_tr.reshape(-1, F)).reshape(X_tr.shape), dtype=torch.float32
        )
        y_tr_s = torch.tensor(scaler_y.transform(y_tr.reshape(-1, 1)).ravel(), dtype=torch.float32)
        has_val = X_va is not None and len(X_va) > 0
        if has_val:
            X_va_s = torch.tensor(
                scaler_X.transform(X_va.reshape(-1, F)).reshape(X_va.shape), dtype=torch.float32
            )
            y_va_s = torch.tensor(scaler_y.transform(y_va.reshape(-1, 1)).ravel(), dtype=torch.float32)

        model = _RNNRegressor(F, hidden_size, num_layers, rnn_type, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_tr_s, y_tr_s), batch_size=self.batch_size, shuffle=True)

        best_val, best_state, no_improve = float("inf"), None, 0
        for _ in range(self.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

            if has_val:
                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(X_va_s), y_va_s).item()
                if val_loss < best_val - 1e-4:
                    best_val, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _predict_original(self, model: nn.Module, X: np.ndarray, scaler_X, scaler_y) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_s = torch.tensor(scaler_X.transform(X.reshape(-1, F)).reshape(X.shape), dtype=torch.float32)
            pred_s = model(X_s).numpy().reshape(-1, 1)
        return scaler_y.inverse_transform(pred_s).ravel()

    # ------------------------------------------------------------------ #
    # Interface
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "LSTMGRUForecaster":
        self._set_seed()
        train_dates = train["date"].drop_duplicates().to_numpy()
        val_dates = val["date"].drop_duplicates().to_numpy() if val is not None else np.array([])
        combined = pd.concat([train, val]) if val is not None else train

        for pid, g in combined.groupby("product_id"):
            g = g.sort_values("date").reset_index(drop=True)
            vals = g[PER_TIMESTEP_FEATURES].to_numpy(np.float32)
            dates = g["date"].to_numpy()
            self.history_[pid] = (vals.copy(), dates.copy())

            X_seq, y_seq, seq_dates = self._make_sequences(vals, dates)
            tr_mask = np.isin(seq_dates, train_dates)
            va_mask = np.isin(seq_dates, val_dates)
            X_tr, y_tr = X_seq[tr_mask], y_seq[tr_mask]
            X_va, y_va = X_seq[va_mask], y_seq[va_mask]

            # Scalers fitted on training data only (avoids leakage).
            scaler_X = StandardScaler().fit(X_tr.reshape(-1, F))
            scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
            self.scalers_X_[pid] = scaler_X
            self.scalers_y_[pid] = scaler_y

            if self.candidates:
                best_model, best_score, best_params = None, float("inf"), None
                for cand in self.candidates:
                    model = self._train_model(X_tr, y_tr, X_va, y_va, scaler_X, scaler_y, cand)
                    pred = self._predict_original(model, X_va, scaler_X, scaler_y)
                    score = mape(y_va, pred)
                    if score < best_score:
                        best_model, best_score, best_params = model, score, cand
                self.models_[pid] = best_model
                self.best_params_[pid] = best_params
            else:
                self.models_[pid] = self._train_model(
                    X_tr, y_tr, X_va, y_va, scaler_X, scaler_y, {}
                )
                self.best_params_[pid] = {}
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for pid, g in test.groupby("product_id"):
            g = g.sort_values("date")
            test_vals = g[PER_TIMESTEP_FEATURES].to_numpy(np.float32)
            test_dates = g["date"].to_numpy()
            hist_vals, hist_dates = self.history_[pid]

            # Prepend train+val history so the lookback window for the first
            # test days is fully populated with past (known) values.
            vals = np.vstack([hist_vals, test_vals])
            dates = np.concatenate([hist_dates, test_dates])
            X_seq, _, seq_dates = self._make_sequences(vals, dates)

            test_mask = np.isin(seq_dates, test_dates)
            X_test = X_seq[test_mask]
            target_dates = seq_dates[test_mask]

            model = self.models_[pid]
            yhat = self._predict_original(model, X_test, self.scalers_X_[pid], self.scalers_y_[pid])
            frames.append(pd.DataFrame({"product_id": pid, "date": target_dates, "yhat": yhat}))
        return pd.concat(frames, ignore_index=True)
