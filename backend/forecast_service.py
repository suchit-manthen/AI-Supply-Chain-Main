"""Forecasting service: turns an uploaded CSV into forecasts + inventory plan.

This module reuses the existing model implementations in ``src/models`` and the
feature engineering in ``src/features/make_dataset.py``. It does NOT reimplement
the models themselves — it only:

1. Validates / normalises the uploaded CSV into the canonical column set.
2. Runs the Step 2 feature engineering (calendar, lags, rolling windows).
3. Chronologically splits into train/val/test.
4. Fits the requested models and scores them on the test set (MAE/RMSE/MAPE).
5. Produces a future demand forecast (recursive for feature-based models).
6. Computes Safety Stock / Reorder Point / EOQ inventory recommendations.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation.metrics import mae, mape, rmse  # noqa: E402
from src.features.make_dataset import (  # noqa: E402
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    split_by_time,
)
from src.models.lstm_model import F, LSTMGRUForecaster, PER_TIMESTEP_FEATURES  # noqa: E402
from src.models.prophet_model import ProphetForecaster  # noqa: E402
from src.models.random_forest_model import RandomForestForecaster  # noqa: E402

FEATURE_COLS = RandomForestForecaster.FEATURE_COLS
from src.models.sarima import SARIMAForecaster  # noqa: E402
from src.models.xgboost_model import XGBoostForecaster  # noqa: E402

MODEL_REGISTRY = {
    "sarima": SARIMAForecaster,
    "prophet": ProphetForecaster,
    "random_forest": RandomForestForecaster,
    "xgboost": XGBoostForecaster,
    "lstm_gru": LSTMGRUForecaster,
}

MODEL_LABELS = {
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lstm_gru": "LSTM/GRU",
}

MODEL_DESCRIPTIONS = {
    "sarima": "Statistical ARIMA with weekly seasonality on historical demand.",
    "prophet": "Additive trend + yearly/weekly seasonality and holiday effects.",
    "random_forest": "Non-linear relationships using engineered features (lags, rolling stats, calendar).",
    "xgboost": "Gradient boosting on complex demand/price/promo/calendar interactions.",
    "lstm_gru": "Recurrent sequence model using historical lookback windows.",
}

# --------------------------------------------------------------------------- #
# Column detection / validation
# --------------------------------------------------------------------------- #

# Canonical column -> accepted aliases (matched case-insensitively, trimmed).
COLUMN_ALIASES = {
    "date": ["date", "day", "timestamp", "ds", "order_date", "sale_date", "sales_date", "order_day", "datetime"],
    "product_id": ["product_id", "sku", "sku_id", "product", "item_id", "item", "product_code", "store_id"],
    "sales": ["sales", "demand", "units", "quantity", "qty", "units_sold", "volume", "sold"],
    "price": ["price", "unit_price", "selling_price", "sale_price", "retail_price"],
    "is_promo": ["is_promo", "promo", "on_promotion", "promotion", "is_promotion", "promo_flag"],
    "discount_pct": ["discount_pct", "discount", "discount_rate", "discount_percent"],
    "is_holiday": ["is_holiday", "holiday", "holiday_flag", "is_public_holiday"],
    "is_holiday_eve": ["is_holiday_eve", "holiday_eve", "holiday_eve_flag"],
    "category": ["category", "department", "dept", "product_category", "class"],
    "base_price": ["base_price", "regular_price", "list_price", "mrp", "original_price"],
    "product_name": ["product_name", "item_name", "name", "description", "title"],
    "on_hand": ["on_hand", "current_stock", "stock", "stock_level", "inventory", "onhand", "inventory_level"],
    "lead_time_days": ["lead_time_days", "lead_time", "leadtime", "replenishment_days", "replenishment_time"],
}

# Canonical columns guaranteed to exist after normalisation.
CANONICAL = [
    "date", "product_id", "sales", "price", "is_promo", "discount_pct",
    "is_holiday", "is_holiday_eve", "category", "base_price", "product_name",
    "on_hand", "lead_time_days",
]

REQUIRED = ["date", "product_id", "sales"]


def _normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map source column names to canonical names (case/whitespace-insensitive)."""
    rename: dict[str, str] = {}
    mapped: dict[str, str] = {}
    lower_to_source = {str(c).strip().lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_to_source:
                src = lower_to_source[alias.lower()]
                rename[src] = canonical
                mapped[canonical] = src
                break
    return df.rename(columns=rename), mapped


def prepare_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate + normalise an uploaded dataframe.

    Returns ``(df, report)`` where ``df`` has the canonical columns and
    ``report`` describes the mapping, defaults applied and a data summary.
    Raises ``ValueError`` with a user-friendly message when a required column
    is missing or the data cannot be parsed.
    """
    if raw is None or raw.empty:
        raise ValueError("The uploaded file is empty.")

    df, mapped = _normalise_columns(raw)

    missing_required = [c for c in REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing_required)
            + ". Expected columns such as 'date', 'product_id'/'sku' and 'sales'/'demand'."
        )

    # --- parse & coerce ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        raise ValueError("Could not parse the 'date' column into valid dates.")

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str).str.strip()

    # --- apply optional columns with sensible defaults ---
    defaulted: dict[str, str] = {}

    if "price" not in df.columns:
        df["price"] = 0.0
        defaulted["price"] = "defaulted to 0 (not provided)"
    else:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    if "base_price" not in df.columns:
        df["base_price"] = df["price"]
        defaulted["base_price"] = "defaulted to 'price' (not provided)"
    else:
        df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce").fillna(df["price"])

    for col, default in [
        ("is_promo", 0),
        ("discount_pct", 0.0),
        ("is_holiday", 0),
        ("is_holiday_eve", 0),
    ]:
        if col not in df.columns:
            df[col] = default
            defaulted[col] = f"defaulted to {default} (not provided)"
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "category" not in df.columns:
        df["category"] = "Unknown"
        defaulted["category"] = "defaulted to 'Unknown' (not provided)"

    if "product_name" not in df.columns:
        df["product_name"] = df["product_id"]
        defaulted["product_name"] = "defaulted to product_id (not provided)"

    for col in ("on_hand", "lead_time_days"):
        if col not in df.columns:
            df[col] = np.nan
            defaulted[col] = "not provided — enter current stock / lead time for reorder recommendations"
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only canonical columns and drop rows with invalid date/sales.
    df = df[CANONICAL].dropna(subset=["date", "sales", "product_id"])
    df = df[df["product_id"] != ""]
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows after cleaning. Check the date/sales/product columns.")

    n_products = df["product_id"].nunique()
    min_days = df.groupby("product_id")["date"].transform("nunique")
    if min_days.min() < 35:
        raise ValueError(
            "Each product needs at least ~35 days of history for the 28-day lag "
            "and rolling windows. Some products have fewer."
        )

    report = _build_report(df, mapped, defaulted, missing_required)
    return df, report


def _build_report(df, mapped, defaulted, missing_required) -> dict:
    n_products = int(df["product_id"].nunique())
    categories = (
        df.groupby("category")["product_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"product_id": "count"})
        .to_dict("records")
    )
    return {
        "mapped": mapped,
        "defaulted": defaulted,
        "missing_required": missing_required,
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "n_rows": int(len(df)),
        "n_products": n_products,
        "n_categories": len(categories),
        "n_days": int(df["date"].nunique()),
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "categories": categories,
        "preview": _preview(df),
    }


def _preview(df: pd.DataFrame, n: int = 8) -> list[dict]:
    out = []
    for _, row in df.head(n).iterrows():
        rec = {}
        for c in df.columns:
            v = row[c]
            if isinstance(v, (pd.Timestamp,)):
                v = v.date().isoformat()
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = round(float(v), 3)
            elif pd.isna(v):
                v = None
            rec[c] = v
        out.append(rec)
    return out


def list_products(df: pd.DataFrame) -> list[dict]:
    rows = []
    for pid, g in df.groupby("product_id"):
        on_hand = _last_value(g, "on_hand")
        lead_time = _last_value(g, "lead_time_days")
        rows.append(
            {
                "product_id": pid,
                "product_name": str(g["product_name"].iloc[0]),
                "category": str(g["category"].iloc[0]),
                "n_days": int(g["date"].nunique()),
                "avg_sales": round(float(g["sales"].mean()), 2),
                "current_stock": on_hand,
                "lead_time": lead_time,
            }
        )
    return rows


def _last_value(g: pd.DataFrame, col: str):
    """Return the last non-null value of ``col`` in the group, or None."""
    if col not in g.columns:
        return None
    s = g[col].dropna()
    if s.empty:
        return None
    v = s.iloc[-1]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 2)
    return None


# --------------------------------------------------------------------------- #
# Config (fast demo settings — disable candidate search)
# --------------------------------------------------------------------------- #


def _build_config():
    config = load_config(str(ROOT / "config" / "config.yaml"))
    # The API path disables validation-based hyperparameter search to keep
    # interactive runs fast. The model classes are untouched.
    config.models.sarima.candidates = []
    config.models.prophet.candidates = []
    config.models.random_forest.candidates = []
    config.models.xgboost.candidates = []
    config.models.lstm_gru.candidates = []
    config.models.lstm_gru.epochs = 15
    return config


# --------------------------------------------------------------------------- #
# Future forecast helpers (recursive for feature-based models)
# --------------------------------------------------------------------------- #


def _ts_future(model, test_dates: pd.DatetimeIndex, horizon: int) -> dict[str, pd.DataFrame]:
    future_dates = pd.date_range(test_dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    all_dates = pd.DatetimeIndex(list(test_dates) + list(future_dates))
    preds = model.predict(all_dates)
    preds = preds[preds["date"].isin(future_dates)]
    return {pid: g[["date", "yhat"]] for pid, g in preds.groupby("product_id")}


def _feature_row(d: pd.Timestamp, sales: list[float], last_price: float) -> dict:
    def last(n):
        return float(sales[-n]) if len(sales) >= n else float("nan")

    def roll(n):
        w = sales[-n:]
        return (float(np.mean(w)), float(np.std(w))) if len(w) == n else (float("nan"), float("nan"))

    r7, s7 = roll(7)
    r14, s14 = roll(14)
    r28, s28 = roll(28)
    return {
        "price": last_price, "is_promo": 0, "discount_pct": 0.0,
        "is_holiday": 0, "is_holiday_eve": 0,
        "year": d.year, "month": d.month, "day": d.day,
        "weekday": d.dayofweek, "is_weekend": int(d.dayofweek >= 5),
        "lag_1": last(1), "lag_7": last(7), "lag_14": last(14), "lag_28": last(28),
        "rolling_mean_7": r7, "rolling_std_7": s7,
        "rolling_mean_14": r14, "rolling_std_14": s14,
        "rolling_mean_28": r28, "rolling_std_28": s28,
    }


def _tree_future(model, df: pd.DataFrame, horizon: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for pid in model.models_:
        g = df[df["product_id"] == pid].sort_values("date")
        if g.empty:
            continue
        sales = g["sales"].astype(float).tolist()
        last_date = g["date"].max()
        last_price = float(g["price"].iloc[-1])
        rows = []
        for i in range(1, horizon + 1):
            d = last_date + pd.Timedelta(days=i)
            X = pd.DataFrame([_feature_row(d, sales, last_price)])[FEATURE_COLS]
            yhat = float(model.models_[pid].predict(X)[0])
            rows.append({"date": d, "yhat": yhat})
            sales.append(yhat)
        out[pid] = pd.DataFrame(rows)
    return out


def _lstm_future(model, df: pd.DataFrame, horizon: int) -> dict[str, pd.DataFrame]:
    import torch

    out: dict[str, pd.DataFrame] = {}
    for pid in model.models_:
        g = df[df["product_id"] == pid].sort_values("date").copy()
        if g.empty:
            continue
        g["weekday"] = g["date"].dt.dayofweek
        g["is_weekend"] = (g["date"].dt.dayofweek >= 5).astype(int)
        g["month"] = g["date"].dt.month
        vals = g[PER_TIMESTEP_FEATURES].to_numpy(np.float32)
        last_date = g["date"].max()
        last_price = float(g["price"].iloc[-1])
        module = model.models_[pid]
        scaler_x = model.scalers_X_[pid]
        scaler_y = model.scalers_y_[pid]
        L = model.lookback
        rows = []
        for i in range(1, horizon + 1):
            window = vals[-L:].reshape(1, L, F)
            with torch.no_grad():
                xs = torch.tensor(
                    scaler_x.transform(window.reshape(-1, F)).reshape(window.shape),
                    dtype=torch.float32,
                )
                pred = module(xs).numpy().reshape(-1, 1)
            yhat = float(scaler_y.inverse_transform(pred).ravel()[0])
            d = last_date + pd.Timedelta(days=i)
            newrow = np.array(
                [[yhat, last_price, 0.0, 0.0, 0.0, 0.0, d.dayofweek, int(d.dayofweek >= 5), d.month]],
                dtype=np.float32,
            )
            vals = np.vstack([vals, newrow])
            rows.append({"date": d, "yhat": yhat})
        out[pid] = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------- #
# Inventory recommendations
# --------------------------------------------------------------------------- #


def _lag(sales: list[float], n: int) -> float:
    return float(sales[-n]) if len(sales) >= n else float("nan")


def _roll(sales: list[float], n: int) -> tuple[float, float]:
    w = sales[-n:]
    if len(w) == n:
        return float(np.mean(w)), float(np.std(w))
    return float("nan"), float("nan")


def _feature_recursive_test(model, hist: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Recursive multi-step forecast of feature-based models over the test set.

    Forecasts are produced one day at a time starting from the end of the
    train+val window; each predicted value is fed back into the lag/rolling
    features. Only *exogenous* test features (price, promo, holiday, calendar)
    are used — the actual test *demand* is never seen.
    """
    frames = []
    for pid in model.models_:
        h = hist[hist["product_id"] == pid].sort_values("date")
        t = test[test["product_id"] == pid].sort_values("date")
        if h.empty or t.empty:
            continue
        sales = h["sales"].astype(float).tolist()
        rows = []
        for tr in t.itertuples(index=False):
            d = tr.date
            r7, s7 = _roll(sales, 7)
            r14, s14 = _roll(sales, 14)
            r28, s28 = _roll(sales, 28)
            row = {
                "price": float(tr.price),
                "is_promo": int(tr.is_promo),
                "discount_pct": float(tr.discount_pct),
                "is_holiday": int(tr.is_holiday),
                "is_holiday_eve": int(tr.is_holiday_eve),
                "year": d.year, "month": d.month, "day": d.day,
                "weekday": d.dayofweek, "is_weekend": int(d.dayofweek >= 5),
                "lag_1": _lag(sales, 1), "lag_7": _lag(sales, 7),
                "lag_14": _lag(sales, 14), "lag_28": _lag(sales, 28),
                "rolling_mean_7": r7, "rolling_std_7": s7,
                "rolling_mean_14": r14, "rolling_std_14": s14,
                "rolling_mean_28": r28, "rolling_std_28": s28,
            }
            X = pd.DataFrame([row])[FEATURE_COLS]
            yhat = float(model.models_[pid].predict(X)[0])
            rows.append({"product_id": pid, "date": d, "yhat": yhat})
            sales.append(yhat)
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["product_id", "date", "yhat"])
    return pd.concat(frames, ignore_index=True)


def _lstm_recursive_test(model, hist: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Recursive multi-step forecast of the LSTM/GRU model over the test set."""
    import torch

    frames = []
    for pid in model.models_:
        h = hist[hist["product_id"] == pid].sort_values("date").copy()
        t = test[test["product_id"] == pid].sort_values("date").copy()
        if h.empty or t.empty:
            continue
        h["weekday"] = h["date"].dt.dayofweek
        h["is_weekend"] = (h["date"].dt.dayofweek >= 5).astype(int)
        h["month"] = h["date"].dt.month
        vals = h[PER_TIMESTEP_FEATURES].to_numpy(np.float32)
        module = model.models_[pid]
        sx = model.scalers_X_[pid]
        sy = model.scalers_y_[pid]
        L = model.lookback
        rows = []
        for tr in t.itertuples(index=False):
            d = tr.date
            window = vals[-L:].reshape(1, L, F)
            with torch.no_grad():
                xs = torch.tensor(
                    sx.transform(window.reshape(-1, F)).reshape(window.shape),
                    dtype=torch.float32,
                )
                pred = module(xs).numpy().reshape(-1, 1)
            yhat = float(sy.inverse_transform(pred).ravel()[0])
            newrow = np.array(
                [[yhat, float(tr.price), float(tr.is_promo), float(tr.discount_pct),
                  float(tr.is_holiday), float(tr.is_holiday_eve),
                  d.dayofweek, int(d.dayofweek >= 5), d.month]],
                dtype=np.float32,
            )
            vals = np.vstack([vals, newrow])
            rows.append({"product_id": pid, "date": d, "yhat": yhat})
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["product_id", "date", "yhat"])
    return pd.concat(frames, ignore_index=True)


def predict_test_recursive(model, train, val, test, test_dates) -> pd.DataFrame:
    """Fair, leak-free test predictions for *all* model families.

    Feature-based models (Random Forest, XGBoost, LSTM/GRU) are rolled forward
    recursively from the end of validation — they only ever use past demand and
    the known exogenous features, never the actual test demand. Time-series
    models (SARIMA/Prophet) natively forecast the full horizon in one step.
    """
    if getattr(model, "needs_features", False):
        hist = pd.concat([train, val])
        if model.name == "lstm_gru":
            return _lstm_recursive_test(model, hist, test)
        return _feature_recursive_test(model, hist, test)
    return model.predict(test_dates)


def build_features_and_split(df, config, selected_ids=None):
    """Feature engineering + chronological split, optionally filtered by SKU."""
    feat = add_calendar_features(df.copy())
    feat = add_lag_features(feat, list(config.features.lags))
    feat = add_rolling_features(feat, list(config.features.rolling_windows))
    feat = feat.dropna(subset=[c for c in feat.columns if c.startswith(("lag_", "rolling_"))])
    train, val, test = split_by_time(feat, config)
    if selected_ids is not None:
        train = train[train["product_id"].isin(selected_ids)]
        val = val[val["product_id"].isin(selected_ids)]
        test = test[test["product_id"].isin(selected_ids)]
    return train, val, test


def compute_inventory(df: pd.DataFrame, service_level: float, lead_time: int,
                      ordering_cost: float, holding_rate: float) -> list[dict]:
    z = NormalDist().inv_cdf(service_level)
    rows = []
    for pid, g in df.groupby("product_id"):
        sales = g["sales"].astype(float)
        avg = float(sales.mean())
        std = float(sales.std(ddof=0))
        annual = avg * 365
        base_price = float(g["base_price"].iloc[0]) if "base_price" in g.columns else float(g["price"].mean())
        holding = base_price * holding_rate
        safety = z * std * math.sqrt(lead_time)
        reorder = avg * lead_time + safety
        eoq = math.sqrt(2 * annual * ordering_cost / holding) if holding else 0.0
        rows.append(
            {
                "product_id": pid,
                "product_name": str(g["product_name"].iloc[0]) if "product_name" in g.columns else pid,
                "category": str(g["category"].iloc[0]) if "category" in g.columns else "Unknown",
                "base_price": round(base_price, 3),
                "avg_daily_demand": round(avg, 2),
                "demand_std": round(std, 2),
                "annual_demand": round(annual, 1),
                "safety_stock": round(safety, 1),
                "reorder_point": round(reorder, 1),
                "eoq": round(eoq, 1),
                "holding_cost_per_unit": round(holding, 3),
            }
        )
    return rows


def compute_inventory_from_forecast(
    df: pd.DataFrame,
    forecast: dict[str, pd.DataFrame],
    service_level: float,
    lead_time_default: int,
    ordering_cost: float,
    holding_rate: float,
    error_std_by_pid: dict[str, float] | None = None,
) -> list[dict]:
    """Manager-facing inventory plan driven by the *forecast* demand.

    For every SKU this computes:
      - expected lead-time demand  (sum of the forecast over the lead time)
      - safety stock / reorder point / EOQ
      - recommended order quantity (reorder point minus current stock)
      - stockout and overstock risk with a plain-language reason

    ``forecast`` maps ``product_id`` -> DataFrame(date, yhat) of future demand.
    ``error_std_by_pid`` is the forecast-error standard deviation per SKU (from
    the test-set residuals); it falls back to the forecast/historical spread.
    """
    z = NormalDist().inv_cdf(service_level)
    error_std_by_pid = error_std_by_pid or {}
    rows = []
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("date")
        name = str(g["product_name"].iloc[0]) if "product_name" in g.columns else pid
        category = str(g["category"].iloc[0]) if "category" in g.columns else "Unknown"
        base_price = (
            float(g["base_price"].iloc[0])
            if "base_price" in g.columns
            else float(g["price"].mean())
        )

        lead_time = _last_value(g, "lead_time_days")
        lead_time = int(lead_time) if lead_time is not None else int(lead_time_default)
        lead_time = max(1, lead_time)

        stock = _last_value(g, "on_hand")
        stock = float(stock) if stock is not None else None

        hist_sales = g["sales"].astype(float)
        intermittency = float((hist_sales == 0).mean())

        f = forecast.get(pid)
        if f is not None and len(f):
            fc = f.sort_values("date")["yhat"].astype(float)
            avg_daily = float(fc.mean())
            lead_demand = float(fc.head(lead_time).sum())
            annual = avg_daily * 365
        else:
            avg_daily = float(hist_sales.mean())
            lead_demand = avg_daily * lead_time
            annual = avg_daily * 365

        # Forecast-error std (for safety stock): prefer the measured test-set
        # residual std, else fall back to the historical daily variability. We
        # deliberately do NOT use the forecast spread (a flat forecast would
        # wrongly imply zero safety stock).
        err_std = error_std_by_pid.get(pid)
        if err_std is None or err_std < 1e-6 or math.isnan(err_std):
            err_std = float(hist_sales.std(ddof=0))

        holding = base_price * holding_rate
        # EOQ is a *reference* order size only — never the automatic order.
        eoq = (
            math.sqrt(2 * annual * ordering_cost / holding)
            if holding > 0 and annual > 0
            else None
        )

        # Safety stock (normal assumption). For intermittent demand the normal
        # approximation overstates the buffer, so it is capped at one lead-time
        # of demand — you don't hold a full extra lead-time of slow movers.
        safety = z * err_std * math.sqrt(lead_time)
        if intermittency >= 0.2:
            safety = min(safety, lead_demand)
        reorder = lead_demand + safety

        # A large safety buffer (vs. lead-time demand) is worth explaining.
        buffer_note = ""
        if safety > 0.5 * lead_demand and lead_demand > 0:
            buffer_note = " The large safety buffer reflects high demand variability or a long lead time."

        # --- zero / negligible demand ---
        if avg_daily <= 0:
            rows.append(_inventory_row(
                pid, name, category, base_price, stock, lead_time, avg_daily, err_std,
                annual, lead_demand, safety, reorder, eoq, None, intermittency,
                "low", "low", "ok", 0,
                "No recent demand — no order needed.",
            ))
            continue

        # --- risk & recommendation (demand-based thresholds, robust to noise) ---
        days_cover = round(stock / avg_daily, 1) if stock is not None and avg_daily > 0 else None

        if stock is None:
            stockout_risk, overstock_risk, status = "unknown", "unknown", "unknown"
            recommended_order = None
            reason = "Current stock not provided — upload on-hand inventory to get a reorder quantity."
        elif stock < lead_demand:
            stockout_risk, overstock_risk, status = "high", "low", "stockout"
            recommended_order = int(math.ceil(reorder - stock))
            reason = (
                f"Current stock ({int(round(stock))}) may not cover expected demand "
                f"({int(round(lead_demand))}) during the {lead_time}-day supplier lead time. "
                f"Order {recommended_order} units now."
            )
        elif stock < reorder:
            stockout_risk, overstock_risk, status = "medium", "low", "reorder"
            recommended_order = int(math.ceil(reorder - stock))
            reason = (
                f"Stock ({int(round(stock))}) is below the reorder point ({int(round(reorder))}). "
                f"Reorder {recommended_order} units."
            )
        elif days_cover is not None and days_cover > max(3 * lead_time, 14):
            # Overstock = holding more than ~3 replenishment cycles (or > 14 days).
            stockout_risk, overstock_risk, status = "low", "high", "overstock"
            recommended_order = 0
            reason = (
                f"Stock ({int(round(stock))}) covers ~{days_cover} days — well above the reorder point "
                f"({int(round(reorder))}). No order needed; consider reducing future orders."
            )
        else:
            stockout_risk, overstock_risk, status = "low", "low", "ok"
            recommended_order = 0
            cover = f"{days_cover} days" if days_cover is not None else "plenty of"
            reason = f"Stock level is healthy ({cover} of cover). No order needed."

        reason += buffer_note

        rows.append(_inventory_row(
            pid, name, category, base_price, stock, lead_time, avg_daily, err_std,
            annual, lead_demand, safety, reorder, eoq, days_cover, intermittency,
            stockout_risk, overstock_risk, status, recommended_order, reason,
        ))
    return rows


def _inventory_row(
    pid, name, category, base_price, stock, lead_time, avg_daily, err_std,
    annual, lead_demand, safety, reorder, eoq, days_cover, intermittency,
    stockout_risk, overstock_risk, status, recommended_order, reason,
) -> dict:
    return {
        "product_id": pid,
        "product_name": name,
        "category": category,
        "base_price": round(base_price, 3),
        "current_stock": int(round(stock)) if stock is not None else None,
        "lead_time": lead_time,
        "avg_daily_demand": round(avg_daily, 2),
        "demand_std": round(err_std, 2),
        "annual_demand": round(annual, 1),
        "expected_demand_lead_time": round(lead_demand, 1),
        "safety_stock": round(safety, 1),
        "reorder_point": round(reorder, 1),
        "eoq": round(eoq, 1) if eoq is not None else None,
        "recommended_order": recommended_order,
        "days_cover": days_cover,
        "intermittency": round(intermittency, 3),
        "stockout_risk": stockout_risk,
        "overstock_risk": overstock_risk,
        "status": status,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_forecast(
    df: pd.DataFrame,
    model_names: list[str],
    product_ids: list[str] | None = None,
    category: str | None = None,
    horizon: int = 30,
    service_level: float = 0.95,
    lead_time: int = 3,
    ordering_cost: float = 50.0,
    holding_rate: float = 0.20,
) -> dict:
    config = _build_config()

    # Select SKUs (product filter + optional category filter).
    selected = df[["product_id", "category"]].drop_duplicates()
    if category and category != "all":
        selected = selected[selected["category"] == category]
    if product_ids:
        selected = selected[selected["product_id"].isin(product_ids)]
    selected_ids = selected["product_id"].unique().tolist()
    if not selected_ids:
        raise ValueError("No products match the selected filters.")

    # Feature engineering (calendar + lags + rolling), then chronological split.
    train, val, test = build_features_and_split(df, config, selected_ids)

    if test.empty:
        raise ValueError("Not enough history to form a test split for the selected products.")

    test_dates = pd.DatetimeIndex(pd.to_datetime(sorted(test["date"].unique())))

    results: dict = {
        "models": model_names,
        "model_labels": {m: MODEL_LABELS[m] for m in model_names},
        "model_descriptions": {m: MODEL_DESCRIPTIONS[m] for m in model_names},
        "products": [p for p in list_products(df) if p["product_id"] in selected_ids],
        "horizon": horizon,
        "test_dates": {"start": test_dates[0].date().isoformat(), "end": test_dates[-1].date().isoformat()},
        "test_metrics": {},
        "sku_metrics": {},
        "actual_vs_predicted": {},
        "future": {},
        "errors": {},
        "runtime_sec": {},
    }

    for name in model_names:
        t0 = time.perf_counter()
        try:
            model = MODEL_REGISTRY[name](config)
            model.fit(train, val)

            preds = predict_test_recursive(model, train, val, test, test_dates)

            merged = test[["product_id", "date", "sales"]].merge(
                preds, on=["product_id", "date"]
            )
            results["test_metrics"][name] = {
                "mae": round(mae(merged["sales"], merged["yhat"]), 3),
                "rmse": round(rmse(merged["sales"], merged["yhat"]), 3),
                "mape": round(mape(merged["sales"], merged["yhat"]), 3),
                "n_rows": int(len(merged)),
            }

            sku_rows = []
            series = {}
            for pid, g in merged.groupby("product_id"):
                sku_rows.append(
                    {
                        "product_id": pid,
                        "mae": round(mae(g["sales"], g["yhat"]), 3),
                        "rmse": round(rmse(g["sales"], g["yhat"]), 3),
                        "mape": round(mape(g["sales"], g["yhat"]), 3),
                    }
                )
                series[pid] = [
                    {
                        "date": r.date.date().isoformat(),
                        "actual": round(float(r.sales), 1),
                        "predicted": round(float(r.yhat), 1),
                    }
                    for r in g.sort_values("date").itertuples(index=False)
                ]
            results["sku_metrics"][name] = sku_rows
            results["actual_vs_predicted"][name] = series

            if getattr(model, "needs_features", False):
                if name == "lstm_gru":
                    future = _lstm_future(model, df, horizon)
                else:
                    future = _tree_future(model, df, horizon)
            else:
                future = _ts_future(model, test_dates, horizon)

            results["future"][name] = {
                pid: [
                    {"date": r.date.date().isoformat(), "predicted": round(float(r.yhat), 1)}
                    for r in fr.sort_values("date").itertuples(index=False)
                ]
                for pid, fr in future.items()
            }
        except Exception as exc:  # pragma: no cover - depends on user data
            results["errors"][name] = f"{type(exc).__name__}: {exc}"
        results["runtime_sec"][name] = round(time.perf_counter() - t0, 2)

    results["inventory"] = compute_inventory(
        df[df["product_id"].isin(selected_ids)],
        service_level, lead_time, ordering_cost, holding_rate,
    )
    results["inventory_params"] = {
        "service_level": service_level,
        "lead_time": lead_time,
        "ordering_cost": ordering_cost,
        "holding_rate": holding_rate,
        "z_score": round(NormalDist().inv_cdf(service_level), 4),
    }
    return results
