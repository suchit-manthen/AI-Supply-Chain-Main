"""Feature engineering and chronological train/validation/test splitting.

Reads ``data/raw/sales.csv`` and produces a feature-engineered dataset plus
time-ordered splits saved under ``data/processed/``.

Design rules enforced here:
- Rows are sorted by (product_id, date) before any feature is created.
- Calendar features are re-derived from ``date`` (day, weekday, month, year,
  weekend, holiday). Holiday flags come from the raw data.
- Lag features (t-1, t-7, t-14, t-28) are computed per SKU.
- Rolling mean/std windows (7, 14, 28) are shifted by one day so they only
  contain *past* observations (never the current day's sales).
- Splits are chronological (train -> val -> test by date), never random.

Hidden data-generating parameters (base_demand, weekend_boost,
seasonality_phase, price_elasticity) are excluded to avoid target leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Raw columns to read. Hidden generator parameters are deliberately omitted so
# models cannot "see" the true demand process (target leakage).
BASE_COLS = [
    "date",
    "product_id",
    "sales",          # target
    "price",          # actual selling price
    "is_promo",
    "discount_pct",
    "is_holiday",
    "is_holiday_eve",
    "category",
    "base_price",     # regular shelf price (observable, static)
]


def load_raw(config: object) -> pd.DataFrame:
    """Load raw sales and keep only the columns usable as model inputs."""
    df = pd.read_csv(config.data.output.raw, parse_dates=["date"])
    df = df[BASE_COLS]
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day/weekday/month/year/weekend features derived from ``date``."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.dayofweek  # Monday=0 ... Sunday=6
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame, lags: list[int], group_col: str = "product_id", value_col: str = "sales"
) -> pd.DataFrame:
    """Add lagged target columns (past information only)."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    group_col: str = "product_id",
    value_col: str = "sales",
) -> pd.DataFrame:
    """Add rolling mean/std using only *past* values.

    ``shift(1)`` is applied first so a window ending at day t only includes
    days t-1 .. t-window (never the current day), preventing leakage.
    """
    df = df.copy()
    prev = df.groupby(group_col)[value_col].shift(1)
    for w in windows:
        roll = prev.groupby(df[group_col]).rolling(window=w, min_periods=w)
        df[f"rolling_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}"] = roll.std().reset_index(level=0, drop=True)
    return df


def build_features(df: pd.DataFrame, config: object) -> pd.DataFrame:
    """Apply calendar, lag and rolling features in a leakage-safe order."""
    df = add_calendar_features(df)
    df = add_lag_features(df, list(config.features.lags))
    df = add_rolling_features(df, list(config.features.rolling_windows))
    return df


def split_by_time(df: pd.DataFrame, config: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split into train/val/test using date cutoffs."""
    s = config.features.split
    dates = np.sort(df["date"].unique())
    n = len(dates)
    train_cut = dates[min(int(n * s.train_frac), n - 1)]
    val_cut = dates[min(int(n * (s.train_frac + s.val_frac)), n - 1)]

    train = df[df["date"] <= train_cut]
    val = df[(df["date"] > train_cut) & (df["date"] <= val_cut)]
    test = df[df["date"] > val_cut]
    return train, val, test


def _report(df: pd.DataFrame, name: str) -> None:
    print(f"\n[{name}]")
    print(f"  shape:      {df.shape}")
    print(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  SKUs:       {df['product_id'].nunique()}")


def main(config: object) -> None:
    raw = load_raw(config)
    df = build_features(raw, config)

    # Drop the warm-up period (first rows per SKU where lags/rolling are NaN).
    n_before = len(df)
    feature_cols = [c for c in df.columns if c not in BASE_COLS and c != "date"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"Warm-up rows dropped (NaNs in features): {n_before - len(df):,}")

    train, val, test = split_by_time(df, config)

    out = config.features.output
    Path(out.features).parent.mkdir(parents=True, exist_ok=True)

    # Full feature-engineered dataset (pre-split) for EDA.
    df.to_csv(out.features, index=False)
    train.to_csv(out.train, index=False)
    val.to_csv(out.val, index=False)
    test.to_csv(out.test, index=False)

    _report(df, "full features")
    _report(train, "train")
    _report(val, "validation")
    _report(test, "test")

    print("\nFeature columns (order):")
    for i, col in enumerate(feature_cols, start=1):
        print(f"  {i:2d}. {col}")

    print("\nWritten ->", out.features)
    print("Written ->", out.train)
    print("Written ->", out.val)
    print("Written ->", out.test)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_config  # noqa: E402

    main(load_config())
