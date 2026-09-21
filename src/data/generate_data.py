"""Generate a synthetic supermarket sales dataset.

The data is generated with realistic demand patterns so that later forecasting
models (ARIMA/SARIMA, Prophet, Random Forest, XGBoost/LightGBM, LSTM/GRU) have
signal to learn from. Demand is modelled as a multiplicative combination of:

    sales = base_demand
            * yearly_seasonality
            * weekly_seasonality (weekend boost)
            * holiday_effect
            * promotion_lift
            * price_effect (elasticity)
            * noise (log-normal)

The output is written in long format (one row per product per day).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Product catalogue
# --------------------------------------------------------------------------- #

# (category, product_name, base_demand, base_price, weekend_boost_override)
# base_demand  -> avg daily units sold at base price, no promo, no holiday
# base_price   -> regular shelf price in EUR
_PRODUCT_SPECS = [
    ("Dairy",       "Whole Milk 1L",            220, 1.19, None),
    ("Dairy",       "Cheddar Cheese 200g",       60, 2.49, None),
    ("Dairy",       "Greek Yogurt 500g",         90, 2.19, None),
    ("Bakery",      "White Bread 800g",         150, 1.49, None),
    ("Bakery",      "Croissant 4-pack",          40, 2.29, 0.45),
    ("Bakery",      "Chocolate Cake",            25, 5.99, 0.50),
    ("Beverages",   "Cola 1.5L",                120, 1.89, None),
    ("Beverages",   "Orange Juice 1L",           80, 2.09, None),
    ("Beverages",   "Coffee Beans 500g",         45, 6.49, None),
    ("Beverages",   "Mineral Water 6x1.5L",     100, 3.49, None),
    ("Produce",     "Bananas 1kg",              180, 1.29, None),
    ("Produce",     "Tomatoes 1kg",              70, 2.99, None),
    ("Produce",     "Avocado 2-pack",            55, 3.49, None),
    ("Produce",     "Iceberg Lettuce",           40, 1.09, None),
    ("Snacks",      "Potato Chips 150g",        110, 1.79, 0.35),
    ("Snacks",      "Chocolate Bar 100g",       130, 1.29, None),
    ("Snacks",      "Salted Peanuts 200g",       50, 1.99, 0.30),
    ("Frozen",      "Frozen Pizza",              60, 3.99, 0.40),
    ("Frozen",      "Vanilla Ice Cream 1L",      70, 4.49, 0.45),
    ("Frozen",      "Frozen Peas 750g",          30, 1.79, None),
    ("Household",   "Toilet Paper 8-roll",       55, 4.99, None),
    ("Household",   "Dish Soap 500ml",           35, 1.99, None),
    ("Household",   "Laundry Detergent 3L",      28, 8.99, None),
    ("Personal",    "Shampoo 400ml",             40, 3.49, None),
]


def build_product_catalogue(rng: np.random.Generator) -> pd.DataFrame:
    """Create a product catalogue DataFrame from the static spec list."""
    rows = []
    for i, (cat, name, base_demand, base_price, weekend_boost) in enumerate(
        _PRODUCT_SPECS, start=1
    ):
        # Each product gets its own phase offset for yearly seasonality so the
        # catalogue does not all peak at the same time.
        phase = rng.uniform(0.0, 2.0 * np.pi)
        rows.append(
            {
                "product_id": f"SKU{i:04d}",
                "category": cat,
                "product_name": name,
                "base_demand": base_demand,
                "base_price": base_price,
                "weekend_boost": (
                    weekend_boost if weekend_boost is not None else 0.25
                ),
                "seasonality_phase": phase,
                "price_elasticity": -1.2,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #


def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive date/time features used both for generation and later modelling."""
    cal = pd.DataFrame({"date": dates})
    cal["year"] = dates.year
    cal["month"] = dates.month
    cal["day"] = dates.day
    cal["dayofweek"] = dates.dayofweek  # Monday=0 ... Sunday=6
    cal["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    cal["dayofyear"] = dates.dayofyear
    cal["week"] = dates.isocalendar().week.to_numpy()
    return cal


def holiday_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Flag major holidays and their eve (common demand spikes for groceries)."""
    df = pd.DataFrame({"date": dates})
    df["is_holiday"] = 0
    df["is_holiday_eve"] = 0

    # Fixed-date holidays (with eve the day before)
    fixed_holidays = {
        "2022": [(1, 1), (12, 24), (12, 25), (12, 31)],
        "2023": [(1, 1), (12, 24), (12, 25), (12, 31)],
        "2024": [(1, 1), (12, 24), (12, 25), (12, 31)],
    }
    for year, days in fixed_holidays.items():
        for month, day in days:
            holiday = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
            eve = holiday - pd.Timedelta(days=1)
            df.loc[df["date"] == holiday, "is_holiday"] = 1
            df.loc[df["date"] == eve, "is_holiday_eve"] = 1

    # Easter (movable) - dates for 2022-2024
    easter = {"2022": "2022-04-17", "2023": "2023-04-09", "2024": "2024-03-31"}
    for year, date_str in easter.items():
        holiday = pd.Timestamp(date_str)
        df.loc[df["date"] == holiday, "is_holiday"] = 1
        df.loc[df["date"] == holiday - pd.Timedelta(days=1), "is_holiday_eve"] = 1

    return df


# --------------------------------------------------------------------------- #
# Demand generation
# --------------------------------------------------------------------------- #


def _yearly_seasonality(dayofyear: np.ndarray, phase: float) -> np.ndarray:
    """Smooth annual cycle using a sinusoid; peak shifted by ``phase``."""
    t = 2.0 * np.pi * dayofyear / 365.25
    return 1.0 + 0.15 * np.sin(t + phase)


def generate_demand(
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    rng: np.random.Generator,
    config: object,
) -> pd.DataFrame:
    """Generate daily sales for every product using a multiplicative model."""
    n_days = len(calendar)
    date_arr = calendar["date"].values
    dayofyear = calendar["dayofyear"].values
    is_weekend = calendar["is_weekend"].values
    is_holiday = calendar["is_holiday"].values
    is_holiday_eve = calendar["is_holiday_eve"].values

    frames = []
    for _, prod in products.iterrows():
        base = prod["base_demand"]
        price = prod["base_price"]
        elast = prod["price_elasticity"]
        phase = prod["seasonality_phase"]
        weekend_boost = prod["weekend_boost"]

        # --- seasonality & calendar effects ---
        yearly = _yearly_seasonality(dayofyear, phase)
        weekly = 1.0 + weekend_boost * is_weekend
        holiday = (
            1.0
            + config.data.demand.holiday_lift * is_holiday
            + 0.5 * config.data.demand.holiday_lift * is_holiday_eve
        )

        # --- promotions & price ---
        promo = rng.random(n_days) < config.data.demand.discount_fraction
        promo = promo & ~(is_holiday.astype(bool))  # no promo on holidays
        discount = np.where(
            promo,
            rng.uniform(0.10, 0.35, size=n_days),
            0.0,
        )
        price_t = price * (1.0 - discount)
        # price effect relative to base price (log-log elasticity)
        price_effect = (price_t / price) ** elast
        promo_lift = np.where(
            promo,
            rng.uniform(
                config.data.demand.promo_lift_low, config.data.demand.promo_lift_high, size=n_days
            ),
            0.0,
        )
        promo_effect = 1.0 + promo_lift

        # --- noise: log-normal multiplicative, slightly autocorrelated ---
        eps = np.exp(rng.normal(0.0, config.data.demand.noise_std, size=n_days))
        # simple AR(1) smoothing so day-to-day sales are not pure white noise
        alpha = 0.3
        smoothed = np.empty_like(eps)
        smoothed[0] = eps[0]
        for t in range(1, n_days):
            smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * eps[t]

        # --- combine into expected demand then sample ---
        expected = base * yearly * weekly * holiday * promo_effect * price_effect
        sales = np.random.poisson(expected * smoothed)

        df = pd.DataFrame(
            {
                "date": date_arr,
                "product_id": prod["product_id"],
                "sales": sales,
                "price": np.round(price_t, 2),
                "is_promo": promo.astype(int),
                "discount_pct": np.round(discount, 2),
            }
        )
        frames.append(df)

    sales = pd.concat(frames, ignore_index=True)
    return sales


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(config: object) -> None:
    rng = np.random.default_rng(config.data.random_seed)

    products = build_product_catalogue(rng)
    products.to_csv(config.data.output.products, index=False)

    dates = pd.date_range(
        start=config.data.start_date, end=config.data.end_date, freq="D"
    )
    calendar = build_calendar(dates)
    calendar = calendar.merge(holiday_flags(dates), on="date")

    sales = generate_demand(products, calendar, rng, config)

    # Merge in calendar + product metadata to produce a ready-to-use dataset.
    full = sales.merge(calendar, on="date").merge(products, on="product_id")

    # Ensure output directories exist.
    Path(config.data.output.raw).parent.mkdir(parents=True, exist_ok=True)
    Path(config.data.output.processed).parent.mkdir(parents=True, exist_ok=True)

    full.to_csv(config.data.output.raw, index=False)

    # Summary for the user
    print(f"Products:        {products['product_id'].nunique()}")
    print(f"Date range:      {full['date'].min().date()} -> {full['date'].max().date()}")
    print(f"Rows generated:  {len(full):,}")
    print(f"Avg daily sales: {full['sales'].mean():.1f}")
    print(f"Promo share:     {full['is_promo'].mean():.3f}")
    print(f"Written -> {config.data.output.raw}")
    print(f"Written -> {config.data.output.products}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_config  # noqa: E402

    main(load_config())
