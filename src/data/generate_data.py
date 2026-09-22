"""Generate a synthetic supermarket sales dataset.

The catalogue is split into eight *demand behaviour* archetypes so the
forecasting layer has genuinely different patterns to learn from:

    stable_weekly       strong, smooth day-of-week rhythm
    yearly_trending     clear annual seasonality + a growth trend
    promo_driven        demand spikes driven by promotions
    price_sensitive     demand reacts strongly to price changes
    holiday_driven      big spikes around public holidays
    volatile            high day-to-day variance
    intermittent        sparse / low-frequency demand (many zero days)
    complex_sequential  weekly + trend + promo + holiday + persistence

Crucially, the *behaviour* is a generator-internal knob. The output CSV only
contains observable columns (date, sales, price, promo, holiday, category, ...)
plus inventory metadata — no ``behavior`` label and no hidden generator
parameters (weekend_boost / seasonality_phase / price_elasticity). The
forecasting system must infer each SKU's pattern from the data itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Product catalogue: (category, name, base_demand, base_price, behaviour)
# --------------------------------------------------------------------------- #
_PRODUCT_SPECS = [
    ("Dairy",       "Whole Milk 1L",            220, 1.19, "stable_weekly"),
    ("Dairy",       "Cheddar Cheese 200g",       60, 2.49, "stable_weekly"),
    ("Dairy",       "Greek Yogurt 500g",         90, 2.19, "yearly_trending"),
    ("Bakery",      "White Bread 800g",         150, 1.49, "stable_weekly"),
    ("Bakery",      "Croissant 4-pack",          40, 2.29, "holiday_driven"),
    ("Bakery",      "Chocolate Cake",            25, 5.99, "holiday_driven"),
    ("Beverages",   "Cola 1.5L",                120, 1.89, "price_sensitive"),
    ("Beverages",   "Orange Juice 1L",           80, 2.09, "yearly_trending"),
    ("Beverages",   "Coffee Beans 500g",         45, 6.49, "yearly_trending"),
    ("Beverages",   "Mineral Water 6x1.5L",     100, 3.49, "price_sensitive"),
    ("Produce",     "Bananas 1kg",              180, 1.29, "volatile"),
    ("Produce",     "Tomatoes 1kg",              70, 2.99, "volatile"),
    ("Produce",     "Avocado 2-pack",            55, 3.49, "volatile"),
    ("Produce",     "Iceberg Lettuce",           40, 1.09, "complex_sequential"),
    ("Snacks",      "Potato Chips 150g",        110, 1.79, "promo_driven"),
    ("Snacks",      "Chocolate Bar 100g",       130, 1.29, "promo_driven"),
    ("Snacks",      "Salted Peanuts 200g",       50, 1.99, "intermittent"),
    ("Frozen",      "Frozen Pizza",              60, 3.99, "promo_driven"),
    ("Frozen",      "Vanilla Ice Cream 1L",      70, 4.49, "holiday_driven"),
    ("Frozen",      "Frozen Peas 750g",          30, 1.79, "complex_sequential"),
    ("Household",   "Toilet Paper 8-roll",       55, 4.99, "complex_sequential"),
    ("Household",   "Dish Soap 500ml",           35, 1.99, "intermittent"),
    ("Household",   "Laundry Detergent 3L",      28, 8.99, "intermittent"),
    ("Personal",    "Shampoo 400ml",             40, 3.49, "price_sensitive"),
]

# Generator-internal behaviour parameters. ``trend`` is the fractional change in
# demand across the whole history; ``elasticity`` is the log-log price
# elasticity; ``markdown_*`` add non-promo price variation so price sensitivity
# is learnable independently of the promotion flag.
_BEHAVIORS = {
    "stable_weekly": dict(
        weekend_boost=0.45, yearly_amp=0.04, trend=0.0,
        promo_freq=0.02, promo_lift=(0.02, 0.06), elasticity=-0.4, holiday_lift=0.08,
        noise=0.06, autocorr=0.35, intermittency=0.0, base_scale=1.0,
        markdown_freq=0.02, markdown_hi=0.05,
    ),
    "yearly_trending": dict(
        weekend_boost=0.05, yearly_amp=0.45, trend=0.60,
        promo_freq=0.03, promo_lift=(0.02, 0.08), elasticity=-0.6, holiday_lift=0.10,
        noise=0.10, autocorr=0.30, intermittency=0.0, base_scale=1.0,
        markdown_freq=0.02, markdown_hi=0.05,
    ),
    "promo_driven": dict(
        weekend_boost=0.05, yearly_amp=0.05, trend=0.0,
        promo_freq=0.20, promo_lift=(0.40, 0.80), elasticity=-0.7, holiday_lift=0.10,
        noise=0.08, autocorr=0.30, intermittency=0.0, base_scale=1.0,
        markdown_freq=0.02, markdown_hi=0.05,
    ),
    "price_sensitive": dict(
        weekend_boost=0.05, yearly_amp=0.05, trend=0.0,
        promo_freq=0.02, promo_lift=(0.02, 0.05), elasticity=-2.8, holiday_lift=0.10,
        noise=0.10, autocorr=0.30, intermittency=0.0, base_scale=1.0,
        markdown_freq=0.30, markdown_hi=0.22,
    ),
    "holiday_driven": dict(
        weekend_boost=0.05, yearly_amp=0.05, trend=0.0,
        promo_freq=0.02, promo_lift=(0.05, 0.15), elasticity=-0.6, holiday_lift=0.90,
        noise=0.10, autocorr=0.30, intermittency=0.0, base_scale=1.0,
        markdown_freq=0.02, markdown_hi=0.05,
    ),
    "volatile": dict(
        weekend_boost=0.08, yearly_amp=0.08, trend=0.0,
        promo_freq=0.02, promo_lift=(0.05, 0.15), elasticity=-0.6, holiday_lift=0.10,
        noise=0.65, autocorr=0.15, intermittency=0.0, base_scale=1.0,
        markdown_freq=0.03, markdown_hi=0.08,
    ),
    "intermittent": dict(
        weekend_boost=0.10, yearly_amp=0.04, trend=0.0,
        promo_freq=0.03, promo_lift=(0.05, 0.15), elasticity=-0.5, holiday_lift=0.10,
        noise=0.15, autocorr=0.15, intermittency=0.55, base_scale=0.35,
        markdown_freq=0.02, markdown_hi=0.05,
    ),
    "complex_sequential": dict(
        weekend_boost=0.10, yearly_amp=0.08, trend=0.10,
        promo_freq=0.03, promo_lift=(0.05, 0.12), elasticity=-0.5, holiday_lift=0.08,
        noise=0.15, autocorr=0.90, intermittency=0.0, base_scale=1.0,
        markdown_freq=0.03, markdown_hi=0.08,
    ),
}

# Replenishment lead time (days) per category (inventory optimisation input).
_LEAD_TIME_BY_CATEGORY = {
    "Dairy": (1, 2), "Bakery": (1, 2), "Produce": (1, 2),
    "Beverages": (2, 3), "Snacks": (2, 3),
    "Frozen": (3, 5), "Household": (3, 5), "Personal": (3, 5),
}

# On-hand stock is seeded relative to a product's lead-time demand so the demo
# produces a realistic mix of stockout / healthy / overstocked SKUs. A dedicated
# RNG keeps the historical *sales* stream independent of these inventory columns.
_INVENTORY_SEED = 777
_ON_HAND_FACTORS = [0.5, 0.8, 1.1, 1.5, 2.2, 3.0]


def build_product_catalogue(rng: np.random.Generator) -> pd.DataFrame:
    """Create an internal catalogue (incl. behaviour + yearly phase)."""
    inv_rng = np.random.default_rng(_INVENTORY_SEED)
    rows = []
    for i, (cat, name, base_demand, base_price, behavior) in enumerate(_PRODUCT_SPECS, start=1):
        lead_low, lead_high = _LEAD_TIME_BY_CATEGORY.get(cat, (2, 4))
        rows.append(
            {
                "product_id": f"SKU{i:04d}",
                "category": cat,
                "product_name": name,
                "base_demand": base_demand,
                "base_price": base_price,
                "lead_time_days": int(inv_rng.integers(lead_low, lead_high + 1)),
                "behavior": behavior,
                "seasonality_phase": rng.uniform(0.0, 2.0 * np.pi),
            }
        )
    return pd.DataFrame(rows)


def _seed_on_hand(products: pd.DataFrame, sales: pd.DataFrame) -> None:
    """Seed on-hand stock per SKU from realised average demand."""
    mean = sales.groupby("product_id")["sales"].mean()
    lead_time = dict(zip(products["product_id"], products["lead_time_days"]))
    inv_rng = np.random.default_rng(_INVENTORY_SEED)
    on_hand = {
        pid: int(round(float(mean[pid]) * int(lead_time[pid]) * float(inv_rng.choice(_ON_HAND_FACTORS))))
        for pid in products["product_id"]
    }
    products["on_hand"] = products["product_id"].map(on_hand)


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #


def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
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
    df = pd.DataFrame({"date": dates})
    df["is_holiday"] = 0
    df["is_holiday_eve"] = 0

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

    easter = {"2022": "2022-04-17", "2023": "2023-04-09", "2024": "2024-03-31"}
    for year, date_str in easter.items():
        holiday = pd.Timestamp(date_str)
        df.loc[df["date"] == holiday, "is_holiday"] = 1
        df.loc[df["date"] == holiday - pd.Timedelta(days=1), "is_holiday_eve"] = 1

    return df


# --------------------------------------------------------------------------- #
# Demand generation
# --------------------------------------------------------------------------- #


def generate_demand(products: pd.DataFrame, calendar: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate daily sales per product using its behaviour archetype."""
    n_days = len(calendar)
    date_arr = calendar["date"].values
    dayofyear = calendar["dayofyear"].values
    is_weekend = calendar["is_weekend"].values
    is_holiday = calendar["is_holiday"].values
    is_holiday_eve = calendar["is_holiday_eve"].values
    t = np.arange(n_days)

    frames = []
    for _, prod in products.iterrows():
        b = _BEHAVIORS[prod["behavior"]]
        base = prod["base_demand"] * b["base_scale"]
        price = prod["base_price"]
        phase = prod["seasonality_phase"]

        yearly = 1.0 + b["yearly_amp"] * np.sin(2.0 * np.pi * dayofyear / 365.25 + phase)
        weekly = 1.0 + b["weekend_boost"] * is_weekend
        holiday = 1.0 + b["holiday_lift"] * is_holiday + 0.5 * b["holiday_lift"] * is_holiday_eve
        trend = 1.0 + b["trend"] * (t / (n_days - 1))

        promo = rng.random(n_days) < b["promo_freq"]
        promo = promo & ~(is_holiday.astype(bool))
        markdown = (rng.random(n_days) < b["markdown_freq"]) & ~promo & ~(is_holiday.astype(bool))

        discount = np.zeros(n_days)
        discount[promo] = rng.uniform(0.10, 0.35, size=int(promo.sum()))
        discount[markdown] = rng.uniform(0.0, b["markdown_hi"], size=int(markdown.sum()))
        price_t = price * (1.0 - discount)
        price_effect = (1.0 - discount) ** b["elasticity"]

        promo_lift = np.zeros(n_days)
        promo_lift[promo] = rng.uniform(b["promo_lift"][0], b["promo_lift"][1], size=int(promo.sum()))
        promo_effect = 1.0 + promo_lift

        # --- multiplicative AR(1) demand noise (stationary, log-space) ---
        # ``autocorr`` is the persistence coefficient; ``noise`` is the log-std,
        # kept constant so more persistent series are *not* artificially shrunk.
        alpha = b["autocorr"]
        noise = b["noise"]
        z = np.zeros(n_days)
        z[0] = rng.normal(0.0, noise)
        for k in range(1, n_days):
            z[k] = alpha * z[k - 1] + rng.normal(0.0, noise * np.sqrt(1.0 - alpha ** 2))
        noise_mult = np.exp(z)

        expected = base * yearly * weekly * trend * holiday * promo_effect * price_effect
        sales = np.random.poisson(expected * noise_mult)
        if b["intermittency"] > 0:
            sales = np.where(rng.random(n_days) < b["intermittency"], 0, sales)

        frames.append(
            pd.DataFrame(
                {
                    "date": date_arr,
                    "product_id": prod["product_id"],
                    "sales": sales,
                    "price": np.round(price_t, 2),
                    "is_promo": promo.astype(int),
                    "discount_pct": np.round(discount, 2),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(config: object) -> None:
    rng = np.random.default_rng(config.data.random_seed)

    products = build_product_catalogue(rng)
    dates = pd.date_range(start=config.data.start_date, end=config.data.end_date, freq="D")
    calendar = build_calendar(dates).merge(holiday_flags(dates), on="date")

    sales = generate_demand(products, calendar, rng)
    _seed_on_hand(products, sales)

    # Observable catalogue output (no behaviour / latent parameters).
    products_out = products[
        ["product_id", "category", "product_name", "base_demand", "base_price", "lead_time_days", "on_hand"]
    ]
    products_out.to_csv(config.data.output.products, index=False)

    # Observable long-format sales output (no base_demand / latent parameters).
    sales_out = sales.merge(calendar, on="date").merge(
        products_out.drop(columns=["base_demand"]), on="product_id"
    )

    Path(config.data.output.raw).parent.mkdir(parents=True, exist_ok=True)
    Path(config.data.output.processed).parent.mkdir(parents=True, exist_ok=True)
    sales_out.to_csv(config.data.output.raw, index=False)

    print(f"Products:        {products['product_id'].nunique()}")
    print(f"Date range:      {sales_out['date'].min().date()} -> {sales_out['date'].max().date()}")
    print(f"Rows generated:  {len(sales_out):,}")
    print(f"Avg daily sales: {sales_out['sales'].mean():.1f}")
    print(f"Promo share:     {sales_out['is_promo'].mean():.3f}")
    print(f"Written -> {config.data.output.raw}")
    print(f"Written -> {config.data.output.products}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_config  # noqa: E402

    main(load_config())
