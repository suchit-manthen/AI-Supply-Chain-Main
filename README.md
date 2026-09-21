# AI Supply Chain — Supermarket Demand Forecasting

An end-to-end prototype for forecasting supermarket product demand and using
the forecasts for inventory optimization (Safety Stock, Reorder Point, EOQ).

## Project structure

```
AI-SUPPLY-CHAIN/
├── config/
│   └── config.yaml          # dataset & model configuration
├── data/
│   ├── raw/                 # generated synthetic sales data
│   └── processed/           # feature-engineered data (later steps)
├── src/
│   ├── config.py            # YAML config loader
│   ├── data/
│   │   └── generate_data.py # synthetic dataset generator
│   ├── features/
│   │   └── make_dataset.py  # feature engineering + chronological split
│   ├── models/              # forecasting models (Step 3)
│   └── evaluation/          # MAE/RMSE/MAPE metrics (Step 3)
├── scripts/
│   ├── generate_dataset.py  # CLI entry point for data generation
│   └── make_dataset.py      # CLI entry point for feature engineering
├── notebooks/               # exploratory analysis
└── requirements.txt
```

## Pipeline (planned)

1. **Step 1 (done)** — Generate synthetic supermarket sales data.
2. **Step 2 (done)** — Feature engineering + chronological train/val/test split.
3. **Step 3** — Fit & compare models: ARIMA/SARIMA, Prophet, Random Forest,
   XGBoost/LightGBM, LSTM/GRU. Evaluate with MAE, RMSE, MAPE and pick the best.
4. **Step 4** — Inventory optimization: Safety Stock, Reorder Point, EOQ.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python scripts/generate_dataset.py
python scripts/make_dataset.py
```

The generator writes:
- `data/raw/sales.csv` — long-format daily sales per product
- `data/raw/products.csv` — product catalogue

## How the synthetic demand is modelled

```
sales ~ base_demand * yearly_seasonality * weekly_seasonality(weekend)
        * holiday_effect * promotion_lift * price_effect(elasticity)
        * log-normal noise (mildly autocorrelated)
```

This gives the ML models genuine signal: weekly cycles, annual seasonality,
holiday spikes, promo-driven lifts and price sensitivity.
