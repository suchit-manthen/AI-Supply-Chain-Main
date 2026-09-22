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
│   ├── models/
│   │   ├── base.py          # common fit/predict interface
│   │   ├── sarima.py        # ARIMA/SARIMA (per-SKU, val-based order selection)
│   │   ├── prophet_model.py # Prophet (per-SKU, val-based hyperparameter selection)
│   │   ├── random_forest_model.py  # Random Forest (per-SKU, feature-based)
│   │   ├── xgboost_model.py # XGBoost (per-SKU, feature-based)
│   │   └── lstm_model.py    # LSTM/GRU (per-SKU, sequence feature-based)
│   └── evaluation/
│       ├── metrics.py       # MAE, RMSE, MAPE
│       └── evaluate.py      # prediction vs actual scoring
├── scripts/
│   ├── generate_dataset.py  # CLI entry point for data generation
│   ├── make_dataset.py      # CLI entry point for feature engineering
│   └── train_model.py       # CLI entry point for training/evaluation
├── notebooks/               # exploratory analysis
└── requirements.txt
```

## Pipeline (planned)

1. **Step 1 (done)** — Generate synthetic supermarket sales data.
2. **Step 2 (done)** — Feature engineering + chronological train/val/test split.
3. **Step 3 (in progress)** — Fit & compare models:
   - SARIMA (done) — MAE/RMSE/MAPE on test, val-based order selection
   - Prophet (done) — MAE/RMSE/MAPE on test, val-based hyperparameter selection
   - Random Forest (done) — feature-based, val-based hyperparameter selection
   - XGBoost (done) — feature-based, val-based hyperparameter selection
   - LSTM/GRU (done) — sequence feature-based, val-based architecture selection
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
