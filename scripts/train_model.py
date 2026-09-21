"""Train and evaluate a single forecasting model (Step 3).

Usage (from project root):

    python scripts/train_model.py --model sarima
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluation.evaluate import evaluate_predictions, evaluate_per_sku  # noqa: E402
from src.models.sarima import SARIMAForecaster  # noqa: E402

MODEL_REGISTRY = {
    "sarima": SARIMAForecaster,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a forecaster.")
    parser.add_argument("--model", default="sarima", choices=list(MODEL_REGISTRY))
    args = parser.parse_args()

    config = load_config()
    out_dir = Path(config.models.output_dir)

    train = pd.read_csv(config.features.output.train, parse_dates=["date"])
    val = pd.read_csv(config.features.output.val, parse_dates=["date"])
    test = pd.read_csv(config.features.output.test, parse_dates=["date"])

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(config)

    print(f"Model: {model.name}")
    t0 = time.perf_counter()
    model.fit(train, val)
    fit_time = time.perf_counter() - t0

    test_dates = pd.DatetimeIndex(pd.to_datetime(test["date"].unique()))
    t0 = time.perf_counter()
    preds = model.predict(test_dates)
    pred_time = time.perf_counter() - t0

    metrics = evaluate_predictions(preds, test)
    per_sku = evaluate_per_sku(preds, test)

    print(f"\nFit time:    {fit_time:6.2f}s")
    print(f"Predict time: {pred_time:6.2f}s")
    print(f"\n=== Test metrics ({args.model}) ===")
    print(f"  MAE : {metrics['mae']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  MAPE: {metrics['mape']:.3f}%")
    print(f"  rows: {metrics['n_rows']}")

    print(f"\nPer-SKU MAPE summary:")
    print(f"  mean : {per_sku['mape'].mean():.2f}%")
    print(f"  min  : {per_sku['mape'].min():.2f}%")
    print(f"  max  : {per_sku['mape'].max():.2f}%")

    if hasattr(model, "best_order_"):
        orders = pd.Series({pid: str(o) for pid, o in model.best_order_.items()})
        print(f"\nSelected orders (validation-based):")
        print(orders.value_counts().to_string())

    # Persist predictions for reproducibility.
    pred_path = out_dir / f"predictions_{args.model}.csv"
    preds.to_csv(pred_path, index=False)
    per_sku_path = out_dir / f"metrics_{args.model}.csv"
    per_sku.to_csv(per_sku_path, index=False)
    print(f"\nWritten -> {pred_path}")
    print(f"Written -> {per_sku_path}")


if __name__ == "__main__":
    main()
