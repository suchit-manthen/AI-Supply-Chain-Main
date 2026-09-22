"""Forecasting orchestration layer (manager-facing pipeline).

This module glues the demand-pattern analysis, the forecasting models and the
inventory recommendation together. It does NOT reimplement any model — it drives
the existing forecasters and turns their output into plain-language guidance.

Flow
----
1. Characterise each SKU's demand (``demand_patterns.analyze_patterns``).
2. Match each SKU to an *appropriate* approach (``recommend_models``) — this is
   based on the demand pattern, not on whichever model scored lowest MAPE.
3. Run the set of models that are actually needed and blend the top-two
   pattern-appropriate forecasts into a single ensemble forecast per SKU.
4. Turn that forecast into an inventory action plan (safety stock, reorder
   point, EOQ, recommended order quantity, stockout/overstock risk).
"""
from __future__ import annotations

import time
from collections import Counter

import pandas as pd

import forecast_service as fs
from demand_patterns import analyze_patterns, recommend_models


def _blend_series(series_by_model: dict[str, dict], keys: list[str], weights: dict[str, float], pid: str) -> pd.Series | None:
    """Score-weighted blend of the per-model ``predicted`` series for ``pid``."""
    frames = []
    ws = []
    for m in keys:
        rows = series_by_model.get(m, {}).get(pid, [])
        if not rows:
            continue
        frames.append(pd.DataFrame(rows).set_index("date")["predicted"])
        ws.append(weights.get(m, 1.0))
    if not frames:
        return None
    blend = pd.concat(frames, axis=1)
    w = pd.Series(ws, index=blend.columns)
    return (blend * w).sum(axis=1) / w.sum()


def _blend_test(actual_vs_predicted: dict[str, dict], keys: list[str], weights: dict[str, float], pid: str):
    """Return (actual, weighted blended prediction) Series on the test period."""
    preds = []
    ws = []
    actual = None
    for m in keys:
        rows = actual_vs_predicted.get(m, {}).get(pid, [])
        if not rows:
            continue
        d = pd.DataFrame(rows)
        if actual is None:
            actual = d.set_index("date")["actual"]
        preds.append(d.set_index("date")["predicted"])
        ws.append(weights.get(m, 1.0))
    if not preds or actual is None:
        return None, None
    blend = pd.concat(preds, axis=1)
    w = pd.Series(ws, index=blend.columns)
    return actual, (blend * w).sum(axis=1) / w.sum()


def _history(df: pd.DataFrame, days: int = 90) -> dict[str, list[dict]]:
    out = {}
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("date").tail(days)
        out[pid] = [
            {"date": r.date.date().isoformat(), "sales": round(float(r.sales), 1)}
            for r in g.itertuples(index=False)
        ]
    return out


def _iso(d) -> str:
    return d.date().isoformat() if hasattr(d, "date") else str(d)


def run_recommend(
    df: pd.DataFrame,
    horizon: int = 30,
    service_level: float = 0.95,
    lead_time_default: int = 3,
    ordering_cost: float = 50.0,
    holding_rate: float = 0.20,
    product_ids: list[str] | None = None,
    category: str | None = None,
) -> dict:
    """Run the full manager pipeline and return a plain-language action plan."""
    t0 = time.perf_counter()

    # --- SKU selection ---
    selected = df[["product_id", "category"]].drop_duplicates()
    if category and category != "all":
        selected = selected[selected["category"] == category]
    if product_ids:
        selected = selected[selected["product_id"].isin(product_ids)]
    selected_ids = sorted(selected["product_id"].unique().tolist())
    if not selected_ids:
        raise ValueError("No products match the selected filters.")

    # --- demand pattern analysis ---
    patterns = analyze_patterns(df[df["product_id"].isin(selected_ids)])
    recommendation = recommend_models(patterns)

    # --- decide which models are actually needed (union of top-2 per SKU) ---
    needed = sorted({m for r in recommendation.values() for m in r["ensemble"]})

    # --- run the models (test metrics + future forecasts) ---
    advanced = fs.run_forecast(
        df,
        model_names=needed,
        product_ids=selected_ids,
        horizon=horizon,
        service_level=service_level,
        lead_time=lead_time_default,
        ordering_cost=ordering_cost,
        holding_rate=holding_rate,
    )

    # --- blend the top-two pattern-appropriate models into one forecast ---
    forecast = {}
    forecast_frames = {}
    error_std = {}
    for pid, r in recommendation.items():
        weights = r["ensemble_weights"]
        blended = _blend_series(advanced["future"], r["ensemble"], weights, pid)
        if blended is not None:
            forecast_frames[pid] = pd.DataFrame({"date": blended.index, "yhat": blended.values})
            forecast[pid] = [
                {"date": _iso(d), "predicted": round(float(v), 1)}
                for d, v in blended.items()
            ]
        actual, pred = _blend_test(advanced["actual_vs_predicted"], r["ensemble"], weights, pid)
        if actual is not None and pred is not None and len(pred) > 1:
            error_std[pid] = float((actual - pred).std(ddof=0))

    # --- forecast-based inventory plan ---
    inventory = fs.compute_inventory_from_forecast(
        df[df["product_id"].isin(selected_ids)],
        forecast_frames,
        service_level,
        lead_time_default,
        ordering_cost,
        holding_rate,
        error_std_by_pid=error_std,
    )

    summary = _summarize(inventory)
    model_usage = dict(Counter(r["primary"] for r in recommendation.values()))

    return {
        "mode": "recommend",
        "horizon": horizon,
        "date_max": df["date"].max().date().isoformat(),
        "model_usage": model_usage,
        "inventory_params": {
            "service_level": service_level,
            "lead_time": lead_time_default,
            "ordering_cost": ordering_cost,
            "holding_rate": holding_rate,
        },
        "products": [p for p in fs.list_products(df) if p["product_id"] in selected_ids],
        "patterns": patterns,
        "recommendation": recommendation,
        "history": _history(df[df["product_id"].isin(selected_ids)]),
        "forecast": forecast,
        "inventory": inventory,
        "summary": summary,
        "advanced": advanced,
        "runtime_sec": round(time.perf_counter() - t0, 2),
    }


def _summarize(inventory: list[dict]) -> dict:
    stockout = [i for i in inventory if i["stockout_risk"] == "high"]
    reorder = [i for i in inventory if i["status"] in ("stockout", "reorder")]
    overstock = [i for i in inventory if i["overstock_risk"] == "high"]
    unknown = [i for i in inventory if i["status"] == "unknown"]
    ok = [i for i in inventory if i["status"] == "ok"]

    # Attention list: stockouts first, then reorders, ordered by risk then size.
    attention = sorted(
        reorder,
        key=lambda i: (
            0 if i["stockout_risk"] == "high" else 1,
            -(i["recommended_order"] or 0),
        ),
    )

    return {
        "n_skus": len(inventory),
        "n_stockout": len(stockout),
        "n_reorder": len(reorder),
        "n_overstock": len(overstock),
        "n_ok": len(ok),
        "n_unknown_stock": len(unknown),
        "total_recommended_order": int(sum(i["recommended_order"] or 0 for i in inventory)),
        "attention": attention,
    }
