"""Demand-pattern analysis and approach recommendation.

Concept (per the product brief): different forecasting algorithms suit
different demand patterns. Instead of picking whichever model happened to have
the lowest MAPE on a single dataset, we characterise each SKU's historical
demand and match it to the algorithm *family* that handles that pattern best.

Pattern signals (computed per SKU from historical daily sales):

  - weekly            strength of a repeating day-of-week cycle
  - yearly            strength of a repeating month-of-year cycle
  - trend             normalised linear trend across the history
  - intermittency     fraction of zero-sales days
  - volatility        coefficient of variation of daily sales
  - promo             how strongly sales react to the promotion flag
  - holiday           how strongly sales react to holidays
  - price             how strongly sales react to price/discount changes
                      (measured on *non-promotion* days, so it isolates the
                      pure price effect from the promotion flag)
  - autocorr          lag-1 autocorrelation (day-to-day persistence / sequential)

These signals are mapped to a score per model family (0..1). The top-scoring
family is the *recommended* approach; the top two are blended into a weighted
ensemble forecast. Forecast accuracy (MAPE etc.) is deliberately *not* used to
choose the approach here — it is reported separately for technical users.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_KEYS = ["sarima", "prophet", "random_forest", "xgboost", "lstm_gru"]

# Human-readable description of the *pattern* each approach is best at.
APPROACH_PATTERN = {
    "sarima": "a repeating weekly pattern",
    "prophet": "seasonal, trending or holiday-driven demand",
    "xgboost": "promotion-, price- or holiday-driven demand",
    "random_forest": "irregular, intermittent or noisy demand",
    "lstm_gru": "sequential demand patterns",
}

# Plain-language tags surfaced to the store manager.
PATTERN_TAGS = {
    "weekly": "Strong weekly pattern",
    "yearly": "Seasonal (time-of-year) pattern",
    "trend_up": "Rising demand",
    "trend_down": "Falling demand",
    "intermittent": "Irregular / intermittent sales",
    "volatile": "High day-to-day variation",
    "promo": "Promotion-sensitive",
    "price": "Price-sensitive",
    "holiday": "Holiday-driven spikes",
    "stable": "Stable, predictable demand",
}


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _safe_autocorr(s: pd.Series, lag: int = 1) -> float:
    if len(s) <= lag or float(s.std()) == 0:
        return 0.0
    v = s.autocorr(lag=lag)
    return float(v) if v is not None and not np.isnan(v) else 0.0


def analyze_patterns(df: pd.DataFrame) -> dict[str, dict]:
    """Compute a demand-pattern profile for every SKU.

    ``df`` is the canonical, cleaned sales frame (columns: date, product_id,
    sales, is_promo, discount_pct, is_holiday, is_holiday_eve, price, ...).
    """
    out: dict[str, dict] = {}
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("date")
        sales = g["sales"].astype(float)
        mean = float(sales.mean())
        std = float(sales.std(ddof=0))
        n = len(sales)
        dow = g["date"].dt.dayofweek

        # --- weekly seasonality: day-of-week peak-to-trough relative to mean.
        #     (Peak-to-trough is used rather than lag-7 autocorrelation because
        #     the latter is also inflated by a smooth yearly cycle / trend.)
        weekly = 0.0
        if n >= 14 and mean > 0:
            dow_means = sales.groupby(dow).mean()
            weekly = _clamp01(float((dow_means.max() - dow_means.min()) / mean))

        # --- yearly seasonality: spread of month-of-year means ---
        yearly = 0.0
        if mean > 0 and n >= 30:
            monthly = sales.groupby(g["date"].dt.month).mean()
            if len(monthly) > 1:
                yearly = _clamp01(float(monthly.std() / mean))

        # --- trend: slope of year-month means, normalised to per-day ---
        trend = 0.0
        if mean > 0:
            ym = sales.groupby(g["date"].dt.to_period("M")).mean()
            if len(ym) >= 12:
                slope_month = float(np.polyfit(np.arange(len(ym)), ym.to_numpy(), 1)[0])
                trend = float((slope_month / mean) / 30.0)

        # --- intermittency: fraction of zero-sales days ---
        intermittency = float((sales == 0).mean())

        # --- volatility: coefficient of variation ---
        volatility = float(std / mean) if mean > 0 else 1.0

        # --- promo sensitivity: lift on promotion days ---
        promo_sens = 0.0
        if "is_promo" in g.columns:
            promo = g["is_promo"].astype(float)
            if promo.sum() > 0:
                on = sales[promo > 0]
                off = sales[promo <= 0]
                if off.mean() > 0:
                    promo_sens = _clamp01(float(on.mean() / off.mean() - 1.0))

        # --- holiday sensitivity: lift on holiday / holiday-eve days ---
        holiday_sens = 0.0
        if {"is_holiday", "is_holiday_eve"}.issubset(g.columns):
            holiday = (g["is_holiday"].astype(float) + g["is_holiday_eve"].astype(float)).clip(upper=1)
            if holiday.sum() > 0 and mean > 0:
                on = sales[holiday > 0]
                off = sales[holiday <= 0]
                if off.mean() > 0:
                    holiday_sens = _clamp01(float(on.mean() / off.mean() - 1.0))

        # --- price sensitivity: sales vs discount on NON-promo days only ---
        price_sens = 0.0
        if "discount_pct" in g.columns and "is_promo" in g.columns:
            non_promo = g[g["is_promo"] == 0]
            if len(non_promo) > 10 and float(non_promo["discount_pct"].std()) > 1e-6:
                price_sens = _clamp01(float(non_promo["sales"].corr(non_promo["discount_pct"])))

        # --- autocorrelation: day-to-day persistence after removing weekly,
        #     yearly AND linear-trend effects (isolates sequential structure).
        autocorr = 0.0
        if mean > 0 and n > 2:
            dow_mean = sales.groupby(dow).transform("mean")
            month_mean = sales.groupby(g["date"].dt.month).transform("mean")
            resid = sales - dow_mean - month_mean + mean
            rv = resid.to_numpy()
            slope, intercept = np.polyfit(np.arange(n), rv, 1)
            detrended = rv - (slope * np.arange(n) + intercept)
            autocorr = _clamp01(_safe_autocorr(pd.Series(detrended), lag=1))

        out[pid] = {
            "product_id": pid,
            "n_days": int(n),
            "mean_sales": round(mean, 2),
            "weekly": round(weekly, 3),
            "yearly": round(yearly, 3),
            "trend": round(trend, 5),
            "intermittency": round(intermittency, 3),
            "volatility": round(volatility, 3),
            "promo": round(promo_sens, 3),
            "holiday": round(holiday_sens, 3),
            "price": round(price_sens, 3),
            "autocorr": round(autocorr, 3),
        }
    return out


def _score_models(p: dict) -> dict[str, float]:
    """Map a pattern profile to a score per model family (0..1).

    The weights encode the "right tool for the right pattern" idea:
      - SARIMA     : strong weekly rhythm, smooth & predictable (gated on weekly)
      - Prophet    : yearly seasonality + trend
      - XGBoost    : promotion / price / holiday interactions
      - Random F.  : intermittent / highly volatile (robust fallback)
      - LSTM/GRU   : sequential persistence (weekly + trend + autocorr + promo)
    """
    wk, yr = p["weekly"], p["yearly"]
    tr = min(abs(p["trend"]) * 365, 1.0)  # per-day trend scaled to a year
    it, vo = p["intermittency"], min(p["volatility"] / 0.8, 1.0)
    pr, ho = p["promo"], p["holiday"]
    ps, ac = p["price"], p["autocorr"]

    scores = {
        "sarima": wk * (0.60 + 0.40 * (1 - vo)),
        "prophet": (0.45 * yr + 0.40 * tr) * (1 - it) + 0.10 * wk,
        "xgboost": 0.40 * pr + 0.30 * ps + 0.20 * ho + 0.10 * vo,
        "random_forest": 0.55 * it + 0.30 * vo,
        "lstm_gru": 0.35 * wk + 0.25 * tr + 0.25 * ac + 0.15 * pr,
    }
    return scores


def _build_tags(p: dict) -> list[str]:
    tags = []
    if p["weekly"] >= 0.15:
        tags.append("weekly")
    if p["yearly"] >= 0.12:
        tags.append("yearly")
    if p["trend"] >= 0.0005:
        tags.append("trend_up")
    elif p["trend"] <= -0.0005:
        tags.append("trend_down")
    if p["intermittency"] >= 0.15:
        tags.append("intermittent")
    if p["volatility"] >= 0.45:
        tags.append("volatile")
    if p["promo"] >= 0.20:
        tags.append("promo")
    if p["price"] >= 0.20:
        tags.append("price")
    if p["holiday"] >= 0.15:
        tags.append("holiday")
    if not tags:
        tags.append("stable")
    return tags


def recommend_models(patterns: dict[str, dict]) -> dict[str, dict]:
    """Choose a primary + secondary approach per SKU based on its pattern.

    Returns, per SKU, the ranked model scores, the primary/secondary pair and
    the (score-normalised) ensemble weights used to blend their forecasts.
    """
    out: dict[str, dict] = {}
    for pid, p in patterns.items():
        scores = _score_models(p)
        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
        primary, secondary = ranked[0], ranked[1]

        s1, s2 = scores[primary], scores[secondary]
        total = (s1 + s2) if (s1 + s2) > 0 else 1.0
        weights = {primary: round(s1 / total, 3), secondary: round(s2 / total, 3)}

        tags = _build_tags(p)
        tag_text = _tags_to_phrase(tags)
        reason = (
            f"{tag_text}. We applied an approach suited to {APPROACH_PATTERN[primary]} "
            f"and cross-checked it against {APPROACH_PATTERN[secondary]}."
        )
        out[pid] = {
            "product_id": pid,
            "scores": {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])},
            "primary": primary,
            "secondary": secondary,
            "ensemble": [primary, secondary],
            "ensemble_weights": weights,
            "tags": tags,
            "tag_labels": [PATTERN_TAGS[t] for t in tags],
            "reason": reason,
        }
    return out


def _tags_to_phrase(tags: list[str]) -> str:
    labels = [PATTERN_TAGS[t] for t in tags]
    if not labels:
        return "Demand looks stable and predictable"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]
