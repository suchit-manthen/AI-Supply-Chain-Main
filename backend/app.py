"""Flask API for the AI supermarket demand-forecasting dashboard.

Workflow endpoints:
    POST /api/upload   -> validate an uploaded CSV, store it, return a summary
    POST /api/run      -> run forecasting on the stored dataset, return results
    GET  /api/meta     -> model list / labels / descriptions + defaults

Run from the backend directory:

    python app.py
"""
from __future__ import annotations

import io
import uuid

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

import forecast_service as fs
import orchestrator

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB
CORS(app)

# In-memory store of uploaded datasets (keyed by dataset_id). Good enough for a
# demo; a production build would persist these or use a job queue.
DATASETS: dict[str, pd.DataFrame] = {}


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/meta")
def meta():
    return jsonify(
        {
            "models": list(fs.MODEL_REGISTRY.keys()),
            "model_labels": fs.MODEL_LABELS,
            "model_descriptions": fs.MODEL_DESCRIPTIONS,
            "defaults": {
                "horizon": 30,
                "service_level": 0.95,
                "lead_time": 3,
                "ordering_cost": 50.0,
                "holding_rate": 0.20,
            },
        }
    )


@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided (expected multipart field 'file')."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty upload."}), 400

    try:
        raw = pd.read_csv(io.BytesIO(file.read()))
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    try:
        df, report = fs.prepare_dataframe(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    dataset_id = uuid.uuid4().hex[:12]
    DATASETS[dataset_id] = df

    return jsonify(
        {
            "dataset_id": dataset_id,
            "report": report,
            "products": fs.list_products(df),
        }
    )


@app.post("/api/demo")
def demo():
    """Load the built-in generated dataset as a convenience for demos."""
    from pathlib import Path

    demo_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "sales.csv"
    if not demo_path.exists():
        return jsonify({"error": "Demo dataset not found. Run scripts/generate_dataset.py first."}), 404

    try:
        df, report = fs.prepare_dataframe(pd.read_csv(demo_path))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    dataset_id = uuid.uuid4().hex[:12]
    DATASETS[dataset_id] = df
    return jsonify({"dataset_id": dataset_id, "report": report, "products": fs.list_products(df)})


@app.post("/api/recommend")
def recommend():
    """Run the manager-facing pipeline: pattern analysis -> forecast -> actions."""
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    if not dataset_id or dataset_id not in DATASETS:
        return jsonify({"error": "Unknown dataset_id. Upload a CSV first."}), 400

    horizon = int(body.get("horizon", 30))
    service_level = float(body.get("service_level", 0.95))
    lead_time = int(body.get("lead_time", 3))
    ordering_cost = float(body.get("ordering_cost", 50.0))
    holding_rate = float(body.get("holding_rate", 0.20))
    products = body.get("products")
    category = body.get("category")

    try:
        results = orchestrator.run_recommend(
            DATASETS[dataset_id],
            horizon=horizon,
            service_level=service_level,
            lead_time_default=lead_time,
            ordering_cost=ordering_cost,
            holding_rate=holding_rate,
            product_ids=products,
            category=category,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Recommendation failed: {exc}"}), 500

    return jsonify(results)


@app.post("/api/run")
def run():
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    if not dataset_id or dataset_id not in DATASETS:
        return jsonify({"error": "Unknown dataset_id. Upload a CSV first."}), 400

    models = body.get("models") or ["random_forest"]
    if models == ["all"] or models == "all":
        models = list(fs.MODEL_REGISTRY.keys())
    unknown = [m for m in models if m not in fs.MODEL_REGISTRY]
    if unknown:
        return jsonify({"error": f"Unknown model(s): {', '.join(unknown)}"}), 400

    products = body.get("products")
    category = body.get("category")

    horizon = int(body.get("horizon", 30))
    service_level = float(body.get("service_level", 0.95))
    lead_time = int(body.get("lead_time", 3))
    ordering_cost = float(body.get("ordering_cost", 50.0))
    holding_rate = float(body.get("holding_rate", 0.20))

    try:
        results = fs.run_forecast(
            DATASETS[dataset_id],
            model_names=models,
            product_ids=products,
            category=category,
            horizon=horizon,
            service_level=service_level,
            lead_time=lead_time,
            ordering_cost=ordering_cost,
            holding_rate=holding_rate,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Forecasting failed: {exc}"}), 500

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
