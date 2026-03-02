"""
Model Serving API

FastAPI server that loads all trained models and serves predictions.
Provides the same analysis endpoint that the agent and dashboard consume,
making it a drop-in replacement for the Claude API analysis.

Endpoints:
  POST /predict/anomaly      — anomaly detection
  POST /predict/forecast     — time-series forecasting
  POST /predict/root_cause   — root cause classification
  POST /predict/log_cluster  — log pattern matching
  POST /analyse              — full pipeline (all models + NLP report)
  GET  /health               — health check
  GET  /models               — loaded model info
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, DEVICE, METRIC_FEATURES

# ── Pydantic Models ──────────────────────────────────────────

class MetricWindow(BaseModel):
    """Input: time-series metric window."""
    timestamps: List[str] = []
    service: str = "unknown"
    metrics: Dict[str, List[float]]  # {metric_name: [values]}

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.metrics)
        df["service"] = self.service
        if self.timestamps:
            df["timestamp"] = pd.to_datetime(self.timestamps[:len(df)])
        for col in METRIC_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        return df


class LogBatch(BaseModel):
    """Input: batch of log messages."""
    messages: List[str]
    service: Optional[str] = None


class AnalyseRequest(BaseModel):
    """Input: full analysis request with all data."""
    metrics: Optional[MetricWindow] = None
    logs: Optional[LogBatch] = None
    stats: Optional[Dict] = None


class PredictionResponse(BaseModel):
    """Standard response wrapper."""
    model: str
    latency_ms: float
    result: Dict


# ── Global Model Registry ────────────────────────────────────
models = {}


def load_models():
    """Load all trained models at startup."""
    global models

    # Anomaly detector
    try:
        from models.anomaly.detector import AnomalyDetector
        detector = AnomalyDetector()
        detector.load()
        models["anomaly"] = detector
        print("[serving] ✓ Anomaly detector loaded")
    except Exception as e:
        print(f"[serving] ✗ Anomaly detector failed: {e}")

    # Forecaster
    try:
        from models.forecasting.forecaster import MetricForecaster
        forecaster = MetricForecaster()
        forecaster.load()
        models["forecaster"] = forecaster
        print("[serving] ✓ Forecaster loaded")
    except Exception as e:
        print(f"[serving] ✗ Forecaster failed: {e}")

    # Root cause classifier
    try:
        from models.root_cause.classifier import RootCauseClassifier
        classifier = RootCauseClassifier()
        classifier.load()
        models["root_cause"] = classifier
        print("[serving] ✓ Root cause classifier loaded")
    except Exception as e:
        print(f"[serving] ✗ Root cause classifier failed: {e}")

    # Log clusterer
    try:
        from models.log_clustering.clusterer import LogClusterer
        clusterer = LogClusterer()
        clusterer.load()
        models["log_cluster"] = clusterer
        print("[serving] ✓ Log clusterer loaded")
    except Exception as e:
        print(f"[serving] ✗ Log clusterer failed: {e}")

    # NLP report generator (load lazily to save VRAM)
    models["nlp"] = None  # loaded on first /analyse call

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[serving] Total VRAM usage: {vram:.1f} GB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    # Cleanup
    if models.get("nlp"):
        models["nlp"].unload()


# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="DevOps AI Agent — ML Model Server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    loaded = {k: v is not None for k, v in models.items()}
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": loaded,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    }


@app.get("/models")
async def model_info():
    info = {}
    for name, model in models.items():
        if model is not None:
            info[name] = {
                "loaded": True,
                "type": type(model).__name__,
            }
        else:
            info[name] = {"loaded": False}
    return info


@app.post("/predict/anomaly", response_model=PredictionResponse)
async def predict_anomaly(data: MetricWindow):
    if "anomaly" not in models or models["anomaly"] is None:
        raise HTTPException(503, "Anomaly detector not loaded")

    t0 = time.perf_counter()
    df = data.to_dataframe()
    result = models["anomaly"].predict(df)
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(model="anomaly_detection", latency_ms=round(latency, 2), result=result)


@app.post("/predict/forecast", response_model=PredictionResponse)
async def predict_forecast(data: MetricWindow):
    if "forecaster" not in models or models["forecaster"] is None:
        raise HTTPException(503, "Forecaster not loaded")

    t0 = time.perf_counter()
    df = data.to_dataframe()
    result = models["forecaster"].predict(df)
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(model="forecasting", latency_ms=round(latency, 2), result=result)


@app.post("/predict/root_cause", response_model=PredictionResponse)
async def predict_root_cause(data: MetricWindow):
    if "root_cause" not in models or models["root_cause"] is None:
        raise HTTPException(503, "Root cause classifier not loaded")

    t0 = time.perf_counter()
    df = data.to_dataframe()
    result = models["root_cause"].predict(df)
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(model="root_cause_classification", latency_ms=round(latency, 2), result=result)


@app.post("/predict/log_cluster", response_model=PredictionResponse)
async def predict_log_cluster(data: LogBatch):
    if "log_cluster" not in models or models["log_cluster"] is None:
        raise HTTPException(503, "Log clusterer not loaded")

    t0 = time.perf_counter()
    results = models["log_cluster"].predict(data.messages)
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        model="log_clustering",
        latency_ms=round(latency, 2),
        result={"predictions": results},
    )


@app.post("/analyse")
async def full_analysis(data: AnalyseRequest):
    """
    Run the full ML pipeline and generate an analysis report.
    This is the drop-in replacement for the Claude API call.
    """
    t0 = time.perf_counter()
    component_latencies = {}

    anomaly_result = None
    forecast_result = None
    root_cause_result = None
    log_cluster_result = None

    # Run metric-based models if metrics provided
    if data.metrics:
        df = data.metrics.to_dataframe()

        if models.get("anomaly"):
            t1 = time.perf_counter()
            anomaly_result = models["anomaly"].predict(df)
            component_latencies["anomaly"] = round((time.perf_counter() - t1) * 1000, 2)

        if models.get("forecaster"):
            t1 = time.perf_counter()
            forecast_result = models["forecaster"].predict(df)
            component_latencies["forecast"] = round((time.perf_counter() - t1) * 1000, 2)

        if models.get("root_cause") and anomaly_result and anomaly_result.get("is_anomaly"):
            t1 = time.perf_counter()
            root_cause_result = models["root_cause"].predict(df)
            component_latencies["root_cause"] = round((time.perf_counter() - t1) * 1000, 2)

    # Run log clustering if logs provided
    if data.logs and models.get("log_cluster"):
        t1 = time.perf_counter()
        cluster_preds = models["log_cluster"].predict(data.logs.messages)
        pattern_summary = models["log_cluster"].get_pattern_summary()
        log_cluster_result = {
            **pattern_summary,
            "recent_predictions": cluster_preds[:20],
        }
        component_latencies["log_cluster"] = round((time.perf_counter() - t1) * 1000, 2)

    # Generate NLP report
    t1 = time.perf_counter()
    if models.get("nlp") is None:
        # Lazy load NLP model
        try:
            from models.nlp.report_generator import ReportGenerator
            models["nlp"] = ReportGenerator()
            models["nlp"].load()
        except Exception as e:
            print(f"[serving] NLP load failed: {e}, using fallback")

    if models.get("nlp") and hasattr(models["nlp"], "generate_report"):
        report = models["nlp"].generate_report(
            anomaly_results=anomaly_result,
            forecast_results=forecast_result,
            root_cause_results=root_cause_result,
            log_cluster_results=log_cluster_result,
            stats=data.stats,
        )
    else:
        # Use the fallback directly
        from models.nlp.report_generator import ReportGenerator
        rg = ReportGenerator()
        report = rg._fallback_report(
            anomaly_results=anomaly_result,
            forecast_results=forecast_result,
            root_cause_results=root_cause_result,
            log_cluster_results=log_cluster_result,
            stats=data.stats,
        )
    component_latencies["nlp"] = round((time.perf_counter() - t1) * 1000, 2)

    total_latency = (time.perf_counter() - t0) * 1000

    return {
        **report,
        "_ml_metadata": {
            "total_latency_ms": round(total_latency, 2),
            "component_latencies_ms": component_latencies,
            "models_used": list(component_latencies.keys()),
            "inference_device": DEVICE,
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ML_SERVER_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
