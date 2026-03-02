"""
ML Pipeline Smoke Test

Validates the full ML pipeline end-to-end with tiny datasets.
Runs on CPU in ~2 minutes. Use this for:
  - CI/CD validation
  - Post-install verification
  - Quick sanity check after code changes

Usage:
  python -m pipeline.smoke_test
"""

import sys
import os
import time
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✓"
FAIL = "✗"
results = []


def test(name, fn):
    """Run a test and record result."""
    try:
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        results.append((PASS, name, f"{elapsed:.1f}s"))
        print(f"  {PASS} {name} ({elapsed:.1f}s)")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL} {name}: {e}")


def test_config():
    from config import (
        METRIC_FEATURES, SERVICES, ROOT_CAUSE_LABELS,
        anomaly_config, forecast_config, root_cause_config,
        log_cluster_config, nlp_config, training_config, DEVICE,
    )
    assert len(METRIC_FEATURES) == 8
    assert len(SERVICES) == 3
    assert len(ROOT_CAUSE_LABELS) == 11
    assert DEVICE in ("cpu", "cuda")


def test_metric_generator():
    from data.generators.metric_generator import generate_training_dataset
    metrics_df, windows_df = generate_training_dataset(
        n_normal_windows=20, n_anomaly_windows=10, window_size=30, seed=99,
    )
    assert len(metrics_df) > 0
    assert len(windows_df) == 30
    assert "label" in windows_df.columns
    assert "is_anomaly" in windows_df.columns
    assert windows_df["is_anomaly"].sum() > 0
    return metrics_df, windows_df


def test_log_generator():
    from data.generators.log_generator import generate_log_dataset
    logs_df = generate_log_dataset(n_logs=200, seed=99)
    assert len(logs_df) == 200
    assert "message" in logs_df.columns
    assert "cluster_label" in logs_df.columns
    assert "severity" in logs_df.columns
    return logs_df


# Store shared data
shared = {}


def test_anomaly_detector():
    from models.anomaly.detector import AnomalyDetector
    from data.generators.metric_generator import generate_training_dataset

    metrics_df, windows_df = generate_training_dataset(
        n_normal_windows=30, n_anomaly_windows=15, window_size=30, seed=42,
    )

    detector = AnomalyDetector()

    # Train Isolation Forest
    if_results = detector.train_isolation_forest(windows_df)
    assert "precision" in if_results
    assert 0 <= if_results["f1"] <= 1

    # Train LSTM-AE (tiny: few epochs)
    from config import anomaly_config
    old_epochs = anomaly_config.lstm_epochs
    old_patience = anomaly_config.lstm_patience
    anomaly_config.lstm_epochs = 5
    anomaly_config.lstm_patience = 3

    lstm_results = detector.train_lstm_autoencoder(metrics_df)
    assert "precision" in lstm_results
    assert detector.reconstruction_threshold > 0

    anomaly_config.lstm_epochs = old_epochs
    anomaly_config.lstm_patience = old_patience

    # Inference
    sample_window = metrics_df[metrics_df["window_id"] == windows_df["window_id"].iloc[0]]
    pred = detector.predict(sample_window)
    assert "is_anomaly" in pred
    assert "anomaly_score" in pred
    assert 0 <= pred["anomaly_score"] <= 1

    # Save/load roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        detector.save(Path(tmpdir))
        detector2 = AnomalyDetector()
        detector2.load(Path(tmpdir))
        pred2 = detector2.predict(sample_window)
        assert abs(pred2["anomaly_score"] - pred["anomaly_score"]) < 0.01

    shared["metrics_df"] = metrics_df
    shared["windows_df"] = windows_df


def test_forecaster():
    from models.forecasting.forecaster import MetricForecaster
    from config import forecast_config

    metrics_df = shared["metrics_df"]

    forecaster = MetricForecaster()

    # Tiny training
    old_epochs = forecast_config.epochs
    old_patience = forecast_config.patience
    forecast_config.epochs = 5
    forecast_config.patience = 3

    results = forecaster.train(metrics_df)
    assert "train_mae" in results

    forecast_config.epochs = old_epochs
    forecast_config.patience = old_patience

    # Inference with MC dropout
    pred = forecaster.predict(metrics_df.head(forecast_config.seq_length + 10), n_mc_samples=3)
    assert "forecasts" in pred
    assert "confidence_lower" in pred
    assert "confidence_upper" in pred
    assert len(pred["forecasts"]) == 8  # 8 features

    # Save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        forecaster.save(Path(tmpdir))
        f2 = MetricForecaster()
        f2.load(Path(tmpdir))
        pred2 = f2.predict(metrics_df.head(forecast_config.seq_length + 10), n_mc_samples=3)
        assert "forecasts" in pred2


def test_root_cause_classifier():
    from models.root_cause.classifier import RootCauseClassifier

    windows_df = shared["windows_df"]
    metrics_df = shared["metrics_df"]

    classifier = RootCauseClassifier()
    results = classifier.train(windows_df, metrics_df, use_cv=False)
    assert "train_accuracy" in results

    # Inference
    sample_wid = windows_df[windows_df["label"] != "normal"]["window_id"].iloc[0]
    sample = metrics_df[metrics_df["window_id"] == sample_wid]
    pred = classifier.predict(sample)
    assert "predicted_cause" in pred
    assert "confidence" in pred
    assert "top_causes" in pred
    assert len(pred["top_causes"]) > 0

    # Save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier.save(Path(tmpdir))
        c2 = RootCauseClassifier()
        c2.load(Path(tmpdir))
        pred2 = c2.predict(sample)
        assert pred2["predicted_cause"] == pred["predicted_cause"]


def test_log_clusterer_encoding():
    """Test log clustering without full SBERT download (just structure)."""
    from models.log_clustering.clusterer import LogClusterer
    clusterer = LogClusterer()

    # Test pattern extraction helper
    assert clusterer.cluster_patterns == {}
    assert clusterer.cluster_map == {}

    # Test get_pattern_summary
    summary = clusterer.get_pattern_summary()
    assert summary["total_patterns"] == 0


def test_nlp_fallback():
    """Test NLP fallback report generation (no model download needed)."""
    from models.nlp.report_generator import ReportGenerator, build_ml_context

    rg = ReportGenerator()

    # Test fallback report
    anomaly_results = {
        "is_anomaly": True,
        "anomaly_score": 0.78,
        "if_score": 0.65,
        "lstm_score": 0.85,
        "details": "Anomaly detected (score: 0.78)",
    }
    forecast_results = {
        "breach_alerts": [{
            "metric": "cpu_usage",
            "step": 5,
            "predicted_value": 0.92,
            "threshold": 0.85,
            "confidence": 0.75,
        }],
    }
    root_cause_results = {
        "predicted_cause": "memory_leak",
        "confidence": 0.82,
        "top_causes": [
            {"cause": "memory_leak", "probability": 0.82},
            {"cause": "jvm_gc_pressure", "probability": 0.12},
        ],
    }

    report = rg._fallback_report(
        anomaly_results=anomaly_results,
        forecast_results=forecast_results,
        root_cause_results=root_cause_results,
    )

    assert report["health_status"] in ("HEALTHY", "DEGRADED", "CRITICAL")
    assert len(report["anomalies"]) > 0
    assert len(report["root_causes"]) > 0
    assert len(report["recommendations"]) > 0
    assert "summary" in report

    # Test context building
    context = build_ml_context(
        anomaly_results=anomaly_results,
        forecast_results=forecast_results,
        root_cause_results=root_cause_results,
    )
    assert "Anomaly Detection" in context
    assert "Forecasting" in context


def test_evaluation_framework():
    from evaluation.benchmark import (
        ModelEvaluator, LatencyBenchmark, CostAnalyser, StatisticalTests,
    )

    evaluator = ModelEvaluator()

    # Anomaly evaluation
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    y_scores = np.array([0.1, 0.2, 0.9, 0.4, 0.15, 0.85])

    results = evaluator.evaluate_anomaly_detector(y_true, y_pred, y_scores)
    assert "precision" in results
    assert "recall" in results
    assert "auc_roc" in results

    # Cost analysis
    cost = CostAnalyser()
    comparison = cost.compare()
    assert "monthly_savings_usd" in comparison
    assert comparison["api_cost"]["monthly_total_cost_usd"] > 0

    # Statistical tests
    tests = StatisticalTests()
    a = np.random.randn(50) + 1
    b = np.random.randn(50)
    t_result = tests.paired_t_test(a, b)
    assert "p_value" in t_result
    ci = tests.bootstrap_ci(a, n_bootstrap=100)
    assert ci["ci_lower"] < ci["ci_upper"]


def test_ml_analyzer_integration():
    """Test the async ML analyzer's context parsing."""
    sys.path.insert(0, str(Path(__file__).parent.parent / ".."))

    # We can't import the async one easily, test the sync agent_integration
    from agent_integration import MLAnalyzer

    analyzer = MLAnalyzer()

    # Test rule-based fallback
    context = {
        "stats": {"total_logs": 1000, "total_errors": 50, "slow_traces": 5},
        "metrics": [],
        "error_logs": [],
        "logs": [],
    }
    result = analyzer._rule_based_fallback(context)
    assert result["health_status"] in ("HEALTHY", "DEGRADED", "CRITICAL")

    # Test metric mapping
    metrics = [
        {"name": "request_rate", "value": 50.0, "timestamp": "2025-01-01T00:00:00Z"},
        {"name": "error_rate", "value": 0.01, "timestamp": "2025-01-01T00:00:01Z"},
    ]
    window = analyzer._metrics_to_window(metrics)
    assert window is not None
    assert "request_rate" in window["metrics"]


def test_serving_schema():
    """Test the serving API models validate correctly."""
    from serving.model_server import MetricWindow, LogBatch, AnalyseRequest

    # MetricWindow
    mw = MetricWindow(
        service="movie-service",
        metrics={"request_rate": [50.0, 51.0], "error_rate": [0.01, 0.02]},
    )
    df = mw.to_dataframe()
    assert len(df) == 2
    assert "request_rate" in df.columns

    # LogBatch
    lb = LogBatch(messages=["Error: connection refused", "Timeout after 30s"])
    assert len(lb.messages) == 2

    # AnalyseRequest
    req = AnalyseRequest(metrics=mw, logs=lb, stats={"total_logs": 100})
    assert req.stats["total_logs"] == 100


def main():
    print("=" * 60)
    print("  ML Pipeline Smoke Test")
    print("=" * 60)
    print()

    test("Config loads correctly", test_config)
    test("Metric data generator", test_metric_generator)
    test("Log data generator", test_log_generator)
    test("Anomaly detector (IF + LSTM-AE)", test_anomaly_detector)
    test("Time-series forecaster (LSTM+Attn)", test_forecaster)
    test("Root cause classifier (XGBoost)", test_root_cause_classifier)
    test("Log clusterer structure", test_log_clusterer_encoding)
    test("NLP fallback report", test_nlp_fallback)
    test("Evaluation framework", test_evaluation_framework)
    test("ML analyzer integration", test_ml_analyzer_integration)
    test("Serving API schemas", test_serving_schema)

    print()
    print("=" * 60)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    print(f"  Results: {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for status, name, detail in results:
            if status == FAIL:
                print(f"  {FAIL} {name}: {detail}")
        sys.exit(1)
    else:
        print("\n  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
