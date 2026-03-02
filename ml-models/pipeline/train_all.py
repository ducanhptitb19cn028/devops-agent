"""
Training Pipeline Orchestrator

Runs the full ML training pipeline:
  1. Generate / load training data
  2. Train each specialised model
  3. Evaluate all models
  4. Save artifacts and metrics
  5. Optionally track experiments with MLflow

Usage:
  python -m pipeline.train_all                    # full pipeline
  python -m pipeline.train_all --model anomaly    # single model
  python -m pipeline.train_all --data-only        # generate data only
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    training_config as tc, DATA_DIR, MODEL_DIR, ARTIFACT_DIR,
    METRIC_FEATURES,
)


def generate_data(force: bool = False):
    """Generate or load synthetic training data."""
    metrics_path = DATA_DIR / "metrics_raw.parquet"
    windows_path = DATA_DIR / "windows_features.parquet"
    logs_path = DATA_DIR / "logs_training.parquet"

    if not force and metrics_path.exists() and windows_path.exists() and logs_path.exists():
        print("[pipeline] Loading existing training data...")
        metrics_df = pd.read_parquet(metrics_path)
        windows_df = pd.read_parquet(windows_path)
        logs_df = pd.read_parquet(logs_path)
    else:
        print("[pipeline] Generating synthetic training data...")
        from data.generators.metric_generator import generate_training_dataset
        from data.generators.log_generator import generate_log_dataset

        metrics_df, windows_df = generate_training_dataset(
            n_normal_windows=3000,
            n_anomaly_windows=1500,
            seed=tc.seed,
        )
        logs_df = generate_log_dataset(n_logs=20000, seed=tc.seed)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(metrics_path, index=False)
        windows_df.to_parquet(windows_path, index=False)
        logs_df.to_parquet(logs_path, index=False)
        print(f"[pipeline] Data saved to {DATA_DIR}")

    return metrics_df, windows_df, logs_df


def split_data(metrics_df, windows_df):
    """Split into train/val/test sets."""
    from sklearn.model_selection import train_test_split

    window_ids = windows_df["window_id"].values
    labels = windows_df["label"].values

    # Stratified split
    train_ids, temp_ids = train_test_split(
        window_ids, test_size=tc.val_split + tc.test_split,
        stratify=labels, random_state=tc.seed,
    )
    temp_labels = windows_df[windows_df["window_id"].isin(temp_ids)]["label"].values
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=tc.test_split / (tc.val_split + tc.test_split),
        stratify=temp_labels, random_state=tc.seed,
    )

    splits = {
        "train": {"windows": windows_df[windows_df["window_id"].isin(train_ids)],
                   "metrics": metrics_df[metrics_df["window_id"].isin(train_ids)]},
        "val":   {"windows": windows_df[windows_df["window_id"].isin(val_ids)],
                   "metrics": metrics_df[metrics_df["window_id"].isin(val_ids)]},
        "test":  {"windows": windows_df[windows_df["window_id"].isin(test_ids)],
                   "metrics": metrics_df[metrics_df["window_id"].isin(test_ids)]},
    }

    for name, data in splits.items():
        print(f"[pipeline] {name}: {len(data['windows'])} windows, "
              f"{len(data['metrics'])} metric samples")

    return splits


def train_anomaly_detector(splits):
    """Train anomaly detection ensemble."""
    from models.anomaly.detector import AnomalyDetector

    detector = AnomalyDetector()
    t0 = time.time()

    if_results = detector.train_isolation_forest(splits["train"]["windows"])
    lstm_results = detector.train_lstm_autoencoder(
        splits["train"]["metrics"],
        splits["val"]["metrics"],
    )

    # Test set evaluation
    test_results = {}
    test_metrics = splits["test"]["metrics"]
    test_windows = splits["test"]["windows"]

    predictions = []
    for wid in test_windows["window_id"].unique():
        window = test_metrics[test_metrics["window_id"] == wid]
        if len(window) >= 30:
            pred = detector.predict(window)
            predictions.append({
                "window_id": wid,
                "predicted_anomaly": pred["is_anomaly"],
                "anomaly_score": pred["anomaly_score"],
                "actual_anomaly": int(test_windows[test_windows["window_id"] == wid]["is_anomaly"].iloc[0]),
            })

    if predictions:
        pred_df = pd.DataFrame(predictions)
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        y_true = pred_df["actual_anomaly"].values
        y_pred = pred_df["predicted_anomaly"].astype(int).values
        y_score = pred_df["anomaly_score"].values

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        test_results = {"test_precision": p, "test_recall": r, "test_f1": f1, "test_auc": auc}
        print(f"[anomaly] TEST — P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    detector.save()
    elapsed = time.time() - t0

    return {
        "model": "anomaly_detection",
        "training_time_s": round(elapsed, 1),
        "isolation_forest": if_results,
        "lstm_autoencoder": lstm_results,
        "test": test_results,
    }


def train_forecaster(splits):
    """Train time-series forecaster."""
    from models.forecasting.forecaster import MetricForecaster

    forecaster = MetricForecaster()
    t0 = time.time()

    results = forecaster.train(
        splits["train"]["metrics"],
        splits["val"]["metrics"],
    )

    forecaster.save()
    elapsed = time.time() - t0

    return {
        "model": "forecasting",
        "training_time_s": round(elapsed, 1),
        **results,
    }


def train_root_cause_classifier(splits):
    """Train root cause classifier."""
    from models.root_cause.classifier import RootCauseClassifier

    classifier = RootCauseClassifier()
    t0 = time.time()

    results = classifier.train(
        splits["train"]["windows"],
        splits["train"]["metrics"],
    )

    # Test evaluation
    test_windows = splits["test"]["windows"]
    anomaly_test = test_windows[test_windows["label"] != "normal"]
    if len(anomaly_test) > 0:
        test_preds = []
        for _, row in anomaly_test.iterrows():
            window = splits["test"]["metrics"][
                splits["test"]["metrics"]["window_id"] == row["window_id"]
            ]
            if len(window) > 0:
                pred = classifier.predict(window)
                test_preds.append({
                    "actual": row["label"],
                    "predicted": pred["predicted_cause"],
                    "confidence": pred["confidence"],
                })

        if test_preds:
            pred_df = pd.DataFrame(test_preds)
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(pred_df["actual"], pred_df["predicted"])
            f1 = f1_score(pred_df["actual"], pred_df["predicted"], average="weighted")
            results["test_accuracy"] = float(acc)
            results["test_f1_weighted"] = float(f1)
            print(f"[root_cause] TEST — Accuracy: {acc:.3f}, F1: {f1:.3f}")

    classifier.save()
    elapsed = time.time() - t0

    return {
        "model": "root_cause_classification",
        "training_time_s": round(elapsed, 1),
        **results,
    }


def train_log_clusterer(logs_df):
    """Train log clustering model."""
    from models.log_clustering.clusterer import LogClusterer

    clusterer = LogClusterer()
    t0 = time.time()
    results = clusterer.train(logs_df)
    clusterer.save()
    elapsed = time.time() - t0

    return {
        "model": "log_clustering",
        "training_time_s": round(elapsed, 1),
        **results,
    }


def run_pipeline(args):
    """Run the full training pipeline."""
    print("=" * 60)
    print("  DevOps AI Agent — ML Training Pipeline")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # 1. Data
    metrics_df, windows_df, logs_df = generate_data(force=args.regenerate_data)
    if args.data_only:
        print("[pipeline] Data generation complete. Exiting.")
        return

    # 2. Split
    splits = split_data(metrics_df, windows_df)

    # 3. Train models
    all_results = {}
    models_to_train = args.model.split(",") if args.model != "all" else [
        "anomaly", "forecasting", "root_cause", "log_clustering",
    ]

    if "anomaly" in models_to_train:
        all_results["anomaly"] = train_anomaly_detector(splits)

    if "forecasting" in models_to_train:
        all_results["forecasting"] = train_forecaster(splits)

    if "root_cause" in models_to_train:
        all_results["root_cause"] = train_root_cause_classifier(splits)

    if "log_clustering" in models_to_train:
        all_results["log_clustering"] = train_log_clusterer(logs_df)

    # 4. Save results
    results_path = ARTIFACT_DIR / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[pipeline] Results saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    for name, res in all_results.items():
        elapsed = res.get("training_time_s", "?")
        print(f"  {name}: {elapsed}s")
        for k, v in res.items():
            if k.startswith("test_"):
                print(f"    {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument("--model", default="all",
                        help="Model to train: all, anomaly, forecasting, root_cause, log_clustering (comma-separated)")
    parser.add_argument("--data-only", action="store_true",
                        help="Only generate data, don't train")
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Force regenerate training data")
    args = parser.parse_args()
    run_pipeline(args)
