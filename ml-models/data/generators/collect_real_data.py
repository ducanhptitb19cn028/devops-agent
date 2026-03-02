"""
Real Data Collector for Model Retraining

Pulls live telemetry data from the DevOps backend API and converts
it into the training format expected by the ML models. This enables
transitioning from synthetic data to real cluster data.

Workflow:
  1. Query backend API for metrics, logs, and traces
  2. Transform into training-format DataFrames
  3. Optionally label anomalous periods (semi-supervised)
  4. Merge with existing synthetic data or replace entirely
  5. Retrain selected models

Usage:
  python -m data.generators.collect_real_data --hours 24
  python -m data.generators.collect_real_data --hours 168 --retrain
"""

import argparse
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_DIR, METRIC_FEATURES, SERVICES, BACKEND_URL


class RealDataCollector:
    """Collects real telemetry from the backend API."""

    def __init__(self, backend_url: str = None):
        self.backend_url = backend_url or BACKEND_URL
        self.session = requests.Session()

    def _get(self, path: str, params: dict = None) -> dict:
        try:
            resp = self.session.get(f"{self.backend_url}{path}", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[collect] Error fetching {path}: {e}")
            return {}

    def collect_metrics(self, since_minutes: int = 1440, limit: int = 10000) -> pd.DataFrame:
        """Pull metric samples from the backend."""
        print(f"[collect] Fetching metrics (last {since_minutes} min)...")
        data = self._get("/api/metrics", {"since_minutes": since_minutes, "limit": limit})

        metrics = data.get("metrics", [])
        if not metrics:
            print("[collect] No metrics found")
            return pd.DataFrame()

        df = pd.DataFrame(metrics)
        print(f"[collect] Got {len(df)} metric samples")

        # Pivot into wide format (one row per timestamp per service)
        if "metric_name" in df.columns and "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            wide = df.pivot_table(
                index=["timestamp", "service"],
                columns="metric_name",
                values="value",
                aggfunc="mean",
            ).reset_index()

            # Map prometheus metric names to our features
            name_map = {
                "http_server_requests_seconds_count": "request_rate",
                "process_cpu_usage": "cpu_usage",
                "jvm_memory_used_bytes": "jvm_heap_used",
                "jvm_gc_pause_seconds_sum": "jvm_gc_pause_seconds",
            }
            wide = wide.rename(columns=name_map)

            # Fill missing features with defaults
            for feat in METRIC_FEATURES:
                if feat not in wide.columns:
                    wide[feat] = 0.0

            print(f"[collect] Pivoted to {len(wide)} rows x {len(wide.columns)} columns")
            return wide

        return df

    def collect_logs(self, since_minutes: int = 1440, limit: int = 10000) -> pd.DataFrame:
        """Pull log entries from the backend."""
        print(f"[collect] Fetching logs (last {since_minutes} min)...")
        data = self._get("/api/logs", {"since_minutes": since_minutes, "limit": limit})

        logs = data.get("logs", [])
        if not logs:
            print("[collect] No logs found")
            return pd.DataFrame()

        df = pd.DataFrame(logs)
        print(f"[collect] Got {len(df)} log entries")
        print(f"[collect] Severity distribution:\n{df['severity'].value_counts().to_string()}")
        return df

    def collect_traces(self, since_minutes: int = 1440, limit: int = 5000) -> pd.DataFrame:
        """Pull trace data from the backend."""
        print(f"[collect] Fetching traces (last {since_minutes} min)...")
        data = self._get("/api/traces", {"since_minutes": since_minutes, "limit": limit})

        traces = data.get("traces", [])
        if not traces:
            return pd.DataFrame()

        df = pd.DataFrame(traces)
        print(f"[collect] Got {len(df)} traces")
        return df

    def collect_analysis_history(self, limit: int = 100) -> List[dict]:
        """Pull past AI analysis results (useful for ground truth labelling)."""
        print(f"[collect] Fetching analysis history...")
        data = self._get("/api/analysis/latest")
        if data:
            print(f"[collect] Got analysis: status={data.get('health_status')}")
            return [data]
        return []


class DataTransformer:
    """Transform raw collected data into training format."""

    @staticmethod
    def create_metric_windows(
        metrics_df: pd.DataFrame,
        window_size: int = 60,
        stride: int = 15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create windowed time-series from continuous metric data.

        Returns:
            raw_windows: DataFrame with window_id column added
            feature_windows: Windowed feature statistics
        """
        if metrics_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        all_windows = []
        feature_records = []
        window_id = 0

        for service in metrics_df["service"].unique():
            svc_data = metrics_df[metrics_df["service"] == service].sort_values("timestamp")

            for start in range(0, len(svc_data) - window_size + 1, stride):
                window = svc_data.iloc[start:start + window_size].copy()
                wid = f"real_{service}_{window_id}"
                window["window_id"] = wid
                window["label"] = "unlabelled"  # needs manual or heuristic labelling
                all_windows.append(window)

                # Compute feature statistics
                feats = {"window_id": wid, "service": service, "label": "unlabelled"}
                for col in METRIC_FEATURES:
                    if col in window.columns:
                        vals = window[col].values.astype(float)
                        feats[f"{col}_mean"] = np.nanmean(vals)
                        feats[f"{col}_std"] = np.nanstd(vals)
                        feats[f"{col}_min"] = np.nanmin(vals)
                        feats[f"{col}_max"] = np.nanmax(vals)
                        feats[f"{col}_trend"] = np.polyfit(np.arange(len(vals)), vals, 1)[0] if len(vals) > 1 else 0
                        feats[f"{col}_roc"] = (vals[-1] - vals[0]) / (vals[0] + 1e-8) if len(vals) > 1 else 0

                feature_records.append(feats)
                window_id += 1

        raw_df = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
        feat_df = pd.DataFrame(feature_records)

        print(f"[transform] Created {window_id} windows from {len(metrics_df)} samples")
        return raw_df, feat_df

    @staticmethod
    def heuristic_labelling(
        feat_df: pd.DataFrame,
        error_rate_threshold: float = 0.05,
        cpu_threshold: float = 0.85,
        latency_threshold: float = 500,
    ) -> pd.DataFrame:
        """
        Apply heuristic labels to unlabelled windows.
        This provides weak supervision for initial retraining.
        """
        df = feat_df.copy()

        for idx, row in df.iterrows():
            label = "normal"

            # Check for anomalous patterns
            if row.get("error_rate_mean", 0) > error_rate_threshold:
                label = "config_error"
            elif row.get("cpu_usage_mean", 0) > cpu_threshold:
                label = "cpu_saturation"
            elif row.get("latency_p99_mean", 0) > latency_threshold:
                label = "downstream_timeout"
            elif row.get("jvm_heap_used_trend", 0) > 1e5:
                label = "memory_leak"
            elif row.get("jvm_gc_pause_seconds_mean", 0) > 0.1:
                label = "jvm_gc_pressure"
            elif row.get("request_rate_roc", 0) > 2.0:
                label = "request_spike"

            df.at[idx, "label"] = label

        df["is_anomaly"] = (df["label"] != "normal").astype(int)

        print(f"[label] Heuristic labelling results:")
        print(f"  {df['label'].value_counts().to_string()}")
        return df


def collect_and_save(hours: int, backend_url: str, retrain: bool = False):
    """Full collection pipeline."""
    collector = RealDataCollector(backend_url)
    transformer = DataTransformer()

    since_minutes = hours * 60

    # Collect
    metrics_df = collector.collect_metrics(since_minutes=since_minutes)
    logs_df = collector.collect_logs(since_minutes=since_minutes)

    # Transform metrics into windows
    if not metrics_df.empty:
        raw_windows, feat_windows = transformer.create_metric_windows(metrics_df)

        if not feat_windows.empty:
            # Apply heuristic labels
            feat_windows = transformer.heuristic_labelling(feat_windows)

            # Save
            out_dir = DATA_DIR / "real"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            raw_path = out_dir / f"metrics_raw_{timestamp}.parquet"
            feat_path = out_dir / f"windows_features_{timestamp}.parquet"
            raw_windows.to_parquet(raw_path, index=False)
            feat_windows.to_parquet(feat_path, index=False)
            print(f"[collect] Saved metrics to {raw_path}")
            print(f"[collect] Saved features to {feat_path}")

    if not logs_df.empty:
        out_dir = DATA_DIR / "real"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_path = out_dir / f"logs_{timestamp}.parquet"
        logs_df.to_parquet(logs_path, index=False)
        print(f"[collect] Saved logs to {logs_path}")

    # Optionally retrain
    if retrain and not metrics_df.empty:
        print("\n[collect] Starting model retraining with real data...")
        retrain_models(raw_windows, feat_windows, logs_df)


def retrain_models(metrics_df, windows_df, logs_df):
    """Retrain models with collected real data."""
    from models.anomaly.detector import AnomalyDetector
    from models.root_cause.classifier import RootCauseClassifier

    # Retrain anomaly detector
    if not metrics_df.empty:
        print("\n[retrain] Retraining anomaly detector...")
        detector = AnomalyDetector()
        try:
            detector.load()
            print("[retrain] Loaded existing model, fine-tuning...")
        except Exception:
            print("[retrain] Training from scratch...")

        detector.train_isolation_forest(windows_df)
        detector.train_lstm_autoencoder(metrics_df)
        detector.save()
        print("[retrain] Anomaly detector updated")

    # Retrain root cause classifier (only if we have labelled anomalies)
    anomaly_windows = windows_df[windows_df["label"] != "normal"]
    if len(anomaly_windows) >= 10:
        print("\n[retrain] Retraining root cause classifier...")
        classifier = RootCauseClassifier()
        classifier.train(windows_df, metrics_df, use_cv=False)
        classifier.save()
        print("[retrain] Root cause classifier updated")
    else:
        print(f"[retrain] Only {len(anomaly_windows)} anomaly windows — skipping root cause retraining")

    print("\n[retrain] Retraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect real data for retraining")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to collect")
    parser.add_argument("--backend", default=None, help="Backend URL override")
    parser.add_argument("--retrain", action="store_true", help="Retrain models after collection")
    args = parser.parse_args()

    collect_and_save(
        hours=args.hours,
        backend_url=args.backend or BACKEND_URL,
        retrain=args.retrain,
    )
