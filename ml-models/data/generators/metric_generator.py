"""
Synthetic Metric Data Generator

Generates realistic time-series metric data for TraceFlix microservices
with injected anomaly patterns for supervised training. Each sample is a
multivariate time-series window with optional anomaly labels.

Anomaly patterns:
  - Memory leak: monotonic heap increase over time
  - CPU saturation: sustained high CPU with micro-bursts
  - Latency spike: sudden P99 jump with gradual recovery
  - Error storm: correlated error rate spike across services
  - GC pressure: increased GC pause frequency and duration
  - Request flood: traffic ramp exceeding capacity
  - Deployment regression: step-change in baseline metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    METRIC_FEATURES, SERVICES, ROOT_CAUSE_LABELS,
    training_config as tc,
)


@dataclass
class MetricBaseline:
    """Normal operating baselines per service."""
    request_rate: float = 50.0          # req/s
    error_rate: float = 0.005           # 0.5%
    latency_p50: float = 15.0           # ms
    latency_p99: float = 120.0          # ms
    jvm_heap_used: float = 256e6        # bytes (256MB)
    jvm_gc_pause_seconds: float = 0.02  # 20ms
    cpu_usage: float = 0.25             # 25%
    memory_usage: float = 0.40          # 40%


# Per-service baseline variations
SERVICE_BASELINES = {
    "movie-service": MetricBaseline(
        request_rate=80, latency_p50=12, latency_p99=95, cpu_usage=0.30,
    ),
    "actor-service": MetricBaseline(
        request_rate=45, latency_p50=18, latency_p99=140, cpu_usage=0.20,
    ),
    "review-service": MetricBaseline(
        request_rate=60, latency_p50=22, latency_p99=180, cpu_usage=0.35,
        jvm_heap_used=320e6,
    ),
}


def _add_noise(series: np.ndarray, level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise proportional to signal magnitude."""
    noise = np.random.normal(0, level * np.abs(series.mean() + 1e-8), size=series.shape)
    return series + noise


def _add_seasonality(n: int, period: int = 96, amplitude: float = 0.15) -> np.ndarray:
    """Add daily seasonality pattern (period=96 for 15s intervals over 24min)."""
    t = np.arange(n)
    return amplitude * np.sin(2 * np.pi * t / period)


def generate_normal_metrics(
    n_steps: int,
    service: str = "movie-service",
    noise_level: float = 0.05,
) -> pd.DataFrame:
    """Generate normal (non-anomalous) metric time-series for a service."""

    baseline = SERVICE_BASELINES.get(service, MetricBaseline())
    seasonality = _add_seasonality(n_steps)

    data = {
        "request_rate": _add_noise(
            baseline.request_rate * (1 + 0.3 * seasonality), noise_level
        ),
        "error_rate": np.clip(
            _add_noise(np.full(n_steps, baseline.error_rate), noise_level * 0.5),
            0, 1,
        ),
        "latency_p50": np.clip(
            _add_noise(baseline.latency_p50 * (1 + 0.1 * seasonality), noise_level),
            1, None,
        ),
        "latency_p99": np.clip(
            _add_noise(baseline.latency_p99 * (1 + 0.2 * seasonality), noise_level),
            5, None,
        ),
        "jvm_heap_used": np.clip(
            _add_noise(
                baseline.jvm_heap_used * (1 + 0.08 * np.sin(2 * np.pi * np.arange(n_steps) / 200)),
                noise_level,
            ),
            0, None,
        ),
        "jvm_gc_pause_seconds": np.clip(
            _add_noise(np.full(n_steps, baseline.jvm_gc_pause_seconds), noise_level),
            0, None,
        ),
        "cpu_usage": np.clip(
            _add_noise(baseline.cpu_usage * (1 + 0.2 * seasonality), noise_level),
            0, 1,
        ),
        "memory_usage": np.clip(
            _add_noise(np.full(n_steps, baseline.memory_usage), noise_level * 0.3),
            0, 1,
        ),
    }

    df = pd.DataFrame(data)
    df["service"] = service
    df["timestamp"] = pd.date_range("2025-01-01", periods=n_steps, freq="15s")
    df["label"] = "normal"
    return df


def inject_memory_leak(df: pd.DataFrame, start: int, duration: int) -> pd.DataFrame:
    """Inject monotonic heap increase pattern."""
    end = min(start + duration, len(df))
    t = np.arange(end - start)
    leak_rate = np.random.uniform(0.5e6, 2e6)  # 0.5-2MB per step
    df.loc[start:end-1, "jvm_heap_used"] += leak_rate * t
    df.loc[start:end-1, "jvm_gc_pause_seconds"] *= (1 + 0.05 * t / duration * 10)
    df.loc[start:end-1, "memory_usage"] = np.clip(
        df.loc[start:end-1, "memory_usage"].values + 0.002 * t, 0, 0.98
    )
    df.loc[start:end-1, "label"] = "memory_leak"
    return df


def inject_cpu_saturation(df: pd.DataFrame, start: int, duration: int) -> pd.DataFrame:
    """Inject sustained high CPU with bursts."""
    end = min(start + duration, len(df))
    n = end - start
    base_high = np.random.uniform(0.85, 0.95)
    bursts = np.random.choice(n, size=max(1, n // 10), replace=False)
    cpu = np.full(n, base_high) + np.random.normal(0, 0.03, n)
    cpu[bursts] = np.clip(cpu[bursts] + 0.1, 0, 1.0)
    df.loc[start:end-1, "cpu_usage"] = np.clip(cpu, 0, 1.0)
    df.loc[start:end-1, "latency_p99"] *= np.linspace(1.0, 2.5, n)
    df.loc[start:end-1, "label"] = "cpu_saturation"
    return df


def inject_latency_spike(df: pd.DataFrame, start: int, duration: int) -> pd.DataFrame:
    """Inject sudden P99 spike with exponential recovery."""
    end = min(start + duration, len(df))
    n = end - start
    spike_magnitude = np.random.uniform(3, 10)
    recovery = spike_magnitude * np.exp(-3 * np.linspace(0, 1, n))
    df.loc[start:end-1, "latency_p99"] *= (1 + recovery)
    df.loc[start:end-1, "latency_p50"] *= (1 + recovery * 0.3)
    df.loc[start:end-1, "error_rate"] = np.clip(
        df.loc[start:end-1, "error_rate"].values * (1 + recovery * 0.5), 0, 1
    )
    df.loc[start:end-1, "label"] = "downstream_timeout"
    return df


def inject_error_storm(df: pd.DataFrame, start: int, duration: int) -> pd.DataFrame:
    """Inject sudden error rate spike."""
    end = min(start + duration, len(df))
    n = end - start
    error_level = np.random.uniform(0.1, 0.5)
    ramp = np.minimum(np.linspace(0, 1, min(10, n)), 1.0)
    full_ramp = np.ones(n)
    full_ramp[:len(ramp)] = ramp
    df.loc[start:end-1, "error_rate"] = error_level * full_ramp + np.random.normal(0, 0.02, n)
    df.loc[start:end-1, "error_rate"] = np.clip(df.loc[start:end-1, "error_rate"], 0, 1)
    df.loc[start:end-1, "label"] = "config_error"
    return df


def inject_gc_pressure(df: pd.DataFrame, start: int, duration: int) -> pd.DataFrame:
    """Inject increased GC pauses and heap oscillation."""
    end = min(start + duration, len(df))
    n = end - start
    gc_multiplier = np.random.uniform(3, 8)
    df.loc[start:end-1, "jvm_gc_pause_seconds"] *= gc_multiplier
    heap = df.loc[start:end-1, "jvm_heap_used"].values
    oscillation = heap.mean() * 0.3 * np.sin(np.linspace(0, 8 * np.pi, n))
    df.loc[start:end-1, "jvm_heap_used"] = np.clip(heap + oscillation, 0, None)
    df.loc[start:end-1, "latency_p99"] *= np.linspace(1.0, 1.8, n)
    df.loc[start:end-1, "label"] = "jvm_gc_pressure"
    return df


def inject_request_flood(df: pd.DataFrame, start: int, duration: int) -> pd.DataFrame:
    """Inject traffic ramp exceeding capacity."""
    end = min(start + duration, len(df))
    n = end - start
    ramp = np.linspace(1.0, np.random.uniform(3, 6), n)
    df.loc[start:end-1, "request_rate"] *= ramp
    df.loc[start:end-1, "cpu_usage"] = np.clip(
        df.loc[start:end-1, "cpu_usage"].values * ramp * 0.6, 0, 1
    )
    df.loc[start:end-1, "latency_p99"] *= np.sqrt(ramp)
    saturation_point = n * 2 // 3
    df.loc[start + saturation_point:end-1, "error_rate"] = np.clip(
        np.linspace(0.01, 0.15, max(1, end - start - saturation_point)), 0, 1
    )
    df.loc[start:end-1, "label"] = "request_spike"
    return df


def inject_deployment_regression(df: pd.DataFrame, start: int, duration: int) -> pd.DataFrame:
    """Inject step-change in baseline after deployment."""
    end = min(start + duration, len(df))
    degradation = np.random.uniform(1.3, 2.0)
    df.loc[start:end-1, "latency_p50"] *= degradation
    df.loc[start:end-1, "latency_p99"] *= degradation * 1.2
    df.loc[start:end-1, "cpu_usage"] = np.clip(
        df.loc[start:end-1, "cpu_usage"] * (degradation * 0.7), 0, 1
    )
    df.loc[start:end-1, "error_rate"] = np.clip(
        df.loc[start:end-1, "error_rate"] * degradation, 0, 1
    )
    df.loc[start:end-1, "label"] = "deployment_regression"
    return df


# Map labels to injection functions
ANOMALY_INJECTORS = {
    "memory_leak": inject_memory_leak,
    "cpu_saturation": inject_cpu_saturation,
    "downstream_timeout": inject_latency_spike,
    "config_error": inject_error_storm,
    "jvm_gc_pressure": inject_gc_pressure,
    "request_spike": inject_request_flood,
    "deployment_regression": inject_deployment_regression,
}


def generate_training_dataset(
    n_normal_windows: int = 3000,
    n_anomaly_windows: int = 1500,
    window_size: int = 60,
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a full training dataset with normal and anomalous windows.

    Returns:
        metrics_df: Full time-series DataFrame with labels
        windows_df: Windowed feature matrix ready for ML training
    """
    np.random.seed(seed)
    all_frames = []

    # ── Normal data ──────────────────────────────────────────
    for i in range(n_normal_windows):
        service = np.random.choice(SERVICES)
        df = generate_normal_metrics(window_size, service, noise_level)
        df["window_id"] = f"normal_{i}"
        all_frames.append(df)

    # ── Anomalous data ───────────────────────────────────────
    anomaly_types = list(ANOMALY_INJECTORS.keys())
    for i in range(n_anomaly_windows):
        service = np.random.choice(SERVICES)
        anomaly_type = np.random.choice(anomaly_types)
        # Generate longer window, inject anomaly in the middle portion
        total_steps = window_size + 30
        df = generate_normal_metrics(total_steps, service, noise_level)
        inject_start = np.random.randint(10, 20)
        inject_duration = np.random.randint(20, window_size)
        df = ANOMALY_INJECTORS[anomaly_type](df, inject_start, inject_duration)
        df = df.iloc[:window_size].reset_index(drop=True)
        df["window_id"] = f"anomaly_{anomaly_type}_{i}"
        all_frames.append(df)

    metrics_df = pd.concat(all_frames, ignore_index=True)

    # ── Build windowed feature matrix ────────────────────────
    records = []
    for wid in metrics_df["window_id"].unique():
        window = metrics_df[metrics_df["window_id"] == wid]
        feat = {}
        for col in METRIC_FEATURES:
            vals = window[col].values
            feat[f"{col}_mean"] = vals.mean()
            feat[f"{col}_std"] = vals.std()
            feat[f"{col}_min"] = vals.min()
            feat[f"{col}_max"] = vals.max()
            feat[f"{col}_last"] = vals[-1]
            feat[f"{col}_first"] = vals[0]
            feat[f"{col}_trend"] = np.polyfit(np.arange(len(vals)), vals, 1)[0]
            # Rate of change
            if len(vals) > 1:
                feat[f"{col}_roc"] = (vals[-1] - vals[0]) / (vals[0] + 1e-8)
            else:
                feat[f"{col}_roc"] = 0.0

        feat["service"] = window["service"].iloc[0]
        feat["label"] = window["label"].mode().iloc[0]
        feat["is_anomaly"] = 0 if feat["label"] == "normal" else 1
        feat["window_id"] = wid
        records.append(feat)

    windows_df = pd.DataFrame(records)

    print(f"[data] Generated {len(metrics_df)} metric samples in {metrics_df['window_id'].nunique()} windows")
    print(f"[data] Label distribution:\n{windows_df['label'].value_counts().to_string()}")

    return metrics_df, windows_df


def generate_cross_service_features(metrics_df: pd.DataFrame, window_id: str) -> Dict[str, float]:
    """
    Compute cross-service correlation features for a single window.
    These capture cascading failure patterns across services.
    """
    window = metrics_df[metrics_df["window_id"] == window_id]
    services_in_window = window["service"].unique()

    features = {}
    if len(services_in_window) < 2:
        # Single service — pad cross-service features with zeros
        for s1 in SERVICES:
            for s2 in SERVICES:
                if s1 != s2:
                    features[f"corr_error_{s1}_{s2}"] = 0.0
                    features[f"corr_latency_{s1}_{s2}"] = 0.0
        return features

    for i, s1 in enumerate(services_in_window):
        for s2 in services_in_window[i+1:]:
            d1 = window[window["service"] == s1]
            d2 = window[window["service"] == s2]
            min_len = min(len(d1), len(d2))
            if min_len > 2:
                features[f"corr_error_{s1}_{s2}"] = np.corrcoef(
                    d1["error_rate"].values[:min_len],
                    d2["error_rate"].values[:min_len],
                )[0, 1]
                features[f"corr_latency_{s1}_{s2}"] = np.corrcoef(
                    d1["latency_p99"].values[:min_len],
                    d2["latency_p99"].values[:min_len],
                )[0, 1]

    return features


if __name__ == "__main__":
    metrics_df, windows_df = generate_training_dataset(
        n_normal_windows=500,
        n_anomaly_windows=250,
        seed=42,
    )
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_parquet(out_dir / "metrics_raw.parquet", index=False)
    windows_df.to_parquet(out_dir / "windows_features.parquet", index=False)
    print(f"[data] Saved to {out_dir}")
