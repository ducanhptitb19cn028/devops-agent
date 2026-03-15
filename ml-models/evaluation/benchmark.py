"""
Evaluation & Benchmarking Framework

Compares the hybrid ML pipeline against the Claude API baseline:
  1. Per-model evaluation metrics
  2. End-to-end analysis quality comparison
  3. Latency and cost analysis
  4. Statistical significance testing

Designed for PhD research — generates publication-ready metrics tables.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, silhouette_score,
    classification_report,
)
from scipy import stats as scipy_stats
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARTIFACT_DIR


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    model_name: str
    metric_name: str
    value: float
    std: float = 0.0
    n_samples: int = 0
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ModelEvaluator:
    """Evaluate individual ML models with standard metrics."""

    @staticmethod
    def evaluate_anomaly_detector(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
    ) -> Dict:
        """Full anomaly detection evaluation."""
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.0

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "false_positive_rate": float(fp / (fp + tn + 1e-8)),
            "miss_rate": float(fn / (fn + tp + 1e-8)),
        }

    @staticmethod
    def evaluate_forecaster(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict:
        """Forecasting evaluation with per-feature breakdown."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAPE (avoiding division by zero)
        mask = np.abs(y_true) > 1e-8
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0

        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true, axis=0) > 0
            pred_direction = np.diff(y_pred, axis=0) > 0
            directional_acc = np.mean(true_direction == pred_direction)
        else:
            directional_acc = 0.0

        result = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "directional_accuracy": float(directional_acc),
        }

        # Per-feature if available
        if feature_names and y_true.ndim == 2:
            for i, feat in enumerate(feature_names):
                result[f"mae_{feat}"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))

        return result

    @staticmethod
    def evaluate_classifier(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        label_names: List[str] = None,
    ) -> Dict:
        """Root cause classification evaluation."""
        acc = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_macro = f1_score(y_true, y_pred, average="macro")

        result = {
            "accuracy": float(acc),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
        }

        # Top-k accuracy
        if y_proba is not None and y_proba.ndim == 2:
            for k in [3, 5]:
                if k <= y_proba.shape[1]:
                    top_k = np.mean([
                        y_true[i] in np.argsort(y_proba[i])[-k:]
                        for i in range(len(y_true))
                    ])
                    result[f"top_{k}_accuracy"] = float(top_k)

        # Full classification report
        if label_names:
            report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
            result["per_class"] = report

        return result

    @staticmethod
    def evaluate_clustering(
        embeddings: np.ndarray,
        pred_labels: np.ndarray,
        true_labels: np.ndarray = None,
    ) -> Dict:
        """Log clustering evaluation."""
        non_noise = pred_labels != -1
        result = {
            "n_clusters": len(set(pred_labels)) - (1 if -1 in pred_labels else 0),
            "noise_ratio": float((~non_noise).mean()),
        }

        if non_noise.sum() > 1 and result["n_clusters"] > 1:
            result["silhouette_score"] = float(
                silhouette_score(embeddings[non_noise], pred_labels[non_noise])
            )

        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            if non_noise.sum() > 0:
                result["adjusted_rand_index"] = float(
                    adjusted_rand_score(true_labels[non_noise], pred_labels[non_noise])
                )
                result["normalised_mutual_info"] = float(
                    normalized_mutual_info_score(true_labels[non_noise], pred_labels[non_noise])
                )

        return result


class LatencyBenchmark:
    """Measure inference latency for individual and end-to-end pipelines."""

    @staticmethod
    def measure(fn, *args, n_runs: int = 50, warmup: int = 5, **kwargs) -> Dict:
        """Run a function multiple times and collect latency stats."""
        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            "mean_ms": float(latencies.mean()),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "std_ms": float(latencies.std()),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "n_runs": n_runs,
        }


class CostAnalyser:
    """Compare operational costs: ML inference vs API calls."""

    def __init__(self):
        # Approximate costs (update with current pricing)
        self.claude_sonnet_input_per_1m = 3.00   # USD per 1M input tokens
        self.claude_sonnet_output_per_1m = 15.00  # USD per 1M output tokens
        self.gpu_cost_per_hour = 0.50             # RTX 3060/3070 electricity + amortisation

    def estimate_api_cost(
        self,
        avg_input_tokens: int = 2000,
        avg_output_tokens: int = 800,
        calls_per_day: int = 288,  # every 5 minutes
    ) -> Dict:
        """Estimate Claude API cost per month."""
        daily_input = avg_input_tokens * calls_per_day
        daily_output = avg_output_tokens * calls_per_day
        monthly_input = daily_input * 30
        monthly_output = daily_output * 30

        input_cost = (monthly_input / 1e6) * self.claude_sonnet_input_per_1m
        output_cost = (monthly_output / 1e6) * self.claude_sonnet_output_per_1m

        return {
            "monthly_input_tokens": monthly_input,
            "monthly_output_tokens": monthly_output,
            "monthly_input_cost_usd": round(input_cost, 2),
            "monthly_output_cost_usd": round(output_cost, 2),
            "monthly_total_cost_usd": round(input_cost + output_cost, 2),
            "daily_calls": calls_per_day,
        }

    def estimate_ml_cost(
        self,
        inference_time_ms: float = 200,
        calls_per_day: int = 288,
        nlp_inference_time_ms: float = 3000,
    ) -> Dict:
        """Estimate local ML inference cost per month."""
        # Total GPU time per day
        ml_time_per_call = (inference_time_ms + nlp_inference_time_ms) / 1000  # seconds
        daily_gpu_seconds = ml_time_per_call * calls_per_day
        daily_gpu_hours = daily_gpu_seconds / 3600
        monthly_gpu_hours = daily_gpu_hours * 30
        monthly_cost = monthly_gpu_hours * self.gpu_cost_per_hour

        return {
            "inference_time_per_call_ms": inference_time_ms + nlp_inference_time_ms,
            "monthly_gpu_hours": round(monthly_gpu_hours, 2),
            "monthly_cost_usd": round(monthly_cost, 2),
            "daily_calls": calls_per_day,
            "gpu_utilisation_pct": round((daily_gpu_hours / 24) * 100, 2),
        }

    def compare(self, **kwargs) -> Dict:
        """Full cost comparison."""
        import inspect
        api = self.estimate_api_cost(**kwargs)
        ml_params = inspect.signature(self.estimate_ml_cost).parameters
        ml = self.estimate_ml_cost(**{k: v for k, v in kwargs.items() if k in ml_params})
        savings = api["monthly_total_cost_usd"] - ml["monthly_cost_usd"]
        savings_pct = (savings / api["monthly_total_cost_usd"]) * 100 if api["monthly_total_cost_usd"] > 0 else 0

        return {
            "api_cost": api,
            "ml_cost": ml,
            "monthly_savings_usd": round(savings, 2),
            "savings_percentage": round(savings_pct, 1),
        }


class StatisticalTests:
    """Statistical significance testing for model comparison."""

    @staticmethod
    def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
        """Paired t-test for comparing two models on same data."""
        t_stat, p_value = scipy_stats.ttest_rel(scores_a, scores_b)
        return {
            "test": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "effect_direction": "A > B" if np.mean(scores_a) > np.mean(scores_b) else "B > A",
        }

    @staticmethod
    def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
        """Wilcoxon signed-rank test (non-parametric alternative)."""
        try:
            stat, p_value = scipy_stats.wilcoxon(scores_a, scores_b)
        except ValueError:
            return {"test": "wilcoxon", "error": "Identical samples"}

        return {
            "test": "wilcoxon_signed_rank",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
        }

    @staticmethod
    def bootstrap_ci(
        scores: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
    ) -> Dict:
        """Bootstrap confidence interval for a metric."""
        boot_means = np.array([
            np.mean(np.random.choice(scores, size=len(scores), replace=True))
            for _ in range(n_bootstrap)
        ])
        alpha = 1 - confidence
        lower = np.percentile(boot_means, alpha / 2 * 100)
        upper = np.percentile(boot_means, (1 - alpha / 2) * 100)

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
        }


def generate_evaluation_report(results: Dict, output_path: Path = None) -> str:
    """Generate a formatted evaluation report."""
    lines = [
        "=" * 70,
        "  DevOps AI Agent — ML vs API Evaluation Report",
        "=" * 70,
        "",
    ]

    for model_name, metrics in results.items():
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  {model_name.upper()}")
        lines.append(f"{'─' * 50}")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k:30s}: {v:.4f}")
                elif isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for kk, vv in v.items():
                        if isinstance(vv, float):
                            lines.append(f"    {kk:28s}: {vv:.4f}")
                        else:
                            lines.append(f"    {kk:28s}: {vv}")
                else:
                    lines.append(f"  {k:30s}: {v}")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[eval] Report saved to {output_path}")

    return report
