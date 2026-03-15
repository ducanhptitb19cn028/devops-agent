"""
Root Cause Classification Model — XGBoost

Maps anomalous metric windows to probable root causes using
gradient-boosted tree classification. Includes multi-window
feature engineering to capture temporal context and cross-service
correlation patterns.

Output: probability distribution over root cause labels with
feature importance for interpretability.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, top_k_accuracy_score,
)
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    root_cause_config as cfg, METRIC_FEATURES,
    ROOT_CAUSE_LABELS, SERVICES,
)


class FeatureEngineer:
    """
    Compute rich features from metric windows for root cause classification.

    Features include:
      - Per-metric statistics (mean, std, min, max, trend, rate-of-change)
      - Multi-scale rolling statistics (5, 15, 30 step windows)
      - Cross-metric ratios (e.g., error_rate / request_rate)
      - Derivative features (acceleration of change)
      - Service-encoded features
    """

    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or cfg.window_sizes
        self.feature_names: List[str] = []
        self.scaler = StandardScaler()
        self.fitted = False

    def compute_features(self, window_df: pd.DataFrame) -> np.ndarray:
        """Compute feature vector from a single metric window."""
        features = {}

        for col in METRIC_FEATURES:
            vals = window_df[col].values.astype(float)

            # Base statistics
            features[f"{col}_mean"] = np.nanmean(vals)
            features[f"{col}_std"] = np.nanstd(vals)
            features[f"{col}_min"] = np.nanmin(vals)
            features[f"{col}_max"] = np.nanmax(vals)
            features[f"{col}_range"] = np.nanmax(vals) - np.nanmin(vals)
            features[f"{col}_last"] = vals[-1] if len(vals) > 0 else 0
            features[f"{col}_first"] = vals[0] if len(vals) > 0 else 0

            # Trend (linear regression slope)
            if len(vals) > 2:
                features[f"{col}_trend"] = np.polyfit(np.arange(len(vals)), vals, 1)[0]
            else:
                features[f"{col}_trend"] = 0.0

            # Rate of change
            features[f"{col}_roc"] = (vals[-1] - vals[0]) / (vals[0] + 1e-8) if len(vals) > 1 else 0.0

            # Second derivative (acceleration)
            if len(vals) > 3:
                first_half_trend = np.polyfit(np.arange(len(vals) // 2), vals[:len(vals) // 2], 1)[0]
                second_half_trend = np.polyfit(np.arange(len(vals) - len(vals) // 2), vals[len(vals) // 2:], 1)[0]
                features[f"{col}_acceleration"] = second_half_trend - first_half_trend
            else:
                features[f"{col}_acceleration"] = 0.0

            # Multi-scale rolling statistics
            for ws in self.window_sizes:
                if len(vals) >= ws:
                    recent = vals[-ws:]
                    features[f"{col}_roll{ws}_mean"] = np.nanmean(recent)
                    features[f"{col}_roll{ws}_std"] = np.nanstd(recent)
                else:
                    features[f"{col}_roll{ws}_mean"] = np.nanmean(vals)
                    features[f"{col}_roll{ws}_std"] = np.nanstd(vals)

            # Coefficient of variation
            features[f"{col}_cv"] = (np.nanstd(vals) / (np.nanmean(vals) + 1e-8))

        # Cross-metric ratios
        err = features.get("error_rate_mean", 0)
        req = features.get("request_rate_mean", 1)
        features["error_per_request"] = err / (req + 1e-8)
        features["latency_ratio_p99_p50"] = (
            features.get("latency_p99_mean", 0) / (features.get("latency_p50_mean", 1) + 1e-8)
        )
        features["heap_utilisation"] = (
            features.get("jvm_heap_used_mean", 0) / (512e6)  # normalise by typical max
        )
        features["cpu_memory_product"] = (
            features.get("cpu_usage_mean", 0) * features.get("memory_usage_mean", 0)
        )
        features["gc_impact"] = (
            features.get("jvm_gc_pause_seconds_mean", 0) * features.get("request_rate_mean", 0)
        )

        return features

    def transform_dataset(self, windows_df: pd.DataFrame, raw_metrics: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute features for all windows in a dataset.

        If raw_metrics is provided, computes features from raw time-series.
        Otherwise uses pre-computed window statistics from windows_df.
        """
        if raw_metrics is not None:
            records = []
            for wid in windows_df["window_id"].unique():
                window = raw_metrics[raw_metrics["window_id"] == wid]
                feats = self.compute_features(window)
                feats["window_id"] = wid
                feats["label"] = windows_df[windows_df["window_id"] == wid]["label"].iloc[0]
                records.append(feats)
            feature_df = pd.DataFrame(records)
        else:
            # Use pre-computed stats already in windows_df
            feature_cols = [c for c in windows_df.columns
                           if c not in ["window_id", "label", "is_anomaly", "service"]]
            feature_df = windows_df.copy()

        self.feature_names = [c for c in feature_df.columns
                              if c not in ["window_id", "label", "is_anomaly", "service"]]
        return feature_df

    def fit_scaler(self, X: np.ndarray):
        self.scaler.fit(X)
        self.fitted = True

    def scale(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(X)


class RootCauseClassifier:
    """
    XGBoost-based root cause classifier with interpretability.
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_engineer = FeatureEngineer()
        self.feature_importance: Optional[Dict] = None

    def train(
        self,
        windows_df: pd.DataFrame,
        raw_metrics: pd.DataFrame = None,
        use_cv: bool = True,
    ) -> Dict:
        """
        Train root cause classifier.

        Args:
            windows_df: Windowed feature DataFrame with 'label' column
            raw_metrics: Optional raw time-series for richer feature engineering
            use_cv: Whether to run stratified k-fold cross-validation
        """
        print("\n[root_cause] Training XGBoost Root Cause Classifier...")

        # Feature engineering
        feature_df = self.feature_engineer.transform_dataset(windows_df, raw_metrics)

        # Filter to anomaly windows only (skip 'normal' label)
        anomaly_df = feature_df[feature_df["label"] != "normal"].copy()
        if len(anomaly_df) < 10:
            print("[root_cause] WARNING: Very few anomaly samples, including normal class")
            anomaly_df = feature_df.copy()

        # Encode labels — fit only on labels present in training data so XGBoost
        # receives contiguous integers [0, 1, ..., n_classes-1]
        self.label_encoder.fit(sorted(anomaly_df["label"].unique()))
        y = self.label_encoder.transform(anomaly_df["label"].values)
        X = anomaly_df[self.feature_engineer.feature_names].fillna(0).values

        # Scale features
        self.feature_engineer.fit_scaler(X)
        X_scaled = self.feature_engineer.scale(X)

        # Handle class imbalance with SMOTE
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples = class_counts.min()

        if min_samples >= 2 and len(unique_classes) > 1:
            try:
                k_neighbors = min(5, min_samples - 1)
                smote = SMOTE(random_state=cfg.random_state, k_neighbors=max(1, k_neighbors))
                X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
                print(f"[root_cause] SMOTE: {len(X_scaled)} → {len(X_resampled)} samples")
            except Exception as e:
                print(f"[root_cause] SMOTE failed ({e}), using original data")
                X_resampled, y_resampled = X_scaled, y
        else:
            X_resampled, y_resampled = X_scaled, y

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            eval_metric="mlogloss",
            tree_method="hist",      # fast for GPU too
            n_jobs=-1,
        )
        self.model.fit(X_resampled, y_resampled)

        # Training metrics
        train_preds = self.model.predict(X_scaled)
        train_acc = accuracy_score(y, train_preds)
        train_f1 = f1_score(y, train_preds, average="weighted")
        print(f"[root_cause] Train — Accuracy: {train_acc:.3f}, F1: {train_f1:.3f}")

        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance = dict(sorted(
            zip(self.feature_engineer.feature_names, importances),
            key=lambda x: x[1], reverse=True,
        ))

        print(f"[root_cause] Top 10 features:")
        for feat, imp in list(self.feature_importance.items())[:10]:
            print(f"  {feat}: {imp:.4f}")

        results = {
            "train_accuracy": float(train_acc),
            "train_f1_weighted": float(train_f1),
            "n_classes": len(unique_classes),
            "n_samples": len(X_scaled),
            "n_samples_resampled": len(X_resampled),
        }

        # Cross-validation
        if use_cv and len(unique_classes) > 1 and min_samples >= 3:
            cv_results = self._cross_validate(X_scaled, y)
            results.update(cv_results)

        print(f"\n[root_cause] Classification Report:\n"
              f"{classification_report(y, train_preds, target_names=self.label_encoder.inverse_transform(unique_classes))}")

        return results

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
        """Run stratified k-fold cross-validation."""
        min_class_count = pd.Series(y).value_counts().min()
        actual_folds = min(n_folds, min_class_count)
        if actual_folds < 2:
            return {"cv_note": "Not enough samples per class for CV"}

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=cfg.random_state)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            model = xgb.XGBClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                random_state=cfg.random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            f1 = f1_score(y[val_idx], preds, average="weighted")
            fold_scores.append(f1)

        print(f"[root_cause] {actual_folds}-fold CV — F1: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
        return {
            "cv_f1_mean": float(np.mean(fold_scores)),
            "cv_f1_std": float(np.std(fold_scores)),
            "cv_folds": actual_folds,
        }

    def predict(self, metrics_window: pd.DataFrame) -> Dict:
        """
        Predict root cause for a single anomalous metric window.

        Returns:
            {
                "predicted_cause": str,
                "confidence": float,
                "top_causes": [{cause, probability}],
                "contributing_features": [{feature, importance, value}],
            }
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        feats = self.feature_engineer.compute_features(metrics_window)
        feat_values = np.array([feats.get(f, 0) for f in self.feature_engineer.feature_names])
        X = self.feature_engineer.scale(feat_values.reshape(1, -1))

        # Predict probabilities
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        # Top causes
        sorted_indices = np.argsort(probs)[::-1]
        top_causes = []
        for idx in sorted_indices[:5]:
            label = self.label_encoder.inverse_transform([idx])[0]
            top_causes.append({
                "cause": label,
                "probability": round(float(probs[idx]), 4),
            })

        # Contributing features (top features with their actual values)
        contributing = []
        if self.feature_importance:
            for feat, imp in list(self.feature_importance.items())[:10]:
                if feat in self.feature_engineer.feature_names:
                    fi = self.feature_engineer.feature_names.index(feat)
                    contributing.append({
                        "feature": feat,
                        "importance": round(float(imp), 4),
                        "value": round(float(feat_values[fi]), 4),
                    })

        return {
            "predicted_cause": pred_label,
            "confidence": round(float(probs[pred_idx]), 4),
            "top_causes": top_causes,
            "contributing_features": contributing,
        }

    def save(self, path: Path = None):
        path = path or cfg.model_path
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "xgboost_model.json"))
        joblib.dump(self.label_encoder, path / "label_encoder.joblib")
        joblib.dump(self.feature_engineer, path / "feature_engineer.joblib")
        with open(path / "feature_importance.json", "w") as f:
            json.dump({k: float(v) for k, v in self.feature_importance.items()}, f, indent=2)
        print(f"[root_cause] Model saved to {path}")

    def load(self, path: Path = None):
        path = path or cfg.model_path
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path / "xgboost_model.json"))
        self.label_encoder = joblib.load(path / "label_encoder.joblib")
        self.feature_engineer = joblib.load(path / "feature_engineer.joblib")
        if (path / "feature_importance.json").exists():
            with open(path / "feature_importance.json") as f:
                self.feature_importance = json.load(f)
        print(f"[root_cause] Model loaded from {path}")
