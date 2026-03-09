"""
Anomaly Detection Model — Isolation Forest + LSTM Autoencoder Ensemble

Two complementary approaches:
  1. Isolation Forest: Fast, interpretable, good at point anomalies
  2. LSTM Autoencoder: Captures temporal patterns, good at contextual anomalies

The ensemble combines both scores with configurable weights for a unified
anomaly probability per time-window.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix,
)
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import anomaly_config as cfg, METRIC_FEATURES, DEVICE


# ═══════════════════════════════════════════════════════════════
# 1. LSTM Autoencoder Architecture
# ═══════════════════════════════════════════════════════════════

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        latent = self.fc_latent(h_n[-1])
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers, seq_length, dropout):
        super().__init__()
        self.seq_length = seq_length
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        expanded = self.fc_expand(latent).unsqueeze(1).repeat(1, self.seq_length, 1)
        decoded, _ = self.lstm(expanded)
        output = self.fc_out(decoded)
        return output


class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder(
            cfg.lstm_input_dim, cfg.lstm_hidden_dim,
            cfg.lstm_latent_dim, cfg.lstm_num_layers, cfg.lstm_dropout,
        )
        self.decoder = LSTMDecoder(
            cfg.lstm_latent_dim, cfg.lstm_hidden_dim,
            cfg.lstm_input_dim, cfg.lstm_num_layers,
            cfg.lstm_seq_length, cfg.lstm_dropout,
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=(1, 2))
        return mse


# ═══════════════════════════════════════════════════════════════
# 2. Dataset
# ═══════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ═══════════════════════════════════════════════════════════════
# 3. Trainer
# ═══════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Ensemble anomaly detector combining Isolation Forest and LSTM Autoencoder.
    """

    def __init__(self):
        self.isolation_forest: Optional[IsolationForest] = None
        self.lstm_ae: Optional[LSTMAutoencoder] = None
        self.scaler = StandardScaler()
        self.reconstruction_threshold: float = 0.0
        self.metrics_history: list = []

    def _prepare_sequences(self, df: pd.DataFrame, window_size: int = None) -> np.ndarray:
        """Convert windowed feature DataFrame into 3D sequences for LSTM."""
        if window_size is None:
            window_size = cfg.lstm_seq_length

        features = df[METRIC_FEATURES].values
        features_scaled = self.scaler.transform(features)

        sequences = []
        for i in range(len(features_scaled) - window_size + 1):
            sequences.append(features_scaled[i:i + window_size])

        return np.array(sequences)

    def _prepare_flat_features(self, windows_df: pd.DataFrame) -> np.ndarray:
        """Prepare flat feature vectors for Isolation Forest from windowed stats."""
        feature_cols = [c for c in windows_df.columns if any(
            c.endswith(s) for s in ["_mean", "_std", "_min", "_max", "_trend", "_roc"]
        )]
        return windows_df[feature_cols].fillna(0).values

    def train_isolation_forest(self, windows_df: pd.DataFrame) -> Dict:
        """Train Isolation Forest on windowed feature statistics."""
        print("\n[IF] Training Isolation Forest...")
        X = self._prepare_flat_features(windows_df)

        self.isolation_forest = IsolationForest(
            n_estimators=cfg.if_n_estimators,
            contamination=cfg.if_contamination,
            max_samples=cfg.if_max_samples,
            random_state=cfg.if_random_state,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X)

        # Evaluate on training data
        scores = self.isolation_forest.decision_function(X)
        preds = self.isolation_forest.predict(X)  # 1 = normal, -1 = anomaly
        preds_binary = (preds == -1).astype(int)

        y_true = windows_df["is_anomaly"].values
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds_binary, average="binary")

        results = {"precision": p, "recall": r, "f1": f1}
        print(f"[IF] Train metrics — P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}")
        return results

    def train_lstm_autoencoder(
        self,
        train_metrics: pd.DataFrame,
        val_metrics: pd.DataFrame = None,
    ) -> Dict:
        """Train LSTM Autoencoder on normal data only (unsupervised)."""
        print("\n[LSTM-AE] Training LSTM Autoencoder...")

        # Fit scaler on ALL training data
        self.scaler.fit(train_metrics[METRIC_FEATURES].values)

        # Filter to normal data only for training
        normal_data = train_metrics[train_metrics["label"] == "normal"]
        sequences = self._prepare_sequences(normal_data)
        print(f"[LSTM-AE] Training sequences: {sequences.shape}")

        dataset = TimeSeriesDataset(sequences)
        loader = DataLoader(dataset, batch_size=cfg.lstm_batch_size, shuffle=True)

        self.lstm_ae = LSTMAutoencoder().to(DEVICE)
        optimizer = torch.optim.Adam(self.lstm_ae.parameters(), lr=cfg.lstm_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=7, factor=0.5,
        )
        criterion = nn.MSELoss()

        # Validation sequences (include anomalies for threshold calibration)
        val_sequences = None
        if val_metrics is not None:
            val_sequences = self._prepare_sequences(val_metrics)

        best_loss = float("inf")
        patience_counter = 0
        history = []

        for epoch in range(cfg.lstm_epochs):
            self.lstm_ae.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                reconstructed = self.lstm_ae(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_ae.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)

            # Validation
            val_loss = None
            if val_sequences is not None:
                self.lstm_ae.eval()
                val_tensor = torch.FloatTensor(val_sequences).to(DEVICE)
                with torch.no_grad():
                    recon = self.lstm_ae(val_tensor)
                    val_loss = criterion(recon, val_tensor).item()

            history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss})

            if epoch % 10 == 0:
                vl = f", val_loss: {val_loss:.6f}" if val_loss else ""
                print(f"[LSTM-AE] Epoch {epoch:3d} — loss: {avg_loss:.6f}{vl}")

            # Early stopping
            check_loss = val_loss if val_loss is not None else avg_loss
            if check_loss < best_loss:
                best_loss = check_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.lstm_patience:
                    print(f"[LSTM-AE] Early stopping at epoch {epoch}")
                    break

        # ── Compute reconstruction threshold ─────────────────
        self.lstm_ae.eval()
        all_sequences = self._prepare_sequences(train_metrics)
        all_labels = []

        for i in range(len(all_sequences)):
            # The label for a sequence is the majority label of its time-steps
            start = i
            end = i + cfg.lstm_seq_length
            if end <= len(train_metrics):
                window_labels = train_metrics.iloc[start:end]["label"].values
                is_anom = (window_labels != "normal").sum() > cfg.lstm_seq_length * 0.3
                all_labels.append(int(is_anom))

        all_tensor = torch.FloatTensor(all_sequences).to(DEVICE)
        errors = self.lstm_ae.get_reconstruction_error(all_tensor).cpu().numpy()

        # Set threshold at configured percentile of NORMAL reconstruction errors
        all_labels = np.array(all_labels[:len(errors)])
        normal_errors = errors[all_labels == 0]
        self.reconstruction_threshold = np.percentile(
            normal_errors, cfg.lstm_threshold_percentile
        )
        print(f"[LSTM-AE] Reconstruction threshold: {self.reconstruction_threshold:.6f}")

        # Evaluate
        preds = (errors > self.reconstruction_threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary")
        try:
            auc = roc_auc_score(all_labels, errors)
        except ValueError:
            auc = 0.0

        results = {"precision": p, "recall": r, "f1": f1, "auc": auc,
                    "threshold": self.reconstruction_threshold}
        print(f"[LSTM-AE] Metrics — P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        return results

    def predict(self, metrics_window: pd.DataFrame) -> Dict:
        """
        Run ensemble prediction on a metric window.

        Returns:
            {
                "is_anomaly": bool,
                "anomaly_score": float (0-1),
                "if_score": float,
                "lstm_score": float,
                "details": str,
            }
        """
        result = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "if_score": 0.0,
            "lstm_score": 0.0,
            "details": "",
        }

        # ── Isolation Forest score ───────────────────────────
        if self.isolation_forest is not None:
            flat_features = self._compute_window_stats(metrics_window)
            if_raw = self.isolation_forest.decision_function(flat_features.reshape(1, -1))[0]
            # Normalize: more negative = more anomalous → invert and clip to [0, 1]
            result["if_score"] = float(np.clip(1 - (if_raw + 0.5), 0, 1))

        # ── LSTM Autoencoder score ───────────────────────────
        if self.lstm_ae is not None and len(metrics_window) >= cfg.lstm_seq_length:
            seq = self._prepare_sequences(metrics_window)
            if len(seq) > 0:
                seq_tensor = torch.FloatTensor(seq[-1:]).to(DEVICE)
                self.lstm_ae.eval()
                error = self.lstm_ae.get_reconstruction_error(seq_tensor).item()
                # Normalize relative to threshold
                result["lstm_score"] = float(np.clip(error / (self.reconstruction_threshold * 2), 0, 1))

        # ── Ensemble ─────────────────────────────────────────
        w = cfg.ensemble_weights
        result["anomaly_score"] = (
            w["isolation_forest"] * result["if_score"] +
            w["lstm_autoencoder"] * result["lstm_score"]
        )
        result["is_anomaly"] = result["anomaly_score"] > 0.5

        if result["is_anomaly"]:
            dominant = "temporal pattern (LSTM)" if result["lstm_score"] > result["if_score"] else "statistical outlier (IF)"
            result["details"] = f"Anomaly detected (score: {result['anomaly_score']:.2f}), dominant signal: {dominant}"
        else:
            result["details"] = f"Normal (score: {result['anomaly_score']:.2f})"

        return result

    def _compute_window_stats(self, window_df: pd.DataFrame) -> np.ndarray:
        """Compute flat statistics from a raw metric window."""
        stats = []
        for col in METRIC_FEATURES:
            vals = window_df[col].values
            stats.extend([
                vals.mean(), vals.std(), vals.min(), vals.max(),
                np.polyfit(np.arange(len(vals)), vals, 1)[0],
                (vals[-1] - vals[0]) / (vals[0] + 1e-8),
            ])
        return np.array(stats)

    def save(self, path: Path = None):
        """Save all model artifacts."""
        path = path or cfg.model_path
        path.mkdir(parents=True, exist_ok=True)

        if self.isolation_forest:
            joblib.dump(self.isolation_forest, path / "isolation_forest.joblib")
        if self.lstm_ae:
            torch.save(self.lstm_ae.state_dict(), path / "lstm_autoencoder.pt")
        joblib.dump(self.scaler, path / "scaler.joblib")
        meta = {
            "reconstruction_threshold": float(self.reconstruction_threshold),
            "ensemble_weights": cfg.ensemble_weights,
            "seq_length": cfg.lstm_seq_length,
            "features": METRIC_FEATURES,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[anomaly] Models saved to {path}")

    def load(self, path: Path = None):
        """Load saved model artifacts."""
        path = path or cfg.model_path

        if (path / "isolation_forest.joblib").exists():
            self.isolation_forest = joblib.load(path / "isolation_forest.joblib")
        if (path / "lstm_autoencoder.pt").exists():
            self.lstm_ae = LSTMAutoencoder().to(DEVICE)
            self.lstm_ae.load_state_dict(torch.load(path / "lstm_autoencoder.pt", map_location=DEVICE))
            self.lstm_ae.eval()
        if (path / "scaler.joblib").exists():
            self.scaler = joblib.load(path / "scaler.joblib")
        if (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                meta = json.load(f)
            self.reconstruction_threshold = meta["reconstruction_threshold"]
        print(f"[anomaly] Models loaded from {path}")


if __name__ == "__main__":
    # Quick standalone training test
    sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "generators"))
    from metric_generator import generate_training_dataset

    metrics_df, windows_df = generate_training_dataset(
        n_normal_windows=500, n_anomaly_windows=250, seed=42,
    )

    detector = AnomalyDetector()
    detector.train_isolation_forest(windows_df)
    detector.train_lstm_autoencoder(metrics_df)
    detector.save()
