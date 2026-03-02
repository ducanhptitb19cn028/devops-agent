"""
Time-Series Forecasting Model — LSTM with Attention

Predicts future values of system metrics to enable proactive alerting.
Uses a multi-head self-attention layer on top of LSTM hidden states
for better long-range dependency capture.

Outputs:
  - Point forecasts for each metric over the forecast horizon
  - Prediction intervals (uncertainty quantification via MC Dropout)
  - Capacity breach probabilities (when will thresholds be crossed)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import forecast_config as cfg, METRIC_FEATURES, DEVICE


# ═══════════════════════════════════════════════════════════════
# 1. Architecture
# ═══════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    """Multi-head self-attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class LSTMForecaster(nn.Module):
    """
    LSTM + Attention forecasting model.

    Input:  (batch, seq_length, input_dim) — historical metric window
    Output: (batch, forecast_horizon, input_dim) — future predictions
    """

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        self.lstm = nn.LSTM(
            cfg.hidden_dim, cfg.hidden_dim, cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )

        self.attention = TemporalAttention(cfg.hidden_dim, cfg.attention_heads)
        self.dropout = nn.Dropout(cfg.dropout)

        self.fc_horizon = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.forecast_horizon * cfg.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Project input features to hidden dim
        x = self.input_proj(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Attention over all time-steps
        attn_out = self.attention(lstm_out)

        # Use last attended hidden state for forecasting
        last_hidden = self.dropout(attn_out[:, -1, :])

        # Project to forecast horizon × features
        forecast = self.fc_horizon(last_hidden)
        forecast = forecast.view(batch_size, cfg.forecast_horizon, cfg.input_dim)
        return forecast


# ═══════════════════════════════════════════════════════════════
# 2. Dataset
# ═══════════════════════════════════════════════════════════════

class ForecastDataset(Dataset):
    """Sliding window dataset: input window → future target."""

    def __init__(self, data: np.ndarray, seq_len: int, horizon: int):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.horizon = horizon
        self.valid_indices = len(data) - seq_len - horizon + 1

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.horizon]
        return x, y


# ═══════════════════════════════════════════════════════════════
# 3. Trainer & Predictor
# ═══════════════════════════════════════════════════════════════

class MetricForecaster:
    """Trains and serves the LSTM forecasting model."""

    def __init__(self):
        self.model: Optional[LSTMForecaster] = None
        self.scaler = StandardScaler()
        self.history: List[Dict] = []

        # Capacity thresholds for proactive alerting
        self.thresholds = {
            "cpu_usage": 0.85,
            "memory_usage": 0.90,
            "error_rate": 0.05,
            "latency_p99": 500,  # ms
            "jvm_heap_used": 450e6,  # bytes
        }

    def train(
        self,
        train_metrics: pd.DataFrame,
        val_metrics: pd.DataFrame = None,
    ) -> Dict:
        """
        Train the forecasting model on continuous time-series data.

        Args:
            train_metrics: DataFrame with METRIC_FEATURES columns, sorted by time
            val_metrics: Optional validation DataFrame
        """
        print("\n[forecast] Training LSTM Forecaster with Attention...")

        # Scale features
        train_values = train_metrics[METRIC_FEATURES].values
        self.scaler.fit(train_values)
        train_scaled = self.scaler.transform(train_values)

        train_dataset = ForecastDataset(train_scaled, cfg.seq_length, cfg.forecast_horizon)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

        val_loader = None
        if val_metrics is not None:
            val_scaled = self.scaler.transform(val_metrics[METRIC_FEATURES].values)
            val_dataset = ForecastDataset(val_scaled, cfg.seq_length, cfg.forecast_horizon)
            val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

        self.model = LSTMForecaster().to(DEVICE)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=cfg.scheduler_patience, factor=cfg.scheduler_factor,
        )
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)

            # Validation
            val_loss = None
            if val_loader:
                self.model.eval()
                val_total = 0.0
                val_n = 0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                        pred_val = self.model(x_val)
                        val_total += criterion(pred_val, y_val).item()
                        val_n += 1
                val_loss = val_total / max(val_n, 1)
                scheduler.step(val_loss)
            else:
                scheduler.step(avg_train)

            self.history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss})

            if epoch % 10 == 0:
                vl = f", val: {val_loss:.6f}" if val_loss else ""
                lr = optimizer.param_groups[0]["lr"]
                print(f"[forecast] Epoch {epoch:3d} — train: {avg_train:.6f}{vl}, lr: {lr:.2e}")

            # Early stopping on validation loss
            check_loss = val_loss if val_loss is not None else avg_train
            if check_loss < best_val_loss:
                best_val_loss = check_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"[forecast] Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(DEVICE)

        # Final evaluation
        results = self._evaluate(train_loader, "train")
        if val_loader:
            val_results = self._evaluate(val_loader, "val")
            results.update(val_results)

        return results

    def _evaluate(self, loader: DataLoader, prefix: str) -> Dict:
        """Evaluate forecast accuracy."""
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                pred = self.model(x).cpu().numpy()
                all_preds.append(pred)
                all_targets.append(y.numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # Inverse transform for interpretable metrics
        n_samples = preds.shape[0] * preds.shape[1]
        preds_flat = self.scaler.inverse_transform(preds.reshape(-1, cfg.input_dim))
        targets_flat = self.scaler.inverse_transform(targets.reshape(-1, cfg.input_dim))

        mae = mean_absolute_error(targets_flat, preds_flat)
        rmse = math.sqrt(mean_squared_error(targets_flat, preds_flat))

        # Per-feature MAE
        per_feature = {}
        for i, feat in enumerate(METRIC_FEATURES):
            per_feature[feat] = float(mean_absolute_error(
                targets_flat[:, i], preds_flat[:, i],
            ))

        results = {
            f"{prefix}_mae": float(mae),
            f"{prefix}_rmse": float(rmse),
            f"{prefix}_per_feature_mae": per_feature,
        }
        print(f"[forecast] {prefix} — MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return results

    def predict(
        self,
        recent_metrics: pd.DataFrame,
        n_mc_samples: int = 20,
    ) -> Dict:
        """
        Forecast future metrics with uncertainty quantification.

        Uses MC Dropout: run multiple forward passes with dropout enabled
        to estimate prediction intervals.

        Returns:
            {
                "forecasts": {metric_name: [horizon values]},
                "confidence_lower": {metric_name: [values]},
                "confidence_upper": {metric_name: [values]},
                "breach_alerts": [{metric, step, predicted_value, threshold}],
            }
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        values = recent_metrics[METRIC_FEATURES].values[-cfg.seq_length:]
        if len(values) < cfg.seq_length:
            padding = np.tile(values[0], (cfg.seq_length - len(values), 1))
            values = np.vstack([padding, values])

        scaled = self.scaler.transform(values)
        x = torch.FloatTensor(scaled).unsqueeze(0).to(DEVICE)

        # MC Dropout — enable dropout at inference for uncertainty estimation
        self.model.train()  # keeps dropout active
        mc_preds = []
        with torch.no_grad():
            for _ in range(n_mc_samples):
                pred = self.model(x).cpu().numpy()[0]
                pred_original = self.scaler.inverse_transform(pred)
                mc_preds.append(pred_original)

        self.model.eval()
        mc_preds = np.array(mc_preds)  # (n_mc, horizon, features)

        mean_pred = mc_preds.mean(axis=0)
        std_pred = mc_preds.std(axis=0)
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred

        # Build per-feature forecasts
        forecasts = {}
        conf_lower = {}
        conf_upper = {}
        for i, feat in enumerate(METRIC_FEATURES):
            forecasts[feat] = mean_pred[:, i].tolist()
            conf_lower[feat] = lower[:, i].tolist()
            conf_upper[feat] = upper[:, i].tolist()

        # Check for threshold breaches
        breach_alerts = []
        for metric, threshold in self.thresholds.items():
            if metric in forecasts:
                for step, val in enumerate(forecasts[metric]):
                    if val > threshold:
                        breach_alerts.append({
                            "metric": metric,
                            "step": step + 1,
                            "predicted_value": round(val, 4),
                            "threshold": threshold,
                            "confidence": round(
                                float(np.mean(mc_preds[:, step, METRIC_FEATURES.index(metric)] > threshold)), 2
                            ),
                        })
                        break  # only first breach per metric

        return {
            "forecasts": forecasts,
            "confidence_lower": conf_lower,
            "confidence_upper": conf_upper,
            "breach_alerts": breach_alerts,
            "horizon_steps": cfg.forecast_horizon,
        }

    def save(self, path: Path = None):
        path = path or cfg.model_path
        path.mkdir(parents=True, exist_ok=True)
        if self.model:
            torch.save(self.model.state_dict(), path / "forecaster.pt")
        joblib.dump(self.scaler, path / "scaler.joblib")
        meta = {
            "seq_length": cfg.seq_length,
            "forecast_horizon": cfg.forecast_horizon,
            "features": METRIC_FEATURES,
            "thresholds": self.thresholds,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[forecast] Model saved to {path}")

    def load(self, path: Path = None):
        path = path or cfg.model_path
        self.model = LSTMForecaster().to(DEVICE)
        self.model.load_state_dict(torch.load(path / "forecaster.pt", map_location=DEVICE))
        self.model.eval()
        self.scaler = joblib.load(path / "scaler.joblib")
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        self.thresholds = meta.get("thresholds", self.thresholds)
        print(f"[forecast] Model loaded from {path}")
