"""
Log Clustering Model — Sentence-BERT + UMAP + HDBSCAN

Clusters log messages into operational patterns without requiring
pre-labelled data. The pipeline:
  1. Encode log messages with Sentence-BERT (all-MiniLM-L6-v2)
  2. Reduce dimensionality with UMAP
  3. Cluster with HDBSCAN (density-based, handles noise)
  4. Extract representative patterns per cluster

Enables:
  - Automatic identification of new/unknown error patterns
  - Frequency analysis of operational patterns
  - Pattern-based alerting (new cluster = new issue type)
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_completeness_v_measure,
)
from collections import Counter
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import log_cluster_config as cfg


class LogClusterer:
    """
    Unsupervised log clustering with pattern extraction.
    """

    def __init__(self):
        self.encoder: Optional[SentenceTransformer] = None
        self.umap_model: Optional[umap.UMAP] = None
        self.hdbscan_model: Optional[hdbscan.HDBSCAN] = None
        self.cluster_patterns: Dict[int, Dict] = {}
        self.cluster_map: Dict[int, str] = {}  # cluster_id → human-readable label

    def _load_encoder(self):
        """Lazy-load Sentence-BERT to save memory when not embedding."""
        if self.encoder is None:
            print(f"[log_cluster] Loading encoder: {cfg.embedding_model}...")
            self.encoder = SentenceTransformer(cfg.embedding_model)
            print(f"[log_cluster] Encoder loaded ({cfg.embedding_dim}d embeddings)")

    def encode(self, messages: List[str]) -> np.ndarray:
        """Encode log messages to dense vectors."""
        self._load_encoder()
        print(f"[log_cluster] Encoding {len(messages)} messages...")
        embeddings = self.encoder.encode(
            messages,
            batch_size=cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def train(
        self,
        logs_df: pd.DataFrame,
        message_col: str = "message",
        label_col: str = "cluster_label",
    ) -> Dict:
        """
        Train the full clustering pipeline.

        Args:
            logs_df: DataFrame with log messages
            message_col: Column name for log messages
            label_col: Column name for ground truth labels (for evaluation)
        """
        print("\n[log_cluster] Training Log Clustering Pipeline...")
        messages = logs_df[message_col].tolist()

        # 1. Encode
        embeddings = self.encode(messages)
        print(f"[log_cluster] Embeddings shape: {embeddings.shape}")

        # 2. UMAP dimensionality reduction
        print(f"[log_cluster] Fitting UMAP ({cfg.embedding_dim}d → {cfg.umap_n_components}d)...")
        self.umap_model = umap.UMAP(
            n_components=cfg.umap_n_components,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            metric=cfg.metric,
            random_state=42,
        )
        reduced = self.umap_model.fit_transform(embeddings)
        print(f"[log_cluster] UMAP reduced: {reduced.shape}")

        # 3. HDBSCAN clustering
        print(f"[log_cluster] Fitting HDBSCAN...")
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            cluster_selection_method=cfg.cluster_selection_method,
            metric=cfg.metric,
            prediction_data=True,
        )
        cluster_labels = self.hdbscan_model.fit_predict(reduced)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = (cluster_labels == -1).sum()
        print(f"[log_cluster] Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(cluster_labels)*100:.1f}%)")

        # 4. Extract patterns
        self._extract_patterns(logs_df, cluster_labels, message_col)

        # 5. Evaluate
        results = {"n_clusters": n_clusters, "n_noise_points": int(n_noise)}

        # Internal metrics
        non_noise_mask = cluster_labels != -1
        if non_noise_mask.sum() > 1 and n_clusters > 1:
            sil = silhouette_score(reduced[non_noise_mask], cluster_labels[non_noise_mask])
            results["silhouette_score"] = float(sil)
            print(f"[log_cluster] Silhouette score: {sil:.3f}")

        # External metrics (if ground truth available)
        if label_col in logs_df.columns:
            true_labels = logs_df[label_col].values
            if non_noise_mask.sum() > 0:
                ari = adjusted_rand_score(true_labels[non_noise_mask], cluster_labels[non_noise_mask])
                nmi = normalized_mutual_info_score(true_labels[non_noise_mask], cluster_labels[non_noise_mask])
                h, c, v = homogeneity_completeness_v_measure(true_labels[non_noise_mask], cluster_labels[non_noise_mask])
                results.update({
                    "adjusted_rand_index": float(ari),
                    "normalised_mutual_info": float(nmi),
                    "homogeneity": float(h),
                    "completeness": float(c),
                    "v_measure": float(v),
                })
                print(f"[log_cluster] ARI: {ari:.3f}, NMI: {nmi:.3f}, V-measure: {v:.3f}")

        return results

    def _extract_patterns(
        self,
        logs_df: pd.DataFrame,
        cluster_labels: np.ndarray,
        message_col: str,
    ):
        """Extract representative patterns and metadata for each cluster."""
        self.cluster_patterns = {}
        logs_df = logs_df.copy()
        logs_df["_cluster"] = cluster_labels

        for cid in sorted(set(cluster_labels)):
            if cid == -1:
                continue

            cluster_logs = logs_df[logs_df["_cluster"] == cid]
            messages = cluster_logs[message_col].tolist()

            # Find most representative message (closest to centroid)
            # Use frequency analysis for pattern extraction
            severity_dist = cluster_logs["severity"].value_counts().to_dict() if "severity" in cluster_logs.columns else {}
            service_dist = cluster_logs["service"].value_counts().to_dict() if "service" in cluster_logs.columns else {}

            # Find common tokens for pattern description
            all_tokens = " ".join(messages).split()
            common_tokens = [t for t, c in Counter(all_tokens).most_common(10) if c > len(messages) * 0.3]

            self.cluster_patterns[int(cid)] = {
                "size": len(messages),
                "representative": messages[0] if messages else "",
                "pattern_keywords": common_tokens[:8],
                "severity_distribution": severity_dist,
                "service_distribution": service_dist,
                "examples": messages[:3],
            }

            # Auto-label based on severity and keywords
            if severity_dist:
                top_severity = max(severity_dist, key=severity_dist.get)
                keywords = " ".join(common_tokens[:3])
                self.cluster_map[int(cid)] = f"{top_severity}: {keywords}"

        logs_df.drop("_cluster", axis=1, inplace=True)
        print(f"[log_cluster] Extracted patterns for {len(self.cluster_patterns)} clusters")

    def predict(self, messages: List[str]) -> List[Dict]:
        """
        Assign new log messages to existing clusters.

        Returns list of:
            {
                "message": str,
                "cluster_id": int (-1 = noise/new pattern),
                "cluster_label": str,
                "is_new_pattern": bool,
                "confidence": float,
            }
        """
        if self.umap_model is None or self.hdbscan_model is None:
            return [{"error": "Model not loaded"}]

        embeddings = self.encode(messages)
        reduced = self.umap_model.transform(embeddings)
        labels, strengths = hdbscan.approximate_predict(self.hdbscan_model, reduced)

        results = []
        for i, (msg, cid, strength) in enumerate(zip(messages, labels, strengths)):
            is_new = cid == -1
            results.append({
                "message": msg,
                "cluster_id": int(cid),
                "cluster_label": self.cluster_map.get(int(cid), "unknown/new_pattern"),
                "is_new_pattern": is_new,
                "confidence": round(float(strength), 4),
                "pattern_info": self.cluster_patterns.get(int(cid), {}),
            })

        return results

    def get_pattern_summary(self) -> Dict:
        """Get summary of all discovered patterns."""
        summary = {
            "total_patterns": len(self.cluster_patterns),
            "patterns": [],
        }
        for cid, info in sorted(self.cluster_patterns.items()):
            summary["patterns"].append({
                "cluster_id": cid,
                "label": self.cluster_map.get(cid, "unknown"),
                "size": info["size"],
                "keywords": info["pattern_keywords"],
                "severities": info["severity_distribution"],
                "representative": info["representative"][:100],
            })
        return summary

    def save(self, path: Path = None):
        path = path or cfg.model_path
        path.mkdir(parents=True, exist_ok=True)
        if self.umap_model:
            joblib.dump(self.umap_model, path / "umap_model.joblib")
        if self.hdbscan_model:
            joblib.dump(self.hdbscan_model, path / "hdbscan_model.joblib")
        with open(path / "cluster_patterns.json", "w") as f:
            json.dump(self.cluster_patterns, f, indent=2, default=str)
        with open(path / "cluster_map.json", "w") as f:
            json.dump(self.cluster_map, f, indent=2)
        meta = {
            "embedding_model": cfg.embedding_model,
            "embedding_dim": cfg.embedding_dim,
            "n_clusters": len(self.cluster_patterns),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[log_cluster] Models saved to {path}")

    def load(self, path: Path = None):
        path = path or cfg.model_path
        self._load_encoder()
        self.umap_model = joblib.load(path / "umap_model.joblib")
        self.hdbscan_model = joblib.load(path / "hdbscan_model.joblib")
        with open(path / "cluster_patterns.json") as f:
            self.cluster_patterns = {int(k): v for k, v in json.load(f).items()}
        with open(path / "cluster_map.json") as f:
            self.cluster_map = {int(k): v for k, v in json.load(f).items()}
        print(f"[log_cluster] Models loaded from {path}")
