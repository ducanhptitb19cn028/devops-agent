"""
ML Agent Integration — Drop-in Replacement for ClaudeAnalyzer

Replaces the Claude API call in the DevOps agent with calls to
the local ML model server. Maintains the same interface so the
agent, backend, and dashboard work without changes.

Usage in agent.py:
    # Before (Claude API):
    # analyzer = ClaudeAnalyzer()
    # result = analyzer.analyze(context)

    # After (ML models):
    from ml_agent_integration import MLAnalyzer
    analyzer = MLAnalyzer()
    result = analyzer.analyze(context)
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://localhost:8001")


@dataclass
class MLAnalyzerConfig:
    """Configuration for the ML analyzer."""
    ml_server_url: str = ML_SERVER_URL
    timeout_seconds: int = 30
    retry_attempts: int = 2
    fallback_to_rules: bool = True  # use rule-based fallback if server is down


class MLAnalyzer:
    """
    Drop-in replacement for ClaudeAnalyzer.

    Sends collected telemetry to the ML model server's /analyse endpoint
    and returns the same JSON schema that the dashboard and agent expect.
    """

    def __init__(self, config: MLAnalyzerConfig = None):
        self.config = config or MLAnalyzerConfig()
        self.last_analysis: Optional[Dict] = None
        self.total_analyses = 0
        self.total_latency_ms = 0

    def analyze(self, context: Dict) -> Dict:
        """
        Analyse telemetry context using the ML model server.

        Args:
            context: The same context dict built by the agent's ContextBuilder,
                     containing metrics, logs, events, traces, stats, tsdb_trends.

        Returns:
            Analysis dict matching the dashboard JSON schema:
            {health_status, confidence, summary, anomalies, root_causes,
             recommendations, incident_timeline}
        """
        t0 = time.time()

        # Build request payload from agent context
        payload = self._build_payload(context)

        # Call ML model server
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = requests.post(
                    f"{self.config.ml_server_url}/analyse",
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                result = response.json()

                # Track metrics
                self.total_analyses += 1
                latency = (time.time() - t0) * 1000
                self.total_latency_ms += latency
                self.last_analysis = result

                ml_meta = result.pop("_ml_metadata", {})
                logger.info(
                    f"ML analysis complete in {latency:.0f}ms "
                    f"(server: {ml_meta.get('total_latency_ms', '?')}ms, "
                    f"models: {ml_meta.get('models_used', [])})"
                )

                return result

            except requests.exceptions.ConnectionError:
                logger.warning(f"ML server connection failed (attempt {attempt + 1})")
                if attempt < self.config.retry_attempts:
                    time.sleep(1)
            except requests.exceptions.Timeout:
                logger.warning(f"ML server timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"ML analysis error: {e}")
                break

        # Fallback to rule-based analysis
        if self.config.fallback_to_rules:
            logger.warning("Falling back to rule-based analysis")
            return self._rule_based_fallback(context)

        return {
            "health_status": "UNKNOWN",
            "confidence": 0.0,
            "summary": "ML model server unavailable",
            "anomalies": [],
            "root_causes": [],
            "recommendations": [{"action": "Check ML model server", "reason": "Server unreachable", "priority": "immediate"}],
            "incident_timeline": [],
        }

    def _build_payload(self, context: Dict) -> Dict:
        """Transform agent context into ML server request format."""
        payload = {"stats": context.get("stats", {})}

        # Metrics → MetricWindow format
        metrics = context.get("metrics", [])
        if metrics:
            metric_window = self._metrics_to_window(metrics)
            if metric_window:
                payload["metrics"] = metric_window

        # Logs → LogBatch format
        logs = context.get("error_logs", []) + context.get("logs", [])
        if logs:
            payload["logs"] = {
                "messages": [l.get("message", "") for l in logs[:200] if l.get("message")],
            }

        return payload

    def _metrics_to_window(self, metrics: List[Dict]) -> Optional[Dict]:
        """Convert agent metric list to MetricWindow format."""
        if not metrics:
            return None

        # Group metrics by name and build time-series
        from collections import defaultdict
        series = defaultdict(list)
        timestamps = []

        for m in metrics:
            ts = m.get("timestamp", "")
            name = m.get("name", "")
            value = m.get("value", 0)

            # Map Prometheus metric names to our standard features
            feature_map = {
                "http_server_requests_seconds_count": "request_rate",
                "http_server_requests_seconds_sum": "latency_p50",
                "request_rate": "request_rate",
                "error_rate": "error_rate",
                "latency_p50": "latency_p50",
                "latency_p99": "latency_p99",
                "jvm_memory_used_bytes": "jvm_heap_used",
                "jvm_gc_pause_seconds_sum": "jvm_gc_pause_seconds",
                "process_cpu_usage": "cpu_usage",
                "jvm_memory_usage_after_gc": "memory_usage",
            }

            mapped = feature_map.get(name, name)
            if mapped in ["request_rate", "error_rate", "latency_p50", "latency_p99",
                          "jvm_heap_used", "jvm_gc_pause_seconds", "cpu_usage", "memory_usage"]:
                series[mapped].append(float(value))
                if ts and len(timestamps) < len(series[mapped]):
                    timestamps.append(ts)

        if not series:
            return None

        # Pad all series to same length
        max_len = max(len(v) for v in series.values())
        for key in series:
            while len(series[key]) < max_len:
                series[key].append(series[key][-1] if series[key] else 0.0)

        service = "unknown"
        if metrics and "labels" in metrics[0]:
            service = metrics[0]["labels"].get("service_name", "unknown")

        return {
            "timestamps": timestamps[:max_len],
            "service": service,
            "metrics": dict(series),
        }

    def _rule_based_fallback(self, context: Dict) -> Dict:
        """Simple rule-based analysis when ML server is unavailable."""
        stats = context.get("stats", {})
        health = "HEALTHY"
        anomalies = []
        recommendations = []

        # Check error rate
        error_count = stats.get("total_errors", 0)
        total_logs = stats.get("total_logs", 1)
        error_rate = error_count / max(total_logs, 1)

        if error_rate > 0.1:
            health = "CRITICAL"
            anomalies.append({
                "title": "High error rate",
                "severity": "critical",
                "detail": f"Error rate at {error_rate:.1%} ({error_count} errors)",
                "affected_resources": [],
                "evidence": f"{error_count} errors out of {total_logs} logs",
            })
        elif error_rate > 0.02:
            health = "DEGRADED"
            anomalies.append({
                "title": "Elevated error rate",
                "severity": "medium",
                "detail": f"Error rate at {error_rate:.1%}",
                "affected_resources": [],
                "evidence": f"{error_count} errors out of {total_logs} logs",
            })

        # Check slow traces
        slow = stats.get("slow_traces", 0)
        if slow > 10:
            if health == "HEALTHY":
                health = "DEGRADED"
            anomalies.append({
                "title": "Slow traces detected",
                "severity": "high" if slow > 20 else "medium",
                "detail": f"{slow} slow traces in the last window",
                "affected_resources": [],
                "evidence": f"{slow} traces exceeded latency threshold",
            })

        return {
            "health_status": health,
            "confidence": 0.6,
            "summary": f"Rule-based analysis: {len(anomalies)} issues detected (ML server offline)",
            "anomalies": anomalies,
            "root_causes": [],
            "recommendations": recommendations,
            "incident_timeline": [],
        }

    def get_stats(self) -> Dict:
        """Return analyzer performance statistics."""
        avg_latency = self.total_latency_ms / max(self.total_analyses, 1)
        return {
            "total_analyses": self.total_analyses,
            "avg_latency_ms": round(avg_latency, 2),
            "ml_server_url": self.config.ml_server_url,
        }
