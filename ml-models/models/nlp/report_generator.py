"""
NLP Report Generator — Phi-3-mini 4-bit Quantised

Takes structured outputs from the 4 specialised ML models and
synthesises them into a natural language analysis report matching
the JSON schema that the dashboard and agent expect.

This replaces the Claude API call with a local LLM that:
  1. Receives structured context (anomaly scores, forecasts, root causes, log patterns)
  2. Generates a coherent health assessment
  3. Outputs the same JSON format the dashboard consumes

VRAM usage: ~3.5GB with 4-bit NF4 quantisation
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typing import Dict, Optional
import json
import re
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import nlp_config as cfg, DEVICE


# ── System prompt for structured DevOps analysis ─────────────
SYSTEM_PROMPT = """You are an expert DevOps AI analyst for a Kubernetes microservices cluster called TraceFlix.
You receive structured telemetry analysis from ML models and must synthesise it into a coherent operations report.

You MUST respond with ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
  "health_status": "HEALTHY|DEGRADED|CRITICAL",
  "confidence": 0.0-1.0,
  "summary": "1-2 sentence overview",
  "anomalies": [
    {
      "title": "short title",
      "severity": "critical|high|medium|low",
      "detail": "explanation",
      "affected_resources": ["service-name"],
      "evidence": "data points supporting this"
    }
  ],
  "root_causes": [
    {
      "issue": "the problem",
      "probable_cause": "explanation",
      "confidence": 0.0-1.0
    }
  ],
  "recommendations": [
    {
      "action": "what to do",
      "reason": "why",
      "priority": "immediate|short_term|long_term",
      "command": "optional kubectl/shell command"
    }
  ],
  "incident_timeline": [
    {"time": "relative time", "event": "what happened"}
  ]
}

Rules:
- health_status is CRITICAL if any anomaly is critical severity
- health_status is DEGRADED if any anomaly is high/medium severity
- health_status is HEALTHY only if no significant anomalies
- Be specific: reference actual metric values, services, and thresholds
- Recommendations should be actionable with real kubectl commands where possible
- Keep summary concise but informative
"""


def build_ml_context(
    anomaly_results: Dict = None,
    forecast_results: Dict = None,
    root_cause_results: Dict = None,
    log_cluster_results: Dict = None,
    stats: Dict = None,
) -> str:
    """Build the context string from all ML model outputs."""
    sections = []

    if stats:
        sections.append(f"## Cluster Statistics\n{json.dumps(stats, indent=2)}")

    if anomaly_results:
        sections.append(f"## Anomaly Detection Results\n{json.dumps(anomaly_results, indent=2)}")

    if forecast_results:
        # Trim forecasts to just alerts and summary
        forecast_summary = {
            "breach_alerts": forecast_results.get("breach_alerts", []),
            "horizon_steps": forecast_results.get("horizon_steps", 0),
        }
        # Add just the last predicted value per metric
        if "forecasts" in forecast_results:
            forecast_summary["predicted_end_values"] = {
                k: round(v[-1], 4) if v else None
                for k, v in forecast_results["forecasts"].items()
            }
        sections.append(f"## Forecasting Results\n{json.dumps(forecast_summary, indent=2)}")

    if root_cause_results:
        sections.append(f"## Root Cause Analysis\n{json.dumps(root_cause_results, indent=2)}")

    if log_cluster_results:
        # Summarise log patterns
        log_summary = {
            "total_patterns": log_cluster_results.get("total_patterns", 0),
            "new_patterns_detected": sum(
                1 for r in log_cluster_results.get("recent_predictions", [])
                if r.get("is_new_pattern", False)
            ),
            "error_patterns": [
                p for p in log_cluster_results.get("patterns", [])
                if "ERROR" in str(p.get("severities", {}))
            ][:5],
        }
        sections.append(f"## Log Pattern Analysis\n{json.dumps(log_summary, indent=2)}")

    return "\n\n".join(sections)


class ReportGenerator:
    """
    Local LLM-based report generator using quantised Phi-3-mini.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        """Load the quantised model. Call once, reuse for inference."""
        if self.loaded:
            return

        print(f"[nlp] Loading {cfg.model_name} (4-bit quantised)...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.loaded = True

        # Report VRAM
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"[nlp] Model loaded. VRAM usage: {vram_used:.1f} GB")
        else:
            print("[nlp] Model loaded on CPU (slow inference expected)")

    def generate_report(
        self,
        anomaly_results: Dict = None,
        forecast_results: Dict = None,
        root_cause_results: Dict = None,
        log_cluster_results: Dict = None,
        stats: Dict = None,
    ) -> Dict:
        """
        Generate a full analysis report from ML model outputs.

        Returns the same JSON schema the dashboard expects.
        """
        if not self.loaded:
            self.load()

        # Build context from ML outputs
        context = build_ml_context(
            anomaly_results, forecast_results,
            root_cause_results, log_cluster_results, stats,
        )

        user_prompt = (
            f"Analyse the following telemetry data from TraceFlix Kubernetes cluster "
            f"and produce a JSON operations report:\n\n{context}"
        )

        # Format for Phi-3 instruct
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                do_sample=cfg.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse JSON from response
        report = self._parse_json_response(response_text)

        if report is None:
            # Fallback: construct report from ML outputs directly
            print("[nlp] WARNING: LLM output wasn't valid JSON, using rule-based fallback")
            report = self._fallback_report(
                anomaly_results, forecast_results,
                root_cause_results, log_cluster_results, stats,
            )

        return report

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Extract and parse JSON from model response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _fallback_report(
        self,
        anomaly_results: Dict = None,
        forecast_results: Dict = None,
        root_cause_results: Dict = None,
        log_cluster_results: Dict = None,
        stats: Dict = None,
    ) -> Dict:
        """
        Rule-based fallback when LLM fails to produce valid JSON.
        Constructs a report directly from ML model outputs.
        """
        report = {
            "health_status": "HEALTHY",
            "confidence": 0.8,
            "summary": "System analysis completed using ML models.",
            "anomalies": [],
            "root_causes": [],
            "recommendations": [],
            "incident_timeline": [],
        }

        # Anomaly → anomalies list
        if anomaly_results and anomaly_results.get("is_anomaly"):
            score = anomaly_results.get("anomaly_score", 0)
            severity = "critical" if score > 0.8 else "high" if score > 0.6 else "medium"
            report["health_status"] = "CRITICAL" if severity == "critical" else "DEGRADED"
            report["anomalies"].append({
                "title": "Metric anomaly detected",
                "severity": severity,
                "detail": anomaly_results.get("details", "Anomalous pattern detected in system metrics"),
                "affected_resources": [],
                "evidence": f"Anomaly score: {score:.2f} (IF: {anomaly_results.get('if_score', 0):.2f}, LSTM: {anomaly_results.get('lstm_score', 0):.2f})",
            })

        # Forecasts → breach alerts as recommendations
        if forecast_results:
            for alert in forecast_results.get("breach_alerts", []):
                report["recommendations"].append({
                    "action": f"Proactive: {alert['metric']} predicted to breach threshold in {alert['step']} steps",
                    "reason": f"Predicted value: {alert['predicted_value']}, threshold: {alert['threshold']}, confidence: {alert['confidence']}",
                    "priority": "immediate" if alert["confidence"] > 0.8 else "short_term",
                    "command": "",
                })
                if report["health_status"] == "HEALTHY":
                    report["health_status"] = "DEGRADED"

        # Root cause → root_causes list
        if root_cause_results and root_cause_results.get("predicted_cause") != "normal":
            rc = root_cause_results
            report["root_causes"].append({
                "issue": rc["predicted_cause"].replace("_", " ").title(),
                "probable_cause": f"ML classifier identified {rc['predicted_cause']} with {rc['confidence']:.0%} confidence",
                "confidence": rc["confidence"],
            })
            if rc.get("top_causes"):
                for tc in rc["top_causes"][1:3]:
                    if tc["probability"] > 0.1:
                        report["root_causes"].append({
                            "issue": tc["cause"].replace("_", " ").title(),
                            "probable_cause": f"Alternative root cause ({tc['probability']:.0%})",
                            "confidence": tc["probability"],
                        })

        # Summary
        n_anomalies = len(report["anomalies"])
        n_alerts = len(forecast_results.get("breach_alerts", [])) if forecast_results else 0
        if n_anomalies > 0 or n_alerts > 0:
            report["summary"] = (
                f"Detected {n_anomalies} anomalies and {n_alerts} forecast alerts. "
                f"System status: {report['health_status']}."
            )
        else:
            report["summary"] = "All metrics within normal parameters. No anomalies or forecast alerts."

        return report

    def unload(self):
        """Free GPU memory."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[nlp] Model unloaded, GPU memory freed")
