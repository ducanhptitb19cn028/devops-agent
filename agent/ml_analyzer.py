"""
Async ML Analyzer — Drop-in async replacement for ClaudeAnalyzer.

Matches the same interface (init, close, analyze) so the agent
can switch between Claude API and ML models with a single env var.

Usage in agent.py:
    ANALYZER_MODE = os.getenv("ANALYZER_MODE", "claude")  # "claude" or "ml"

    if ANALYZER_MODE == "ml":
        analyzer = MLAsyncAnalyzer()
    else:
        analyzer = ClaudeAnalyzer()
"""

import os
import json
import logging
import aiohttp
from typing import Optional

logger = logging.getLogger("ai-agent.ml")

ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://devops-ml-server.devops-agent.svc.cluster.local:8001")
ML_TIMEOUT = int(os.getenv("ML_TIMEOUT", "30"))


class MLAsyncAnalyzer:
    """
    Async ML model server client.
    Same interface as ClaudeAnalyzer: init(), close(), analyze(context_str).
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.total_analyses = 0
        self.total_errors = 0

    async def init(self):
        self.session = aiohttp.ClientSession()
        logger.info(f"ML Analyzer initialised — server: {ML_SERVER_URL}")

        # Health check
        try:
            async with self.session.get(
                f"{ML_SERVER_URL}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    health = await resp.json()
                    logger.info(f"ML server healthy — models: {health.get('models_loaded', {})}, device: {health.get('device')}")
                else:
                    logger.warning(f"ML server health check returned {resp.status}")
        except Exception as e:
            logger.warning(f"ML server not reachable at startup: {e} (will retry on analysis)")

    async def close(self):
        if self.session:
            await self.session.close()

    async def analyze(self, context: str) -> dict:
        """
        Analyse telemetry context using ML model server.

        Accepts the same context string that ClaudeAnalyzer receives,
        parses it to extract structured data, and sends to /analyse.
        """
        if not self.session:
            return self._fallback("ML analyzer not initialised")

        # Parse context string into structured payload
        payload = self._parse_context_to_payload(context)

        try:
            async with self.session.post(
                f"{ML_SERVER_URL}/analyse",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=ML_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"ML server {resp.status}: {err[:300]}")
                    return self._fallback(f"ML server error {resp.status}")

                result = await resp.json()
                self.total_analyses += 1

                # Extract ML metadata for logging
                ml_meta = result.pop("_ml_metadata", {})
                logger.info(
                    f"ML analysis #{self.total_analyses} — "
                    f"latency: {ml_meta.get('total_latency_ms', '?')}ms, "
                    f"models: {ml_meta.get('models_used', [])}, "
                    f"status: {result.get('health_status', '?')}"
                )

                return result

        except aiohttp.ClientError as e:
            self.total_errors += 1
            logger.error(f"ML server connection error: {e}")
            return self._fallback(str(e))
        except Exception as e:
            self.total_errors += 1
            logger.error(f"ML analysis failed: {e}")
            return self._fallback(str(e))

    def _parse_context_to_payload(self, context: str) -> dict:
        """
        Parse the text context string (built by build_context()) into
        structured data for the ML model server.

        This bridges the gap between the agent's text-based context
        and the ML server's JSON payload format.
        """
        payload = {"stats": {}, "metrics": None, "logs": None}

        # Extract stats from the context header
        stats = {}
        for line in context.split("\n"):
            line = line.strip()
            if line.startswith("Total logs:"):
                stats["total_logs"] = self._extract_int(line)
            elif line.startswith("Error logs:"):
                stats["total_errors"] = self._extract_int(line)
            elif line.startswith("Total traces:"):
                stats["total_traces"] = self._extract_int(line)
            elif line.startswith("Slow traces"):
                stats["slow_traces"] = self._extract_int(line)
            elif line.startswith("Error traces:"):
                stats["error_traces"] = self._extract_int(line)
            elif line.startswith("K8s warning events:"):
                stats["k8s_events"] = self._extract_int(line)
        payload["stats"] = stats

        # Extract error log messages
        error_messages = []
        in_errors = False
        for line in context.split("\n"):
            if "ERROR LOGS" in line:
                in_errors = True
                continue
            if in_errors:
                if line.startswith("---") or line.startswith("==="):
                    in_errors = False
                    continue
                if line.strip():
                    # Extract message part after service name
                    parts = line.split(":", 2)
                    msg = parts[-1].strip() if len(parts) > 1 else line.strip()
                    if msg:
                        error_messages.append(msg)

        # Add warning messages too
        in_warns = False
        for line in context.split("\n"):
            if "WARNING LOGS" in line:
                in_warns = True
                continue
            if in_warns:
                if line.startswith("---") or line.startswith("==="):
                    in_warns = False
                    continue
                if line.strip():
                    parts = line.split(":", 2)
                    msg = parts[-1].strip() if len(parts) > 1 else line.strip()
                    if msg:
                        error_messages.append(msg)

        if error_messages:
            payload["logs"] = {"messages": error_messages[:200]}

        return payload

    @staticmethod
    def _extract_int(line: str) -> int:
        """Extract first integer from a line."""
        for part in line.split():
            try:
                return int(part.replace(",", ""))
            except ValueError:
                continue
        return 0

    @staticmethod
    def _fallback(error: str) -> dict:
        return {
            "health_status": "UNKNOWN",
            "confidence": 0.0,
            "summary": f"ML analysis failed: {error}",
            "anomalies": [],
            "root_causes": [],
            "performance": {},
            "recommendations": [{
                "action": "Check ML model server connectivity",
                "reason": f"Error: {error}",
                "priority": "immediate",
            }],
            "incident_timeline": [],
        }
