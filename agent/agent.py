"""
DevOps AI Agent — Intelligence Service (Step 4)
Runs as a K8s CronJob or continuous Deployment.
Workflow:
  1. Query backend API for recent logs, metrics, traces, events
  2. Build structured context from TraceFlix observability data
  3. Send to Claude API or local ML model server for analysis
  4. Store analysis results back via backend POST /api/analysis

Analyzer mode (set via ANALYZER_MODE env var):
  - "claude": Uses Claude API (requires ANTHROPIC_API_KEY)
  - "ml":     Uses local ML model server (requires ML_SERVER_URL)
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

# ── Configuration ────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://devops-backend.devops-agent.svc.cluster.local:8000")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
ANALYSIS_INTERVAL = int(os.getenv("ANALYSIS_INTERVAL", "300"))
RUN_MODE = os.getenv("RUN_MODE", "continuous")  # "continuous" or "once"
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Analyzer mode: "claude" uses Claude API, "ml" uses local ML model server
ANALYZER_MODE = os.getenv("ANALYZER_MODE", "claude")  # "claude" or "ml"
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://devops-ml-server.devops-agent.svc.cluster.local:8001")

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ai-agent")


# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert DevOps AI Agent analyzing a Kubernetes-based microservices platform called TraceFlix.

TraceFlix Architecture:
- movie-service (Spring Boot) — Orchestrator, calls actor-service + review-service via REST
- actor-service (Spring Boot) — Serves actor data, endpoint: /api/actors/{id}
- review-service (Spring Boot) — Serves reviews, endpoint: /api/reviews?movieId=X
- All services are instrumented with OpenTelemetry Java Agent
- Telemetry pipeline: OTel Collector → Tempo (traces), Prometheus (metrics), Loki (logs)
- Namespace: on-demand-observability

You receive structured telemetry data (logs, metrics, traces, K8s events) and must provide actionable analysis.

Respond ONLY with valid JSON matching this schema:
{
  "health_status": "HEALTHY|DEGRADED|CRITICAL",
  "confidence": 0.0-1.0,
  "summary": "2-3 sentence overall assessment",
  "anomalies": [
    {
      "severity": "low|medium|high|critical",
      "title": "Short description",
      "detail": "Root cause + impact explanation",
      "affected_resources": ["namespace/pod or service name"],
      "evidence": "Specific metric values, log excerpts, or trace data"
    }
  ],
  "root_causes": [
    {
      "issue": "What's going wrong",
      "probable_cause": "Why it's happening",
      "confidence": 0.0-1.0,
      "evidence": ["Supporting data points"]
    }
  ],
  "performance": {
    "movie_service": "Assessment of movie-service health + latency",
    "actor_service": "Assessment of actor-service health + latency",
    "review_service": "Assessment of review-service health + latency",
    "inter_service_calls": "Assessment of REST call chain: movie → actor + review",
    "bottlenecks": ["Identified bottlenecks"],
    "trends": ["Notable trends or patterns"]
  },
  "recommendations": [
    {
      "priority": "immediate|short_term|long_term",
      "action": "Specific action to take",
      "reason": "Why this helps",
      "command": "kubectl command or config snippet if applicable"
    }
  ],
  "incident_timeline": [
    {"time": "ISO timestamp or relative", "event": "What happened"}
  ]
}

Analysis priorities for TraceFlix:
1. Inter-service latency: movie-service makes sequential calls to actor-service (N+1 pattern) — flag if slow
2. Error cascading: If actor-service or review-service is down, movie-service will fail
3. JVM health: Memory pressure, GC pauses, thread pool exhaustion
4. K8s events: OOMKills, CrashLoopBackOffs, restart counts
5. Trace analysis: Slow spans, error propagation across services
6. Resource utilization: CPU/memory headroom before scaling is needed
7. TSDB Trends: Long-range time-series data from VictoriaMetrics — detect degradation trends, memory leaks (monotonic heap increase), latency drift, error rate acceleration, and capacity saturation. The "trend_pct" field shows percentage change between first and second half of the window; "direction" is increasing/decreasing/stable; "volatility_cv" is coefficient of variation"""


# ── Data Fetcher ─────────────────────────────────────────────────────────────
class DataFetcher:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def _get(self, path: str, params: dict = None) -> dict:
        try:
            async with self.session.get(
                f"{BACKEND_URL}{path}", params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return await resp.json() if resp.status == 200 else {}
        except Exception as e:
            logger.error(f"Fetch {path} failed: {e}")
            return {}

    async def _post(self, path: str, payload: dict) -> bool:
        try:
            async with self.session.post(
                f"{BACKEND_URL}{path}", json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"POST {path} failed: {e}")
            return False

    async def get_stats(self) -> dict:
        return await self._get("/api/stats")

    async def get_logs(self, severity: str = None) -> dict:
        params = {"since_minutes": LOOKBACK_MINUTES, "limit": 500}
        if severity:
            params["severity"] = severity
        return await self._get("/api/logs", params)

    async def get_traces(self, slow_only: bool = False, errors_only: bool = False) -> dict:
        params = {"since_minutes": LOOKBACK_MINUTES, "limit": 200}
        if slow_only:
            params["slow_only"] = "true"
        if errors_only:
            params["errors_only"] = "true"
        return await self._get("/api/traces", params)

    async def get_metrics(self) -> dict:
        return await self._get("/api/metrics", {"since_minutes": LOOKBACK_MINUTES, "limit": 300})

    async def get_events(self) -> dict:
        return await self._get("/api/events", {"since_minutes": LOOKBACK_MINUTES, "limit": 100})

    async def get_tsdb_trends(self) -> dict:
        return await self._get("/api/tsdb/trends/summary")

    async def store_analysis(self, analysis: dict) -> bool:
        return await self._post("/api/analysis", analysis)


# ── Claude API Client ────────────────────────────────────────────────────────
class ClaudeAnalyzer:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not ANTHROPIC_API_KEY:
            logger.error("ANTHROPIC_API_KEY not set!")
            return
        self.session = aiohttp.ClientSession(headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        })

    async def close(self):
        if self.session:
            await self.session.close()

    async def analyze(self, context: str) -> dict:
        if not self.session:
            return self._fallback("No API key configured")

        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": context}],
        }

        try:
            async with self.session.post(
                ANTHROPIC_API_URL, json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"Claude API {resp.status}: {err[:300]}")
                    return self._fallback(f"API error {resp.status}")

                data = await resp.json()
                raw = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
                return self._parse(raw)

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return self._fallback(str(e))

    def _parse(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            result = json.loads(cleaned.strip())
            result["_raw_response"] = raw
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            return self._fallback(f"Parse error: {e}", raw)

    @staticmethod
    def _fallback(error: str, raw: str = "") -> dict:
        return {
            "health_status": "UNKNOWN", "confidence": 0.0,
            "summary": f"Analysis failed: {error}",
            "anomalies": [], "root_causes": [],
            "performance": {}, "recommendations": [],
            "incident_timeline": [], "_raw_response": raw,
        }


# ── Context Builder ──────────────────────────────────────────────────────────
def build_context(stats: dict, error_logs: dict, all_logs: dict,
                  traces: dict, slow_traces: dict, error_traces: dict,
                  metrics: dict, events: dict, tsdb_trends: dict) -> str:
    s = []
    now = datetime.now(timezone.utc).isoformat()

    s.append(f"=== TRACEFLIX CLUSTER ANALYSIS ===")
    s.append(f"Timestamp: {now}")
    s.append(f"Window: last {LOOKBACK_MINUTES} minutes\n")

    # Stats summary
    if stats:
        s.append("--- DASHBOARD SUMMARY ---")
        s.append(f"Total logs: {stats.get('total_logs', 'N/A')}")
        s.append(f"Error logs: {stats.get('total_errors', 'N/A')}")
        s.append(f"Total traces: {stats.get('total_traces', 'N/A')}")
        s.append(f"Slow traces (>500ms): {stats.get('slow_traces', 'N/A')}")
        s.append(f"Error traces: {stats.get('error_traces', 'N/A')}")
        s.append(f"K8s warning events: {stats.get('k8s_events', 'N/A')}")
        if stats.get("errors_by_service"):
            s.append("Errors by service:")
            for svc in stats["errors_by_service"]:
                s.append(f"  {svc.get('service', '?')}: {svc.get('cnt', 0)} errors")
        if stats.get("latency_by_service"):
            s.append("Latency by service:")
            for svc in stats["latency_by_service"]:
                s.append(f"  {svc.get('service', '?')}: avg={svc.get('avg_ms', 0):.1f}ms, max={svc.get('max_ms', 0):.1f}ms")
        s.append("")

    # Error logs
    err_list = error_logs.get("logs", [])
    if err_list:
        s.append(f"--- ERROR LOGS ({len(err_list)} entries) ---")
        for log in err_list[:40]:
            s.append(f"[{log.get('timestamp', '?')}] {log.get('service', '?')}: {str(log.get('message', ''))[:300]}")
        s.append("")

    # Slow traces
    slow_list = slow_traces.get("traces", [])
    if slow_list:
        s.append(f"--- SLOW TRACES >500ms ({len(slow_list)} entries) ---")
        for t in slow_list[:20]:
            s.append(f"  {t.get('service', '?')} {t.get('operation', '?')} — {t.get('duration_ms', 0)}ms (trace: {t.get('trace_id', '?')[:16]})")
        s.append("")

    # Error traces
    err_traces = error_traces.get("traces", [])
    if err_traces:
        s.append(f"--- ERROR TRACES ({len(err_traces)} entries) ---")
        for t in err_traces[:20]:
            s.append(f"  {t.get('service', '?')} {t.get('operation', '?')} — {t.get('duration_ms', 0)}ms")
        s.append("")

    # K8s Events
    evt_list = events.get("events", [])
    k8s_evts = [e for e in evt_list if e.get("source") == "kubernetes"]
    if k8s_evts:
        s.append(f"--- K8S WARNING EVENTS ({len(k8s_evts)} entries) ---")
        for ev in k8s_evts[:20]:
            s.append(f"  {ev.get('reason', '?')}: {ev.get('pod', '?')} — {str(ev.get('message', ''))[:200]}")
        s.append("")

    # Metrics summary
    met_list = metrics.get("metrics", [])
    if met_list:
        s.append(f"--- METRICS ({len(met_list)} samples) ---")
        by_name: dict = {}
        for m in met_list[:200]:
            n = m.get("metric_name", "?")
            by_name.setdefault(n, []).append(m.get("value", ""))
        for name, values in list(by_name.items())[:15]:
            s.append(f"  {name}: {len(values)} samples, latest={values[0] if values else 'N/A'}")
        s.append("")

    # Warning logs
    warns = [l for l in all_logs.get("logs", []) if l.get("severity") == "WARN"]
    if warns:
        s.append(f"--- WARNING LOGS ({len(warns)} entries) ---")
        for log in warns[:20]:
            s.append(f"[{log.get('timestamp', '?')}] {log.get('service', '?')}: {str(log.get('message', ''))[:200]}")
        s.append("")

    # TSDB Trend Analysis (VictoriaMetrics long-range data)
    if tsdb_trends:
        degrading = tsdb_trends.get("degrading", [])
        improving = tsdb_trends.get("improving", [])
        stable = tsdb_trends.get("stable", [])

        s.append(f"--- TSDB TREND ANALYSIS (VictoriaMetrics) ---")
        s.append(f"Total monitored series: {tsdb_trends.get('total_series', 0)}")
        s.append(f"Degrading: {len(degrading)} | Stable: {len(stable)} | Improving: {len(improving)}")
        s.append("")

        if degrading:
            s.append("⚠ DEGRADING TRENDS (require attention):")
            for t in degrading[:15]:
                labels = t.get("labels", {})
                svc = labels.get("service_name", labels.get("service", "unknown"))
                s.append(f"  [{t.get('query', '?')}] {svc} ({t.get('range', '?')} window)")
                s.append(f"    Direction: {t.get('direction')} | Change: +{t.get('trend_pct', 0):.1f}%")
                s.append(f"    Latest: {t.get('latest')} | Avg: {t.get('avg')} | Volatility(CV): {t.get('volatility_cv', 0):.1f}%")
                s.append(f"    Description: {t.get('description', '')}")
            s.append("")

        if improving:
            s.append("✓ IMPROVING TRENDS:")
            for t in improving[:10]:
                labels = t.get("labels", {})
                svc = labels.get("service_name", labels.get("service", "unknown"))
                s.append(f"  [{t.get('query', '?')}] {svc}: {t.get('trend_pct', 0):.1f}% ({t.get('direction')})")
            s.append("")

    s.append("=== END OF DATA — ANALYZE NOW ===")

    context = "\n".join(s)
    logger.info(f"Context built: ~{len(context) // 4} tokens")
    return context


# ── Analysis Pipeline ────────────────────────────────────────────────────────
async def run_analysis(fetcher: DataFetcher, analyzer: ClaudeAnalyzer):
    start = datetime.now(timezone.utc)
    logger.info("Starting analysis cycle...")

    # Gather all data concurrently
    stats, error_logs, all_logs, traces, slow_traces, error_traces, metrics, events, tsdb_trends = await asyncio.gather(
        fetcher.get_stats(),
        fetcher.get_logs(severity="ERROR"),
        fetcher.get_logs(),
        fetcher.get_traces(),
        fetcher.get_traces(slow_only=True),
        fetcher.get_traces(errors_only=True),
        fetcher.get_metrics(),
        fetcher.get_events(),
        fetcher.get_tsdb_trends(),
    )

    # Check if there's data to analyze
    total = sum(len(d.get(k, [])) for d, k in [
        (all_logs, "logs"), (traces, "traces"), (metrics, "metrics"), (events, "events"),
    ])
    if total == 0:
        logger.info("No data available — skipping cycle")
        return

    # Build context & analyze
    context = build_context(stats, error_logs, all_logs, traces, slow_traces, error_traces, metrics, events, tsdb_trends)
    analysis = await analyzer.analyze(context)

    # Store result
    dur = (datetime.now(timezone.utc) - start).total_seconds()
    analysis["_analysis_duration_seconds"] = dur
    stored = await fetcher.store_analysis(analysis)

    status = analysis.get("health_status", "UNKNOWN")
    anomaly_count = len(analysis.get("anomalies", []))
    rec_count = len(analysis.get("recommendations", []))
    logger.info(f"Analysis complete in {dur:.1f}s — Status: {status}, Anomalies: {anomaly_count}, Recommendations: {rec_count}, Stored: {stored}")


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    logger.info("=" * 60)
    logger.info("DevOps AI Agent — Intelligence Service starting")
    logger.info(f"  Backend:  {BACKEND_URL}")
    logger.info(f"  Analyzer: {ANALYZER_MODE.upper()}")
    if ANALYZER_MODE == "claude":
        logger.info(f"  Model:    {CLAUDE_MODEL}")
    else:
        logger.info(f"  ML Server: {ML_SERVER_URL}")
    logger.info(f"  Mode:     {RUN_MODE}")
    logger.info(f"  Interval: {ANALYSIS_INTERVAL}s")
    logger.info(f"  Lookback: {LOOKBACK_MINUTES}min")
    logger.info("=" * 60)

    fetcher = DataFetcher()

    # Select analyzer based on mode
    if ANALYZER_MODE == "ml":
        from ml_analyzer import MLAsyncAnalyzer
        analyzer = MLAsyncAnalyzer()
        logger.info("Using ML model server for analysis")
    else:
        analyzer = ClaudeAnalyzer()
        logger.info("Using Claude API for analysis")

    await fetcher.init()
    await analyzer.init()

    try:
        if RUN_MODE == "once":
            await run_analysis(fetcher, analyzer)
        else:
            while True:
                await run_analysis(fetcher, analyzer)
                logger.info(f"Next analysis in {ANALYSIS_INTERVAL}s...")
                await asyncio.sleep(ANALYSIS_INTERVAL)
    finally:
        await fetcher.close()
        await analyzer.close()
        logger.info("Agent shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main()) 