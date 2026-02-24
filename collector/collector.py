"""
DevOps AI Agent — Real-Time Collector (Step 1)
Integrates with the existing TraceFlix observability stack:
  - Prometheus (metrics) at prometheus:9090
  - Loki (logs) at loki:3100
  - Tempo (traces) at tempo:3200
  - K8s API (pod events, health)
Publishes all telemetry to Redis Streams for downstream processing.
"""

import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
import redis.asyncio as aioredis

# ── Configuration ────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis.devops-agent.svc.cluster.local")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus.on-demand-observability.svc.cluster.local:9090")
LOKI_URL = os.getenv("LOKI_URL", "http://loki.on-demand-observability.svc.cluster.local:3100")
TEMPO_URL = os.getenv("TEMPO_URL", "http://tempo.on-demand-observability.svc.cluster.local:3200")
VICTORIA_METRICS_URL = os.getenv("VICTORIA_METRICS_URL", "http://victoriametrics.devops-agent.svc.cluster.local:8428")
K8S_API = os.getenv("K8S_API", "https://kubernetes.default.svc")
K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
K8S_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

TARGET_NAMESPACE = os.getenv("TARGET_NAMESPACE", "on-demand-observability")
TRACEFLIX_SERVICES = ["movie-service", "actor-service", "review-service"]

STREAM_METRICS = "stream:metrics"
STREAM_LOGS = "stream:logs"
STREAM_TRACES = "stream:traces"
STREAM_EVENTS = "stream:events"
STREAM_TSDB_TRENDS = "stream:tsdb_trends"

METRICS_INTERVAL = int(os.getenv("METRICS_INTERVAL", "15"))
LOGS_INTERVAL = int(os.getenv("LOGS_INTERVAL", "10"))
TRACES_INTERVAL = int(os.getenv("TRACES_INTERVAL", "20"))
EVENTS_INTERVAL = int(os.getenv("EVENTS_INTERVAL", "10"))
TSDB_TRENDS_INTERVAL = int(os.getenv("TSDB_TRENDS_INTERVAL", "60"))
STREAM_MAXLEN = int(os.getenv("STREAM_MAXLEN", "50000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("collector")

shutdown_event = asyncio.Event()

def _handle_signal(sig, _):
    logger.info(f"Signal {sig} received, shutting down...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Redis Publisher ──────────────────────────────────────────────────────────
class RedisPublisher:
    def __init__(self):
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        self.client = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            password=REDIS_PASSWORD or None,
            decode_responses=True, socket_connect_timeout=10, retry_on_timeout=True,
        )
        await self.client.ping()
        logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")

    async def publish(self, stream: str, data: dict):
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "data": json.dumps(data, default=str)}
        await self.client.xadd(stream, entry, maxlen=STREAM_MAXLEN)

    async def close(self):
        if self.client:
            await self.client.aclose()


# ── Prometheus Collector ─────────────────────────────────────────────────────
class PrometheusCollector:
    """Queries Prometheus for OTel-exported metrics from TraceFlix services."""

    QUERIES = {
        "http_request_rate": 'sum(rate(http_server_request_duration_seconds_count{job="otel-collector"}[2m])) by (service_name)',
        "http_latency_p50": 'histogram_quantile(0.50, sum(rate(http_server_request_duration_seconds_bucket{job="otel-collector"}[2m])) by (le, service_name))',
        "http_latency_p99": 'histogram_quantile(0.99, sum(rate(http_server_request_duration_seconds_bucket{job="otel-collector"}[2m])) by (le, service_name))',
        "http_error_rate": 'sum(rate(http_server_request_duration_seconds_count{job="otel-collector",http_response_status_code=~"5.."}[2m])) by (service_name)',
        "jvm_memory_used": 'jvm_memory_used_bytes{job="otel-collector"}',
        "jvm_gc_duration": 'rate(jvm_gc_duration_seconds_sum{job="otel-collector"}[2m])',
        "active_threads": 'jvm_threads_live_threads{job="otel-collector"}',
    }

    def __init__(self, session: aiohttp.ClientSession, publisher: RedisPublisher):
        self.session = session
        self.publisher = publisher

    async def run(self):
        logger.info(f"Prometheus collector started (every {METRICS_INTERVAL}s)")
        while not shutdown_event.is_set():
            try:
                await self._collect()
            except Exception as e:
                logger.error(f"Prometheus error: {e}")
            await asyncio.sleep(METRICS_INTERVAL)

    async def _collect(self):
        collected = {}
        for name, query in self.QUERIES.items():
            try:
                async with self.session.get(
                    f"{PROMETHEUS_URL}/api/v1/query", params={"query": query},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        collected[name] = [
                            {"labels": r.get("metric", {}), "value": r.get("value", [None, None])[1]}
                            for r in data.get("data", {}).get("result", [])
                        ]
            except asyncio.TimeoutError:
                logger.warning(f"Prometheus query '{name}' timed out")

        if collected:
            await self.publisher.publish(STREAM_METRICS, {
                "source": "prometheus", "namespace": TARGET_NAMESPACE, "metrics": collected,
            })
            logger.debug(f"Published {sum(len(v) for v in collected.values())} metric data points")


# ── Loki Log Collector ───────────────────────────────────────────────────────
class LokiCollector:
    """Queries Loki for logs from TraceFlix microservices."""

    def __init__(self, session: aiohttp.ClientSession, publisher: RedisPublisher):
        self.session = session
        self.publisher = publisher

    async def run(self):
        logger.info(f"Loki log collector started (every {LOGS_INTERVAL}s)")
        while not shutdown_event.is_set():
            try:
                for svc in TRACEFLIX_SERVICES + ["otel-collector"]:
                    await self._query_service(svc)
            except Exception as e:
                logger.error(f"Loki error: {e}")
            await asyncio.sleep(LOGS_INTERVAL)

    async def _query_service(self, service: str):
        now = datetime.now(timezone.utc)
        since = now - timedelta(seconds=LOGS_INTERVAL + 5)
        logql = f'{{namespace="{TARGET_NAMESPACE}", app="{service}"}}'

        try:
            async with self.session.get(
                f"{LOKI_URL}/loki/api/v1/query_range",
                params={
                    "query": logql,
                    "start": str(int(since.timestamp() * 1e9)),
                    "end": str(int(now.timestamp() * 1e9)),
                    "limit": 100, "direction": "backward",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return

                data = await resp.json()
                entries = []
                for stream in data.get("data", {}).get("result", []):
                    labels = stream.get("stream", {})
                    for ts, line in stream.get("values", []):
                        entries.append({
                            "service": service, "labels": labels,
                            "timestamp_ns": ts, "severity": self._severity(line),
                            "message": line[:4096],
                        })

                if entries:
                    await self.publisher.publish(STREAM_LOGS, {
                        "source": "loki", "service": service,
                        "namespace": TARGET_NAMESPACE, "count": len(entries),
                        "entries": entries,
                    })
        except asyncio.TimeoutError:
            logger.warning(f"Loki query timed out for {service}")

    @staticmethod
    def _severity(msg: str) -> str:
        ml = msg.lower()
        if any(k in ml for k in ["error", "exception", "stacktrace", "fatal"]):
            return "ERROR"
        if any(k in ml for k in ["warn", "warning"]):
            return "WARN"
        if any(k in ml for k in ["debug", "trace"]):
            return "DEBUG"
        return "INFO"


# ── Tempo Trace Collector ────────────────────────────────────────────────────
class TempoCollector:
    """Queries Tempo for recent traces, flags slow/error spans."""

    def __init__(self, session: aiohttp.ClientSession, publisher: RedisPublisher):
        self.session = session
        self.publisher = publisher

    async def run(self):
        logger.info(f"Tempo trace collector started (every {TRACES_INTERVAL}s)")
        while not shutdown_event.is_set():
            try:
                for svc in TRACEFLIX_SERVICES:
                    await self._search(svc)
            except Exception as e:
                logger.error(f"Tempo error: {e}")
            await asyncio.sleep(TRACES_INTERVAL)

    async def _search(self, service: str):
        now = datetime.now(timezone.utc)
        since = now - timedelta(seconds=TRACES_INTERVAL + 10)

        try:
            async with self.session.get(
                f"{TEMPO_URL}/api/search",
                params={
                    "tags": f"service.name={service}",
                    "start": str(int(since.timestamp())),
                    "end": str(int(now.timestamp())),
                    "limit": 20,
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return

                data = await resp.json()
                summaries = []
                for t in data.get("traces", []):
                    dur = t.get("durationMs", 0)
                    summaries.append({
                        "trace_id": t.get("traceID", ""),
                        "service": t.get("rootServiceName", service),
                        "operation": t.get("rootTraceName", ""),
                        "duration_ms": dur,
                        "is_slow": dur > 500,
                        "has_error": t.get("statusCode", 0) == 2,
                    })

                if summaries:
                    await self.publisher.publish(STREAM_TRACES, {
                        "source": "tempo", "service": service,
                        "namespace": TARGET_NAMESPACE, "count": len(summaries),
                        "traces": summaries,
                    })
        except asyncio.TimeoutError:
            logger.warning(f"Tempo search timed out for {service}")


# ── K8s Event Collector ──────────────────────────────────────────────────────
class K8sEventCollector:
    """Watches K8s events for OOMKills, CrashLoops, scheduling failures."""

    IMPORTANT = {"OOMKilled", "CrashLoopBackOff", "BackOff", "Failed",
                 "FailedScheduling", "Evicted", "Unhealthy", "ImagePullBackOff", "ErrImagePull"}

    def __init__(self, session: aiohttp.ClientSession, publisher: RedisPublisher):
        self.session = session
        self.publisher = publisher
        self.k8s_token: Optional[str] = None
        self._seen: set = set()

    async def init(self):
        try:
            with open(K8S_TOKEN_PATH) as f:
                self.k8s_token = f.read().strip()
        except FileNotFoundError:
            self.k8s_token = os.getenv("K8S_TOKEN", "")

    async def run(self):
        if not self.k8s_token:
            logger.warning("No K8s token — event collection disabled")
            return
        logger.info(f"K8s event collector started (every {EVENTS_INTERVAL}s)")
        while not shutdown_event.is_set():
            try:
                await self._poll_events()
                await self._poll_pod_status()
            except Exception as e:
                logger.error(f"K8s error: {e}")
            await asyncio.sleep(EVENTS_INTERVAL)

    async def _k8s_get(self, path: str) -> dict:
        import ssl
        ssl_ctx = ssl.create_default_context(cafile=K8S_CA_PATH) if os.path.exists(K8S_CA_PATH) else False
        url = f"{K8S_API}{path}"
        headers = {"Authorization": f"Bearer {self.k8s_token}"}
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as s:
            async with s.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
                return await r.json() if r.status == 200 else {}

    async def _poll_events(self):
        data = await self._k8s_get(f"/api/v1/namespaces/{TARGET_NAMESPACE}/events")
        new = []
        for ev in data.get("items", []):
            uid = ev["metadata"]["uid"]
            if uid in self._seen:
                continue
            self._seen.add(uid)
            reason = ev.get("reason", "")
            if reason in self.IMPORTANT or ev.get("type") == "Warning":
                new.append({
                    "pod": ev.get("involvedObject", {}).get("name", "?"),
                    "reason": reason, "message": ev.get("message", "")[:500],
                    "type": ev.get("type", "Normal"), "count": ev.get("count", 1),
                })
        if new:
            await self.publisher.publish(STREAM_EVENTS, {
                "source": "kubernetes", "namespace": TARGET_NAMESPACE, "events": new,
            })
        if len(self._seen) > 10000:
            self._seen = set(list(self._seen)[-5000:])

    async def _poll_pod_status(self):
        data = await self._k8s_get(f"/api/v1/namespaces/{TARGET_NAMESPACE}/pods")
        pods = []
        for p in data.get("items", []):
            containers = [{
                "name": c["name"], "ready": c.get("ready", False),
                "restarts": c.get("restartCount", 0),
            } for c in p.get("status", {}).get("containerStatuses", [])]
            pods.append({
                "name": p["metadata"]["name"],
                "app": p["metadata"].get("labels", {}).get("app", "?"),
                "phase": p.get("status", {}).get("phase", "Unknown"),
                "containers": containers,
            })
        if pods:
            await self.publisher.publish(STREAM_EVENTS, {
                "source": "k8s_pod_status", "namespace": TARGET_NAMESPACE, "pods": pods,
            })


# ── VictoriaMetrics TSDB Trend Collector ─────────────────────────────────────
class VictoriaMetricsTrendCollector:
    """
    Queries VictoriaMetrics for long-range trend data.
    Unlike the PrometheusCollector (which gets point-in-time snapshots),
    this queries range data over 1h/6h/24h windows to detect:
      - Latency degradation trends
      - Memory leak patterns (monotonic increase)
      - Error rate acceleration
      - Capacity saturation curves
    Publishes pre-computed trend summaries to a dedicated stream.
    """

    # Range queries with step sizes for trend analysis
    TREND_QUERIES = {
        "request_rate_1h": {
            "query": 'sum(rate(http_server_request_duration_seconds_count{job="otel-collector"}[5m])) by (service_name)',
            "range": "1h", "step": "1m",
            "description": "Request rate per service over last hour",
        },
        "latency_p99_1h": {
            "query": 'histogram_quantile(0.99, sum(rate(http_server_request_duration_seconds_bucket{job="otel-collector"}[5m])) by (le, service_name))',
            "range": "1h", "step": "1m",
            "description": "P99 latency trend over last hour",
        },
        "latency_p50_1h": {
            "query": 'histogram_quantile(0.50, sum(rate(http_server_request_duration_seconds_bucket{job="otel-collector"}[5m])) by (le, service_name))',
            "range": "1h", "step": "1m",
            "description": "P50 latency trend over last hour",
        },
        "error_rate_1h": {
            "query": 'sum(rate(http_server_request_duration_seconds_count{job="otel-collector",http_response_status_code=~"5.."}[5m])) by (service_name)',
            "range": "1h", "step": "1m",
            "description": "5xx error rate trend over last hour",
        },
        "jvm_heap_used_1h": {
            "query": 'jvm_memory_used_bytes{job="otel-collector",area="heap"}',
            "range": "1h", "step": "1m",
            "description": "JVM heap memory trend (detect leaks)",
        },
        "jvm_gc_pause_1h": {
            "query": 'rate(jvm_gc_duration_seconds_sum{job="otel-collector"}[5m])',
            "range": "1h", "step": "1m",
            "description": "GC pause rate trend",
        },
        "request_rate_24h": {
            "query": 'sum(rate(http_server_request_duration_seconds_count{job="otel-collector"}[15m])) by (service_name)',
            "range": "24h", "step": "15m",
            "description": "Request rate over 24 hours (daily pattern)",
        },
        "latency_p99_24h": {
            "query": 'histogram_quantile(0.99, sum(rate(http_server_request_duration_seconds_bucket{job="otel-collector"}[15m])) by (le, service_name))',
            "range": "24h", "step": "15m",
            "description": "P99 latency over 24 hours",
        },
    }

    def __init__(self, session: aiohttp.ClientSession, publisher: RedisPublisher):
        self.session = session
        self.publisher = publisher

    async def run(self):
        logger.info(f"VictoriaMetrics trend collector started (every {TSDB_TRENDS_INTERVAL}s)")
        # Wait for VictoriaMetrics to ingest initial data
        await asyncio.sleep(30)
        while not shutdown_event.is_set():
            try:
                await self._collect_trends()
            except Exception as e:
                logger.error(f"VictoriaMetrics trend error: {e}")
            await asyncio.sleep(TSDB_TRENDS_INTERVAL)

    async def _collect_trends(self):
        now = datetime.now(timezone.utc)
        trends = {}

        for name, config in self.TREND_QUERIES.items():
            range_delta = self._parse_range(config["range"])
            start = now - range_delta
            result = await self._query_range(
                config["query"], start, now, config["step"]
            )

            if result:
                analyzed = self._analyze_trend(result)
                trends[name] = {
                    "description": config["description"],
                    "range": config["range"],
                    "step": config["step"],
                    "series_count": len(result),
                    "analysis": analyzed,
                    "raw_series": [
                        {
                            "labels": r.get("metric", {}),
                            "values_count": len(r.get("values", [])),
                            "first_value": r.get("values", [[0, "0"]])[0][1] if r.get("values") else None,
                            "last_value": r.get("values", [[0, "0"]])[-1][1] if r.get("values") else None,
                        }
                        for r in result[:10]  # Cap raw series for message size
                    ],
                }

        if trends:
            await self.publisher.publish(STREAM_TSDB_TRENDS, {
                "source": "victoriametrics",
                "namespace": TARGET_NAMESPACE,
                "query_count": len(trends),
                "trends": trends,
            })
            logger.debug(f"Published {len(trends)} TSDB trend analyses")

    async def _query_range(self, query: str, start: datetime, end: datetime, step: str) -> list:
        """Execute a range query against VictoriaMetrics PromQL-compatible API."""
        url = f"{VICTORIA_METRICS_URL}/api/v1/query_range"
        params = {
            "query": query,
            "start": str(int(start.timestamp())),
            "end": str(int(end.timestamp())),
            "step": step,
        }
        try:
            async with self.session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", {}).get("result", [])
                else:
                    logger.debug(f"VM query returned {resp.status} for: {query[:60]}")
        except asyncio.TimeoutError:
            logger.warning(f"VictoriaMetrics query timed out: {query[:60]}")
        except Exception as e:
            logger.debug(f"VM query error: {e}")
        return []

    def _analyze_trend(self, series: list) -> dict:
        """Compute basic trend statistics from range query results."""
        analysis = {"series": []}

        for s in series[:10]:
            values = s.get("values", [])
            if not values:
                continue

            floats = []
            for _, v in values:
                try:
                    floats.append(float(v))
                except (ValueError, TypeError):
                    continue

            if not floats:
                continue

            n = len(floats)
            avg = sum(floats) / n
            min_v = min(floats)
            max_v = max(floats)
            latest = floats[-1]

            # Simple linear trend: positive = increasing, negative = decreasing
            if n >= 2:
                first_half_avg = sum(floats[:n // 2]) / (n // 2)
                second_half_avg = sum(floats[n // 2:]) / (n - n // 2)
                trend_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg != 0 else 0
            else:
                trend_pct = 0

            # Volatility: coefficient of variation
            if avg != 0:
                variance = sum((x - avg) ** 2 for x in floats) / n
                stddev = variance ** 0.5
                cv = (stddev / abs(avg)) * 100
            else:
                cv = 0

            direction = "increasing" if trend_pct > 5 else "decreasing" if trend_pct < -5 else "stable"

            analysis["series"].append({
                "labels": s.get("metric", {}),
                "data_points": n,
                "avg": round(avg, 4),
                "min": round(min_v, 4),
                "max": round(max_v, 4),
                "latest": round(latest, 4),
                "trend_pct": round(trend_pct, 2),
                "direction": direction,
                "volatility_cv": round(cv, 2),
            })

        return analysis

    @staticmethod
    def _parse_range(range_str: str) -> timedelta:
        num = int(range_str[:-1])
        unit = range_str[-1]
        if unit == "h":
            return timedelta(hours=num)
        elif unit == "d":
            return timedelta(days=num)
        elif unit == "m":
            return timedelta(minutes=num)
        return timedelta(hours=1)


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    logger.info("=" * 60)
    logger.info("DevOps AI Agent — Collector starting")
    logger.info(f"  Target:     {TARGET_NAMESPACE}")
    logger.info(f"  Services:   {TRACEFLIX_SERVICES}")
    logger.info(f"  Prometheus: {PROMETHEUS_URL}")
    logger.info(f"  Loki:       {LOKI_URL}")
    logger.info(f"  Tempo:      {TEMPO_URL}")
    logger.info(f"  VictoriaM:  {VICTORIA_METRICS_URL}")
    logger.info("=" * 60)

    publisher = RedisPublisher()
    await publisher.connect()
    session = aiohttp.ClientSession()

    k8s = K8sEventCollector(session, publisher)
    await k8s.init()

    tasks = [
        asyncio.create_task(PrometheusCollector(session, publisher).run()),
        asyncio.create_task(LokiCollector(session, publisher).run()),
        asyncio.create_task(TempoCollector(session, publisher).run()),
        asyncio.create_task(k8s.run()),
        asyncio.create_task(VictoriaMetricsTrendCollector(session, publisher).run()),
    ]

    try:
        await shutdown_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await session.close()
        await publisher.close()
        logger.info("Collector shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
