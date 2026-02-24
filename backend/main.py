"""
DevOps AI Agent — Backend Service (Step 3)
FastAPI application that:
  - Consumes logs/metrics/traces/events from Redis Streams (consumer groups)
  - Stores data in PostgreSQL with retention policies
  - Serves REST API for querying data
  - Provides WebSocket for real-time log streaming to dashboard
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Configuration ────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis.devops-agent.svc.cluster.local")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres.devops-agent.svc.cluster.local")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "devops_agent")
POSTGRES_USER = os.getenv("POSTGRES_USER", "devops")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

CONSUMER_GROUP = "backend-consumers"
CONSUMER_NAME = os.getenv("HOSTNAME", "backend-0")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "7"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backend")

# ── Database Schema ──────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS logs (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    namespace   TEXT NOT NULL,
    service     TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'INFO',
    message     TEXT NOT NULL,
    labels      JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS metrics (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    namespace   TEXT,
    metric_name TEXT NOT NULL,
    labels      JSONB DEFAULT '{}',
    value       TEXT,
    raw_data    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS traces (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trace_id    TEXT NOT NULL,
    service     TEXT NOT NULL,
    operation   TEXT,
    duration_ms FLOAT DEFAULT 0,
    is_slow     BOOLEAN DEFAULT FALSE,
    has_error   BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    namespace   TEXT,
    source      TEXT NOT NULL,
    pod         TEXT,
    reason      TEXT,
    message     TEXT,
    event_type  TEXT,
    raw_data    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS analysis (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    health_status   TEXT NOT NULL DEFAULT 'UNKNOWN',
    confidence      FLOAT DEFAULT 0.0,
    summary         TEXT NOT NULL,
    anomalies       JSONB DEFAULT '[]',
    root_causes     JSONB DEFAULT '[]',
    performance     JSONB DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',
    incident_timeline JSONB DEFAULT '[]',
    raw_response    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tsdb_trends (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    namespace   TEXT,
    query_name  TEXT NOT NULL,
    description TEXT,
    range_window TEXT,
    step        TEXT,
    series_count INT DEFAULT 0,
    analysis    JSONB DEFAULT '{}',
    raw_series  JSONB DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_svc ON logs (service);
CREATE INDEX IF NOT EXISTS idx_logs_severity ON logs (severity);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traces_svc ON traces (service);
CREATE INDEX IF NOT EXISTS idx_traces_slow ON traces (is_slow) WHERE is_slow = TRUE;
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_ts ON analysis (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tsdb_trends_ts ON tsdb_trends (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tsdb_trends_name ON tsdb_trends (query_name);
"""

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[aioredis.Redis] = None


async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host=POSTGRES_HOST, port=POSTGRES_PORT, database=POSTGRES_DB,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD, min_size=5, max_size=20,
    )
    async with db_pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
    logger.info("PostgreSQL initialized")


async def init_redis():
    global redis_client
    redis_client = aioredis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD or None, decode_responses=True,
    )
    await redis_client.ping()
    for stream in ["stream:logs", "stream:metrics", "stream:traces", "stream:events", "stream:tsdb_trends"]:
        try:
            await redis_client.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
    logger.info("Redis consumer groups initialized")


# ── Stream Consumers ─────────────────────────────────────────────────────────
class StreamConsumer:
    def __init__(self, stream: str, handler):
        self.stream = stream
        self.handler = handler
        self._running = True

    async def run(self):
        logger.info(f"Consumer started: {self.stream}")
        while self._running:
            try:
                msgs = await redis_client.xreadgroup(
                    groupname=CONSUMER_GROUP, consumername=CONSUMER_NAME,
                    streams={self.stream: ">"}, count=BATCH_SIZE, block=2000,
                )
                if msgs:
                    for _, entries in msgs:
                        for msg_id, data in entries:
                            try:
                                await self.handler(data)
                                await redis_client.xack(self.stream, CONSUMER_GROUP, msg_id)
                            except Exception as e:
                                logger.error(f"Process error {msg_id}: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer {self.stream} error: {e}")
                await asyncio.sleep(2)

    def stop(self):
        self._running = False


def _parse_ts(ts_str) -> datetime:
    if not ts_str:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


async def handle_logs(data: dict):
    raw = json.loads(data.get("data", "{}"))
    entries = raw.get("entries", [])
    ts = _parse_ts(data.get("timestamp"))
    async with db_pool.acquire() as conn:
        for entry in entries[:100]:
            await conn.execute(
                "INSERT INTO logs (timestamp, namespace, service, severity, message, labels) VALUES ($1,$2,$3,$4,$5,$6)",
                ts, raw.get("namespace", ""), entry.get("service", ""),
                entry.get("severity", "INFO"), entry.get("message", "")[:4096],
                json.dumps(entry.get("labels", {})),
            )
    # Broadcast to WebSocket
    await ws_manager.broadcast({"type": "logs", "service": raw.get("service"), "count": len(entries)})


async def handle_metrics(data: dict):
    raw = json.loads(data.get("data", "{}"))
    ts = _parse_ts(data.get("timestamp"))
    async with db_pool.acquire() as conn:
        for name, results in raw.get("metrics", {}).items():
            for r in results[:50]:
                await conn.execute(
                    "INSERT INTO metrics (timestamp, namespace, metric_name, labels, value, raw_data) VALUES ($1,$2,$3,$4,$5,$6)",
                    ts, raw.get("namespace", ""), name,
                    json.dumps(r.get("labels", {})), str(r.get("value", "")),
                    json.dumps(r),
                )


async def handle_traces(data: dict):
    raw = json.loads(data.get("data", "{}"))
    ts = _parse_ts(data.get("timestamp"))
    async with db_pool.acquire() as conn:
        for t in raw.get("traces", [])[:50]:
            await conn.execute(
                "INSERT INTO traces (timestamp, trace_id, service, operation, duration_ms, is_slow, has_error) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                ts, t.get("trace_id", ""), t.get("service", ""),
                t.get("operation", ""), t.get("duration_ms", 0),
                t.get("is_slow", False), t.get("has_error", False),
            )


async def handle_events(data: dict):
    raw = json.loads(data.get("data", "{}"))
    ts = _parse_ts(data.get("timestamp"))
    source = raw.get("source", "unknown")
    async with db_pool.acquire() as conn:
        if source == "kubernetes":
            for ev in raw.get("events", [])[:50]:
                await conn.execute(
                    "INSERT INTO events (timestamp, namespace, source, pod, reason, message, event_type, raw_data) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                    ts, raw.get("namespace", ""), source,
                    ev.get("pod", ""), ev.get("reason", ""), ev.get("message", ""),
                    ev.get("type", "Normal"), json.dumps(ev),
                )
        elif source == "k8s_pod_status":
            for pod in raw.get("pods", [])[:50]:
                await conn.execute(
                    "INSERT INTO events (timestamp, namespace, source, pod, reason, message, raw_data) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                    ts, raw.get("namespace", ""), source,
                    pod.get("name", ""), "PodStatus", pod.get("phase", ""),
                    json.dumps(pod),
                )


async def handle_tsdb_trends(data: dict):
    raw = json.loads(data.get("data", "{}"))
    ts = _parse_ts(data.get("timestamp"))
    trends = raw.get("trends", {})
    async with db_pool.acquire() as conn:
        for name, trend_data in trends.items():
            await conn.execute(
                """INSERT INTO tsdb_trends (timestamp, namespace, query_name, description,
                   range_window, step, series_count, analysis, raw_series)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                ts, raw.get("namespace", ""), name,
                trend_data.get("description", ""),
                trend_data.get("range", ""),
                trend_data.get("step", ""),
                trend_data.get("series_count", 0),
                json.dumps(trend_data.get("analysis", {})),
                json.dumps(trend_data.get("raw_series", [])),
            )


# ── WebSocket Manager ────────────────────────────────────────────────────────
class WSManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

ws_manager = WSManager()


# ── Retention Cleanup ────────────────────────────────────────────────────────
async def cleanup_loop():
    while True:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
            async with db_pool.acquire() as conn:
                for table in ["logs", "metrics", "traces", "events", "tsdb_trends"]:
                    await conn.execute(f"DELETE FROM {table} WHERE created_at < $1", cutoff)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        await asyncio.sleep(3600)


# ── FastAPI App ──────────────────────────────────────────────────────────────
consumers: list[StreamConsumer] = []
bg_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()

    consumers.extend([
        StreamConsumer("stream:logs", handle_logs),
        StreamConsumer("stream:metrics", handle_metrics),
        StreamConsumer("stream:traces", handle_traces),
        StreamConsumer("stream:events", handle_events),
        StreamConsumer("stream:tsdb_trends", handle_tsdb_trends),
    ])
    for c in consumers:
        bg_tasks.append(asyncio.create_task(c.run()))
    bg_tasks.append(asyncio.create_task(cleanup_loop()))

    logger.info("Backend service started")
    yield

    for c in consumers:
        c.stop()
    for t in bg_tasks:
        t.cancel()
    await asyncio.gather(*bg_tasks, return_exceptions=True)
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.aclose()


app = FastAPI(title="DevOps AI Agent Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/logs")
async def get_logs(
    service: Optional[str] = None, severity: Optional[str] = None,
    since_minutes: int = Query(default=60, ge=1, le=1440),
    limit: int = Query(default=200, ge=1, le=5000),
):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    q = "SELECT * FROM logs WHERE timestamp >= $1"
    params: list = [cutoff]
    idx = 2
    if service:
        q += f" AND service = ${idx}"; params.append(service); idx += 1
    if severity:
        q += f" AND severity = ${idx}"; params.append(severity.upper()); idx += 1
    q += f" ORDER BY timestamp DESC LIMIT ${idx}"; params.append(limit)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(q, *params)
    return {"count": len(rows), "logs": [dict(r) for r in rows]}


@app.get("/api/metrics")
async def get_metrics(
    metric_name: Optional[str] = None,
    since_minutes: int = Query(default=60, ge=1, le=1440),
    limit: int = Query(default=200, ge=1, le=5000),
):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    q = "SELECT * FROM metrics WHERE timestamp >= $1"
    params: list = [cutoff]
    idx = 2
    if metric_name:
        q += f" AND metric_name = ${idx}"; params.append(metric_name); idx += 1
    q += f" ORDER BY timestamp DESC LIMIT ${idx}"; params.append(limit)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(q, *params)
    return {"count": len(rows), "metrics": [dict(r) for r in rows]}


@app.get("/api/traces")
async def get_traces(
    service: Optional[str] = None, slow_only: bool = False, errors_only: bool = False,
    since_minutes: int = Query(default=60, ge=1, le=1440),
    limit: int = Query(default=100, ge=1, le=1000),
):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    q = "SELECT * FROM traces WHERE timestamp >= $1"
    params: list = [cutoff]
    idx = 2
    if service:
        q += f" AND service = ${idx}"; params.append(service); idx += 1
    if slow_only:
        q += " AND is_slow = TRUE"
    if errors_only:
        q += " AND has_error = TRUE"
    q += f" ORDER BY timestamp DESC LIMIT ${idx}"; params.append(limit)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(q, *params)
    return {"count": len(rows), "traces": [dict(r) for r in rows]}


@app.get("/api/events")
async def get_events(
    source: Optional[str] = None, reason: Optional[str] = None,
    since_minutes: int = Query(default=60, ge=1, le=1440),
    limit: int = Query(default=100, ge=1, le=1000),
):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    q = "SELECT * FROM events WHERE timestamp >= $1"
    params: list = [cutoff]
    idx = 2
    if source:
        q += f" AND source = ${idx}"; params.append(source); idx += 1
    if reason:
        q += f" AND reason = ${idx}"; params.append(reason); idx += 1
    q += f" ORDER BY timestamp DESC LIMIT ${idx}"; params.append(limit)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(q, *params)
    return {"count": len(rows), "events": [dict(r) for r in rows]}


@app.get("/api/analysis")
async def get_analysis(limit: int = Query(default=10, ge=1, le=50)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM analysis ORDER BY timestamp DESC LIMIT $1", limit)
    return {"count": len(rows), "analyses": [dict(r) for r in rows]}


@app.get("/api/analysis/latest")
async def latest_analysis():
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM analysis ORDER BY timestamp DESC LIMIT 1")
    if not row:
        raise HTTPException(404, "No analysis available yet")
    return dict(row)


@app.post("/api/analysis")
async def store_analysis(payload: dict):
    """Store an AI analysis result (called by the Agent)."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO analysis (health_status, confidence, summary, anomalies,
               root_causes, performance, recommendations, incident_timeline, raw_response)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
            payload.get("health_status", "UNKNOWN"),
            payload.get("confidence", 0.0),
            payload.get("summary", ""),
            json.dumps(payload.get("anomalies", [])),
            json.dumps(payload.get("root_causes", [])),
            json.dumps(payload.get("performance", {})),
            json.dumps(payload.get("recommendations", [])),
            json.dumps(payload.get("incident_timeline", [])),
            payload.get("_raw_response", ""),
        )
    return {"status": "stored"}


@app.get("/api/tsdb/trends")
async def get_tsdb_trends(
    query_name: Optional[str] = None,
    since_hours: int = Query(default=6, ge=1, le=168),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Query TSDB trend analyses from VictoriaMetrics."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    q = "SELECT * FROM tsdb_trends WHERE timestamp >= $1"
    params: list = [cutoff]
    idx = 2
    if query_name:
        q += f" AND query_name = ${idx}"; params.append(query_name); idx += 1
    q += f" ORDER BY timestamp DESC LIMIT ${idx}"; params.append(limit)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(q, *params)
    return {"count": len(rows), "trends": [dict(r) for r in rows]}


@app.get("/api/tsdb/trends/latest")
async def get_latest_trends():
    """Get the most recent trend snapshot for each query name."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT DISTINCT ON (query_name) *
               FROM tsdb_trends ORDER BY query_name, timestamp DESC"""
        )
    return {"count": len(rows), "trends": [dict(r) for r in rows]}


@app.get("/api/tsdb/trends/summary")
async def get_trend_summary():
    """Aggregated trend summary for the AI agent — highlights degrading metrics."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT DISTINCT ON (query_name) query_name, description, range_window,
                      analysis, timestamp
               FROM tsdb_trends ORDER BY query_name, timestamp DESC"""
        )

    degrading = []
    stable = []
    improving = []

    for r in rows:
        analysis = r["analysis"] if isinstance(r["analysis"], dict) else json.loads(r["analysis"] or "{}")
        series_list = analysis.get("series", [])
        for s in series_list:
            entry = {
                "query": r["query_name"],
                "description": r["description"],
                "range": r["range_window"],
                "labels": s.get("labels", {}),
                "latest": s.get("latest"),
                "avg": s.get("avg"),
                "trend_pct": s.get("trend_pct", 0),
                "direction": s.get("direction", "stable"),
                "volatility_cv": s.get("volatility_cv", 0),
            }
            if s.get("direction") == "increasing" and s.get("trend_pct", 0) > 10:
                degrading.append(entry)
            elif s.get("direction") == "decreasing":
                improving.append(entry)
            else:
                stable.append(entry)

    return {
        "degrading": sorted(degrading, key=lambda x: x["trend_pct"], reverse=True),
        "stable": stable,
        "improving": improving,
        "total_series": len(degrading) + len(stable) + len(improving),
    }


@app.get("/api/stats")
async def get_stats():
    async with db_pool.acquire() as conn:
        hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        log_count = await conn.fetchval("SELECT COUNT(*) FROM logs WHERE timestamp >= $1", hour_ago)
        error_count = await conn.fetchval("SELECT COUNT(*) FROM logs WHERE timestamp >= $1 AND severity='ERROR'", hour_ago)
        trace_count = await conn.fetchval("SELECT COUNT(*) FROM traces WHERE timestamp >= $1", hour_ago)
        slow_traces = await conn.fetchval("SELECT COUNT(*) FROM traces WHERE timestamp >= $1 AND is_slow=TRUE", hour_ago)
        error_traces = await conn.fetchval("SELECT COUNT(*) FROM traces WHERE timestamp >= $1 AND has_error=TRUE", hour_ago)
        event_count = await conn.fetchval("SELECT COUNT(*) FROM events WHERE timestamp >= $1 AND source='kubernetes'", hour_ago)

        svc_errors = await conn.fetch(
            "SELECT service, COUNT(*) as cnt FROM logs WHERE timestamp >= $1 AND severity='ERROR' GROUP BY service ORDER BY cnt DESC LIMIT 10",
            hour_ago,
        )
        svc_latency = await conn.fetch(
            "SELECT service, AVG(duration_ms) as avg_ms, MAX(duration_ms) as max_ms FROM traces WHERE timestamp >= $1 GROUP BY service",
            hour_ago,
        )

    return {
        "period": "last_hour",
        "total_logs": log_count, "total_errors": error_count,
        "total_traces": trace_count, "slow_traces": slow_traces, "error_traces": error_traces,
        "k8s_events": event_count,
        "errors_by_service": [dict(r) for r in svc_errors],
        "latency_by_service": [dict(r) for r in svc_latency],
    }


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
