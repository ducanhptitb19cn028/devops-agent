"""
Synthetic Log Data Generator

Generates realistic log messages from TraceFlix microservices for
training the log clustering model. Logs follow common Spring Boot
patterns and include correlated error sequences.

Log templates are parameterised to create diverse but clusterable
messages that map to operational patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SERVICES, LOG_SEVERITIES

# ── Log Templates ────────────────────────────────────────────
# Each template: (severity, template_str, cluster_label, params_fn)

INFO_TEMPLATES = [
    ("INFO", "Received GET /api/{endpoint} from {ip}", "api_request",
     lambda: {"endpoint": np.random.choice(["movies", "actors", "reviews", "movies/popular", "actors/search"]),
              "ip": f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"}),
    ("INFO", "Successfully fetched {count} records from {table} table", "db_query_success",
     lambda: {"count": np.random.randint(1, 500), "table": np.random.choice(["movies", "actors", "reviews", "ratings"])}),
    ("INFO", "Cache hit for key {key}, ttl={ttl}s remaining", "cache_hit",
     lambda: {"key": f"{np.random.choice(['movie', 'actor', 'review'])}:{np.random.randint(1, 10000)}",
              "ttl": np.random.randint(30, 3600)}),
    ("INFO", "Health check passed: db={db_status}, redis={redis_status}", "health_check",
     lambda: {"db_status": "UP", "redis_status": "UP"}),
    ("INFO", "Processing batch of {size} events from stream {stream}", "batch_processing",
     lambda: {"size": np.random.randint(10, 200), "stream": np.random.choice(["events", "updates", "notifications"])}),
    ("INFO", "Request completed in {duration}ms with status {status}", "request_complete",
     lambda: {"duration": np.random.randint(5, 200), "status": np.random.choice([200, 201, 204])}),
    ("INFO", "Connection pool stats: active={active}, idle={idle}, max={max}", "pool_stats",
     lambda: {"active": np.random.randint(1, 15), "idle": np.random.randint(0, 10), "max": 20}),
]

WARN_TEMPLATES = [
    ("WARN", "Slow query detected: {query} took {duration}ms (threshold: {threshold}ms)", "slow_query",
     lambda: {"query": f"SELECT * FROM {np.random.choice(['movies', 'actors', 'reviews'])} WHERE ...",
              "duration": np.random.randint(500, 5000), "threshold": 500}),
    ("WARN", "Connection pool nearing capacity: {active}/{max} active connections", "pool_exhaustion",
     lambda: {"active": np.random.randint(15, 20), "max": 20}),
    ("WARN", "Retry attempt {attempt}/{max_retries} for {operation}", "retry_attempt",
     lambda: {"attempt": np.random.randint(1, 4), "max_retries": 3,
              "operation": np.random.choice(["db_write", "cache_set", "http_call", "stream_publish"])}),
    ("WARN", "Response time {duration}ms exceeds SLO threshold of {slo}ms for {endpoint}", "slo_breach",
     lambda: {"duration": np.random.randint(500, 3000), "slo": 500,
              "endpoint": np.random.choice(["/api/movies", "/api/actors", "/api/reviews"])}),
    ("WARN", "JVM heap usage at {pct}% ({used}MB / {total}MB), GC may be triggered", "heap_warning",
     lambda: {"pct": np.random.randint(75, 92), "used": np.random.randint(384, 470), "total": 512}),
    ("WARN", "Circuit breaker {state} for {service}: {failures} failures in last {window}s", "circuit_breaker",
     lambda: {"state": np.random.choice(["HALF_OPEN", "OPEN"]),
              "service": np.random.choice(SERVICES), "failures": np.random.randint(3, 15), "window": 60}),
]

ERROR_TEMPLATES = [
    ("ERROR", "Connection refused to {host}:{port} after {timeout}ms timeout", "connection_refused",
     lambda: {"host": np.random.choice(["postgres-0", "redis-0", "actor-service", "review-service"]),
              "port": np.random.choice([5432, 6379, 8080]), "timeout": np.random.randint(3000, 30000)}),
    ("ERROR", "NullPointerException in {class}.{method} at line {line}", "null_pointer",
     lambda: {"class": np.random.choice(["MovieController", "ActorService", "ReviewRepository", "CacheManager"]),
              "method": np.random.choice(["getById", "findAll", "update", "processRequest"]),
              "line": np.random.randint(50, 500)}),
    ("ERROR", "OutOfMemoryError: Java heap space - requested {size}MB, available {avail}MB", "oom_error",
     lambda: {"size": np.random.randint(64, 256), "avail": np.random.randint(1, 32)}),
    ("ERROR", "Database connection pool exhausted: {active}/{max} connections in use, {queued} queued", "db_pool_exhausted",
     lambda: {"active": 20, "max": 20, "queued": np.random.randint(5, 50)}),
    ("ERROR", "HTTP 503 from {service}: Service Unavailable after {retries} retries", "service_unavailable",
     lambda: {"service": np.random.choice(SERVICES), "retries": 3}),
    ("ERROR", "Transaction rolled back: {reason}", "tx_rollback",
     lambda: {"reason": np.random.choice([
         "deadlock detected", "lock wait timeout exceeded",
         "constraint violation on reviews.user_id", "serialization failure",
     ])}),
    ("ERROR", "SSL handshake failed with {host}: certificate expired", "ssl_error",
     lambda: {"host": np.random.choice(["redis-0.devops-agent.svc", "postgres-0.devops-agent.svc"])}),
]

FATAL_TEMPLATES = [
    ("FATAL", "Application startup failed: {reason}", "startup_failure",
     lambda: {"reason": np.random.choice([
         "Cannot connect to database postgres-0:5432",
         "Port 8080 already in use",
         "Missing required config: SPRING_DATASOURCE_URL",
         "Bean creation exception: circularReference in MovieService",
     ])}),
]

ALL_TEMPLATES = INFO_TEMPLATES + WARN_TEMPLATES + ERROR_TEMPLATES + FATAL_TEMPLATES


def _render_template(template_str: str, params: dict) -> str:
    """Render a log template with parameters."""
    return template_str.format(**params)


def generate_log_dataset(
    n_logs: int = 20000,
    severity_weights: dict = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a log dataset with cluster labels for training.

    Args:
        n_logs: Number of log entries to generate
        severity_weights: Override default severity distribution
        seed: Random seed

    Returns:
        DataFrame with columns: timestamp, service, severity, message,
        cluster_label, template_id
    """
    np.random.seed(seed)

    if severity_weights is None:
        severity_weights = {"INFO": 0.60, "WARN": 0.20, "ERROR": 0.15, "FATAL": 0.05}

    templates_by_severity = {}
    for sev, tmpl, label, params_fn in ALL_TEMPLATES:
        templates_by_severity.setdefault(sev, []).append((tmpl, label, params_fn))

    records = []
    base_time = pd.Timestamp("2025-01-01")

    for i in range(n_logs):
        # Choose severity
        sev = np.random.choice(
            list(severity_weights.keys()),
            p=list(severity_weights.values()),
        )
        # Choose template
        if sev in templates_by_severity:
            tmpl_str, cluster_label, params_fn = templates_by_severity[sev][
                np.random.randint(len(templates_by_severity[sev]))
            ]
        else:
            tmpl_str, cluster_label, params_fn = templates_by_severity["INFO"][0]
            sev = "INFO"

        params = params_fn()
        message = _render_template(tmpl_str, params)

        records.append({
            "timestamp": base_time + pd.Timedelta(seconds=i * np.random.uniform(0.1, 2.0)),
            "service": np.random.choice(SERVICES),
            "severity": sev,
            "message": message,
            "cluster_label": cluster_label,
            "template_id": tmpl_str,
        })

    df = pd.DataFrame(records)
    print(f"[log_gen] Generated {len(df)} log entries")
    print(f"[log_gen] Severity distribution:\n{df['severity'].value_counts().to_string()}")
    print(f"[log_gen] Cluster distribution:\n{df['cluster_label'].value_counts().to_string()}")
    return df


def generate_incident_log_sequences(
    n_incidents: int = 500,
    sequence_length: int = 20,
    seed: int = 42,
) -> List[Tuple[str, List[dict]]]:
    """
    Generate correlated log sequences representing incidents.
    Each incident is a sequence of logs that tell a causal story.

    Returns:
        List of (incident_type, [log_entries]) tuples
    """
    np.random.seed(seed)

    incident_patterns = {
        "memory_leak_sequence": [
            ("WARN", "JVM heap usage at {pct}% ({used}MB / {total}MB), GC may be triggered"),
            ("WARN", "JVM heap usage at {pct}% ({used}MB / {total}MB), GC may be triggered"),
            ("INFO", "GC pause: {type} collection took {duration}ms"),
            ("WARN", "JVM heap usage at {pct}% ({used}MB / {total}MB), GC may be triggered"),
            ("ERROR", "OutOfMemoryError: Java heap space - requested {size}MB, available {avail}MB"),
            ("FATAL", "Application startup failed: {reason}"),
        ],
        "cascade_failure_sequence": [
            ("WARN", "Retry attempt {attempt}/{max_retries} for {operation}"),
            ("WARN", "Circuit breaker {state} for {service}: {failures} failures in last {window}s"),
            ("ERROR", "HTTP 503 from {service}: Service Unavailable after {retries} retries"),
            ("ERROR", "Connection refused to {host}:{port} after {timeout}ms timeout"),
            ("WARN", "Response time {duration}ms exceeds SLO threshold of {slo}ms for {endpoint}"),
            ("ERROR", "Database connection pool exhausted: {active}/{max} connections in use, {queued} queued"),
        ],
        "db_overload_sequence": [
            ("WARN", "Slow query detected: {query} took {duration}ms (threshold: {threshold}ms)"),
            ("WARN", "Connection pool nearing capacity: {active}/{max} active connections"),
            ("WARN", "Slow query detected: {query} took {duration}ms (threshold: {threshold}ms)"),
            ("ERROR", "Database connection pool exhausted: {active}/{max} connections in use, {queued} queued"),
            ("ERROR", "Transaction rolled back: {reason}"),
        ],
    }

    incidents = []
    for _ in range(n_incidents):
        incident_type = np.random.choice(list(incident_patterns.keys()))
        pattern = incident_patterns[incident_type]
        service = np.random.choice(SERVICES)

        sequence = []
        for sev, tmpl_str in pattern:
            # Find matching params function
            for s, t, _, pf in ALL_TEMPLATES:
                if t == tmpl_str:
                    params = pf()
                    break
            else:
                params = {}

            try:
                message = tmpl_str.format(**params)
            except KeyError:
                message = tmpl_str

            sequence.append({
                "severity": sev,
                "service": service,
                "message": message,
                "template": tmpl_str,
            })

        incidents.append((incident_type, sequence))

    return incidents


if __name__ == "__main__":
    df = generate_log_dataset(n_logs=5000, seed=42)
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "logs_training.parquet", index=False)
    print(f"[log_gen] Saved to {out_dir / 'logs_training.parquet'}")
