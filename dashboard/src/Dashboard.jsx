import { useState, useEffect, useCallback, useRef } from "react";

// Backend URL — override via REACT_APP_BACKEND_URL env var or nginx reverse proxy
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const WS_URL = BACKEND_URL.replace(/^http/, "ws") + "/ws/live";
const REFRESH_INTERVAL = 15000; // 15 seconds

// ── Theme ────────────────────────────────────────────────────
const theme = {
  bg: "#0a0e17",
  surface: "#111827",
  surfaceHover: "#1a2332",
  border: "#1e293b",
  borderAccent: "#334155",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
  textDim: "#64748b",
  green: "#22c55e",
  greenDim: "#166534",
  red: "#ef4444",
  redDim: "#7f1d1d",
  amber: "#f59e0b",
  amberDim: "#78350f",
  blue: "#3b82f6",
  blueDim: "#1e3a5f",
  purple: "#a855f7",
  cyan: "#06b6d4",
};

const statusColors = {
  HEALTHY: { bg: "#052e16", border: "#166534", text: "#22c55e", icon: "✓" },
  DEGRADED: { bg: "#451a03", border: "#78350f", text: "#f59e0b", icon: "⚠" },
  CRITICAL: { bg: "#450a0a", border: "#7f1d1d", text: "#ef4444", icon: "✕" },
  UNKNOWN: { bg: "#1e293b", border: "#334155", text: "#94a3b8", icon: "?" },
};

const severityColors = {
  ERROR: theme.red,
  WARN: theme.amber,
  INFO: theme.blue,
  DEBUG: theme.textDim,
  critical: theme.red,
  high: "#f97316",
  medium: theme.amber,
  low: theme.blue,
};

// ── API helpers ──────────────────────────────────────────────
async function apiFetch(path, params = {}) {
  const url = new URL(`${BACKEND_URL}${path}`);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  });
  const resp = await fetch(url.toString());
  if (!resp.ok) throw new Error(`API ${resp.status}: ${path}`);
  return resp.json();
}

// ── Helper Components ───────────────────────────────────────
function StatCard({ label, value, subValue, color, icon }) {
  return (
    <div style={{
      background: theme.surface, border: `1px solid ${theme.border}`,
      borderRadius: 8, padding: "16px 20px", flex: 1, minWidth: 140,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ color: theme.textDim, fontSize: 12, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</span>
        {icon && <span style={{ fontSize: 14 }}>{icon}</span>}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || theme.text, fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>{value}</div>
      {subValue && <div style={{ fontSize: 12, color: theme.textDim, marginTop: 4 }}>{subValue}</div>}
    </div>
  );
}

function SeverityBadge({ severity }) {
  const color = severityColors[severity] || theme.textMuted;
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 11,
      fontWeight: 600, fontFamily: "monospace", color,
      background: color + "18", border: `1px solid ${color}40`,
    }}>{severity}</span>
  );
}

function SectionHeader({ title, count, icon }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, paddingBottom: 12, borderBottom: `1px solid ${theme.border}` }}>
      {icon && <span style={{ fontSize: 18 }}>{icon}</span>}
      <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: theme.text }}>{title}</h2>
      {count !== undefined && (
        <span style={{
          background: theme.blueDim, color: theme.blue, fontSize: 11, fontWeight: 600,
          padding: "2px 8px", borderRadius: 10, fontFamily: "monospace",
        }}>{count}</span>
      )}
    </div>
  );
}

function LoadingSpinner({ message }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: 60, color: theme.textDim, gap: 12,
    }}>
      <div style={{
        width: 32, height: 32, border: `3px solid ${theme.border}`,
        borderTop: `3px solid ${theme.blue}`, borderRadius: "50%",
        animation: "spin 1s linear infinite",
      }}></div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <span style={{ fontSize: 13 }}>{message || "Loading..."}</span>
    </div>
  );
}

function ErrorBanner({ error, onRetry }) {
  return (
    <div style={{
      background: theme.redDim, border: `1px solid ${theme.red}40`,
      borderRadius: 8, padding: "12px 16px", margin: "0 0 16px",
      display: "flex", alignItems: "center", justifyContent: "space-between",
    }}>
      <span style={{ color: theme.red, fontSize: 13 }}>Connection error: {error}</span>
      {onRetry && (
        <button onClick={onRetry} style={{
          background: theme.red + "20", border: `1px solid ${theme.red}40`,
          color: theme.red, borderRadius: 4, padding: "4px 12px", cursor: "pointer",
          fontSize: 12, fontWeight: 600,
        }}>Retry</button>
      )}
    </div>
  );
}

// ── Main Dashboard ──────────────────────────────────────────
export default function DevOpsDashboard() {
  const [tab, setTab] = useState("overview");
  const [stats, setStats] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [logs, setLogs] = useState([]);
  const [traces, setTraces] = useState([]);
  const [trends, setTrends] = useState(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const retryTimerRef = useRef(null);

  // ── Fetch all data from backend API ──────────────────────
  const fetchAll = useCallback(async () => {
    try {
      const [statsRes, analysisRes, logsRes, tracesRes, trendsRes] = await Promise.all([
        apiFetch("/api/stats"),
        apiFetch("/api/analysis/latest").catch(() => null),
        apiFetch("/api/logs", { since_minutes: 60, limit: 200 }),
        apiFetch("/api/traces", { since_minutes: 60, limit: 100 }),
        apiFetch("/api/tsdb/trends/summary").catch(() => null),
      ]);

      setStats(statsRes);
      if (analysisRes && analysisRes.health_status) setAnalysis(analysisRes);
      setLogs(logsRes.logs || []);
      setTraces(tracesRes.traces || []);
      if (trendsRes) setTrends(trendsRes);
      setLastUpdate(new Date().toISOString());
      setError(null);
      setLoading(false);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message);
      setLoading(false);
    }
  }, []);

  // ── Initial fetch + auto-refresh interval ────────────────
  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchAll]);

  // ── WebSocket for live log streaming ─────────────────────
  useEffect(() => {
    let ws = null;
    let reconnectDelay = 1000;

    function connect() {
      try {
        ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          reconnectDelay = 1000;
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "log" && msg.data) {
              setLogs((prev) => [msg.data, ...prev].slice(0, 500));
            }
          } catch (e) { /* ignore non-JSON */ }
        };

        ws.onclose = () => {
          setConnected(false);
          retryTimerRef.current = setTimeout(() => {
            reconnectDelay = Math.min(reconnectDelay * 2, 30000);
            connect();
          }, reconnectDelay);
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch (e) {
        setConnected(false);
      }
    }

    connect();

    return () => {
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      if (ws) ws.close();
    };
  }, []);

  const statusConf = statusColors[analysis?.health_status] || statusColors.UNKNOWN;

  const tabs = [
    { id: "overview", label: "Overview", icon: "◉" },
    { id: "analysis", label: "AI Analysis", icon: "⬡" },
    { id: "trends", label: "TSDB Trends", icon: "📈" },
    { id: "logs", label: "Logs", icon: "≡" },
    { id: "traces", label: "Traces", icon: "⤳" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: theme.bg, color: theme.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>
      {/* Header */}
      <header style={{
        background: theme.surface, borderBottom: `1px solid ${theme.border}`,
        padding: "12px 24px", display: "flex", alignItems: "center", justifyContent: "space-between",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: `linear-gradient(135deg, ${theme.blue}, ${theme.purple})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 800, color: "#fff",
          }}>A</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.01em" }}>DevOps AI Agent</div>
            <div style={{ fontSize: 11, color: theme.textDim }}>TraceFlix Cluster Monitor</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: connected ? theme.green : theme.red, boxShadow: connected ? `0 0 8px ${theme.green}60` : "none" }}></div>
            <span style={{ color: theme.textMuted }}>{connected ? "Live" : "Disconnected"}</span>
          </div>
          {lastUpdate && (
            <div style={{ fontSize: 11, color: theme.textDim, fontFamily: "monospace" }}>
              {new Date(lastUpdate).toLocaleTimeString()}
            </div>
          )}
        </div>
      </header>

      {/* Tabs */}
      <nav style={{
        background: theme.surface, borderBottom: `1px solid ${theme.border}`,
        padding: "0 24px", display: "flex", gap: 2,
      }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: "none", border: "none", cursor: "pointer",
            padding: "12px 16px", fontSize: 13, fontWeight: 500,
            color: tab === t.id ? theme.text : theme.textDim,
            borderBottom: tab === t.id ? `2px solid ${theme.blue}` : "2px solid transparent",
            display: "flex", alignItems: "center", gap: 6,
            transition: "all 0.15s",
          }}>
            <span style={{ fontSize: 14 }}>{t.icon}</span>{t.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main style={{ padding: 24, maxWidth: 1400, margin: "0 auto" }}>
        {error && <ErrorBanner error={error} onRetry={fetchAll} />}
        {loading ? (
          <LoadingSpinner message="Connecting to backend API..." />
        ) : (
          <>
            {tab === "overview" && <OverviewTab stats={stats} analysis={analysis} statusConf={statusConf} />}
            {tab === "analysis" && <AnalysisTab analysis={analysis} statusConf={statusConf} />}
            {tab === "trends" && <TrendsTab trends={trends} />}
            {tab === "logs" && <LogsTab logs={logs} connected={connected} />}
            {tab === "traces" && <TracesTab traces={traces} />}
          </>
        )}
      </main>
    </div>
  );
}

// ── Overview Tab ─────────────────────────────────────────────
function OverviewTab({ stats, analysis, statusConf }) {
  if (!stats) return <LoadingSpinner message="Loading stats..." />;

  return (
    <div>
      {/* Health banner */}
      {analysis && (
        <div style={{
          background: statusConf.bg, border: `1px solid ${statusConf.border}`,
          borderRadius: 8, padding: "16px 20px", marginBottom: 20,
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 24 }}>{statusConf.icon}</span>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: statusConf.text }}>{analysis.health_status}</div>
              <div style={{ fontSize: 13, color: theme.textMuted, marginTop: 2 }}>{analysis.summary}</div>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 11, color: theme.textDim }}>Confidence</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: statusConf.text, fontFamily: "monospace" }}>
              {((analysis.confidence || 0) * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      )}

      {/* Stat cards */}
      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        <StatCard label="Total Logs" value={stats.total_logs ?? "—"} icon="📝" />
        <StatCard label="Errors" value={stats.total_errors ?? "—"} color={stats.total_errors > 0 ? theme.red : theme.green} icon="⚠" />
        <StatCard label="Traces" value={stats.total_traces ?? "—"} icon="⤳" />
        <StatCard label="Slow Traces" value={stats.slow_traces ?? "—"} color={stats.slow_traces > 0 ? theme.amber : theme.green} icon="🐢" />
        <StatCard label="K8s Events" value={stats.total_events ?? "—"} icon="☸" />
      </div>

      {/* Latency by service */}
      {stats.latency_by_service && stats.latency_by_service.length > 0 && (
        <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 20, marginBottom: 20 }}>
          <SectionHeader title="Latency by Service" icon="⏱" />
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
            {stats.latency_by_service.map((s, i) => (
              <div key={i} style={{ flex: 1, minWidth: 200 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: theme.cyan, marginBottom: 6 }}>{s.service}</div>
                <div style={{ display: "flex", gap: 16 }}>
                  <div>
                    <div style={{ fontSize: 10, color: theme.textDim }}>P50</div>
                    <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "monospace" }}>{s.p50_ms ?? "—"}ms</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: theme.textDim }}>P99</div>
                    <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "monospace", color: (s.p99_ms || 0) > 500 ? theme.red : theme.text }}>{s.p99_ms ?? "—"}ms</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top recommendations */}
      {analysis?.recommendations && analysis.recommendations.length > 0 && (
        <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 20 }}>
          <SectionHeader title="Top Recommendations" count={analysis.recommendations.length} icon="💡" />
          {analysis.recommendations.slice(0, 3).map((r, i) => (
            <div key={i} style={{
              padding: 12, borderBottom: i < 2 ? `1px solid ${theme.border}` : "none",
              display: "flex", gap: 12, alignItems: "flex-start",
            }}>
              <span style={{
                fontSize: 10, fontWeight: 700, fontFamily: "monospace", padding: "2px 8px", borderRadius: 4,
                background: r.priority === "immediate" ? theme.redDim : r.priority === "short_term" ? theme.amberDim : theme.blueDim,
                color: r.priority === "immediate" ? theme.red : r.priority === "short_term" ? theme.amber : theme.blue,
                whiteSpace: "nowrap",
              }}>{r.priority}</span>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>{r.action}</div>
                <div style={{ fontSize: 12, color: theme.textDim, marginTop: 2 }}>{r.reason}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Analysis Tab ────────────────────────────────────────────
function AnalysisTab({ analysis, statusConf }) {
  if (!analysis) return <LoadingSpinner message="Waiting for first AI analysis cycle..." />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Summary */}
      <div style={{
        background: statusConf.bg, border: `1px solid ${statusConf.border}`,
        borderRadius: 8, padding: 20,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: 20 }}>{statusConf.icon}</span>
          <span style={{ fontSize: 18, fontWeight: 700, color: statusConf.text }}>{analysis.health_status}</span>
          <span style={{ fontSize: 13, color: theme.textDim, marginLeft: "auto", fontFamily: "monospace" }}>
            confidence: {((analysis.confidence || 0) * 100).toFixed(0)}%
          </span>
        </div>
        <p style={{ fontSize: 14, color: theme.text, lineHeight: 1.5, margin: 0 }}>{analysis.summary}</p>
      </div>

      {/* Anomalies */}
      {analysis.anomalies && analysis.anomalies.length > 0 && (
        <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 20 }}>
          <SectionHeader title="Anomalies Detected" count={analysis.anomalies.length} icon="🔍" />
          {analysis.anomalies.map((a, i) => (
            <div key={i} style={{ padding: "12px 0", borderBottom: i < analysis.anomalies.length - 1 ? `1px solid ${theme.border}` : "none" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <SeverityBadge severity={a.severity} />
                <span style={{ fontSize: 14, fontWeight: 600 }}>{a.title}</span>
              </div>
              <p style={{ fontSize: 13, color: theme.textMuted, margin: "0 0 4px", lineHeight: 1.5 }}>{a.detail}</p>
              {a.affected_resources && (
                <div style={{ fontSize: 11, color: theme.cyan, fontFamily: "monospace" }}>
                  Affected: {a.affected_resources.join(", ")}
                </div>
              )}
              {a.evidence && <div style={{ fontSize: 11, color: theme.textDim, marginTop: 4, fontStyle: "italic" }}>{a.evidence}</div>}
            </div>
          ))}
        </div>
      )}

      {/* Root Causes */}
      {analysis.root_causes && analysis.root_causes.length > 0 && (
        <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 20 }}>
          <SectionHeader title="Root Cause Analysis" count={analysis.root_causes.length} icon="🎯" />
          {analysis.root_causes.map((rc, i) => (
            <div key={i} style={{ padding: "12px 0", borderBottom: i < analysis.root_causes.length - 1 ? `1px solid ${theme.border}` : "none" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{rc.issue}</div>
                <span style={{ fontSize: 11, fontFamily: "monospace", color: theme.amber, whiteSpace: "nowrap" }}>
                  {((rc.confidence || 0) * 100).toFixed(0)}% confidence
                </span>
              </div>
              <p style={{ fontSize: 13, color: theme.textMuted, margin: "4px 0 0", lineHeight: 1.5 }}>{rc.probable_cause}</p>
            </div>
          ))}
        </div>
      )}

      {/* Recommendations */}
      {analysis.recommendations && analysis.recommendations.length > 0 && (
        <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 20 }}>
          <SectionHeader title="Recommendations" count={analysis.recommendations.length} icon="💡" />
          {analysis.recommendations.map((r, i) => (
            <div key={i} style={{ padding: "12px 0", borderBottom: i < analysis.recommendations.length - 1 ? `1px solid ${theme.border}` : "none" }}>
              <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span style={{
                  fontSize: 10, fontWeight: 700, fontFamily: "monospace", padding: "2px 8px", borderRadius: 4,
                  background: r.priority === "immediate" ? theme.redDim : r.priority === "short_term" ? theme.amberDim : theme.blueDim,
                  color: r.priority === "immediate" ? theme.red : r.priority === "short_term" ? theme.amber : theme.blue,
                  whiteSpace: "nowrap", marginTop: 2,
                }}>{r.priority}</span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{r.action}</div>
                  <div style={{ fontSize: 12, color: theme.textDim, marginTop: 2 }}>{r.reason}</div>
                  {r.command && (
                    <pre style={{
                      marginTop: 6, padding: "6px 10px", borderRadius: 4,
                      background: theme.bg, fontSize: 11, color: theme.cyan,
                      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                      whiteSpace: "pre-wrap", wordBreak: "break-all",
                      border: `1px solid ${theme.border}`,
                    }}>{r.command}</pre>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Incident Timeline */}
      {analysis.incident_timeline && analysis.incident_timeline.length > 0 && (
        <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 20 }}>
          <SectionHeader title="Incident Timeline" count={analysis.incident_timeline.length} icon="🕐" />
          {analysis.incident_timeline.map((ev, i) => (
            <div key={i} style={{
              display: "flex", gap: 12, padding: "8px 0",
              borderBottom: i < analysis.incident_timeline.length - 1 ? `1px solid ${theme.border}` : "none",
            }}>
              <span style={{ fontSize: 12, fontFamily: "monospace", color: theme.amber, minWidth: 70, whiteSpace: "nowrap" }}>{ev.time}</span>
              <span style={{ fontSize: 13, color: theme.text }}>{ev.event}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Logs Tab ────────────────────────────────────────────────
function LogsTab({ logs, connected }) {
  const [filter, setFilter] = useState("ALL");
  const filtered = filter === "ALL" ? logs : logs.filter(l => l.severity === filter);

  return (
    <div>
      {/* Severity filter bar */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, alignItems: "center" }}>
        {["ALL", "ERROR", "WARN", "INFO", "DEBUG"].map(sev => (
          <button key={sev} onClick={() => setFilter(sev)} style={{
            background: filter === sev ? (severityColors[sev] || theme.blue) + "20" : "transparent",
            border: `1px solid ${filter === sev ? (severityColors[sev] || theme.blue) + "60" : theme.border}`,
            color: filter === sev ? (severityColors[sev] || theme.text) : theme.textDim,
            borderRadius: 4, padding: "4px 12px", cursor: "pointer",
            fontSize: 12, fontWeight: 600, fontFamily: "monospace",
          }}>{sev}</button>
        ))}
        <span style={{ marginLeft: "auto", fontSize: 11, color: theme.textDim }}>
          {filtered.length} entries {connected && <span style={{ color: theme.green }}>· streaming live</span>}
        </span>
      </div>

      {/* Log entries */}
      <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, overflow: "hidden" }}>
        {filtered.length === 0 ? (
          <div style={{ padding: 40, textAlign: "center", color: theme.textDim, fontSize: 13 }}>No logs matching filter</div>
        ) : filtered.slice(0, 200).map((log, i) => (
          <div key={i} style={{
            padding: "10px 16px", borderBottom: `1px solid ${theme.border}`,
            background: log.severity === "ERROR" ? theme.redDim + "15" : "transparent",
            display: "grid", gridTemplateColumns: "75px 55px 130px 1fr", gap: 12, alignItems: "center",
          }}>
            <span style={{ fontSize: 11, fontFamily: "monospace", color: theme.textDim }}>
              {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : "—"}
            </span>
            <SeverityBadge severity={log.severity} />
            <span style={{ fontSize: 12, color: theme.cyan, fontFamily: "monospace" }}>{log.service || "?"}</span>
            <span style={{ fontSize: 12, color: theme.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Traces Tab ──────────────────────────────────────────────
function TracesTab({ traces }) {
  if (!traces || traces.length === 0) return <LoadingSpinner message="Waiting for trace data..." />;

  return (
    <div>
      <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, overflow: "hidden" }}>
        <div style={{
          display: "grid", gridTemplateColumns: "120px 1fr 200px 90px 70px 70px",
          gap: 12, padding: "10px 16px", borderBottom: `1px solid ${theme.border}`,
          fontSize: 11, fontWeight: 600, color: theme.textDim, textTransform: "uppercase", letterSpacing: "0.05em",
        }}>
          <span>Trace ID</span><span>Operation</span><span>Service</span><span>Duration</span><span>Slow</span><span>Error</span>
        </div>
        {traces.map((t, i) => (
          <div key={i} style={{
            display: "grid", gridTemplateColumns: "120px 1fr 200px 90px 70px 70px",
            gap: 12, padding: "12px 16px", borderBottom: `1px solid ${theme.border}`,
            background: t.has_error ? theme.redDim + "20" : t.is_slow ? theme.amberDim + "20" : "transparent",
          }}>
            <span style={{ fontFamily: "monospace", fontSize: 11, color: theme.purple }}>{(t.trace_id || "").slice(0, 12)}…</span>
            <span style={{ fontSize: 13, fontWeight: 500 }}>{t.operation}</span>
            <span style={{ fontSize: 12, color: theme.cyan }}>{t.service}</span>
            <div>
              <span style={{
                fontFamily: "monospace", fontSize: 13, fontWeight: 600,
                color: (t.duration_ms || 0) > 1000 ? theme.red : (t.duration_ms || 0) > 500 ? theme.amber : theme.green,
              }}>{t.duration_ms}ms</span>
              <div style={{ marginTop: 4, height: 3, background: theme.bg, borderRadius: 2, overflow: "hidden" }}>
                <div style={{
                  height: "100%", borderRadius: 2,
                  width: `${Math.min(((t.duration_ms || 0) / 2000) * 100, 100)}%`,
                  background: (t.duration_ms || 0) > 1000 ? theme.red : (t.duration_ms || 0) > 500 ? theme.amber : theme.green,
                }}></div>
              </div>
            </div>
            <span>{t.is_slow ? "🔶" : "—"}</span>
            <span>{t.has_error ? "🔴" : "—"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── TSDB Trends Tab ─────────────────────────────────────────
function TrendCard({ trend, type }) {
  const isBytes = (trend.query || "").includes("heap") || (trend.query || "").includes("memory");
  const formatVal = (v) => {
    if (v == null) return "—";
    if (isBytes) return `${(v / 1024 / 1024).toFixed(0)} MB`;
    if (v < 0.01) return v.toFixed(4);
    if (v < 1) return v.toFixed(3);
    if (v > 1000) return v.toFixed(0);
    return v.toFixed(2);
  };

  const svc = trend.labels?.service_name || trend.labels?.service || "unknown";
  const borderColor = type === "degrading" ? theme.red : type === "improving" ? theme.green : theme.border;
  const bgTint = type === "degrading" ? theme.redDim + "30" : type === "improving" ? theme.greenDim + "30" : "transparent";

  return (
    <div style={{
      background: theme.surface, border: `1px solid ${borderColor}`,
      borderRadius: 8, padding: 16, borderLeft: `3px solid ${borderColor}`,
      backgroundColor: bgTint,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: theme.text }}>{trend.description}</div>
          <div style={{ fontSize: 11, color: theme.textDim, marginTop: 2 }}>
            <span style={{ color: theme.cyan }}>{svc}</span>
            <span style={{ margin: "0 6px" }}>·</span>
            <span style={{ fontFamily: "monospace" }}>{trend.query}</span>
            <span style={{ margin: "0 6px" }}>·</span>
            <span>{trend.range} window</span>
          </div>
        </div>
        <div style={{
          padding: "3px 10px", borderRadius: 12, fontSize: 12, fontWeight: 700, fontFamily: "monospace",
          color: type === "degrading" ? theme.red : type === "improving" ? theme.green : theme.textMuted,
          background: type === "degrading" ? theme.redDim : type === "improving" ? theme.greenDim : theme.surfaceHover,
        }}>
          {(trend.trend_pct || 0) > 0 ? "+" : ""}{(trend.trend_pct || 0).toFixed(1)}%
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <div>
          <div style={{ fontSize: 10, color: theme.textDim, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>Latest</div>
          <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: theme.text }}>{formatVal(trend.latest)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: theme.textDim, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>Average</div>
          <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: theme.textMuted }}>{formatVal(trend.avg)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: theme.textDim, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>Direction</div>
          <div style={{
            fontSize: 13, fontWeight: 600,
            color: trend.direction === "increasing" ? theme.red : trend.direction === "decreasing" ? theme.green : theme.textMuted,
          }}>
            {trend.direction === "increasing" ? "▲" : trend.direction === "decreasing" ? "▼" : "—"} {trend.direction}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: theme.textDim, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>Volatility</div>
          <div style={{
            fontSize: 13, fontWeight: 600, fontFamily: "monospace",
            color: (trend.volatility_cv || 0) > 40 ? theme.amber : theme.textMuted,
          }}>
            CV {(trend.volatility_cv || 0).toFixed(1)}%
          </div>
        </div>
      </div>
      <div style={{ marginTop: 10, height: 4, background: theme.bg, borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", borderRadius: 2,
          width: `${Math.min(Math.abs(trend.trend_pct || 0), 100)}%`,
          background: type === "degrading" ? theme.red : type === "improving" ? theme.green : theme.blue,
        }}></div>
      </div>
    </div>
  );
}

function TrendsTab({ trends }) {
  if (!trends) return <LoadingSpinner message="Waiting for TSDB trend data..." />;

  const degrading = trends.degrading || [];
  const stable = trends.stable || [];
  const improving = trends.improving || [];

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
        <StatCard label="Total Series" value={trends.total_series || 0} icon="📈" color={theme.text} />
        <StatCard label="Degrading" value={degrading.length} icon="🔴" color={degrading.length > 0 ? theme.red : theme.green} subValue={degrading.length > 0 ? "Requires attention" : "All clear"} />
        <StatCard label="Stable" value={stable.length} icon="🟢" color={theme.green} />
        <StatCard label="Improving" value={improving.length} icon="📉" color={theme.cyan} />
      </div>

      {degrading.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <SectionHeader title="Degrading Trends" count={degrading.length} icon="⚠" />
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {degrading.map((t, i) => <TrendCard key={`d-${i}`} trend={t} type="degrading" />)}
          </div>
        </div>
      )}

      {improving.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <SectionHeader title="Improving Trends" count={improving.length} icon="✓" />
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {improving.map((t, i) => <TrendCard key={`i-${i}`} trend={t} type="improving" />)}
          </div>
        </div>
      )}

      {stable.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <SectionHeader title="Stable Metrics" count={stable.length} icon="—" />
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {stable.map((t, i) => <TrendCard key={`s-${i}`} trend={t} type="stable" />)}
          </div>
        </div>
      )}

      <div style={{
        background: theme.surfaceHover, border: `1px solid ${theme.border}`,
        borderRadius: 8, padding: 14, fontSize: 12, color: theme.textDim,
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <span>ℹ</span>
        <span>
          Trend data sourced from <strong style={{ color: theme.text }}>VictoriaMetrics TSDB</strong> via PromQL range queries.
          Metrics ingested via Prometheus remote_write with 30-day retention and automatic downsampling.
          Trends update every 60 seconds.
        </span>
      </div>
    </div>
  );
}
