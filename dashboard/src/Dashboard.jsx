import { useState, useEffect, useCallback, useRef } from "react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, PieChart, Pie,
} from "recharts";

// Backend URL — nginx reverse-proxies /api/* and /ws/* to the backend service.
// Use an explicit REACT_APP_BACKEND_URL only for local dev (npm start).
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "";
const WS_URL = BACKEND_URL
  ? BACKEND_URL.replace(/^http/, "ws") + "/ws/live"
  : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/live`;
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

const SERVICE_COLORS = {
  "movie-service":  theme.blue,
  "actor-service":  theme.cyan,
  "review-service": theme.purple,
};
const SERVICE_LIST = ["movie-service", "actor-service", "review-service"];

const statusColors = {
  HEALTHY:  { bg: "#052e16", border: "#166534", text: "#22c55e", icon: "✓" },
  DEGRADED: { bg: "#451a03", border: "#78350f", text: "#f59e0b", icon: "⚠" },
  CRITICAL: { bg: "#450a0a", border: "#7f1d1d", text: "#ef4444", icon: "✕" },
  UNKNOWN:  { bg: "#1e293b", border: "#334155", text: "#94a3b8", icon: "?" },
};

const severityColors = {
  ERROR: theme.red,
  WARN:  theme.amber,
  INFO:  theme.blue,
  DEBUG: theme.textDim,
  critical: theme.red,
  high:     "#f97316",
  medium:   theme.amber,
  low:      theme.blue,
};

// ── JSONB helpers ─────────────────────────────────────────────
function parseJsonField(val, fallback) {
  if (val === null || val === undefined) return fallback;
  if (typeof val === "string") { try { return JSON.parse(val); } catch { return fallback; } }
  return val;
}

function toArray(val, fallback = []) {
  const v = parseJsonField(val, fallback);
  return Array.isArray(v) ? v : fallback;
}

function normalizeAnalysis(a) {
  if (!a) return a;
  return {
    ...a,
    anomalies:         toArray(a.anomalies),
    root_causes:       toArray(a.root_causes),
    recommendations:   toArray(a.recommendations),
    incident_timeline: toArray(a.incident_timeline),
    performance:       parseJsonField(a.performance, {}),
  };
}

// ── API helpers ──────────────────────────────────────────────
async function apiFetch(path, params = {}) {
  const url = new URL(BACKEND_URL ? `${BACKEND_URL}${path}` : path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  });
  const resp = await fetch(url.toString());
  if (!resp.ok) throw new Error(`API ${resp.status}: ${path}`);
  return resp.json();
}

// ── Chart helpers ─────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label, unit }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#1a2332", border: `1px solid ${theme.border}`,
      borderRadius: 6, padding: "8px 12px", fontSize: 12,
    }}>
      <div style={{ color: theme.textMuted, marginBottom: 6, fontSize: 11 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: p.color, flexShrink: 0 }}></span>
          <span style={{ color: theme.textMuted }}>{p.name}:</span>
          <span style={{ color: theme.text, fontFamily: "monospace", fontWeight: 600 }}>
            {p.value != null ? `${p.value}${unit || ""}` : "—"}
          </span>
        </div>
      ))}
    </div>
  );
};

const axisStyle = { fill: theme.textDim, fontSize: 10 };
const gridProps = { stroke: theme.border, strokeDasharray: "3 3" };

// ── Helper Components ────────────────────────────────────────
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

function SectionHeader({ title, count, icon, subtitle }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, paddingBottom: 12, borderBottom: `1px solid ${theme.border}` }}>
      {icon && <span style={{ fontSize: 18 }}>{icon}</span>}
      <div>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: theme.text }}>{title}</h2>
        {subtitle && <div style={{ fontSize: 11, color: theme.textDim, marginTop: 2 }}>{subtitle}</div>}
      </div>
      {count !== undefined && (
        <span style={{
          background: theme.blueDim, color: theme.blue, fontSize: 11, fontWeight: 600,
          padding: "2px 8px", borderRadius: 10, fontFamily: "monospace", marginLeft: "auto",
        }}>{count}</span>
      )}
    </div>
  );
}

function PanelHeader({ title, icon, badge, badgeColor }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10, padding: "14px 20px",
      borderBottom: `1px solid ${theme.border}`, background: theme.surfaceHover,
      borderRadius: "8px 8px 0 0",
    }}>
      <span style={{ fontSize: 18 }}>{icon}</span>
      <span style={{ fontSize: 15, fontWeight: 700, color: theme.text }}>{title}</span>
      {badge && (
        <span style={{
          marginLeft: "auto", fontSize: 11, fontWeight: 600, fontFamily: "monospace",
          padding: "2px 10px", borderRadius: 10,
          background: (badgeColor || theme.blue) + "20",
          color: badgeColor || theme.blue,
          border: `1px solid ${(badgeColor || theme.blue)}40`,
        }}>{badge}</span>
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
  const [tab, setTab]                   = useState("overview");
  const [stats, setStats]               = useState(null);
  const [analysis, setAnalysis]         = useState(null);
  const [logs, setLogs]                 = useState([]);
  const [traces, setTraces]             = useState([]);
  const [trends, setTrends]             = useState(null);
  // Pipeline Monitor data
  const [trendDetail, setTrendDetail]         = useState([]);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [metricsRaw, setMetricsRaw]           = useState([]);
  const [connected, setConnected]       = useState(false);
  const [lastUpdate, setLastUpdate]     = useState(null);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const wsRef         = useRef(null);
  const retryTimerRef = useRef(null);

  // ── Fetch all data from backend API ──────────────────────
  const fetchAll = useCallback(async () => {
    try {
      const [statsRes, analysisRes, logsRes, tracesRes, trendsRes,
             trendDetailRes, analysisHistRes, metricsRawRes] = await Promise.all([
        apiFetch("/api/stats"),
        apiFetch("/api/analysis/latest").catch(() => null),
        apiFetch("/api/logs",   { since_minutes: 60, limit: 200 }),
        apiFetch("/api/traces", { since_minutes: 60, limit: 100 }),
        apiFetch("/api/tsdb/trends/summary").catch(() => null),
        apiFetch("/api/tsdb/trends", { since_hours: 6, limit: 500 }).catch(() => null),
        apiFetch("/api/analysis",    { limit: 20 }).catch(() => null),
        apiFetch("/api/metrics",     { since_minutes: 60, limit: 500 }).catch(() => null),
      ]);

      setStats(statsRes);
      if (analysisRes) setAnalysis(normalizeAnalysis(analysisRes));
      setLogs(logsRes.logs || []);
      setTraces(tracesRes.traces || []);
      if (trendsRes) setTrends(trendsRes);
      if (trendDetailRes?.trends) setTrendDetail(trendDetailRes.trends);
      if (analysisHistRes?.analyses) setAnalysisHistory(analysisHistRes.analyses.map(normalizeAnalysis));
      if (metricsRawRes?.metrics) setMetricsRaw(metricsRawRes.metrics);
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

        ws.onopen = () => { setConnected(true); reconnectDelay = 1000; };

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

        ws.onerror = () => { ws.close(); };
      } catch (e) { setConnected(false); }
    }

    connect();
    return () => {
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      if (ws) ws.close();
    };
  }, []);

  const statusConf = statusColors[analysis?.health_status] || statusColors.UNKNOWN;

  const tabs = [
    { id: "overview", label: "Overview",         icon: "◉" },
    { id: "pipeline", label: "Pipeline Monitor", icon: "⬡" },
    { id: "analysis", label: "AI Analysis",      icon: "🤖" },
    { id: "trends",   label: "TSDB Trends",      icon: "📈" },
    { id: "logs",     label: "Logs",             icon: "≡" },
    { id: "traces",   label: "Traces",           icon: "⤳" },
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
            {tab === "overview"  && <OverviewTab  stats={stats} analysis={analysis} statusConf={statusConf} />}
            {tab === "pipeline"  && <PipelineTab  trendDetail={trendDetail} analysisHistory={analysisHistory} metricsRaw={metricsRaw} logs={logs} traces={traces} stats={stats} />}
            {tab === "analysis"  && <AnalysisTab  analysis={analysis} statusConf={statusConf} />}
            {tab === "trends"    && <TrendsTab    trends={trends} />}
            {tab === "logs"      && <LogsTab      logs={logs} connected={connected} />}
            {tab === "traces"    && <TracesTab    traces={traces} />}
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

      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        <StatCard label="Total Logs"  value={stats.total_logs   ?? "—"} icon="📝" />
        <StatCard label="Errors"      value={stats.total_errors ?? "—"} color={stats.total_errors > 0 ? theme.red : theme.green} icon="⚠" />
        <StatCard label="Traces"      value={stats.total_traces ?? "—"} icon="⤳" />
        <StatCard label="Slow Traces" value={stats.slow_traces  ?? "—"} color={stats.slow_traces > 0 ? theme.amber : theme.green} icon="🐢" />
        <StatCard label="K8s Events"  value={stats.total_events ?? "—"} icon="☸" />
      </div>

      {stats.latency_by_service?.length > 0 && (
        <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 20, marginBottom: 20 }}>
          <SectionHeader title="Latency by Service" icon="⏱" />
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
            {stats.latency_by_service.map((s, i) => (
              <div key={i} style={{ flex: 1, minWidth: 200 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: theme.cyan, marginBottom: 6 }}>{s.service}</div>
                <div style={{ display: "flex", gap: 16 }}>
                  <div>
                    <div style={{ fontSize: 10, color: theme.textDim }}>AVG</div>
                    <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "monospace" }}>{s.avg_ms ?? "—"}ms</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: theme.textDim }}>MAX</div>
                    <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "monospace", color: (s.max_ms || 0) > 500 ? theme.red : theme.text }}>{s.max_ms ?? "—"}ms</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {Array.isArray(analysis?.recommendations) && analysis.recommendations.length > 0 && (
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

// ══════════════════════════════════════════════════════════════
// ── Pipeline Monitor Tab ─────────────────────────────────────
// ══════════════════════════════════════════════════════════════

function PipelineTab({ trendDetail, analysisHistory, metricsRaw, logs, traces, stats }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <CollectorPanel trendDetail={trendDetail} />
      <BackendPanel   metricsRaw={metricsRaw} logs={logs} traces={traces} stats={stats} />
      <AgentPanel     analysisHistory={analysisHistory} />
    </div>
  );
}

// ── Data transforms ──────────────────────────────────────────

function buildTrendTimeSeries(trendDetail, queryName) {
  const records = (trendDetail || [])
    .filter(r => r.query_name === queryName)
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  return records.map(r => {
    const point = {
      time: new Date(r.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    (r.analysis?.series || []).forEach(s => {
      const svc = s.labels?.service_name || s.labels?.job || "unknown";
      point[svc] = s.latest != null ? parseFloat(s.latest.toFixed(3)) : null;
    });
    return point;
  });
}

function buildSeverityData(logs) {
  const counts = { ERROR: 0, WARN: 0, INFO: 0, DEBUG: 0 };
  (logs || []).forEach(l => { if (l.severity in counts) counts[l.severity]++; });
  return [
    { name: "ERROR", value: counts.ERROR, color: theme.red   },
    { name: "WARN",  value: counts.WARN,  color: theme.amber },
    { name: "INFO",  value: counts.INFO,  color: theme.blue  },
    { name: "DEBUG", value: counts.DEBUG, color: theme.textDim },
  ];
}

function buildLatencyData(stats) {
  return (stats?.latency_by_service || []).map(s => ({
    service:  (s.service || "").replace("-service", ""),
    "Avg ms": parseFloat((s.avg_ms || 0).toFixed(1)),
    "Max ms": parseFloat((s.max_ms || 0).toFixed(1)),
  }));
}

function buildDurationBuckets(traces) {
  const buckets = [
    { name: "0–100",    min: 0,    max: 100,      count: 0 },
    { name: "100–500",  min: 100,  max: 500,      count: 0 },
    { name: "500–1k",   min: 500,  max: 1000,     count: 0 },
    { name: "1k–2k",    min: 1000, max: 2000,     count: 0 },
    { name: ">2k",      min: 2000, max: Infinity,  count: 0 },
  ];
  (traces || []).forEach(t => {
    const d = t.duration_ms || 0;
    const b = buckets.find(bk => d >= bk.min && d < bk.max);
    if (b) b.count++;
  });
  return buckets.map(({ name, count }) => ({ name, count }));
}

function buildAgentTimeSeries(analysisHistory) {
  return (analysisHistory || [])
    .slice().reverse()
    .map(a => ({
      time:       new Date(a.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      confidence: parseFloat(((a.confidence || 0) * 100).toFixed(1)),
      anomalies:  (a.anomalies || []).length,
      roots:      (a.root_causes || []).length,
    }));
}

function buildHealthTimeline(analysisHistory) {
  const statusOrder = ["HEALTHY", "DEGRADED", "CRITICAL", "UNKNOWN"];
  return (analysisHistory || [])
    .slice().reverse()
    .map(a => ({
      time:   new Date(a.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      status: a.health_status || "UNKNOWN",
      value:  4 - statusOrder.indexOf(a.health_status || "UNKNOWN"),
      fill:   (statusColors[a.health_status] || statusColors.UNKNOWN).text,
    }));
}

// ── Mini chart card ───────────────────────────────────────────
function ChartCard({ title, subtitle, children, height = 220 }) {
  return (
    <div style={{
      background: theme.surface, border: `1px solid ${theme.border}`,
      borderRadius: 8, padding: "16px 20px",
    }}>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: theme.text }}>{title}</div>
        {subtitle && <div style={{ fontSize: 11, color: theme.textDim, marginTop: 2 }}>{subtitle}</div>}
      </div>
      <div style={{ height }}>{children}</div>
    </div>
  );
}

// ── Collector Panel ──────────────────────────────────────────
function CollectorPanel({ trendDetail }) {
  const hasData = trendDetail && trendDetail.length > 0;

  const reqRate   = buildTrendTimeSeries(trendDetail, "request_rate_1h");
  const latP99    = buildTrendTimeSeries(trendDetail, "latency_p99_1h");
  const errRate   = buildTrendTimeSeries(trendDetail, "error_rate_1h");
  const jvmHeap   = buildTrendTimeSeries(trendDetail, "jvm_heap_used_1h");
  const gcPause   = buildTrendTimeSeries(trendDetail, "jvm_gc_pause_1h");
  const reqRate24 = buildTrendTimeSeries(trendDetail, "request_rate_24h");

  // Latest snapshot summary for bar chart
  const latestSnap = {};
  (trendDetail || []).forEach(r => {
    const ts = r.timestamp;
    if (!latestSnap[r.query_name] || ts > latestSnap[r.query_name].timestamp) {
      latestSnap[r.query_name] = r;
    }
  });
  const currentValues = Object.entries(latestSnap).flatMap(([qn, r]) =>
    (r.analysis?.series || []).map(s => ({
      metric:    qn.replace(/_1h|_24h/, "").replace(/_/g, " "),
      service:   (s.labels?.service_name || s.labels?.job || "?").replace("-service", ""),
      latest:    s.latest,
      avg:       s.avg,
      trend_pct: s.trend_pct,
      direction: s.direction,
    }))
  );

  const formatBytes = v => v != null ? `${(v / 1024 / 1024).toFixed(0)}MB` : "—";
  const formatMs    = v => v != null ? `${v.toFixed(1)}ms` : "—";
  const formatRate  = v => v != null ? v.toFixed(3) : "—";

  return (
    <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8 }}>
      <PanelHeader
        title="VictoriaMetrics Collector"
        icon="📊"
        badge={hasData ? `${trendDetail.length} records · 6h window` : "No data"}
        badgeColor={hasData ? theme.cyan : theme.amber}
      />
      <div style={{ padding: 20 }}>
        {!hasData ? (
          <LoadingSpinner message="Waiting for TSDB trend data..." />
        ) : (
          <>
            {/* Service legend */}
            <div style={{ display: "flex", gap: 20, marginBottom: 20, flexWrap: "wrap" }}>
              {SERVICE_LIST.map(svc => (
                <div key={svc} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 2, background: SERVICE_COLORS[svc] }}></div>
                  <span style={{ fontSize: 12, color: theme.textMuted }}>{svc}</span>
                </div>
              ))}
              <span style={{ marginLeft: "auto", fontSize: 11, color: theme.textDim }}>Collected every 60s via PromQL range queries</span>
            </div>

            {/* 2×3 chart grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16, marginBottom: 20 }}>

              <ChartCard title="Request Rate (req/s)" subtitle="request_rate_1h — per service">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={reqRate} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <defs>
                      {SERVICE_LIST.map(svc => (
                        <linearGradient key={svc} id={`g-${svc}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor={SERVICE_COLORS[svc]} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={SERVICE_COLORS[svc]} stopOpacity={0} />
                        </linearGradient>
                      ))}
                    </defs>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                    <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    {SERVICE_LIST.map(svc => (
                      <Area key={svc} type="monotone" dataKey={svc} stroke={SERVICE_COLORS[svc]}
                        fill={`url(#g-${svc})`} strokeWidth={2} dot={false} connectNulls />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="P99 Latency (ms)" subtitle="latency_p99_1h — per service">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={latP99} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                    <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip unit="ms" />} />
                    {SERVICE_LIST.map(svc => (
                      <Line key={svc} type="monotone" dataKey={svc} stroke={SERVICE_COLORS[svc]}
                        strokeWidth={2} dot={false} connectNulls />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Error Rate" subtitle="error_rate_1h — 5xx errors per service">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={errRate} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <defs>
                      {SERVICE_LIST.map(svc => (
                        <linearGradient key={svc} id={`ge-${svc}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor={SERVICE_COLORS[svc]} stopOpacity={0.25} />
                          <stop offset="95%" stopColor={SERVICE_COLORS[svc]} stopOpacity={0} />
                        </linearGradient>
                      ))}
                    </defs>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                    <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    {SERVICE_LIST.map(svc => (
                      <Area key={svc} type="monotone" dataKey={svc} stroke={SERVICE_COLORS[svc]}
                        fill={`url(#ge-${svc})`} strokeWidth={2} dot={false} connectNulls />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="JVM Heap Used (MB)" subtitle="jvm_heap_used_1h — memory leak detection">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={jvmHeap} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <defs>
                      {SERVICE_LIST.map(svc => (
                        <linearGradient key={svc} id={`gh-${svc}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor={SERVICE_COLORS[svc]} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={SERVICE_COLORS[svc]} stopOpacity={0} />
                        </linearGradient>
                      ))}
                    </defs>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                    <YAxis tick={axisStyle} tickLine={false} axisLine={false}
                      tickFormatter={v => `${(v / 1048576).toFixed(0)}MB`} />
                    <Tooltip content={<ChartTooltip formatter={formatBytes} />} />
                    {SERVICE_LIST.map(svc => (
                      <Area key={svc} type="monotone" dataKey={svc} stroke={SERVICE_COLORS[svc]}
                        fill={`url(#gh-${svc})`} strokeWidth={2} dot={false} connectNulls />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="JVM GC Pause Rate" subtitle="jvm_gc_pause_1h — GC pressure">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={gcPause} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                    <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip unit="s" />} />
                    {SERVICE_LIST.map(svc => (
                      <Line key={svc} type="monotone" dataKey={svc} stroke={SERVICE_COLORS[svc]}
                        strokeWidth={2} dot={false} connectNulls />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Request Rate — 24h Pattern" subtitle="request_rate_24h · step 15m">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={reqRate24} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <defs>
                      {SERVICE_LIST.map(svc => (
                        <linearGradient key={svc} id={`g24-${svc}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor={SERVICE_COLORS[svc]} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={SERVICE_COLORS[svc]} stopOpacity={0} />
                        </linearGradient>
                      ))}
                    </defs>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                    <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    {SERVICE_LIST.map(svc => (
                      <Area key={svc} type="monotone" dataKey={svc} stroke={SERVICE_COLORS[svc]}
                        fill={`url(#g24-${svc})`} strokeWidth={2} dot={false} connectNulls />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>

            {/* Current values table */}
            {currentValues.length > 0 && (
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: theme.textMuted, marginBottom: 10 }}>Current Snapshot</div>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: theme.surfaceHover }}>
                        {["Metric", "Service", "Latest", "Avg", "Trend %", "Direction"].map(h => (
                          <th key={h} style={{
                            padding: "8px 12px", textAlign: "left", fontWeight: 600,
                            color: theme.textDim, fontSize: 11, textTransform: "uppercase",
                            letterSpacing: "0.04em", borderBottom: `1px solid ${theme.border}`,
                          }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {currentValues.map((row, i) => (
                        <tr key={i} style={{ borderBottom: `1px solid ${theme.border}` }}>
                          <td style={{ padding: "7px 12px", color: theme.textMuted, fontFamily: "monospace", fontSize: 11 }}>{row.metric}</td>
                          <td style={{ padding: "7px 12px", color: SERVICE_COLORS[row.service + "-service"] || theme.cyan, fontWeight: 600 }}>{row.service}</td>
                          <td style={{ padding: "7px 12px", fontFamily: "monospace", color: theme.text }}>{row.latest?.toFixed(3) ?? "—"}</td>
                          <td style={{ padding: "7px 12px", fontFamily: "monospace", color: theme.textMuted }}>{row.avg?.toFixed(3) ?? "—"}</td>
                          <td style={{ padding: "7px 12px", fontFamily: "monospace", color: (row.trend_pct || 0) > 5 ? theme.red : (row.trend_pct || 0) < -5 ? theme.green : theme.textMuted }}>
                            {row.trend_pct != null ? `${row.trend_pct > 0 ? "+" : ""}${row.trend_pct.toFixed(1)}%` : "—"}
                          </td>
                          <td style={{ padding: "7px 12px" }}>
                            <span style={{
                              fontSize: 11, fontWeight: 600,
                              color: row.direction === "increasing" ? theme.red : row.direction === "decreasing" ? theme.green : theme.textMuted,
                            }}>
                              {row.direction === "increasing" ? "▲ " : row.direction === "decreasing" ? "▼ " : "— "}
                              {row.direction || "stable"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Backend Panel ─────────────────────────────────────────────
function BackendPanel({ metricsRaw, logs, traces, stats }) {
  const severityData = buildSeverityData(logs);
  const latencyData  = buildLatencyData(stats);
  const durationData = buildDurationBuckets(traces);

  const totalLogs    = (logs   || []).length;
  const totalMetrics = (metricsRaw || []).length;
  const totalTraces  = (traces || []).length;
  const slowCount    = (traces || []).filter(t => t.is_slow).length;
  const errorCount   = (logs   || []).filter(l => l.severity === "ERROR").length;

  return (
    <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8 }}>
      <PanelHeader
        title="Backend API & Data Ingestion"
        icon="🗄"
        badge="Last 60 min"
        badgeColor={theme.blue}
      />
      <div style={{ padding: 20 }}>

        {/* Ingestion stat cards */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
          <StatCard label="Logs Ingested"    value={totalLogs}    icon="📝" subValue="last 60 min" />
          <StatCard label="Metrics Ingested" value={totalMetrics} icon="📊" subValue="last 60 min" />
          <StatCard label="Traces Ingested"  value={totalTraces}  icon="⤳" subValue="last 60 min" />
          <StatCard label="Slow Traces"      value={slowCount}    icon="🐢" color={slowCount  > 0 ? theme.amber : theme.green} subValue=">500ms" />
          <StatCard label="Error Logs"       value={errorCount}   icon="⚠"  color={errorCount > 0 ? theme.red   : theme.green} subValue="severity=ERROR" />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>

          {/* Log Severity Breakdown */}
          <ChartCard title="Log Severity Breakdown" subtitle={`${totalLogs} log entries`} height={240}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={severityData} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                <CartesianGrid {...gridProps} horizontal={false} />
                <XAxis type="number" tick={axisStyle} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" tick={{ ...axisStyle, fontSize: 11, fontFamily: "monospace" }} tickLine={false} axisLine={false} width={44} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} label={{ position: "right", fill: theme.textMuted, fontSize: 11 }}>
                  {severityData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Service Latency Comparison */}
          <ChartCard title="Service Latency" subtitle="avg vs max (ms)" height={240}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="service" tick={axisStyle} tickLine={false} />
                <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTooltip unit="ms" />} />
                <Legend wrapperStyle={{ fontSize: 11, color: theme.textMuted }} />
                <Bar dataKey="Avg ms" fill={theme.blue}  radius={[3, 3, 0, 0]} />
                <Bar dataKey="Max ms" fill={theme.amber} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Trace Duration Distribution */}
          <ChartCard title="Trace Duration Distribution" subtitle="histogram (ms buckets)" height={240}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={durationData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="name" tick={axisStyle} tickLine={false} />
                <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="count" radius={[3, 3, 0, 0]} label={{ position: "top", fill: theme.textMuted, fontSize: 10 }}>
                  {durationData.map((entry, i) => (
                    <Cell key={i} fill={i < 2 ? theme.green : i === 2 ? theme.amber : theme.red} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Metrics ingested by name */}
        {metricsRaw && metricsRaw.length > 0 && (() => {
          const bySvc = {};
          metricsRaw.forEach(m => {
            const svc = m.labels?.service_name || m.labels?.job || "other";
            bySvc[svc] = (bySvc[svc] || 0) + 1;
          });
          const data = Object.entries(bySvc).map(([name, count]) => ({ name: name.replace("-service",""), count }));
          return (
            <div style={{ marginTop: 16 }}>
              <ChartCard title="Metrics Ingested by Service" subtitle="count of metric records" height={180}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="name" tick={axisStyle} tickLine={false} />
                    <YAxis tick={axisStyle} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {data.map((_, i) => (
                        <Cell key={i} fill={[theme.blue, theme.cyan, theme.purple, theme.green][i % 4]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
          );
        })()}
      </div>
    </div>
  );
}

// ── Agent Panel ───────────────────────────────────────────────
function AgentPanel({ analysisHistory }) {
  const hasData    = analysisHistory && analysisHistory.length > 0;
  const agentSeries = buildAgentTimeSeries(analysisHistory);
  const healthLine  = buildHealthTimeline(analysisHistory);

  const latestHealth = analysisHistory?.[0];
  const healthConf   = statusColors[latestHealth?.health_status] || statusColors.UNKNOWN;

  // Anomaly severity distribution across all analyses
  const sevCounts = { critical: 0, high: 0, medium: 0, low: 0 };
  (analysisHistory || []).forEach(a => {
    (a.anomalies || []).forEach(anom => {
      if (anom.severity in sevCounts) sevCounts[anom.severity]++;
    });
  });
  const sevData = Object.entries(sevCounts)
    .map(([name, value]) => ({ name, value, color: severityColors[name] || theme.textMuted }))
    .filter(d => d.value > 0);

  // Recommendation priority breakdown
  const priCounts = { immediate: 0, short_term: 0, long_term: 0 };
  (analysisHistory || []).forEach(a => {
    (a.recommendations || []).forEach(r => {
      if (r.priority in priCounts) priCounts[r.priority]++;
    });
  });
  const priData = [
    { name: "Immediate",   value: priCounts.immediate,  color: theme.red   },
    { name: "Short-term",  value: priCounts.short_term, color: theme.amber },
    { name: "Long-term",   value: priCounts.long_term,  color: theme.blue  },
  ].filter(d => d.value > 0);

  return (
    <div style={{ background: theme.surface, border: `1px solid ${theme.border}`, borderRadius: 8 }}>
      <PanelHeader
        title="AI Agent Analysis"
        icon="🤖"
        badge={hasData ? `${analysisHistory.length} analysis cycles` : "No data"}
        badgeColor={hasData ? healthConf.text : theme.amber}
      />
      <div style={{ padding: 20 }}>
        {!hasData ? (
          <LoadingSpinner message="Waiting for AI analysis data..." />
        ) : (
          <>
            {/* Current health status */}
            {latestHealth && (
              <div style={{
                background: healthConf.bg, border: `1px solid ${healthConf.border}`,
                borderRadius: 8, padding: "14px 18px", marginBottom: 20,
                display: "flex", alignItems: "center", justifyContent: "space-between",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span style={{ fontSize: 22 }}>{healthConf.icon}</span>
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 700, color: healthConf.text }}>{latestHealth.health_status}</div>
                    <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 2 }}>{latestHealth.summary}</div>
                  </div>
                </div>
                <div style={{ textAlign: "right", fontFamily: "monospace" }}>
                  <div style={{ fontSize: 11, color: theme.textDim }}>Confidence</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: healthConf.text }}>
                    {((latestHealth.confidence || 0) * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: 16 }}>

              {/* Confidence + Anomalies over time */}
              <ChartCard title="Confidence & Anomaly Count Over Time" subtitle={`${agentSeries.length} analysis cycles`} height={240}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={agentSeries} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                    <YAxis yAxisId="conf" tick={axisStyle} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                    <YAxis yAxisId="anom" orientation="right" tick={axisStyle} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11, color: theme.textMuted }} />
                    <Line yAxisId="conf" type="monotone" dataKey="confidence" stroke={theme.green}  strokeWidth={2} dot={{ r: 3, fill: theme.green }}  name="Confidence %" />
                    <Line yAxisId="anom" type="monotone" dataKey="anomalies"  stroke={theme.red}    strokeWidth={2} dot={{ r: 3, fill: theme.red }}    name="Anomalies" />
                    <Line yAxisId="anom" type="monotone" dataKey="roots"      stroke={theme.amber}  strokeWidth={2} dot={{ r: 3, fill: theme.amber }}  name="Root Causes" strokeDasharray="4 2" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* Anomaly severity pie */}
              <ChartCard title="Anomaly Severity" subtitle="all cycles combined" height={240}>
                {sevData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="45%"
                        outerRadius={80} innerRadius={40} paddingAngle={3}
                        label={({ name, value }) => `${name}: ${value}`}
                        labelLine={false}
                      >
                        {sevData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip content={<ChartTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: theme.green, fontSize: 13 }}>
                    ✓ No anomalies detected
                  </div>
                )}
              </ChartCard>

              {/* Recommendation priority pie */}
              <ChartCard title="Recommendation Priority" subtitle="all cycles combined" height={240}>
                {priData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={priData} dataKey="value" nameKey="name" cx="50%" cy="45%"
                        outerRadius={80} innerRadius={40} paddingAngle={3}
                        label={({ name, value }) => `${value}`}
                        labelLine={false}
                      >
                        {priData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip content={<ChartTooltip />} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: theme.textDim, fontSize: 13 }}>
                    No recommendations yet
                  </div>
                )}
              </ChartCard>
            </div>

            {/* Health Status Timeline */}
            {healthLine.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <ChartCard title="Health Status Timeline" subtitle="HEALTHY=4 · DEGRADED=3 · CRITICAL=2 · UNKNOWN=1" height={160}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={healthLine} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                      <CartesianGrid {...gridProps} />
                      <XAxis dataKey="time" tick={axisStyle} tickLine={false} />
                      <YAxis tick={axisStyle} tickLine={false} axisLine={false} domain={[0, 4]}
                        tickFormatter={v => ["","UNKNOWN","CRITICAL","DEGRADED","HEALTHY"][v] || v} />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          const d = payload[0]?.payload;
                          return (
                            <div style={{ background: "#1a2332", border: `1px solid ${theme.border}`, borderRadius: 6, padding: "8px 12px", fontSize: 12 }}>
                              <div style={{ color: theme.textMuted, fontSize: 11, marginBottom: 4 }}>{label}</div>
                              <span style={{ color: d?.fill, fontWeight: 700 }}>{d?.status}</span>
                            </div>
                          );
                        }}
                      />
                      <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                        {healthLine.map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Analysis Tab ────────────────────────────────────────────
function AnalysisTab({ analysis, statusConf }) {
  if (!analysis) return <LoadingSpinner message="Waiting for first AI analysis cycle..." />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div style={{ background: statusConf.bg, border: `1px solid ${statusConf.border}`, borderRadius: 8, padding: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: 20 }}>{statusConf.icon}</span>
          <span style={{ fontSize: 18, fontWeight: 700, color: statusConf.text }}>{analysis.health_status}</span>
          <span style={{ fontSize: 13, color: theme.textDim, marginLeft: "auto", fontFamily: "monospace" }}>
            confidence: {((analysis.confidence || 0) * 100).toFixed(0)}%
          </span>
        </div>
        <p style={{ fontSize: 14, color: theme.text, lineHeight: 1.5, margin: 0 }}>{analysis.summary}</p>
      </div>

      {analysis.anomalies?.length > 0 && (
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
                <div style={{ fontSize: 11, color: theme.cyan, fontFamily: "monospace" }}>Affected: {a.affected_resources.join(", ")}</div>
              )}
              {a.evidence && <div style={{ fontSize: 11, color: theme.textDim, marginTop: 4, fontStyle: "italic" }}>{a.evidence}</div>}
            </div>
          ))}
        </div>
      )}

      {analysis.root_causes?.length > 0 && (
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

      {Array.isArray(analysis?.recommendations) && analysis.recommendations.length > 0 && (
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

      {analysis.incident_timeline?.length > 0 && (
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
            <span>{t.is_slow   ? "🔶" : "—"}</span>
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
    if (v < 1)    return v.toFixed(3);
    if (v > 1000) return v.toFixed(0);
    return v.toFixed(2);
  };

  const svc         = trend.labels?.service_name || trend.labels?.service || "unknown";
  const borderColor = type === "degrading" ? theme.red : type === "improving" ? theme.green : theme.border;
  const bgTint      = type === "degrading" ? theme.redDim + "30" : type === "improving" ? theme.greenDim + "30" : "transparent";

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
        {[
          { label: "Latest",    val: formatVal(trend.latest),   color: theme.text    },
          { label: "Average",   val: formatVal(trend.avg),      color: theme.textMuted },
          { label: "Direction", val: `${trend.direction === "increasing" ? "▲" : trend.direction === "decreasing" ? "▼" : "—"} ${trend.direction}`, color: trend.direction === "increasing" ? theme.red : trend.direction === "decreasing" ? theme.green : theme.textMuted },
          { label: "Volatility CV", val: `${(trend.volatility_cv || 0).toFixed(1)}%`, color: (trend.volatility_cv || 0) > 40 ? theme.amber : theme.textMuted },
        ].map(({ label, val, color }) => (
          <div key={label}>
            <div style={{ fontSize: 10, color: theme.textDim, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>{label}</div>
            <div style={{ fontSize: 15, fontWeight: 700, fontFamily: "monospace", color }}>{val}</div>
          </div>
        ))}
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
  const stable    = trends.stable    || [];
  const improving = trends.improving || [];

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
        <StatCard label="Total Series" value={trends.total_series || 0}  icon="📈" color={theme.text} />
        <StatCard label="Degrading"    value={degrading.length}           icon="🔴" color={degrading.length  > 0 ? theme.red  : theme.green} subValue={degrading.length > 0 ? "Requires attention" : "All clear"} />
        <StatCard label="Stable"       value={stable.length}              icon="🟢" color={theme.green} />
        <StatCard label="Improving"    value={improving.length}           icon="📉" color={theme.cyan} />
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
          Metrics ingested via Prometheus remote_write. Trends update every 60 seconds.
          For full time-series charts, see the <strong style={{ color: theme.text }}>Pipeline Monitor</strong> tab.
        </span>
      </div>
    </div>
  );
}
