import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";

export default function AblationInsights({ apiBase }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetch(`${apiBase}/insights/ablation`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch ablation insights`);
        return res.json();
      })
      .then((json) => {
        if (active) setData(json);
      })
      .catch((err) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [apiBase]);

  if (loading) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P4</span> Model Ablation Study</div>
        <h2>Ablation Study</h2>
        <div className="loading-state">
          <span className="spinner" /> Loading ablation metrics from data/processed/ablation.json…
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P4</span> Model Ablation Study</div>
        <h2>Ablation Study</h2>
        <div className="error-box">
          <p><strong>Error loading ablation data:</strong> {error}</p>
        </div>
      </section>
    );
  }

  if (!data || !data.available) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P4</span> Model Ablation Study</div>
        <h2>Ablation Study</h2>
        <div className="empty-box">
          <p>Ablation artifact (<code>data/processed/ablation.json</code>) is not available on disk.</p>
        </div>
      </section>
    );
  }

  const queries = data.queries || [];
  const totals = data.totals || {};

  const chartData = queries.map((q) => ({
    name: `Q${q.query_idx}`,
    fullQuery: q.query,
    goldTotal: q.total_gold,
    retrievalHits: q.retrieval_hits,
    trainedGatHits: q.trained_gat_hits,
    untrainedGatHits: q.untrained_gat_hits,
    llmContextHits: q.llm_context_hits,
    llmContextSize: q.llm_context_size,
  }));

  return (
    <section className="panel insights-panel">
      <div className="panel-kicker"><span>P4</span> Model Ablation Study</div>
      <div className="insights-header-row">
        <div>
          <h2>Ablation Analysis: Retrieval vs GAT vs Untrained Baseline</h2>
          <p className="muted">
            Evaluation across all ground-truth annotated queries in meetings with aligned labels (<code>data/processed/ablation.json</code>).
          </p>
        </div>
      </div>

      {/* Main Grouped Bar Chart */}
      <div className="chart-card full-width">
        <div className="chart-card-header">
          <h3>Ground-Truth Evidence Hits per Query</h3>
          <span className="badge">Top-5 Selections</span>
        </div>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
              <XAxis dataKey="name" tick={{ fill: "#665", fontSize: 11 }} />
              <YAxis tick={{ fill: "#665", fontSize: 11 }} allowDecimals={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                formatter={(val, name, props) => [
                  `${val} hit${val !== 1 ? "s" : ""}`,
                  name,
                ]}
                labelFormatter={(label, items) => {
                  const item = items?.[0]?.payload;
                  return item ? `${label}: ${item.fullQuery} (Total Gold: ${item.goldTotal})` : label;
                }}
              />
              <Legend />
              <Bar dataKey="retrievalHits" name="(a) Retrieval Only Top-5" fill="#457b9d" radius={[3, 3, 0, 0]} />
              <Bar dataKey="trainedGatHits" name="(b) Trained GAT Top-5" fill="#e28c3e" radius={[3, 3, 0, 0]} />
              <Bar dataKey="untrainedGatHits" name="(c) Untrained GAT (seed 42)" fill="#718096" radius={[3, 3, 0, 0]} />
              <Bar dataKey="llmContextHits" name="(d) LLM Context Set" fill="#319795" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="chart-caption muted">
          Number of ground-truth evidence utterances recovered in the top-5 ranking candidates for each query.
        </p>
      </div>

      {/* Query Detail & Totals Summary Under Chart */}
      <div className="ablation-summary-section">
        <h3>Study Scope & Aggregate Totals</h3>
        <div className="ablation-stats-grid">
          <div className="stat-card">
            <span className="stat-label">Evaluated Queries</span>
            <span className="stat-value">{data.n_queries ?? queries.length}</span>
            <span className="stat-sub">Ground-truth annotated</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">Evaluated Meetings</span>
            <span className="stat-value">{data.n_meetings ?? (data.meetings?.length || 1)}</span>
            <span className="stat-sub">{(data.meetings || []).join(", ")}</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">Total Gold Annotations</span>
            <span className="stat-value">{totals.total_gold ?? 0}</span>
            <span className="stat-sub">Across all test queries</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">Retrieval Only Hits</span>
            <span className="stat-value">{totals.retrieval_hits ?? 0}</span>
            <span className="stat-sub">Top-5 candidates</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">Trained GAT Hits</span>
            <span className="stat-value">{totals.trained_gat_hits ?? 0}</span>
            <span className="stat-sub">Top-5 candidates</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">Untrained Baseline Hits</span>
            <span className="stat-value">{totals.untrained_gat_hits ?? 0}</span>
            <span className="stat-sub">Random seed 42</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">LLM Context Set Hits</span>
            <span className="stat-value">{totals.llm_context_hits ?? 0}</span>
            <span className="stat-sub">Deduplicated pool</span>
          </div>
        </div>

        {/* Detailed Query Breakdown Table */}
        <div className="query-table-card">
          <h4>Per-Query Breakdown</h4>
          <div className="table-responsive">
            <table className="ablation-table">
              <thead>
                <tr>
                  <th>Query</th>
                  <th>Total Gold</th>
                  <th>Retrieval Top-5</th>
                  <th>Trained GAT Top-5</th>
                  <th>Untrained GAT Top-5</th>
                  <th>LLM Context Set</th>
                </tr>
              </thead>
              <tbody>
                {queries.map((q) => (
                  <tr key={q.query_idx}>
                    <td>
                      <strong>Q{q.query_idx}</strong>: <span className="muted">{q.query}</span>
                    </td>
                    <td>{q.total_gold}</td>
                    <td>{q.retrieval_hits}</td>
                    <td>{q.trained_gat_hits}</td>
                    <td>{q.untrained_gat_hits}</td>
                    <td>
                      {q.llm_context_hits} <span className="small muted">(size {q.llm_context_size})</span>
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <td><strong>TOTAL</strong></td>
                  <td><strong>{totals.total_gold}</strong></td>
                  <td><strong>{totals.retrieval_hits}</strong></td>
                  <td><strong>{totals.trained_gat_hits}</strong></td>
                  <td><strong>{totals.untrained_gat_hits}</strong></td>
                  <td><strong>{totals.llm_context_hits}</strong></td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
