import { memo, useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Cell,
} from "recharts";

function DatasetInsights({ apiBase }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetch(`${apiBase}/insights/dataset`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch dataset insights`);
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
        <div className="panel-kicker"><span>P2</span> Real Dataset Alignment</div>
        <h2>Dataset & Preprocessing</h2>
        <div className="loading-state">
          <span className="spinner" /> Loading real dataset artifacts from disk…
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P2</span> Real Dataset Alignment</div>
        <h2>Dataset & Preprocessing</h2>
        <div className="error-box">
          <p><strong>Error loading dataset data:</strong> {error}</p>
        </div>
      </section>
    );
  }

  if (!data || !data.available) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P2</span> Real Dataset Alignment</div>
        <h2>Dataset & Preprocessing</h2>
        <div className="empty-box">
          <p>Dataset artifacts (<code>*_aligned_labels.json</code>) are not available on disk.</p>
        </div>
      </section>
    );
  }

  const totals = data.totals || {};
  const meetings = data.meetings || [];

  // Funnel data
  const funnelData = [
    { stage: "QMSum Raw Turns", count: totals.total_qmsum_turns ?? 0, fill: "#4a5568" },
    { stage: "Kept After Alignment", count: totals.total_kept_turns ?? 0, fill: "#319795" },
  ];

  // Similarity histogram data
  const simHistData = totals.similarity_histogram || [];

  // Segments per meeting data
  const meetingData = meetings.map((m) => ({
    id: m.id,
    segments: m.segment_count,
    durationMin: Number((m.duration_s / 60).toFixed(1)),
    kept: m.kept_alignment_count,
  }));

  // Gold evidence per query across meetings
  const allQueries = [];
  meetings.forEach((m) => {
    (m.queries || []).forEach((q, idx) => {
      allQueries.push({
        name: `Q${idx + 1} (${m.id})`,
        fullQuery: q.query,
        count: q.gold_evidence_count,
      });
    });
  });

  // Top entities data
  const topEntitiesData = (totals.top_entities || []).map((e) => ({
    entity: e.entity,
    count: e.count,
  }));

  return (
    <section className="panel insights-panel">
      <div className="panel-kicker"><span>P2</span> Real Dataset Alignment</div>
      <div className="insights-header-row">
        <div>
          <h2>Real Dataset Artifacts & Alignment</h2>
          <p className="muted">
            Computed on-the-fly from <code>data/processed/</code> and <code>data/qmsum_product/</code>. Zero synthetic data.
          </p>
        </div>
      </div>

      {/* 3 Core Authenticity Cards */}
      <div className="authenticity-grid">
        <div className="auth-card">
          <div className="auth-icon">🎙️</div>
          <div className="auth-content">
            <span className="auth-title">AMI Real Audio</span>
            <p>Real-world multi-party meeting recordings with overlapping speech and authentic acoustic conditions.</p>
          </div>
        </div>

        <div className="auth-card">
          <div className="auth-icon">📋</div>
          <div className="auth-content">
            <span className="auth-title">QMSum Human Questions</span>
            <p>Complex natural-language queries and grounded evidence spans authored by human annotators.</p>
          </div>
        </div>

        <div className="auth-card">
          <div className="auth-icon">🛡️</div>
          <div className="auth-content">
            <span className="auth-title">No Synthetic Data</span>
            <p>Strictly zero LLM-generated fake dialogues or artificial transcripts in both training and evaluation.</p>
          </div>
        </div>
      </div>

      {/* Overview Stat Badges */}
      <div className="summary-strip">
        <div className="summary-badge">
          <span>QMSum Turns</span>
          <strong>{totals.total_qmsum_turns ?? "not available"}</strong>
        </div>
        <div className="summary-badge">
          <span>Kept Aligned Turns</span>
          <strong>{totals.total_kept_turns ?? "not available"}</strong>
        </div>
        <div className="summary-badge">
          <span>Filtered Ratio</span>
          <strong>{totals.percent_filtered != null ? `${totals.percent_filtered}%` : "not available"}</strong>
        </div>
        <div className="summary-badge">
          <span>Mean Similarity</span>
          <strong>{totals.mean_similarity != null ? Number(totals.mean_similarity.toFixed(4)) : "not available"}</strong>
        </div>
        <div className="summary-badge">
          <span>Confidence Threshold</span>
          <strong>{totals.threshold_value ?? 0.4}</strong>
        </div>
        <div className="summary-badge">
          <span>Train / Val Split</span>
          <strong>
            {totals.train_val_split ? `${totals.train_val_split.train} train / ${totals.train_val_split.val} val` : "not available"}
          </strong>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Funnel: Turns -> Kept */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Turn Filtering Funnel</h3>
            <span className="badge">{totals.percent_filtered}% Filtered</span>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={funnelData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                <XAxis dataKey="stage" tick={{ fill: "#665", fontSize: 11 }} />
                <YAxis tick={{ fill: "#665", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                />
                <Bar dataKey="count" name="Turn Count" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {funnelData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption muted">
            Speech-to-text alignment filters low-confidence audio turns to ensure high precision evidence graphs.
          </p>
        </div>

        {/* Similarity Histogram */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Alignment Similarity Distribution</h3>
            <span className="badge">Threshold = {totals.threshold_value ?? 0.4}</span>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={simHistData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                <XAxis dataKey="range" tick={{ fill: "#665", fontSize: 10 }} />
                <YAxis tick={{ fill: "#665", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                />
                <ReferenceLine
                  x="0.40 - 0.50"
                  stroke="#c53030"
                  strokeDasharray="4 4"
                  label={{ value: "Cutoff", fill: "#c53030", fontSize: 10 }}
                />
                <Bar dataKey="count" name="Matches" fill="#2b6cb0" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption muted">
            Semantic cosine similarity of transcript segments aligned with QMSum ground-truth turns.
          </p>
        </div>

        {/* Segments per Meeting */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Segments & Duration per Meeting</h3>
            <span className="badge">{meetings.length} Meeting(s)</span>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={meetingData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                <XAxis dataKey="id" tick={{ fill: "#665", fontSize: 11 }} />
                <YAxis tick={{ fill: "#665", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                  formatter={(val, name) => [val, name === "segments" ? "Segments" : name === "durationMin" ? "Duration (min)" : "Kept Turns"]}
                />
                <Bar dataKey="segments" name="Total Segments" fill="#d69e2e" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="kept" name="Kept Turns" fill="#319795" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption muted">
            Transcribed meeting segments vs aligned labeled turns.
          </p>
        </div>

        {/* Gold Evidence per Query (Log scale) */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Gold Evidence per Query (Log Scale)</h3>
            <span className="badge">{allQueries.length} Queries</span>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={allQueries} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                <XAxis dataKey="name" tick={{ fill: "#665", fontSize: 10 }} />
                <YAxis
                  scale="log"
                  domain={[1, "auto"]}
                  allowDataOverflow
                  tick={{ fill: "#665", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                  formatter={(val, name, props) => [
                    `${val} gold segment${val > 1 ? "s" : ""}`,
                    props.payload.fullQuery.slice(0, 50) + "...",
                  ]}
                />
                <Bar dataKey="count" name="Gold Segments" fill="#805ad5" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption muted">
            Number of ground-truth evidence utterances annotated per query (ranging from focused questions to full-meeting summaries).
          </p>
        </div>

        {/* Top Entities Bar */}
        <div className="chart-card full-width">
          <div className="chart-card-header">
            <h3>Top Extracted Named Entities</h3>
            <span className="badge">spaCy NER (en_core_web_sm)</span>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={topEntitiesData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                <XAxis dataKey="entity" tick={{ fill: "#665", fontSize: 11 }} />
                <YAxis tick={{ fill: "#665", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                />
                <Bar dataKey="count" name="Occurrences" fill="#dd6b20" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption muted">
            Most frequent entities used to form Type-1 entity-co-occurrence edges in the evidence graph.
          </p>
        </div>
      </div>
    </section>
  );
}

export default memo(DatasetInsights);
