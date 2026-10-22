import { memo, useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";

const formatTime = (seconds) => {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
};

function ThisRecordingInsights({ meeting, result, apiBase }) {
  const [traceData, setTraceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeClip, setActiveClip] = useState(null);

  const meetingId = meeting?.meeting_id;

  useEffect(() => {
    let active = true;
    if (!meetingId) {
      setTraceData(null);
      return;
    }

    setLoading(true);
    setError(null);
    fetch(`${apiBase}/meetings/${meetingId}/trace`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch meeting trace`);
        return res.json();
      })
      .then((json) => {
        if (active) setTraceData(json);
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
  }, [meetingId, apiBase]);

  if (!meetingId) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P3</span> Active Recording Diagnostics</div>
        <h2>This Recording Diagnostics</h2>
        <div className="empty-box">
          <p>
            <strong>No active meeting selected.</strong> Upload a meeting audio or video file in the <strong>Ask</strong> tab to inspect real-time extraction timings, timeline segmentation, and GAT graph traces.
          </p>
        </div>
      </section>
    );
  }

  if (loading) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P3</span> Active Recording Diagnostics</div>
        <h2>This Recording Diagnostics</h2>
        <div className="loading-state">
          <span className="spinner" /> Loading recording trace and vector indices…
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P3</span> Active Recording Diagnostics</div>
        <h2>This Recording Diagnostics</h2>
        <div className="error-box">
          <p><strong>Error loading recording trace:</strong> {error}</p>
        </div>
      </section>
    );
  }

  const queryTrace = result?.trace;
  const stageTimings = traceData?.stage_timings;
  const duration = traceData?.duration_s ?? 0;
  const segments = traceData?.segments ?? [];

  // Sets for highlighting in timeline
  const retrievedSet = new Set((queryTrace?.retrieval_candidates || []).map((c) => c.segment_id));
  const gatTop5Set = new Set(
    (queryTrace?.gat_candidates || [])
      .slice(0, 5)
      .map((c) => c.segment_id)
  );
  const goldSet = new Set(queryTrace?.gold_segment_ids || []);

  // Play audio snippet
  const playClip = (seg) => {
    setActiveClip({
      segment_id: seg.segment_id,
      start: seg.start_time,
      end: seg.end_time,
      text: seg.text,
      url: `${apiBase}/meetings/${meetingId}/audio?start=${seg.start_time}&end=${seg.end_time}`,
    });
  };

  // Rank shifts data
  const rankShiftData = (queryTrace?.rank_shifts || []).map((r, idx) => ({
    name: `Cand ${idx + 1}`,
    segmentId: r.segment_id,
    retrievalRank: r.retrieval_rank,
    gatRank: r.gat_rank,
    shift: r.rank_shift,
  }));

  // Score comparison data (retrieval vs GAT)
  const scoreCompData = (queryTrace?.rank_shifts || []).map((r, idx) => ({
    name: `Cand ${idx + 1}`,
    segmentId: r.segment_id,
    retrievalScore: r.retrieval_score,
    gatScore: r.gat_score,
  }));

  // Edge counts
  const edgeCounts = queryTrace?.edge_counts_by_type || {
    question_link: 0,
    shared_entity: 0,
    semantic_similarity: 0,
  };

  return (
    <section className="panel insights-panel">
      <div className="panel-kicker"><span>P3</span> Active Recording Diagnostics</div>
      <div className="insights-header-row">
        <div>
          <h2>Recording Trace: {traceData?.original_filename || meetingId.slice(0, 8)}</h2>
          <p className="muted">
            End-to-end pipeline metrics, stage timings, timeline segmentation, and GAT graph execution traces.
          </p>
        </div>
      </div>

      {/* Pipeline Stepper (a) */}
      <div className="stepper-section">
        <h3>(a) Ingestion & Preprocessing Pipeline</h3>
        <div className="stepper-grid">
          <div className="step-card">
            <div className="step-num">01</div>
            <div className="step-body">
              <span className="step-title">Audio Extraction</span>
              <span className="step-detail">ffmpeg mono 16kHz</span>
              <span className="step-timing">
                {stageTimings?.audio_extract_s != null ? `${stageTimings.audio_extract_s}s` : "not available"}
              </span>
            </div>
          </div>

          <div className="step-card">
            <div className="step-num">02</div>
            <div className="step-body">
              <span className="step-title">Whisper ASR</span>
              <span className="step-detail">{traceData?.n_segments ?? 0} segments</span>
              <span className="step-timing">
                {stageTimings?.asr_s != null ? `${stageTimings.asr_s}s` : "not available"}
              </span>
            </div>
          </div>

          <div className="step-card">
            <div className="step-num">03</div>
            <div className="step-body">
              <span className="step-title">spaCy NER</span>
              <span className="step-detail">{traceData?.n_entities ?? 0} entities</span>
              <span className="step-timing">
                {stageTimings?.ner_s != null ? `${stageTimings.ner_s}s` : "not available"}
              </span>
            </div>
          </div>

          <div className="step-card">
            <div className="step-num">04</div>
            <div className="step-body">
              <span className="step-title">Embed & Index</span>
              <span className="step-detail">{traceData?.embedding_dim}D · {traceData?.index_size} vectors</span>
              <span className="step-timing">
                {stageTimings?.embed_index_s != null ? `${stageTimings.embed_index_s}s` : "not available"}
              </span>
            </div>
          </div>
        </div>

        {/* Latency Stepper if a question has been asked */}
        {queryTrace?.stage_latency_ms && (
          <div className="query-latency-strip">
            <span className="badge">Query Execution Latency</span>
            <div className="latency-chips">
              <div className="latency-chip">
                <span>Retrieve</span>
                <strong>{queryTrace.stage_latency_ms.retrieval_ms} ms</strong>
              </div>
              <div className="latency-chip">
                <span>Graph Build</span>
                <strong>{queryTrace.stage_latency_ms.graph_ms} ms</strong>
              </div>
              <div className="latency-chip">
                <span>GAT Inference</span>
                <strong>{queryTrace.stage_latency_ms.gat_ms} ms</strong>
              </div>
              <div className="latency-chip">
                <span>LLM Gen</span>
                <strong>{queryTrace.stage_latency_ms.llm_ms} ms</strong>
              </div>
              <div className="latency-chip">
                <span>SHAP XAI</span>
                <strong>{queryTrace.stage_latency_ms.shap_ms} ms</strong>
              </div>
              <div className="latency-chip total">
                <span>Total</span>
                <strong>{queryTrace.stage_latency_ms.total_ms} ms</strong>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Timeline Strip (b) */}
      <div className="timeline-section">
        <div className="section-header-inline">
          <h3>(b) Recording Timeline & Evidence Segments</h3>
          <span className="badge">Total Duration: {formatTime(duration)}</span>
        </div>
        <p className="muted small">
          Click any segment to stream the exact audio snippet. Colored blocks indicate candidate selection and GAT reasoning.
        </p>

        {/* Timeline Legend */}
        <div className="timeline-legend">
          <span className="leg-item"><span className="swatch default" /> Transcript Segment</span>
          <span className="leg-item"><span className="swatch retrieved" /> Retrieved Top-15</span>
          <span className="leg-item"><span className="swatch gat" /> GAT Top-5 Evidence</span>
          {queryTrace?.gold_segment_ids && (
            <span className="leg-item"><span className="swatch gold" /> QMSum Gold Label</span>
          )}
        </div>

        {/* Visual Timeline Bar */}
        <div className="timeline-bar-wrapper">
          <div className="timeline-bar">
            {segments.map((seg, idx) => {
              const startPct = duration > 0 ? (seg.start_time / duration) * 100 : 0;
              const widthPct = duration > 0 ? Math.max(0.4, ((seg.end_time - seg.start_time) / duration) * 100) : 1;
              const isRet = retrievedSet.has(seg.segment_id);
              const isGat = gatTop5Set.has(seg.segment_id);
              const isGold = goldSet.has(seg.segment_id);

              let cls = "seg-tick";
              if (isGat) cls += " is-gat";
              else if (isRet) cls += " is-retrieved";
              if (isGold) cls += " is-gold";

              return (
                <button
                  key={seg.segment_id || idx}
                  className={cls}
                  style={{ left: `${startPct}%`, width: `${widthPct}%` }}
                  title={`[${formatTime(seg.start_time)} - ${formatTime(seg.end_time)}] ${seg.text?.slice(0, 80)}`}
                  onClick={() => playClip(seg)}
                  type="button"
                />
              );
            })}
          </div>
          <div className="timeline-ticks">
            <span>0:00</span>
            <span>{formatTime(duration * 0.25)}</span>
            <span>{formatTime(duration * 0.5)}</span>
            <span>{formatTime(duration * 0.75)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Active Audio Clip Player */}
        {activeClip && (
          <div className="clip-player-card">
            <div className="clip-header">
              <span className="badge">Playing Audio Snippet</span>
              <span>{formatTime(activeClip.start)} – {formatTime(activeClip.end)}</span>
              <button type="button" className="close-btn" onClick={() => setActiveClip(null)}>✕</button>
            </div>
            <p className="clip-text">"{activeClip.text}"</p>
            <audio controls autoPlay src={activeClip.url} className="audio-control" />
          </div>
        )}
      </div>

      {/* Graph Stats Card (e) */}
      {queryTrace ? (
        <div className="graph-stats-section">
          <h3>(e) Evidence Graph Structure</h3>
          <div className="graph-stats-grid">
            <div className="stat-card">
              <span className="stat-label">Graph Nodes</span>
              <span className="stat-value">{queryTrace.node_count ?? 0}</span>
              <span className="stat-sub">Question + Segments</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Total Edges</span>
              <span className="stat-value">{queryTrace.edge_count ?? 0}</span>
              <span className="stat-sub">Bidirectional relations</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Type 0: Question Link</span>
              <span className="stat-value">{edgeCounts.question_link ?? 0}</span>
              <span className="stat-sub">Question ↔ Candidate</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Type 1: Shared Entity</span>
              <span className="stat-value">{edgeCounts.shared_entity ?? 0}</span>
              <span className="stat-sub">≥1 Named entity overlap</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Type 2: Semantic Sim</span>
              <span className="stat-value">{edgeCounts.semantic_similarity ?? 0}</span>
              <span className="stat-sub">Cosine similarity &gt; threshold</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="query-prompt-box">
          <p>
            💡 <em>Ask a question in the <strong>Ask</strong> tab to generate the live evidence graph, rank shifts, and score comparison charts for this recording.</em>
          </p>
        </div>
      )}

      {/* Query Diagnostics Charts (c & d) */}
      {queryTrace && (
        <div className="charts-grid">
          {/* Rank Shifts Chart (c) */}
          <div className="chart-card">
            <div className="chart-card-header">
              <h3>(c) Rank-Shift Dynamics</h3>
              <span className="badge">Retrieval vs GAT Rank</span>
            </div>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={rankShiftData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                  <XAxis dataKey="name" tick={{ fill: "#665", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#665", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                    formatter={(val, name, props) => [
                      val,
                      name === "retrievalRank"
                        ? "Retrieval Rank"
                        : name === "gatRank"
                        ? "GAT Rank"
                        : "Rank Shift",
                    ]}
                  />
                  <Legend />
                  <Bar dataKey="retrievalRank" name="Retrieval Rank" fill="#4a5568" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="gatRank" name="GAT Rank" fill="#3182ce" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="chart-caption muted">
              Comparison of bi-encoder initial ranking vs GAT graph attention re-ranking. Lower rank number = higher relevance.
            </p>
          </div>

          {/* Scores Comparison (d) */}
          <div className="chart-card">
            <div className="chart-card-header">
              <h3>(d) Retrieval Score vs GAT Score</h3>
              <span className="badge">Candidate Evidence</span>
            </div>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={scoreCompData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                  <XAxis dataKey="name" tick={{ fill: "#665", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#665", fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                    formatter={(val, name) => [
                      val,
                      name === "retrievalScore" ? "Cosine Similarity" : "GAT Sigmoid Score",
                    ]}
                  />
                  <Legend />
                  <Bar dataKey="retrievalScore" name="Cosine Similarity" fill="#319795" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="gatScore" name="GAT Attention Score" fill="#d69e2e" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="chart-caption muted">
              Raw vector similarity (bi-encoder) compared with learned graph relational attention scores.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

export default memo(ThisRecordingInsights);
