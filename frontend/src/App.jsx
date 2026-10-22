import { useEffect, useRef, useState } from "react";
import TrainingInsights from "./components/TrainingInsights";
import DatasetInsights from "./components/DatasetInsights";
import ThisRecordingInsights from "./components/ThisRecordingInsights";
import AblationInsights from "./components/AblationInsights";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api/v1";
const API_ORIGIN = new URL(API_BASE).origin;

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const formatTime = (seconds) => `${Number(seconds).toFixed(1)}s`;

function Icon({ children }) {
  return <span className="icon" aria-hidden="true">{children}</span>;
}

function Status({ state }) {
  if (!state) return null;
  return (
    <div className={`status ${state.kind}`} role="status">
      {state.loading && <span className="spinner" />}
      {state.message}
    </div>
  );
}

function UploadCard({ onUploaded, onTaskChange, busy }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null);
  const [availableMeetings, setAvailableMeetings] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/meetings`)
      .then((res) => (res.ok ? res.json() : []))
      .then((list) => {
        if (Array.isArray(list)) setAvailableMeetings(list);
      })
      .catch(() => {});
  }, []);

  const loadExistingMeeting = async (mid) => {
    if (!mid) return;
    onTaskChange("Loading meeting…");
    setStatus({ kind: "working", loading: true, message: "Loading pre-processed meeting from disk…" });
    try {
      const response = await fetch(`${API_BASE}/meetings/${mid}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Failed to load meeting");
      onUploaded(data);
      setStatus({ kind: "success", message: `${data.segment_count} transcript segments ready.` });
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    } finally {
      onTaskChange("");
    }
  };

  const upload = async (event) => {
    event.preventDefault();
    if (!file) return;
    onTaskChange("Processing recording…");
    setStatus({ kind: "working", loading: true, message: "Transcribing and indexing your meeting…" });
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_BASE}/meetings/upload`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed");
      onUploaded(data);
      setStatus({ kind: "success", message: `${data.segment_count} transcript segments are ready.` });
    } catch (error) {
      setStatus({ kind: "error", message: error.message });
    } finally {
      onTaskChange("");
    }
  };

  return (
    <section className="panel upload-panel">
      <div className="panel-kicker"><span>01</span> Your source</div>
      <h2>Bring in a meeting.</h2>
      <p className="muted">Audio and video become a searchable, explainable transcript.</p>
      <form onSubmit={upload}>
        <button className="dropzone" type="button" onClick={() => inputRef.current?.click()}>
          <span className="drop-icon"><Icon>↑</Icon></span>
          <span>
            <strong>{file ? file.name : "Choose a recording"}</strong>
            <small>{file ? `${Math.ceil(file.size / 1024 / 1024)} MB selected` : "MP3, WAV, M4A, MP4 and more"}</small>
          </span>
        </button>
        <input
          ref={inputRef}
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          type="file"
          accept="audio/*,video/*"
          hidden
        />
        <button className="action-button" disabled={!file || busy} type="submit">
          {busy ? "Working…" : "Process meeting"}<Icon>→</Icon>
        </button>
      </form>

      {availableMeetings.length > 0 && (
        <div className="stored-meetings-picker">
          <label className="picker-label" htmlFor="meeting-select">
            Or load a processed meeting:
          </label>
          <select
            id="meeting-select"
            className="meeting-select"
            defaultValue=""
            disabled={busy}
            onChange={(e) => {
              if (e.target.value) loadExistingMeeting(e.target.value);
            }}
          >
            <option value="" disabled>Select a meeting ({availableMeetings.length} on disk)…</option>
            {availableMeetings.map((m) => (
              <option key={m.meeting_id} value={m.meeting_id}>
                {m.original_filename || m.meeting_id.slice(0, 8)} ({m.segment_count} segments)
              </option>
            ))}
          </select>
        </div>
      )}

      <Status state={status} />
    </section>
  );
}

function QuestionCard({ meeting, onAsk, busy }) {
  const [question, setQuestion] = useState("");

  const submit = (event) => {
    event.preventDefault();
    if (question.trim()) onAsk(question.trim());
  };

  return (
    <section className={`panel question-panel ${!meeting ? "locked" : ""}`}>
      <div className="panel-kicker"><span>02</span> Ask LISTEN</div>
      <h2>What do you want to know?</h2>
      <p className="muted">LISTEN traces the answer back through the conversation.</p>

      <form onSubmit={submit} className="question-form">
        <textarea
          disabled={!meeting || busy}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={meeting ? "What decision did the team make?" : "Process a meeting first…"}
        />
        <button
          className="round-button"
          disabled={!meeting || busy || !question.trim()}
          type="submit"
          aria-label="Ask question"
        >
          <Icon>→</Icon>
        </button>
      </form>

      {meeting && (
        <div className="meeting-status-footer">
          <div className="meeting-chip">
            <Icon>✓</Icon> {meeting.segment_count} segments indexed
          </div>
          <a href="#transcript-section" className="transcript-jump">
            Explore transcript below ↓
          </a>
        </div>
      )}
    </section>
  );
}

function TranscriptPanel({ meeting }) {
  const [isOpen, setIsOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [filterText, setFilterText] = useState("");
  const segments = meeting?.segments ?? [];
  if (!segments.length) return null;

  const filtered = filterText.trim()
    ? segments.filter((s) => s.text?.toLowerCase().includes(filterText.toLowerCase()))
    : segments;
  const visibleSegments = expanded ? filtered : filtered.slice(0, 40);

  return (
    <section className="panel transcript-section" id="transcript-section">
      <div
        className="transcript-header"
        onClick={() => setIsOpen(!isOpen)}
        role="button"
        tabIndex={0}
      >
        <div className="transcript-title-wrap">
          <div className="panel-kicker"><span>TX</span> Transcript</div>
          <h2>Read the conversation.</h2>
          <span className="badge">{segments.length} segments</span>
        </div>
        <button type="button" className="outline-button transcript-toggle-btn">
          {isOpen ? "Hide transcript ▲" : "View transcript ▼"}
        </button>
      </div>

      {isOpen && (
        <div className="transcript-body">
          <p className="muted">The transcript below is the same ground truth LISTEN uses to retrieve evidence.</p>
          <div className="transcript-search-bar">
            <input
              type="text"
              placeholder="Search transcript text..."
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
          </div>
          <div className="transcript-list">
            {visibleSegments.map((segment, index) => (
              <article className="transcript-turn" key={segment.segment_id ?? `${segment.start_time}-${index}`}>
                <div className="transcript-meta">
                  <span>{formatTime(segment.start_time)} - {formatTime(segment.end_time)}</span>
                  {segment.speaker && <span className="speaker-chip">{segment.speaker}</span>}
                  <span>#{index + 1}</span>
                </div>
                <p>{segment.text || "[No speech recognized]"}</p>
                {segment.entities?.length > 0 && (
                  <div className="entity-list">
                    {segment.entities.map((entity, entityIndex) => (
                      <span key={`${entity.text ?? entity}-${entityIndex}`}>{entity.text ?? entity}</span>
                    ))}
                  </div>
                )}
              </article>
            ))}
            {filtered.length === 0 && (
              <div className="empty-transcript">No segments match your search.</div>
            )}
          </div>
          {filtered.length > 40 && (
            <button
              type="button"
              className="outline-button transcript-toggle"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? "Show less transcript" : `Show all ${filtered.length} segments`}
              <Icon>{expanded ? "▲" : "▼"}</Icon>
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function StaticEvidenceGraph({ graph }) {
  const entries = Object.entries(graph?.node_scores ?? {});
  const top = new Set(graph?.top_evidence_ids ?? []);
  if (!entries.length) return <div className="empty-graph">No evidence nodes returned for this question.</div>;
  const max = Math.max(...entries.map(([, score]) => Math.abs(score)), 0.001);
  return (
    <div className="graph" aria-label="Evidence graph">
      <svg viewBox="0 0 700 250" preserveAspectRatio="none" aria-hidden="true">
        {(graph?.edge_scores ? Object.keys(graph.edge_scores) : []).slice(0, 12).map((key, index) => (
          <line
            key={key}
            x1={`${12 + (index % 5) * 19}%`}
            y1={`${25 + (index % 3) * 22}%`}
            x2={`${30 + ((index + 2) % 5) * 15}%`}
            y2={`${38 + ((index + 1) % 3) * 22}%`}
          />
        ))}
      </svg>
      {entries.slice(0, 12).map(([id, score], index) => {
        const x = 8 + ((index * 29) % 80);
        const y = 13 + ((index * 37) % 62);
        const size = 42 + clamp(Math.abs(score) / max, 0, 1) * 26;
        return (
          <div
            key={id}
            className={`graph-node ${top.has(id) ? "top" : ""}`}
            style={{ left: `${x}%`, top: `${y}%`, width: size, height: size }}
            title={`${id}: ${(score * 100).toFixed(1)}%`}
          >
            <span>{top.has(id) ? "★" : id.slice(-2)}</span>
          </div>
        );
      })}
    </div>
  );
}

function EvidenceGraph({ graph }) {
  const containerRef = useRef(null);
  const entries = Object.entries(graph?.node_scores ?? {});

  useEffect(() => {
    let network;
    let disposed = false;
    const renderNetwork = () => {
      if (disposed || !containerRef.current || !window.vis) return;
      const topIds = new Set(graph?.top_evidence_ids ?? []);
      const nodeIds = new Set(entries.map(([id]) => id));
      const nodes = entries.map(([id, score]) => ({
        id,
        label: id.length > 20 ? id.slice(-8) : id,
        value: Math.max(Math.abs(score) * 100, 12),
        color: topIds.has(id) ? "#ef4444" : "#3b82f6",
        title: `Score: ${(score * 100).toFixed(1)}%\nID: ${id}`,
      }));
      const edges = Object.entries(graph?.edge_scores ?? {}).flatMap(([key, score]) => {
        const [from, to] = key.split("::");
        if (!nodeIds.has(from) || !nodeIds.has(to)) return [];
        return [
          {
            from,
            to,
            value: Math.max(Math.abs(score) * 20, 1),
            title: `Edge weight: ${(score * 100).toFixed(1)}%`,
            color: { color: "rgba(16, 185, 129, 0.4)", highlight: "rgba(16, 185, 129, 0.7)" },
          },
        ];
      });
      network = new window.vis.Network(
        containerRef.current,
        { nodes: new window.vis.DataSet(nodes), edges: new window.vis.DataSet(edges) },
        {
          nodes: { shape: "dot", font: { color: "#ffffff", size: 12 } },
          edges: { smooth: { type: "continuous" } },
          physics: {
            stabilization: { iterations: 100 },
            barnesHut: { gravitationalConstant: -8000, springLength: 100 },
          },
          interaction: { hover: true, tooltipDelay: 100 },
        }
      );
    };
    if (window.vis) renderNetwork();
    else {
      const existing = document.querySelector("script[data-listen-vis-network]");
      const script = existing ?? document.createElement("script");
      if (!existing) {
        script.src = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js";
        script.async = true;
        script.dataset.listenVisNetwork = "true";
        document.head.appendChild(script);
      }
      script.addEventListener("load", renderNetwork, { once: true });
    }
    return () => {
      disposed = true;
      network?.destroy();
    };
  }, [graph]);

  if (!entries.length) return <div className="empty-graph">No evidence nodes returned for this question.</div>;
  return <div ref={containerRef} className="evidence-network" aria-label="Interactive evidence graph" />;
}

function EvidenceText({ words, fallback }) {
  const maximum = Math.max(...words.map((word) => Math.abs(word.score)), 0.001);
  if (!words.length) return <p>{fallback || "Transcript text is unavailable."}</p>;
  return (
    <p>
      {words.map((word, index) => (
        <mark
          key={`${word.word}-${index}`}
          style={{ "--strength": clamp(Math.max(word.score, 0) / maximum, 0, 1) }}
          title={`SHAP score: ${word.score.toFixed(3)}`}
        >
          {word.word}{" "}
        </mark>
      ))}
    </p>
  );
}

function Results({ result, onCounterfactual, counterfactual, busy }) {
  const [copied, setCopied] = useState(false);
  const topIds = result?.gat_output?.top_evidence_ids ?? [];
  const refs = new Map((result?.audio_refs ?? []).map((ref) => [ref.segment_id, ref]));
  const supported = result.answer_supported !== false;

  useEffect(() => {
    document.body.classList.toggle("answer-unsupported", !supported);
    return () => document.body.classList.remove("answer-unsupported");
  }, [supported]);

  const copyAnswer = () => {
    if (!result?.answer) return;
    navigator.clipboard.writeText(result.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <section className="answer panel" id="answer-section">
        <div className="answer-header">
          <div className="panel-kicker">
            <span className="answer-num">03</span>
            <span className="answer-title">Synthesized Answer</span>
          </div>
          <div className="answer-header-actions">
            <span className="evidence-pill">
              <span className="evidence-dot" />
              {topIds.length ? `${topIds.length} Evidence Sources` : "Ground Truth"}
            </span>
            <button
              type="button"
              className="copy-btn"
              onClick={copyAnswer}
              title="Copy answer to clipboard"
            >
              {copied ? "✓ Copied" : "⎘ Copy"}
            </button>
          </div>
        </div>

        <div className="answer-copy">
          <div className="answer-star" aria-hidden="true">✦</div>
          <p className="answer-text">{result.answer}</p>
        </div>

        <div className="answer-footer">
          <div className="answer-badges">
            <span className="answer-tag">GAT Attention Graph</span>
            <span className="answer-tag">SHAP Token Attributed</span>
          </div>
          <a href="#evidence-details" className="answer-jump-link">
            Inspect Evidence Trail ↓
          </a>
        </div>
      </section>

      {supported && (
        <div id="evidence-details">
          <section className="results-grid">
            <section className="panel graph-panel">
              <div className="section-heading">
                <div>
                  <div className="panel-kicker"><span>GAT</span> Evidence path</div>
                  <h2>How LISTEN connected it.</h2>
                </div>
                <span className="badge">{topIds.length} key segments</span>
              </div>
              <EvidenceGraph graph={result.gat_output} />
              <p className="graph-note">
                <span className="dot top-dot" /> Top evidence <span className="dot" /> Supporting context
              </p>
            </section>

            <section className="panel validation-panel">
              <div className="panel-kicker"><span>CF</span> Counterfactual check</div>
              <h2>Would the answer hold?</h2>
              <p className="muted">Remove the strongest evidence node and see whether the conclusion changes.</p>
              <button
                className="outline-button"
                onClick={onCounterfactual}
                disabled={!topIds.length || busy}
              >
                {busy ? "Checking…" : "Test top evidence"}<Icon>→</Icon>
              </button>
              {counterfactual && (
                <div className={`counterfactual ${counterfactual.answer_changed ? "changed" : "stable"}`}>
                  <strong>{counterfactual.answer_changed ? "The answer changed." : "The answer held."}</strong>
                  <span>Node {counterfactual.removed_node_id.slice(-8)} was removed.</span>
                  <p>{counterfactual.counterfactual_answer}</p>
                </div>
              )}
            </section>
          </section>

          <section className="evidence-section">
            <div className="section-heading">
              <div>
                <div className="panel-kicker"><span>SHAP</span> Word-level evidence</div>
                <h2>See what mattered.</h2>
              </div>
              <p className="muted">Brighter words carried more retrieval influence.</p>
            </div>
            <div className="evidence-list">
              {topIds.map((id, index) => {
                const ref = refs.get(id);
                const words = result.shap_highlights?.[id] ?? [];
                const score = result.gat_output?.node_scores?.[id] ?? 0;
                return (
                  <article className="evidence-card" key={id}>
                    <div className="evidence-number">0{index + 1}</div>
                    <div className="evidence-body">
                      <div className="evidence-meta">
                        <span>{formatTime(ref?.start_time ?? 0)} – {formatTime(ref?.end_time ?? 0)}</span>
                        <span>GAT {Math.round(score * 100)}%</span>
                      </div>
                      <EvidenceText words={words} fallback={ref?.text} />
                      {ref && <audio controls preload="metadata" src={`${API_ORIGIN}${ref.url}`} />}
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        </div>
      )}
    </>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("ask");
  const [visitedTabs, setVisitedTabs] = useState(() => new Set(["ask"]));
  const [meeting, setMeeting] = useState(null);
  const [result, setResult] = useState(null);
  const [counterfactual, setCounterfactual] = useState(null);
  const [state, setState] = useState({ mode: null, message: "" });
  const [backgroundTask, setBackgroundTask] = useState("");
  const busy = state.mode !== null || backgroundTask !== "";
  const operationMessage = backgroundTask
    || (state.mode === "query" ? state.message : state.mode === "counterfactual" ? "Validating evidence…" : "");

  const openTab = (tab) => {
    setVisitedTabs((visited) => {
      if (visited.has(tab)) return visited;
      return new Set(visited).add(tab);
    });
    setActiveTab(tab);
  };

  const ask = async (question) => {
    setState({ mode: "query", message: "Following the evidence trail…" });
    setResult(null);
    setCounterfactual(null);
    try {
      const response = await fetch(`${API_BASE}/query/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ meeting_id: meeting.meeting_id, question }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Question failed");
      setResult({ ...data, question });
    } catch (error) {
      alert(error.message);
    } finally {
      setState({ mode: null, message: "" });
    }
  };

  const validate = async () => {
    const node = result?.gat_output?.top_evidence_ids?.[0];
    if (!node) return;
    setState({ mode: "counterfactual", message: "" });
    try {
      const response = await fetch(`${API_BASE}/counterfactual/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          meeting_id: meeting.meeting_id,
          question: result.question,
          node_id_to_remove: node,
          original_answer: result.answer,
          original_evidence_graph: result.gat_output,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Counterfactual validation failed");
      setCounterfactual(data);
    } catch (error) {
      alert(error.message);
    } finally {
      setState({ mode: null, message: "" });
    }
  };

  return (
    <main>
      <header className="topbar">
        <div className="topbar-left">
          <a
            className="brand"
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              openTab("ask");
            }}
          >
            LI<span>ST</span>EN<span className="brand-dot">.</span>
          </a>
        </div>

        <nav className="topbar-nav" aria-label="Main Navigation">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === "ask" ? "active" : ""}`}
            onClick={() => openTab("ask")}
          >
            Ask
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === "dataset" ? "active" : ""}`}
            onClick={() => openTab("dataset")}
          >
            Dataset
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === "training" ? "active" : ""}`}
            onClick={() => openTab("training")}
          >
            Training
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === "recording" ? "active" : ""}`}
            onClick={() => openTab("recording")}
          >
            This Recording {meeting ? <span className="tab-dot" /> : ""}
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === "ablation" ? "active" : ""}`}
            onClick={() => openTab("ablation")}
          >
            Ablation
          </button>
        </nav>

        <div className="topbar-right" role="status">
          <span className="live-dot" /> {operationMessage || "Evidence-first meeting intelligence"}
        </div>
      </header>

      {/* TAB 1: ASK (EXISTING WORKFLOW) */}
      {visitedTabs.has("ask") && (
        <div className="tab-pane-fade" hidden={activeTab !== "ask"}>
          <div className="hero" id="top">
            <div>
              <p className="eyebrow">Layered interpretable speech-to-text network</p>
              <h1>Hear the <em>why</em><br />behind every answer.</h1>
              <p className="hero-copy">Upload a conversation. Ask anything. LISTEN shows the evidence, word by word.</p>
            </div>
            <div className="scribble-card">
              <span>listening...</span>
              <div className="wave">⌁⌁⌁⌁⌁</div>
              <small>Audio → Evidence → Answer</small>
            </div>
          </div>

          <div className="workspace">
            <UploadCard onUploaded={setMeeting} onTaskChange={setBackgroundTask} busy={busy} />
            <QuestionCard meeting={meeting} onAsk={ask} busy={busy} />
          </div>

          {meeting && <TranscriptPanel meeting={meeting} />}

          {state.mode === "query" && (
            <div className="loading-line">
              <span />{state.message}
            </div>
          )}

          {result && (
            <Results
              result={result}
              onCounterfactual={validate}
              counterfactual={counterfactual}
              busy={busy}
            />
          )}
        </div>
      )}

      {/* TAB 2: DATASET */}
      {visitedTabs.has("dataset") && (
        <div className="tab-pane-fade" hidden={activeTab !== "dataset"}>
          <DatasetInsights apiBase={API_BASE} />
        </div>
      )}

      {/* TAB 3: TRAINING */}
      {visitedTabs.has("training") && (
        <div className="tab-pane-fade" hidden={activeTab !== "training"}>
          <TrainingInsights apiBase={API_BASE} />
        </div>
      )}

      {/* TAB 4: THIS RECORDING */}
      {visitedTabs.has("recording") && (
        <div className="tab-pane-fade" hidden={activeTab !== "recording"}>
          <ThisRecordingInsights meeting={meeting} result={result} apiBase={API_BASE} />
        </div>
      )}

      {/* TAB 5: ABLATION */}
      {visitedTabs.has("ablation") && (
        <div className="tab-pane-fade" hidden={activeTab !== "ablation"}>
          <AblationInsights apiBase={API_BASE} />
        </div>
      )}

      <footer>
        LISTEN makes every answer inspectable. <span>Built for conversations that matter.</span>
      </footer>
    </main>
  );
}
