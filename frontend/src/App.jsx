import { useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api/v1";
const API_ORIGIN = new URL(API_BASE).origin;

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const formatTime = (seconds) => `${Number(seconds).toFixed(1)}s`;
function Icon({ children }) {
  return <span className="icon" aria-hidden="true">{children}</span>;
}

function Status({ state }) {
  if (!state) return null;
  return <div className={`status ${state.kind}`} role="status">{state.loading && <span className="spinner" />}{state.message}</div>;
}

function UploadCard({ onUploaded, busy }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null);

  const upload = async (event) => {
    event.preventDefault();
    if (!file) return;
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
    }
  };

  return <section className="panel upload-panel">
    <div className="panel-kicker"><span>01</span> Your source</div>
    <h2>Bring in a meeting.</h2>
    <p className="muted">Audio and video become a searchable, explainable transcript.</p>
    <form onSubmit={upload}>
      <button className="dropzone" type="button" onClick={() => inputRef.current?.click()}>
        <span className="drop-icon"><Icon>↑</Icon></span>
        <span><strong>{file ? file.name : "Choose a recording"}</strong><small>{file ? `${Math.ceil(file.size / 1024 / 1024)} MB selected` : "MP3, WAV, M4A, MP4 and more"}</small></span>
      </button>
      <input ref={inputRef} onChange={(event) => setFile(event.target.files?.[0] ?? null)} type="file" accept="audio/*,video/*" hidden />
      <button className="action-button" disabled={!file || busy} type="submit">{busy ? "Working…" : "Process meeting"}<Icon>→</Icon></button>
    </form>
    <Status state={status} />
  </section>;
}

function QuestionCard({ meeting, onAsk, busy }) {
  const [question, setQuestion] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  useEffect(() => {
    let active = true;
    if (!meeting?.meeting_id) { setSuggestions([]); return undefined; }
    setSuggestions([]); setSuggestionsLoading(true);
    fetch(`${API_BASE}/meetings/${meeting.meeting_id}/suggested-questions`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not generate suggested questions.");
        return Array.isArray(data.questions) ? data.questions : [];
      })
      .then((questions) => { if (active) setSuggestions(questions); })
      .catch(() => { if (active) setSuggestions([]); })
      .finally(() => { if (active) setSuggestionsLoading(false); });
    return () => { active = false; };
  }, [meeting?.meeting_id]);
  const submit = (event) => { event.preventDefault(); if (question.trim()) onAsk(question.trim()); };
  const chooseSuggestion = (suggestion) => { setQuestion(suggestion); onAsk(suggestion); };
  return <section className={`panel question-panel ${!meeting ? "locked" : ""}`}>
    <div className="panel-kicker"><span>02</span> Ask LISTEN</div>
    <h2>What do you want to know?</h2>
    <p className="muted">LISTEN traces the answer back through the conversation.</p>
    {meeting && (suggestionsLoading || suggestions.length > 0) && <div className="question-suggestions"><p>{suggestionsLoading ? "Finding grounded meeting questions..." : "Suggested from this transcript"}</p>{suggestions.length > 0 && <div>{suggestions.map((suggestion) => <button key={suggestion} type="button" disabled={busy} onClick={() => chooseSuggestion(suggestion)}>{suggestion}</button>)}</div>}</div>}
    <form onSubmit={submit} className="question-form">
      <textarea disabled={!meeting || busy} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={meeting ? "What decision did the team make?" : "Process a meeting first…"} />
      <button className="round-button" disabled={!meeting || busy || !question.trim()} type="submit" aria-label="Ask question"><Icon>→</Icon></button>
    </form>
    {meeting && <div className="meeting-chip"><Icon>✓</Icon> {meeting.segment_count} segments indexed</div>}
    {meeting && <TranscriptPanel meeting={meeting} />}
  </section>;
}

function TranscriptPanel({ meeting }) {
  const [expanded, setExpanded] = useState(false);
  const segments = meeting?.segments ?? [];
  if (!segments.length) return null;
  const visibleSegments = expanded ? segments : segments.slice(0, 80);
  return <section className="transcript-panel"><div className="section-heading"><div><div className="panel-kicker"><span>TX</span> Transcript</div><h2>Read the conversation.</h2></div><span className="badge">{segments.length} segments</span></div><p className="muted">The transcript below is the same source LISTEN uses to retrieve evidence.</p><div className="transcript-list">{visibleSegments.map((segment, index) => <article className="transcript-turn" key={segment.segment_id ?? `${segment.start_time}-${index}`}><div className="transcript-meta"><span>{formatTime(segment.start_time)} - {formatTime(segment.end_time)}</span><span>#{index + 1}</span></div><p>{segment.text || "[No speech recognized]"}</p>{segment.entities?.length > 0 && <div className="entity-list">{segment.entities.map((entity, entityIndex) => <span key={`${entity.text ?? entity}-${entityIndex}`}>{entity.text ?? entity}</span>)}</div>}</article>)}</div>{segments.length > 80 && <button type="button" className="outline-button transcript-toggle" onClick={() => setExpanded(!expanded)}>{expanded ? "Show less transcript" : `Show all ${segments.length} segments`}<Icon>v</Icon></button>}</section>;
}

function StaticEvidenceGraph({ graph }) {
  const entries = Object.entries(graph?.node_scores ?? {});
  const top = new Set(graph?.top_evidence_ids ?? []);
  if (!entries.length) return <div className="empty-graph">No evidence nodes returned for this question.</div>;
  const max = Math.max(...entries.map(([, score]) => Math.abs(score)), 0.001);
  return <div className="graph" aria-label="Evidence graph">
    <svg viewBox="0 0 700 250" preserveAspectRatio="none" aria-hidden="true">
      {(graph?.edge_scores ? Object.keys(graph.edge_scores) : []).slice(0, 12).map((key, index) => <line key={key} x1={`${12 + (index % 5) * 19}%`} y1={`${25 + (index % 3) * 22}%`} x2={`${30 + ((index + 2) % 5) * 15}%`} y2={`${38 + ((index + 1) % 3) * 22}%`} />)}
    </svg>
    {entries.slice(0, 12).map(([id, score], index) => {
      const x = 8 + ((index * 29) % 80);
      const y = 13 + ((index * 37) % 62);
      const size = 42 + clamp(Math.abs(score) / max, 0, 1) * 26;
      return <div key={id} className={`graph-node ${top.has(id) ? "top" : ""}`} style={{ left: `${x}%`, top: `${y}%`, width: size, height: size }} title={`${id}: ${(score * 100).toFixed(1)}%`}><span>{top.has(id) ? "★" : id.slice(-2)}</span></div>;
    })}
  </div>;
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
        title: `Score: ${(score * 100).toFixed(1)}%\\nID: ${id}`,
      }));
      const edges = Object.entries(graph?.edge_scores ?? {}).flatMap(([key, score]) => {
        const [from, to] = key.split("::");
        if (!nodeIds.has(from) || !nodeIds.has(to)) return [];
        return [{ from, to, value: Math.max(Math.abs(score) * 20, 1), title: `Edge weight: ${(score * 100).toFixed(1)}%`, color: { color: "rgba(16, 185, 129, 0.4)", highlight: "rgba(16, 185, 129, 0.7)" } }];
      });
      network = new window.vis.Network(containerRef.current, { nodes: new window.vis.DataSet(nodes), edges: new window.vis.DataSet(edges) }, {
        nodes: { shape: "dot", font: { color: "#ffffff", size: 12 } },
        edges: { smooth: { type: "continuous" } },
        physics: { stabilization: { iterations: 100 }, barnesHut: { gravitationalConstant: -8000, springLength: 100 } },
        interaction: { hover: true, tooltipDelay: 100 },
      });
    };
    if (window.vis) renderNetwork();
    else {
      const existing = document.querySelector("script[data-listen-vis-network]");
      const script = existing ?? document.createElement("script");
      if (!existing) { script.src = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"; script.async = true; script.dataset.listenVisNetwork = "true"; document.head.appendChild(script); }
      script.addEventListener("load", renderNetwork, { once: true });
    }
    return () => { disposed = true; network?.destroy(); };
  }, [graph]);

  if (!entries.length) return <div className="empty-graph">No evidence nodes returned for this question.</div>;
  return <div ref={containerRef} className="evidence-network" aria-label="Interactive evidence graph" />;
}

function EvidenceText({ words, fallback }) {
  const maximum = Math.max(...words.map((word) => Math.abs(word.score)), 0.001);
  if (!words.length) return <p>{fallback || "Transcript text is unavailable."}</p>;
  return <p>{words.map((word, index) => <mark key={`${word.word}-${index}`} style={{ "--strength": clamp(Math.max(word.score, 0) / maximum, 0, 1) }} title={`SHAP score: ${word.score.toFixed(3)}`}>{word.word} </mark>)}</p>;
}

function Results({ result, onCounterfactual, counterfactual, busy }) {
  const topIds = result?.gat_output?.top_evidence_ids ?? [];
  const refs = new Map((result?.audio_refs ?? []).map((ref) => [ref.segment_id, ref]));
  const supported = result.answer_supported !== false;
  useEffect(() => {
    document.body.classList.toggle("answer-unsupported", !supported);
    return () => document.body.classList.remove("answer-unsupported");
  }, [supported]);
  return <>
    <section className="answer panel">
      <div className="panel-kicker"><span>03</span> Answer with receipts</div>
      <div className="answer-copy"><div className="answer-star">✦</div><p>{result.answer}</p></div>
    </section>
    {supported && <>
    <section className="results-grid">
      <section className="panel graph-panel"><div className="section-heading"><div><div className="panel-kicker"><span>GAT</span> Evidence path</div><h2>How LISTEN connected it.</h2></div><span className="badge">{topIds.length} key segments</span></div><EvidenceGraph graph={result.gat_output} /><p className="graph-note"><span className="dot top-dot" /> Top evidence <span className="dot" /> Supporting context</p></section>
      <section className="panel validation-panel"><div className="panel-kicker"><span>CF</span> Counterfactual check</div><h2>Would the answer hold?</h2><p className="muted">Remove the strongest evidence node and see whether the conclusion changes.</p><button className="outline-button" onClick={onCounterfactual} disabled={!topIds.length || busy}>{busy ? "Checking…" : "Test top evidence"}<Icon>↗</Icon></button>{counterfactual && <div className={`counterfactual ${counterfactual.answer_changed ? "changed" : "stable"}`}><strong>{counterfactual.answer_changed ? "The answer changed." : "The answer held."}</strong><span>Node {counterfactual.removed_node_id.slice(-8)} was removed.</span><p>{counterfactual.counterfactual_answer}</p></div>}</section>
    </section>
    <section className="evidence-section"><div className="section-heading"><div><div className="panel-kicker"><span>SHAP</span> Word-level evidence</div><h2>See what mattered.</h2></div><p className="muted">Brighter words carried more retrieval influence.</p></div><div className="evidence-list">{topIds.map((id, index) => { const ref = refs.get(id); const words = result.shap_highlights?.[id] ?? []; const score = result.gat_output.node_scores?.[id] ?? 0; return <article className="evidence-card" key={id}><div className="evidence-number">0{index + 1}</div><div className="evidence-body"><div className="evidence-meta"><span>{formatTime(ref?.start_time ?? 0)} – {formatTime(ref?.end_time ?? 0)}</span><span>GAT {Math.round(score * 100)}%</span></div><EvidenceText words={words} fallback={ref?.text} />{ref && <audio controls preload="metadata" src={`${API_ORIGIN}${ref.url}`} />}</div></article>; })}</div></section>
    </>}
  </>;
}

export default function App() {
  const [meeting, setMeeting] = useState(null);
  const [result, setResult] = useState(null);
  const [counterfactual, setCounterfactual] = useState(null);
  const [state, setState] = useState({ mode: null, message: "" });
  const busy = state.mode !== null;
  const ask = async (question) => {
    setState({ mode: "query", message: "Following the evidence trail…" }); setResult(null); setCounterfactual(null);
    try { const response = await fetch(`${API_BASE}/query/`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ meeting_id: meeting.meeting_id, question }) }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Question failed"); setResult({ ...data, question }); } catch (error) { alert(error.message); } finally { setState({ mode: null, message: "" }); }
  };
  const validate = async () => {
    const node = result?.gat_output?.top_evidence_ids?.[0]; if (!node) return;
    setState({ mode: "counterfactual", message: "" });
    try { const response = await fetch(`${API_BASE}/counterfactual/`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ meeting_id: meeting.meeting_id, question: result.question, node_id_to_remove: node, original_answer: result.answer, original_evidence_graph: result.gat_output }) }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Counterfactual validation failed"); setCounterfactual(data); } catch (error) { alert(error.message); } finally { setState({ mode: null, message: "" }); }
  };
  return <main><header className="topbar"><a className="brand" href="#top">LI<span>ST</span>EN<span className="brand-dot">.</span></a><div className="topbar-right"><span className="live-dot" /> Evidence-first meeting intelligence</div></header><div className="hero" id="top"><div><p className="eyebrow">Layered interpretable speech-to-text network</p><h1>Hear the <em>why</em><br />behind every answer.</h1><p className="hero-copy">Upload a conversation. Ask anything. LISTEN shows the evidence, word by word.</p></div><div className="scribble-card"><span>listening...</span><div className="wave">⌁⌁⌁⌁⌁</div><small>Audio → Evidence → Answer</small></div></div><div className="workspace"><UploadCard onUploaded={setMeeting} busy={busy} /><QuestionCard meeting={meeting} onAsk={ask} busy={busy} /></div>{state.mode === "query" && <div className="loading-line"><span />{state.message}</div>}{result && <Results result={result} onCounterfactual={validate} counterfactual={counterfactual} busy={busy} />}<footer>LISTEN makes every answer inspectable. <span>Built for conversations that matter.</span></footer></main>;
}
