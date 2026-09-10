const API_BASE = "http://127.0.0.1:8000/api/v1";
const API_ORIGIN = new URL(API_BASE).origin;

// State
let currentMeetingId = null;

// DOM Elements
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("media-file");
const fileMsg = document.querySelector(".file-msg");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");
const meetingDetails = document.getElementById("meeting-details");
const meetingIdDisplay = document.getElementById("meeting-id-display");
const segmentCountDisplay = document.getElementById("segment-count-display");
const toggleTranscriptBtn = document.getElementById("toggle-transcript-btn");
const fullTranscriptContainer = document.getElementById("full-transcript-container");

let currentMeetingSegments = [];
let latestQueryResponse = null;

toggleTranscriptBtn.addEventListener("click", () => {
    if (fullTranscriptContainer.classList.contains("hidden")) {
        fullTranscriptContainer.classList.remove("hidden");
        toggleTranscriptBtn.textContent = "Hide Full Transcript";
    } else {
        fullTranscriptContainer.classList.add("hidden");
        toggleTranscriptBtn.textContent = "Show Full Transcript";
    }
});

const querySection = document.getElementById("query-section");
const queryForm = document.getElementById("query-form");
const questionInput = document.getElementById("question-input");
const queryBtn = document.getElementById("query-btn");
const queryStatus = document.getElementById("query-status");

const resultsSection = document.getElementById("results-section");
const answerDisplay = document.getElementById("answer-display");
const evidenceContainer = document.getElementById("evidence-container");
const counterfactualBtn = document.getElementById("counterfactual-btn");
const counterfactualStatus = document.getElementById("counterfactual-status");

// Update file drop text
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        fileMsg.textContent = e.target.files[0].name;
    } else {
        fileMsg.textContent = "Drag & drop a media file or click to select";
    }
});

function setStatus(element, message, type = "info", isLoading = false) {
    element.classList.remove("hidden", "error", "success");
    if (type) element.classList.add(type);
    
    element.innerHTML = isLoading 
        ? `<div class="loader"></div> ${message}`
        : message;
}

// 1. Upload Meeting - Fixed endpoint
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    if (fileInput.files.length === 0) return;
    const file = fileInput.files[0];
    
    const formData = new FormData();
    formData.append("file", file);

    uploadBtn.disabled = true;
    setStatus(uploadStatus, "Uploading and processing (SLP pipeline)... This may take a minute.", "info", true);

    try {
        // FIXED: Correct endpoint is /meetings/upload
        const response = await fetch(`${API_BASE}/meetings/upload`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.detail || "Upload failed");

        currentMeetingId = data.meeting_id;
        currentMeetingSegments = data.segments || [];
        
        setStatus(uploadStatus, "Processing complete!", "success");
        meetingIdDisplay.textContent = currentMeetingId;
        segmentCountDisplay.textContent = data.segment_count;
        meetingDetails.classList.remove("hidden");

        // Render full transcript
        fullTranscriptContainer.innerHTML = "";
        currentMeetingSegments.forEach((seg, i) => {
            const div = document.createElement("div");
            div.style.marginBottom = "0.75rem";
            div.style.paddingBottom = "0.75rem";
            div.style.borderBottom = "1px solid rgba(255,255,255,0.1)";
            
            const timeStr = `${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s`;
            div.innerHTML = `
                <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.25rem;">
                    <strong>Segment ${i+1}</strong> [${timeStr}]
                </div>
                <div style="line-height: 1.5;">${seg.text}</div>
            `;
            fullTranscriptContainer.appendChild(div);
        });

        // Enable Query section
        querySection.classList.remove("disabled");
        questionInput.disabled = false;
        queryBtn.disabled = false;
        
    } catch (error) {
        setStatus(uploadStatus, error.message, "error");
    } finally {
        uploadBtn.disabled = false;
    }
});

// 2. Ask Question - Fixed to match actual API schema
queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question || !currentMeetingId) return;

    queryBtn.disabled = true;
    resultsSection.classList.add("hidden");
    setStatus(queryStatus, "Retrieving segments, building graph, and generating answer...", "info", true);

    try {
        // FIXED: Endpoint is /query/ and schema matches QueryRequest
        const response = await fetch(`${API_BASE}/query/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                meeting_id: currentMeetingId,
                question: question
            })
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.detail || "Query failed");

        latestQueryResponse = data;

        queryStatus.classList.add("hidden");
        
        // Display Answer
        answerDisplay.textContent = data.answer;
        
        // --- Render Evidence Segments with SHAP ---
        evidenceContainer.innerHTML = "";
        
        // FIXED: Use actual response schema
        const topIds = data.gat_output.top_evidence_ids || [];
        const nodeScores = data.gat_output.node_scores || {};
        
        topIds.forEach((segId, index) => {
            const score = nodeScores[segId] || 0;
            const audioRef = data.audio_refs.find(ref => ref.segment_id === segId);
            const shapWords = data.shap_highlights[segId] || [];
            
            // Reconstruct text from SHAP words and highlight them
            let highlightedText = "";
            if (shapWords.length > 0) {
                let maxShap = Math.max(...shapWords.map(w => Math.abs(w.score)), 0.001);
                
                shapWords.forEach(w => {
                    const intensity = Math.min(Math.max(w.score / maxShap, 0), 1);
                    const bg = `rgba(239, 68, 68, ${intensity * 0.6})`;
                    highlightedText += `<span style="background-color: ${bg}; border-radius: 2px; padding: 0 2px;" title="SHAP: ${w.score.toFixed(3)}">${w.word}</span> `;
                });
            } else if (audioRef) {
                highlightedText = audioRef.text;
            }
            
            const item = document.createElement("div");
            item.className = "evidence-item";
            
            let timeStr = audioRef ? `${audioRef.start_time.toFixed(1)}s - ${audioRef.end_time.toFixed(1)}s` : "Unknown";
            
            item.innerHTML = `
                <div class="evidence-meta">
                    <span>Rank ${index + 1} | ${segId.substring(segId.length - 12)} | ${timeStr}</span>
                    <span class="evidence-score">GAT: ${(score * 100).toFixed(1)}%</span>
                </div>
                <div class="evidence-text" style="line-height: 1.8;">
                    ${highlightedText || "<em>No text available</em>"}
                </div>
                ${audioRef ? `<audio controls preload="metadata" src="${API_ORIGIN}${audioRef.url}" style="width: 100%; margin-top: 0.75rem;">Your browser does not support audio playback.</audio>` : ""}
            `;
            evidenceContainer.appendChild(item);
        });

        // --- Render Evidence Graph ---
        const nodes = [];
        const edges = [];
        
        // Add nodes from GAT output
        Object.entries(nodeScores).forEach(([id, weight]) => {
            const isTop = topIds.includes(id);
            const label = id.length > 20 ? id.substring(id.length - 8) : id;
            
            nodes.push({
                id: id,
                label: label,
                value: weight * 100,
                color: isTop ? "#ef4444" : "#3b82f6",
                title: `Score: ${(weight*100).toFixed(1)}%\nID: ${id}`
            });
        });
        
        // Add edges from GAT output
        const edgeScores = data.gat_output.edge_scores || {};
        Object.entries(edgeScores).forEach(([edgeStr, weight]) => {
            // Edge format: "source_id::target_id"
            const parts = edgeStr.split("::");
            if (parts.length === 2) {
                edges.push({
                    from: parts[0],
                    to: parts[1],
                    value: Math.max(weight * 20, 1),
                    title: `Edge weight: ${(weight*100).toFixed(1)}%`,
                    color: { color: "rgba(16, 185, 129, 0.4)", highlight: "rgba(16, 185, 129, 0.7)" }
                });
            }
        });
        
        const graphContainer = document.getElementById("network-container");
        if (!graphContainer) {
            throw new Error("Evidence graph container is missing from the page.");
        }

        if (nodes.length > 0) {
            const graphData = {
                nodes: new vis.DataSet(nodes),
                edges: new vis.DataSet(edges)
            };
            const options = {
                nodes: {
                    shape: "dot",
                    font: { color: "#ffffff", size: 12 }
                },
                edges: {
                    smooth: { type: "continuous" }
                },
                physics: {
                    stabilization: { iterations: 100 },
                    barnesHut: { gravitationalConstant: -8000, springLength: 100 }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 100
                }
            };
            new vis.Network(graphContainer, graphData, options);
        } else {
            graphContainer.innerHTML = '<div style="padding: 2rem; text-align: center; color: #94a3b8;">No graph nodes to display</div>';
        }

        resultsSection.classList.remove("hidden");

    } catch (error) {
        setStatus(queryStatus, error.message, "error");
    } finally {
        queryBtn.disabled = false;
    }
});

counterfactualBtn.addEventListener("click", async () => {
    const nodeId = latestQueryResponse?.gat_output?.top_evidence_ids?.[0];
    if (!currentMeetingId || !nodeId) return;

    counterfactualBtn.disabled = true;
    setStatus(counterfactualStatus, "Removing top evidence and re-running reasoning...", "info", true);
    try {
        const response = await fetch(`${API_BASE}/counterfactual/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                meeting_id: currentMeetingId,
                question: questionInput.value.trim(),
                node_id_to_remove: nodeId,
                original_answer: latestQueryResponse.answer,
                original_evidence_graph: latestQueryResponse.gat_output
            })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Counterfactual request failed");
        setStatus(
            counterfactualStatus,
            `Removed ${nodeId}. Answer changed: ${data.answer_changed ? "yes" : "no"}. ${data.counterfactual_answer}`,
            "success"
        );
    } catch (error) {
        setStatus(counterfactualStatus, error.message, "error");
    } finally {
        counterfactualBtn.disabled = false;
    }
});
