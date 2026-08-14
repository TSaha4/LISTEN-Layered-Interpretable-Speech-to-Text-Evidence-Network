const API_BASE = "http://127.0.0.1:8000/api/v1";

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
const networkContainer = document.getElementById("network-container");

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

// 1. Upload Meeting
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    if (fileInput.files.length === 0) return;
    const file = fileInput.files[0];
    
    const formData = new FormData();
    formData.append("file", file);

    uploadBtn.disabled = true;
    setStatus(uploadStatus, "Uploading and processing (SLP pipeline)... This may take a minute.", "info", true);

    try {
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
            
            const speaker = seg.speaker || `Speaker ${i % 2 === 0 ? 'A' : 'B'}`;
            div.innerHTML = `
                <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.25rem;">
                    <strong>${speaker}</strong> [${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s]
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

// 2. Ask Question
queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question || !currentMeetingId) return;

    queryBtn.disabled = true;
    resultsSection.classList.add("hidden");
    setStatus(queryStatus, "Retrieving segments, building graph, and generating answer...", "info", true);

    try {
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

        queryStatus.classList.add("hidden");
        
        // Display Answer
        answerDisplay.textContent = data.answer;
        
        // --- Render XAI SHAP Highlights ---
        evidenceContainer.innerHTML = "";
        const topIds = data.evidence_graph.top_evidence_ids;
        const nodeScores = data.evidence_graph.node_scores;
        
        topIds.forEach((segId, index) => {
            const score = nodeScores[segId] || 0;
            const audioRef = data.audio_refs[segId];
            const shapWords = data.shap_highlights[segId] || [];
            
            // Reconstruct text from SHAP words and highlight them based on score
            let highlightedText = "";
            let maxShap = Math.max(...shapWords.map(w => Math.abs(w.score)), 0.001);
            
            shapWords.forEach(w => {
                // Calculate opacity for red highlight based on score weight
                const intensity = Math.min(Math.max(w.score / maxShap, 0), 1);
                const bg = `rgba(239, 68, 68, ${intensity * 0.6})`;
                highlightedText += `<span style="background-color: ${bg}; border-radius: 2px; padding: 0 2px;" title="SHAP Score: ${w.score.toFixed(3)}">${w.word}</span> `;
            });
            
            const item = document.createElement("div");
            item.className = "evidence-item";
            
            let timeStr = audioRef ? `${audioRef.start_time.toFixed(1)}s - ${audioRef.end_time.toFixed(1)}s` : "Unknown time";
            
            item.innerHTML = `
                <div class="evidence-meta">
                    <span>Rank ${index + 1} | ID: ${segId} | Time: ${timeStr}</span>
                    <span class="evidence-score">GAT Score: ${(score * 100).toFixed(1)}%</span>
                </div>
                <div class="evidence-text" style="line-height: 1.8;">
                    ${highlightedText || "<em>No text available</em>"}
                </div>
            `;
            evidenceContainer.appendChild(item);
        });

        // --- Render Vis-Network Graph ---
        const nodes = [];
        const edges = [];
        
        // Add Question Node
        nodes.push({ id: "Question", label: "Question", color: "#eab308", size: 30, shape: "box" });
        
        // Add Segment Nodes
        for (const [id, weight] of Object.entries(nodeScores)) {
            const isTop = topIds.includes(id);
            nodes.push({
                id: id,
                label: id,
                value: weight * 100,
                color: isTop ? "#ef4444" : "#3b82f6",
                title: `Weight: ${(weight*100).toFixed(1)}%`
            });
            
            // Link everything to Question (simplified representation)
            edges.push({
                from: "Question",
                to: id,
                value: weight * 10,
                color: "rgba(255,255,255,0.1)"
            });
        }
        
        // Add Segment-to-Segment Edges
        for (const [edgeStr, weight] of Object.entries(data.evidence_graph.edge_scores)) {
            const parts = edgeStr.split("-");
            if (parts.length === 2) {
                edges.push({
                    from: parts[0],
                    to: parts[1],
                    value: weight * 20,
                    title: `Attention: ${(weight*100).toFixed(1)}%`,
                    color: "rgba(16, 185, 129, 0.4)"
                });
            }
        }
        
        const graphData = {
            nodes: new vis.DataSet(nodes),
            edges: new vis.DataSet(edges)
        };
        const options = {
            nodes: {
                shape: "dot",
                font: { color: "#ffffff" }
            },
            physics: {
                stabilization: false,
                barnesHut: { springLength: 100 }
            }
        };
        new vis.Network(networkContainer, graphData, options);

        resultsSection.classList.remove("hidden");

    } catch (error) {
        setStatus(queryStatus, error.message, "error");
    } finally {
        queryBtn.disabled = false;
    }
});
