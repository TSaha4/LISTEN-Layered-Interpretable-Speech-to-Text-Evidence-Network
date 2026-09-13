# LISTEN Demo Guide

Quick guide to run and demo the LISTEN system.

---

## Prerequisites

1. **Python 3.12** with virtual environment activated
2. **All dependencies installed:** `pip install -r requirements.txt`
3. **spaCy model downloaded:** `python -m spacy download en_core_web_sm`
4. **Environment variables set:** Copy `.env.example` to `.env` and add your Gemini API key

---

## Running the System

### Option 1: Quick Start (Recommended)

```powershell
# Terminal 1: Start Backend
.venv\Scripts\activate.ps1
python -m app.main

# Terminal 2: Serve Frontend (in new terminal)
cd test_frontend
python -m http.server 3000
```

Then open: **http://localhost:3000**

---

### Option 2: With Uvicorn Options

```powershell
# Backend with auto-reload
.venv\Scripts\activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (same as above)
cd test_frontend
python -m http.server 3000
```

---

## Demo Steps

### Step 1: Upload a Meeting Recording

1. Open http://localhost:3000
2. Click "Choose File" or drag & drop an audio/video file
3. Try: `data/audio/raw/ES2002a.wav` (sample AMI meeting recording)
4. Click "Upload & Process"
5. Wait for SLP pipeline (ASR + NER) - takes ~30-60 seconds for a 5-10 min audio
6. **Success:** Meeting ID displays, transcript shows segments with timestamps

**Supported formats:**
- Audio: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`
- Video: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm` (audio extracted automatically)

---

### Step 2: Ask a Question

1. Enter a natural language question in the query box
2. Example questions for ES2002a:
   - "What was discussed about the budget?"
   - "Who talked about the project timeline?"
   - "What decisions were made in this meeting?"
3. Click "Ask Question"
4. Wait for retrieval + GAT + LLM pipeline (~5-10 seconds)

---

### Step 3: Explore Results

**Answer Section:**
- Generated answer from Gemini (or configured LLM)

**Evidence Segments:**
- Top-K segments ranked by GAT node scores
- SHAP word highlighting (red intensity = importance to retrieval)
- GAT score percentage (how relevant the GAT thinks this segment is)
- Timestamps and segment IDs

**Evidence Graph:**
- Interactive network visualization (vis.js)
- **Red nodes:** Top evidence segments selected for the answer
- **Blue nodes:** Other retrieved candidates
- **Green edges:** Relationships (entity overlap, temporal adjacency, semantic similarity)
- Hover over nodes/edges to see scores
- Drag nodes to rearrange

---

## API Endpoints

### 1. Upload Meeting
```http
POST /api/v1/meetings/upload
Content-Type: multipart/form-data

file: <audio or video file>
```

**Response:**
```json
{
  "meeting_id": "20260819_123456",
  "segment_count": 42,
  "segments": [
    {
      "segment_id": "20260819_123456_seg_0000",
      "start_time": 0.0,
      "end_time": 5.2,
      "text": "Alice discussed the project budget with the team.",
      "entities": ["Alice"]
    },
    ...
  ],
  "message": "Processing complete."
}
```

---

### 2. Ask Question
```http
POST /api/v1/query/
Content-Type: application/json

{
  "meeting_id": "20260819_123456",
  "question": "What was discussed about the budget?"
}
```

**Response:**
```json
{
  "answer": "Alice mentioned that the project budget needs to increase for hardware costs...",
  "gat_output": {
    "node_scores": {
      "20260819_123456_seg_0000": 0.85,
      "20260819_123456_seg_0003": 0.72,
      ...
    },
    "edge_scores": {
      "20260819_123456_seg_0000_20260819_123456_seg_0003": 0.68,
      ...
    },
    "top_evidence_ids": [
      "20260819_123456_seg_0000",
      "20260819_123456_seg_0003",
      ...
    ]
  },
  "shap_highlights": {
    "20260819_123456_seg_0000": [
      {"word": "Alice", "score": 0.15},
      {"word": "budget", "score": 0.52},
      {"word": "project", "score": 0.31},
      ...
    ]
  },
  "audio_refs": [
    {
      "segment_id": "20260819_123456_seg_0000",
      "start_time": 0.0,
      "end_time": 5.2,
      "text": "Alice discussed the project budget with the team.",
      "url": "/api/v1/meetings/20260819_123456/audio?start=0.0&end=5.2"
    },
    ...
  ]
}
```

---

### 3. Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

## Troubleshooting

### Backend won't start
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Try different port
uvicorn app.main:app --port 8001
# Update API_BASE in test_frontend/app.js
```

### Frontend shows CORS error
Add to `app/main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Upload fails with "Whisper model not loaded"
```powershell
# Ensure faster-whisper is installed
pip install faster-whisper
```

### Entity extraction fails
```powershell
# Download spaCy model
python -m spacy download en_core_web_sm
```

### LLM generation fails
Check `.env` file has valid API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
PRIMARY_LLM_PROVIDER=gemini
```

---

## System Architecture

```
User uploads audio/video
    ↓
SLP Pipeline (app/slp/)
├── Extract audio from video (if needed)
├── ASR: Whisper transcription → segments
└── NER: spaCy entity extraction → entities per segment
    ↓
Store meeting + build FAISS index (app/services/meeting_store.py)
    ↓
User asks question
    ↓
Query Pipeline (app/services/pipeline.py)
├── 1. Retrieval (app/dl/retrieval.py)
│   └── Hybrid: BM25 + semantic + key terms → Top-K candidates
├── 2. Graph Builder (app/dl/graph_builder.py)
│   └── Nodes=segments, Edges=entity/temporal/semantic
├── 3. GAT Inference (app/dl/inference.py)
│   └── Multi-hop attention → node scores + edge scores
├── 4. Answer Generation (app/dl/answer_generation.py)
│   └── LLM (Gemini) + context from top evidence → answer
└── 5. XAI (app/xai/)
    ├── SHAP: word-level importance scores
    └── Counterfactual: remove node, re-run (optional)
    ↓
Return JSON to frontend
```

---

## Performance Notes

**Expected Latencies:**
- Upload + SLP: ~30-60s for 5-10 min audio (Whisper base model on CPU)
- Query pipeline: ~5-10s (retrieval + GAT + LLM generation)
- FAISS indexing: <1s for typical meeting (50-100 segments)

**Hardware Requirements:**
- Minimum: 4GB RAM, 2 CPU cores (CPU-only mode)
- Recommended: 8GB RAM, 4 CPU cores, GPU for faster Whisper (change WHISPER_DEVICE="cuda")

**Scalability:**
- Current: Single meeting, local storage
- Production: Add Redis for session state, S3 for audio, PostgreSQL for metadata

---

## Configuration

Edit `app/config.py` or `.env`:

**LLM Provider:**
```python
PRIMARY_LLM_PROVIDER = "gemini"  # or "openai", "groq"
```

**Retrieval:**
```python
RETRIEVAL_TOP_K = 20  # Increase for higher recall
```

**GAT:**
```python
GAT_HIDDEN_DIM = 256
GAT_NUM_LAYERS = 2
GAT_TOP_EVIDENCE_K = 5  # How many segments to show
```

**Whisper:**
```python
WHISPER_MODEL = "base"  # or "tiny", "small", "medium", "large"
WHISPER_DEVICE = "cpu"  # or "cuda" for GPU
```

---

## Known Limitations

1. **Audio Playback:** URL returned but actual audio serving not implemented
2. **Single Meeting:** No multi-meeting search yet
3. **GAT Performance:** Val F1 ~0.29, retrieval recall is the bottleneck
4. **No Authentication:** Anyone can upload/query (add auth for production)
5. **No Persistence:** Meetings stored in `data/meetings/`, cleared on restart if not saved

---

## Demo Talking Points (for Faculty)

**What LISTEN Does:**
- Converts meeting recordings to searchable QA system using graph neural networks
- Not just keyword search - understands relationships between statements via evidence graphs
- Explains *why* segments are relevant using SHAP word highlighting

**Technical Highlights:**
- **Hybrid Retrieval:** BM25 lexical + semantic embeddings + key term matching
- **Graph Attention Network:** Multi-hop reasoning over entity/temporal/semantic edges
- **XAI:** SHAP-based word importance scores for transparency

**Real vs. Demo:**
- ✅ End-to-end pipeline works on real AMI meeting audio
- ✅ GAT trained on QMSum meeting QA dataset (not toy data)
- ⚠️ GAT performance modest (F1 0.29) - retrieval is bottleneck, not graph model
- ⚠️ Audio playback URL exists but not implemented (UI placeholder)

**Current Limitations:**
- Single meeting scope (no cross-meeting search)
- CPU-only deployment (production would use GPU)
- No speaker diarization (who said what)
- GAT scores based on structure, not always aligned with human judgment

---

**End of Guide**
