# LISTEN Backend

Explainable multi-hop question answering over **real meeting recordings** (audio/video).

Pipeline: **Audio/Video → ASR → NER → Embed/Index → Retrieve → Evidence Graph → GAT → LLM Answer → SHAP → Counterfactual validation**

This repository contains the **backend only**. The `frontend/` folder is a placeholder for a future UI.

---

## Architecture

| Layer | Modules | Libraries |
|-------|---------|-----------|
| **SLP** | `app/slp/` | faster-whisper, spaCy, ffmpeg |
| **DL** | `app/dl/` | sentence-transformers, FAISS, PyTorch Geometric GATConv, OpenAI API |
| **XAI** | `app/xai/` | SHAP, counterfactual re-run |

### DL design vs. HGN/GATH (HotpotQA)

LISTEN uses a **single-level** evidence graph inspired by HGN/GATH's graph-attention idea, but scoped for meeting speech:

| Aspect | HGN / GATH (paper) | LISTEN (this project) |
|--------|-------------------|------------------------|
| Graph structure | Hierarchical (entity / sentence / doc nodes) | **Single node type**: ASR transcript segments |
| Node features | Pre-trained QA / lexical features | **Sentence embeddings** (all-MiniLM-L6-v2) |
| Data source | Wikipedia paragraphs (HotpotQA) | **Real AMI / QMSum meeting transcripts** |
| Reasoning depth | Multi-level hops across node types | **Configurable GAT stack** over segment graph |

See `app/dl/gat_model.py` module docstring for viva-ready design notes.

---

## Setup

**Requirements:** Python 3.11, ffmpeg on PATH, ~1 GB RAM (small model defaults).

```bash
cd listen-backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional environment variables:

```bash
export OPENAI_API_KEY=sk-...       # Required for LLM answers (offline fallback otherwise)
```

### PyTorch Geometric (if pip install fails)

On some platforms, install PyG wheels matching your torch version:

```bash
pip install torch==2.5.1
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-geometric==2.6.1
```

---

## Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/api/v1/meetings/upload` | POST | Upload audio/video → SLP pipeline |
| `/api/v1/query/` | POST | Ask a question over a meeting |
| `/api/v1/counterfactual/` | POST | Remove evidence node & re-run |

Interactive docs: `http://localhost:8000/docs`

### Example: upload + query

```bash
curl -X POST "http://localhost:8000/api/v1/meetings/upload" \
  -F "file=@meeting.wav"

curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"meeting_id": "<uuid>", "question": "What was decided about the budget?"}'
```

---

## Workflow 1 — Train the GAT (QMSum)

1. **Download QMSum** from [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum) and copy JSON files:

   ```
   data/qmsum/train/*.json
   data/qmsum/val/*.json
   ```

   Or run: `python scripts/download_data.py --dataset qmsum-info`

2. **Train**:

   ```bash
   python -m app.dl.train_gat --data-dir data/qmsum --epochs 20
   ```

3. **Outputs**:
   - Checkpoint: `data/checkpoints/gat_qmsum.pt`
   - Metrics log: `data/checkpoints/gat_train_log.json` (loss curves, val accuracy)

Training uses QMSum `specific_query_list` gold `relevant_text_span` labels for node-level binary supervision.

---

## Workflow 2 — Inference only (pre-trained checkpoint)

1. Place a trained checkpoint at `data/checkpoints/gat_qmsum.pt` (from training above).
2. Start the API — the GAT loads **lazily** on the first query.
3. Upload a meeting and ask questions; no training step required.

If no checkpoint exists, the API still runs with randomly initialised GAT weights (useful for integration testing, not production).

---

## Datasets

| Dataset | Loader | Usage |
|---------|--------|-------|
| **AMI** | `app/data/loaders.py` → HuggingFace `edinburghcstr/ami` | Real meeting audio metadata |
| **QMSum** | Local JSON files under `data/qmsum/` | GAT training & evaluation |

```bash
python scripts/download_data.py --dataset ami --split train --max-meetings 1
```

---

## Configuration

All model choices and hyperparameters live in **`app/config.py`**:

- `WHISPER_MODEL`, `EMBEDDING_MODEL`, `GAT_*`, `OPENAI_MODEL`, etc.
- Change defaults there — pipeline code stays unchanged.

---

## Testing

```bash
pytest tests/ -v
```

DL tests (`tests/test_dl.py`) cover graph construction, retrieval contracts, and GAT forward-pass shapes without loading heavy models.

---

## Project structure

```
listen-backend/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # All hyperparameters & paths
│   ├── api/                    # upload, query, counterfactual
│   ├── slp/                    # audio, ASR, NER
│   ├── dl/                     # embeddings, retrieval, graph, GAT, train, inference, LLM
│   ├── xai/                    # SHAP, counterfactual
│   ├── data/                   # AMI & QMSum loaders
│   ├── models/schemas.py       # Pydantic data contracts
│   └── services/               # meeting store, pipeline orchestration
├── tests/
├── scripts/
├── frontend/                   # Empty — UI added later
└── requirements.txt
```

---

## API response contract

```json
{
  "answer": "...",
  "evidence_graph": {
    "node_scores": {"seg_id": 0.87},
    "edge_scores": {"src_tgt": 0.65},
    "top_evidence_ids": ["seg_id"]
  },
  "shap_highlights": {
    "seg_id": [{"word": "budget", "score": 0.12}]
  },
  "audio_refs": {
    "seg_id": {"start_time": 0.0, "end_time": 5.0, "url": "..."}
  }
}
```

---

## Team notes

- **DL layer** (`app/dl/`): individually evaluated — fully typed, documented, and tested.
- **SLP / XAI**: implemented to production patterns but secondary priority.
- **Frontend**: consume `/api/v1/*` endpoints; no backend UI included.
