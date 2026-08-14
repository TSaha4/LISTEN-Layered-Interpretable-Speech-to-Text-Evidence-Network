"""Central configuration for the LISTEN backend.

All model choices, hyperparameters, and filesystem paths live here so the
pipeline code stays provider- and checkpoint-agnostic.
"""

from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
UPLOAD_DIR: Path = DATA_DIR / "uploads"
AUDIO_DIR: Path = DATA_DIR / "audio"
INDEX_DIR: Path = DATA_DIR / "indices"
CHECKPOINT_DIR: Path = DATA_DIR / "checkpoints"
MEETING_STORE_DIR: Path = DATA_DIR / "meetings"
QMSUM_DIR: Path = DATA_DIR / "qmsum"

for _d in (UPLOAD_DIR, AUDIO_DIR, INDEX_DIR, CHECKPOINT_DIR, MEETING_STORE_DIR, QMSUM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SLP — Speech & Language Processing
# ---------------------------------------------------------------------------
WHISPER_MODEL: str = "base"
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"
SPACY_MODEL: str = "en_core_web_sm"
FFMPEG_AUDIO_CODEC: str = "pcm_s16le"
FFMPEG_SAMPLE_RATE: int = 16000

# ---------------------------------------------------------------------------
# DL — Embeddings & retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
RETRIEVAL_TOP_K: int = 10
FAISS_INDEX_TYPE: Literal["flat", "ivf"] = "flat"

# Graph construction (weights should sum to <= 1.0; remainder is semantic)
GRAPH_ENTITY_EDGE_WEIGHT: float = 0.35
GRAPH_TEMPORAL_EDGE_WEIGHT: float = 0.25
GRAPH_SEMANTIC_EDGE_WEIGHT: float = 0.40
GRAPH_SEMANTIC_EDGE_THRESHOLD: float = 0.30
GRAPH_MAX_EDGES_PER_NODE: int = 5

# GAT architecture (single-level evidence graph; inspired by HGN/GATH but
# without hierarchical node types — see gat_model.py docstrings)
GAT_HIDDEN_DIM: int = 128
GAT_NUM_HEADS: int = 4
GAT_NUM_LAYERS: int = 2
GAT_DROPOUT: float = 0.1
GAT_TOP_EVIDENCE_K: int = 5

# GAT training (QMSum)
GAT_LEARNING_RATE: float = 1e-3
GAT_WEIGHT_DECAY: float = 1e-4
GAT_BATCH_SIZE: int = 8
GAT_EPOCHS: int = 20
GAT_VAL_SPLIT: float = 0.15
GAT_CHECKPOINT_NAME: str = "gat_qmsum.pt"
GAT_TRAIN_LOG: str = "gat_train_log.json"

# ---------------------------------------------------------------------------
# LLM — answer generation
# ---------------------------------------------------------------------------
LLM_PROVIDER: Literal["openai"] = "openai"
OPENAI_MODEL: str = "gpt-4o-mini"
OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"
LLM_MAX_TOKENS: int = 512
LLM_TEMPERATURE: float = 0.2

# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------
SHAP_MAX_EVALS: int = 50
SHAP_BACKGROUND_SAMPLES: int = 20

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_PREFIX: str = "/api/v1"
MAX_UPLOAD_SIZE_MB: int = 100
ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".flac", ".m4a", ".ogg"})
ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm"})
