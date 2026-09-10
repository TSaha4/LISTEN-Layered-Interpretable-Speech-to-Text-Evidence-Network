"""Central configuration for the LISTEN backend.

Merged configuration from Person A (SLP + API) and Person B (DL hyperparameters),
now including multi-provider LLM support via python-dotenv.
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

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
# SLP — Speech & Language Processing (Person A)
# ---------------------------------------------------------------------------
WHISPER_MODEL: str = "base"
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"
SPACY_MODEL: str = "en_core_web_sm"
FFMPEG_AUDIO_CODEC: str = "pcm_s16le"
FFMPEG_SAMPLE_RATE: int = 16000

# ---------------------------------------------------------------------------
# DL — Embeddings & retrieval (Person B)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
RETRIEVAL_TOP_K: int = 15  # Number of segments to retrieve per question
FAISS_INDEX_TYPE: Literal["flat", "ivf"] = "flat"

# Graph construction (Person B)
SIMILARITY_THRESHOLD: float = 0.45   # Cosine sim threshold for segment-segment edges
MIN_SHARED_ENTITIES: int = 1       # Min shared entities for entity-based edges
MAX_SEGMENTS_PER_GRAPH: int = 25      # Max nodes in a single evidence graph

# Graph edge weighting (should sum to <= 1.0)
GRAPH_ENTITY_EDGE_WEIGHT: float = 0.35
GRAPH_TEMPORAL_EDGE_WEIGHT: float = 0.25
GRAPH_SEMANTIC_EDGE_WEIGHT: float = 0.40
GRAPH_SEMANTIC_EDGE_THRESHOLD: float = 0.30
GRAPH_MAX_EDGES_PER_NODE: int = 5

# GAT architecture (Person B)
GAT_INPUT_DIM: int = EMBEDDING_DIM    # 384 from bi-encoder
GAT_HIDDEN_DIM: int = 128
GAT_OUTPUT_DIM: int = 128
GAT_NUM_HEADS: int = 4                # Multi-head attention (HGN uses multi-head)
GAT_NUM_LAYERS: int = 2               # 2-layer GAT
GAT_DROPOUT: float = 0.2
NUM_EDGE_TYPES: int = 3               # question↔seg, entity-shared, similarity
GAT_TOP_EVIDENCE_K: int = 5

GAT_CHECKPOINT_NAME: str = "gat_qmsum.pt"

# ---------------------------------------------------------------------------
# LLM — Multi-Provider Answer Generation
# ---------------------------------------------------------------------------
PRIMARY_LLM_PROVIDER: str = os.environ.get("PRIMARY_LLM_PROVIDER", "gemini").lower()

# OpenAI
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Gemini
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Groq
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# Generation params
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
