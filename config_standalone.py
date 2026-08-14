"""
LISTEN DL Component — Configuration
All hyperparameters, model paths, and API settings in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
HOTPOTQA_DIR = DATA_DIR / "hotpotqa"
QMSUM_DIR = DATA_DIR / "qmsum"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Create directories if they don't exist
for d in [DATA_DIR, HOTPOTQA_DIR, QMSUM_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Phase 1: Bi-Encoder Retrieval
# ──────────────────────────────────────────────
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K_RETRIEVAL = 15  # Number of segments to retrieve per question


# ──────────────────────────────────────────────
# Phase 2: Graph Construction + GAT
# ──────────────────────────────────────────────
# Graph construction
SIMILARITY_THRESHOLD = 0.45   # Cosine sim threshold for segment-segment edges
MIN_SHARED_ENTITIES = 1       # Min shared entities for entity-based edges

# GAT architecture (following HGN §3.3)
GAT_INPUT_DIM = EMBEDDING_DIM    # 384 from bi-encoder
GAT_HIDDEN_DIM = 256
GAT_OUTPUT_DIM = 128
GAT_NUM_HEADS = 4                # Multi-head attention (HGN uses multi-head)
GAT_NUM_LAYERS = 2               # 2-layer GAT
GAT_DROPOUT = 0.2
NUM_EDGE_TYPES = 3               # question↔seg, entity-shared, similarity

# Training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS_HOTPOTQA = 20             # Pre-train on HotpotQA
EPOCHS_QMSUM = 10               # Fine-tune on QMSum
MAX_SEGMENTS_PER_GRAPH = 25      # Max nodes in a single evidence graph
TRAIN_VAL_SPLIT = 0.9


# ──────────────────────────────────────────────
# Phase 3: LLM Answer Generation (Gemini)
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_TOKENS = 512
TOP_K_FOR_ANSWER = 5             # Top-weighted segments sent to LLM


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
EVAL_BATCH_SIZE = 32
