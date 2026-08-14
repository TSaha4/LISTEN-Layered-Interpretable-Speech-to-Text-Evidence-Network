"""
LISTEN Phase 1 — Bi-Encoder for Semantic Embedding

Uses sentence-transformers (all-MiniLM-L6-v2) to encode transcript segments
and questions into dense 384-dim vectors for retrieval.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config


class BiEncoder:
    """Wraps a sentence-transformer model for encoding segments and questions.

    This is the embedding backbone used by:
      - Phase 1 retrieval (question↔segment cosine similarity)
      - Phase 2 graph construction (segment↔segment similarity edges)
      - Phase 2 GAT node features (segment embeddings as input)
    """

    def __init__(self, model_name: str = config.BI_ENCODER_MODEL, device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace sentence-transformer model identifier.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = config.EMBEDDING_DIM

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts into dense embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) — L2-normalized embeddings.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine sim via dot product
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string.

        Returns:
            np.ndarray of shape (embedding_dim,).
        """
        return self.encode([text])[0]

    def cosine_similarity(self, query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query and a corpus of embeddings.

        Since embeddings are L2-normalized, cosine similarity = dot product.

        Args:
            query_emb: Shape (embedding_dim,) — single query embedding.
            corpus_embs: Shape (N, embedding_dim) — corpus embeddings.

        Returns:
            np.ndarray of shape (N,) — similarity scores.
        """
        return corpus_embs @ query_emb

    def pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Shape (N, embedding_dim).

        Returns:
            np.ndarray of shape (N, N) — symmetric similarity matrix.
        """
        return embeddings @ embeddings.T
