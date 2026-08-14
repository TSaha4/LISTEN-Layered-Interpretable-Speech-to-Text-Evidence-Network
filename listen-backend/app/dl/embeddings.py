"""Segment embedding and FAISS vector indexing (lazy-loaded models)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.models.schemas import SLPSegment

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer


class _EmbedderHolder:
    """Lazy singleton for the sentence-transformer model."""

    model: SentenceTransformer | None = None


_embedder_holder = _EmbedderHolder()


def _get_embedder() -> SentenceTransformer:
    """Load the bi-encoder only when first needed."""
    if _embedder_holder.model is None:
        from sentence_transformers import SentenceTransformer

        _embedder_holder.model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder_holder.model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a batch of strings into L2-normalised embeddings.

    Args:
        texts: Plain-text strings to embed.

    Returns:
        Float32 array of shape ``(len(texts), embedding_dim)``.
    """
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

    model = _get_embedder()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def embed_segments(segments: list[SLPSegment]) -> dict[str, np.ndarray]:
    """Embed all segment texts and return an id → vector mapping."""
    if not segments:
        return {}

    texts = [s.text for s in segments]
    vectors = embed_texts(texts)
    return {seg.segment_id: vectors[i] for i, seg in enumerate(segments)}


class SegmentIndex:
    """FAISS-backed index over meeting transcript segments."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._segment_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    @property
    def segment_ids(self) -> list[str]:
        """Ordered segment ids aligned with FAISS row indices."""
        return list(self._segment_ids)

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return len(self._segment_ids)

    def build(self, segments: list[SLPSegment]) -> None:
        """Build a flat inner-product index from segment embeddings.

        Args:
            segments: SLP segments whose ``text`` fields will be embedded.
        """
        import faiss

        if not segments:
            self._index = None
            self._segment_ids = []
            self._embeddings = np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
            return

        self._segment_ids = [s.segment_id for s in segments]
        self._embeddings = embed_texts([s.text for s in segments])

        self._index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self._index.add(self._embeddings)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Retrieve top-k segments by cosine similarity (via inner product).

        Args:
            query: Natural-language question.
            top_k: Number of neighbours; defaults to ``config.RETRIEVAL_TOP_K``.

        Returns:
            List of ``(segment_id, score)`` tuples sorted by descending score.
        """
        if self._index is None or self.size == 0:
            return []

        k = min(top_k or config.RETRIEVAL_TOP_K, self.size)
        query_vec = embed_texts([query])
        scores, indices = self._index.search(query_vec, k)

        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self._segment_ids[int(idx)], float(score)))
        return results

    def get_embedding(self, segment_id: str) -> np.ndarray | None:
        """Return the stored embedding for a segment id, if present."""
        if self._embeddings is None:
            return None
        try:
            idx = self._segment_ids.index(segment_id)
        except ValueError:
            return None
        return self._embeddings[idx]

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return all indexed embeddings keyed by segment id."""
        if self._embeddings is None:
            return {}
        return {
            sid: self._embeddings[i]
            for i, sid in enumerate(self._segment_ids)
        }

    def save(self, directory: str | Path) -> None:
        """Persist FAISS index and segment-id mapping to disk."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(directory / "index.faiss"))

        meta = {
            "segment_ids": self._segment_ids,
            "embedding_dim": config.EMBEDDING_DIM,
        }
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        if self._embeddings is not None:
            np.save(directory / "embeddings.npy", self._embeddings)

    @classmethod
    def load(cls, directory: str | Path) -> SegmentIndex:
        """Load a previously saved index from disk."""
        import faiss

        directory = Path(directory)
        instance = cls()

        meta_path = directory / "meta.json"
        if not meta_path.exists():
            return instance

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instance._segment_ids = meta["segment_ids"]

        index_path = directory / "index.faiss"
        if index_path.exists():
            instance._index = faiss.read_index(str(index_path))

        emb_path = directory / "embeddings.npy"
        if emb_path.exists():
            instance._embeddings = np.load(emb_path)

        return instance


def reset_embedder() -> None:
    """Release cached embedder (useful in tests)."""
    _embedder_holder.model = None
