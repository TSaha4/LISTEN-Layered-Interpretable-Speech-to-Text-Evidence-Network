"""Persistent commitment graph spanning every processed meeting.

Nodes are commitments; edges link commitments about the same topic and say
whether the later one *supersedes* (conflicts with) or *reaffirms* the
earlier one. Supersession is decided from three cues, in order:

1. opposite polarity (one negates/cancels what the other affirmed),
2. different concrete values (dates, amounts, counts) on the same topic,
3. a later restatement with a different owner.

Only the newest non-superseded commitment on a topic is ``active``. When two
commitments conflict and the later one is far less reliable than the
earlier one (badly heard, heavily hedged) the earlier one is kept active and
the later one is attached as a *disputed* reaffirmation instead — the
uncertainty layer deciding what counts as the record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import config
from app.dl.embeddings import embed_texts
from app.models.schemas import Commitment, CommitmentEdge, CommitmentTimeline


def _norm_set(items: list[str]) -> set[str]:
    return {i.lower().strip() for i in items if i and i.strip()}


class CommitmentGraph:
    """File-backed graph of commitments across meetings."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.COMMITMENT_GRAPH_PATH
        self.nodes: dict[str, Commitment] = {}
        self.edges: list[CommitmentEdge] = []
        self._embeddings: dict[str, np.ndarray] = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {c["commitment_id"]: Commitment(**c) for c in payload.get("nodes", [])}
        self.edges = [CommitmentEdge(**e) for e in payload.get("edges", [])]
        for cid, vec in payload.get("embeddings", {}).items():
            self._embeddings[cid] = np.asarray(vec, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [c.model_dump() for c in self.nodes.values()],
            "edges": [e.model_dump() for e in self.edges],
            "embeddings": {cid: vec.tolist() for cid, vec in self._embeddings.items()},
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    # ── topic matching ────────────────────────────────────────────────────

    def _embedding(self, c: Commitment) -> np.ndarray:
        if c.commitment_id not in self._embeddings:
            self._embeddings[c.commitment_id] = embed_texts([c.text])[0]
        return self._embeddings[c.commitment_id]

    def _topic_similarity(self, a: Commitment, b: Commitment) -> float:
        cos = float(np.dot(self._embedding(a), self._embedding(b)))
        shared = _norm_set(a.topic_entities) & _norm_set(b.topic_entities)
        # Shared named topics are strong evidence of "same subject".
        return min(1.0, cos + (0.15 if shared else 0.0))

    @staticmethod
    def _conflict_reason(earlier: Commitment, later: Commitment) -> str | None:
        if earlier.polarity != later.polarity:
            return "polarity flipped"
        ev, lv = _norm_set(earlier.values), _norm_set(later.values)
        if ev and lv and ev != lv:
            return f"value changed {sorted(ev)} -> {sorted(lv)}"
        if earlier.owner and later.owner and earlier.owner.lower() != later.owner.lower():
            return f"owner changed {earlier.owner} -> {later.owner}"
        return None

    # ── mutation ──────────────────────────────────────────────────────────

    def add(self, commitment: Commitment) -> list[CommitmentEdge]:
        """Insert a commitment, link it to related ones, and resolve status."""
        if commitment.commitment_id in self.nodes:
            return []
        new_edges: list[CommitmentEdge] = []
        self.nodes[commitment.commitment_id] = commitment
        self._embedding(commitment)

        for other in list(self.nodes.values()):
            if other.commitment_id == commitment.commitment_id:
                continue
            sim = self._topic_similarity(other, commitment)
            if sim < config.COMMITMENT_TOPIC_SIMILARITY:
                continue

            earlier, later = sorted((other, commitment), key=lambda c: (c.meeting_time, c.offset_s))
            reason = self._conflict_reason(earlier, later)
            if reason is None:
                new_edges.append(CommitmentEdge(
                    source=later.commitment_id, target=earlier.commitment_id,
                    relation="reaffirms", reason="same stance restated", similarity=sim,
                ))
                continue

            # Conflict: later wins unless it is markedly less reliable.
            if later.reliability + 0.25 < earlier.reliability:
                new_edges.append(CommitmentEdge(
                    source=later.commitment_id, target=earlier.commitment_id,
                    relation="disputes",
                    reason=f"{reason}; later statement unreliable ({later.reliability:.2f} vs {earlier.reliability:.2f})",
                    similarity=sim,
                ))
                continue

            new_edges.append(CommitmentEdge(
                source=later.commitment_id, target=earlier.commitment_id,
                relation="supersedes", reason=reason, similarity=sim,
            ))
            if earlier.status == "active":
                self.nodes[earlier.commitment_id] = earlier.model_copy(update={
                    "status": "superseded", "superseded_by": later.commitment_id,
                })

        self.edges.extend(new_edges)
        return new_edges

    def add_many(self, commitments: list[Commitment]) -> list[CommitmentEdge]:
        edges: list[CommitmentEdge] = []
        for c in sorted(commitments, key=lambda c: (c.meeting_time, c.offset_s)):
            edges.extend(self.add(c))
        return edges

    def remove_meeting(self, meeting_id: str) -> None:
        """Drop a meeting's commitments (used when re-processing)."""
        drop = {cid for cid, c in self.nodes.items() if c.meeting_id == meeting_id}
        self.nodes = {cid: c for cid, c in self.nodes.items() if cid not in drop}
        self.edges = [e for e in self.edges if e.source not in drop and e.target not in drop]
        for cid in drop:
            self._embeddings.pop(cid, None)

    # ── queries ───────────────────────────────────────────────────────────

    def timeline(self, topic: str, top_k: int = 10) -> CommitmentTimeline:
        """Return the commitments about ``topic``, newest first, with status."""
        if not self.nodes:
            return CommitmentTimeline(topic=topic, current=None, history=[], edges=[], reliability=0.0)

        q = embed_texts([topic])[0]
        scored = [
            (float(np.dot(q, self._embedding(c))), c)
            for c in self.nodes.values()
        ]
        scored = [(s, c) for s, c in scored if s >= config.COMMITMENT_TOPIC_SIMILARITY * 0.75]
        scored.sort(key=lambda sc: (-sc[1].meeting_time, -sc[1].offset_s))
        history = [c for _, c in scored[:top_k]]
        ids = {c.commitment_id for c in history}
        edges = [e for e in self.edges if e.source in ids and e.target in ids]

        active = [c for c in history if c.status == "active"]
        current = active[0] if active else None
        reliability = current.reliability if current else 0.0
        return CommitmentTimeline(topic=topic, current=current, history=history, edges=edges, reliability=reliability)

    def for_meeting(self, meeting_id: str) -> list[Commitment]:
        return sorted(
            (c for c in self.nodes.values() if c.meeting_id == meeting_id),
            key=lambda c: c.offset_s,
        )


_graph: CommitmentGraph | None = None


def get_commitment_graph() -> CommitmentGraph:
    """Process-wide commitment graph singleton."""
    global _graph
    if _graph is None:
        _graph = CommitmentGraph()
    return _graph


def reset_commitment_graph() -> None:
    global _graph
    _graph = None
