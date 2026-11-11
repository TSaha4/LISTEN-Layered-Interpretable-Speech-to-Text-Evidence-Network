"""Load the trained QMSum GAT checkpoint and score evidence graphs.

Scores are passed through a ``ReliabilityGate`` so segments the ASR could
not hear well, or the speaker was unsure about, are down-weighted before
answer generation. Both the raw and gated scores are returned.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from app import config
from app.dl.gat_model import EvidenceGAT, ReliabilityGate
from app.models.schemas import GATOutput


class GATInference:
    """Lazy, CPU-backed wrapper around the strict QMSum checkpoint loader."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path or config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME)
        self._model: EvidenceGAT | None = None
        self._gate: ReliabilityGate | None = None
        self._device = torch.device("cpu")

    @property
    def model(self) -> EvidenceGAT:
        if self._model is None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"GAT checkpoint not found: {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
            state_dict = state["model_state_dict"] if "model_state_dict" in state else state
            self._model = EvidenceGAT()
            self._model.load_state_dict(state_dict, strict=True)
            self._model.to(self._device)
            self._model.eval()
            # Optional fine-tuned gate stored alongside the checkpoint.
            gate = ReliabilityGate()
            if isinstance(state, dict) and "gate_state_dict" in state:
                gate.load_state_dict(state["gate_state_dict"])
            gate.eval()
            self._gate = gate
        return self._model

    @property
    def gate(self) -> ReliabilityGate:
        _ = self.model
        assert self._gate is not None
        return self._gate

    def run(self, data: Data, metadata: dict, top_k: int | None = None) -> GATOutput:
        """Score graph nodes and edges using learned checkpoint weights."""
        top_k = top_k or config.GAT_TOP_EVIDENCE_K
        with torch.no_grad():
            node_logits, edge_logits = self.model(data.x, data.edge_index, getattr(data, "edge_attr", None))
            node_probs = torch.sigmoid(node_logits.squeeze(-1))
            edge_probs = torch.sigmoid(edge_logits.squeeze(-1))

            node_rel = getattr(data, "node_reliability", None)
            edge_conf = getattr(data, "edge_confidence", None)
            if node_rel is None:
                node_rel = torch.ones_like(node_probs)
            if edge_conf is None:
                edge_conf = torch.ones_like(edge_probs)

            if config.GAT_RELIABILITY_GATING:
                gated_nodes, gated_edges = self.gate(node_probs, node_rel, edge_probs, edge_conf)
            else:
                gated_nodes, gated_edges = node_probs, edge_probs

        node_id_map = metadata["node_id_map"]
        raw = node_probs.cpu().numpy()
        gated = gated_nodes.cpu().numpy()
        rel = node_rel.cpu().numpy()
        raw_node_scores = {node_id_map[i]: float(raw[i]) for i in range(data.num_nodes)}
        node_scores = {node_id_map[i]: float(gated[i]) for i in range(data.num_nodes)}
        node_reliability = {node_id_map[i]: float(rel[i]) for i in range(data.num_nodes)}
        evidence_only = [(node_id, score) for node_id, score in node_scores.items() if node_id != "__question__"]
        top_evidence_ids = [node_id for node_id, _ in sorted(evidence_only, key=lambda item: item[1], reverse=True)[:top_k]]

        edge_index = data.edge_index.cpu().numpy()
        edge_weights = gated_edges.cpu().numpy()
        edge_conf_np = edge_conf.cpu().numpy()
        edge_scores: dict[str, float] = {}
        edge_confidence: dict[str, float] = {}
        for i in range(edge_index.shape[1]):
            key = f"{node_id_map[edge_index[0, i]]}::{node_id_map[edge_index[1, i]]}"
            edge_scores[key] = float(edge_weights[i])
            edge_confidence[key] = float(edge_conf_np[i])
        return GATOutput(
            node_scores=node_scores,
            edge_scores=edge_scores,
            top_evidence_ids=top_evidence_ids,
            raw_node_scores=raw_node_scores,
            node_reliability=node_reliability,
            edge_confidence=edge_confidence,
        )


_inference: GATInference | None = None


def get_gat_inference() -> GATInference:
    """Return the process-wide inference singleton."""
    global _inference
    if _inference is None:
        _inference = GATInference()
    return _inference


def reset_gat_inference() -> None:
    """Clear the cached model for tests."""
    global _inference
    _inference = None
