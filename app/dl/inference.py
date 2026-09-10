"""Load the trained QMSum GAT checkpoint and score evidence graphs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from app import config
from app.dl.gat_model import EvidenceGAT
from app.models.schemas import GATOutput


class GATInference:
    """Lazy, CPU-backed wrapper around the strict QMSum checkpoint loader."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path or config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME)
        self._model: EvidenceGAT | None = None
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
        return self._model

    def run(self, data: Data, metadata: dict, top_k: int | None = None) -> GATOutput:
        """Score graph nodes and edges using learned checkpoint weights."""
        top_k = top_k or config.GAT_TOP_EVIDENCE_K
        with torch.no_grad():
            node_logits, edge_logits = self.model(data.x, data.edge_index, getattr(data, "edge_attr", None))

        node_id_map = metadata["node_id_map"]
        node_weights = torch.sigmoid(node_logits.squeeze(-1)).cpu().numpy()
        node_scores = {node_id_map[i]: float(node_weights[i]) for i in range(data.num_nodes)}
        evidence_only = [(node_id, score) for node_id, score in node_scores.items() if node_id != "__question__"]
        top_evidence_ids = [node_id for node_id, _ in sorted(evidence_only, key=lambda item: item[1], reverse=True)[:top_k]]

        edge_index = data.edge_index.cpu().numpy()
        edge_weights = torch.sigmoid(edge_logits.squeeze(-1)).cpu().numpy()
        edge_scores = {
            f"{node_id_map[edge_index[0, i]]}::{node_id_map[edge_index[1, i]]}": float(edge_weights[i])
            for i in range(edge_index.shape[1])
        }
        return GATOutput(node_scores=node_scores, edge_scores=edge_scores, top_evidence_ids=top_evidence_ids)


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
