"""Load a trained GAT checkpoint and run inference on evidence graphs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from app import config
from app.dl.gat_model import EvidenceGAT
from app.models.schemas import GATOutput


class GATInference:
    """Lazy-loaded GAT inference wrapper."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(
            checkpoint_path or config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME
        )
        self._model: EvidenceGAT | None = None
        self._device = torch.device("cpu")

    @property
    def model(self) -> EvidenceGAT:
        """Load model weights on first access."""
        if self._model is None:
            self._model = EvidenceGAT()
            if self.checkpoint_path.exists():
                # Person B's checkpoint dict format
                state = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
                if "model_state_dict" in state:
                    self._model.load_state_dict(state["model_state_dict"])
                else:
                    self._model.load_state_dict(state)
            else:
                # Untrained weights — still runnable for integration tests.
                self._model.apply(self._init_weights)
            self._model.to(self._device)
            self._model.eval()
        return self._model

    @staticmethod
    def _init_weights(module: torch.nn.Module) -> None:
        """Xavier init for linear layers when no checkpoint is available."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def run(self, data: Data, metadata: dict, top_k: int | None = None) -> GATOutput:
        """Score nodes in an evidence graph.

        Args:
            data: PyG Data from EvidenceGraphBuilder.
            metadata: Metadata dict from EvidenceGraphBuilder mapping node index to segment ID.
            top_k: Override for number of top evidence segments.

        Returns:
            GATOutput with node/edge relevance scores.
        """
        top_k = top_k or config.GAT_TOP_EVIDENCE_K

        with torch.no_grad():
            outputs = self.model(data)
        
        node_weights = outputs["node_weights"].cpu().numpy()
        node_id_map = metadata["node_id_map"]
        
        node_scores = {}
        # Include all nodes (even Question node 0) so frontend can plot them
        for i in range(data.num_nodes):
            seg_id = node_id_map[i]
            node_scores[seg_id] = float(node_weights[i])

        # Sort and get top K (excluding the question node itself)
        evidence_only_scores = [(k, v) for k, v in node_scores.items() if k != "__question__"]
        sorted_scores = sorted(evidence_only_scores, key=lambda x: x[1], reverse=True)
        top_evidence_ids = [k for k, v in sorted_scores[:top_k]]

        # Edge scores: PyG model doesn't explicitly output a single edge score per edge
        # We can extract attention weights or just use 1.0 for visualization.
        # Let's provide basic edge scores based on the graph structure for the frontend.
        edge_scores = {}
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i]
            tgt_idx = edge_index[1, i]
            src_id = node_id_map[src_idx]
            tgt_id = node_id_map[tgt_idx]
            # Simple average of node importance for edge score
            score = (node_scores[src_id] + node_scores[tgt_id]) / 2.0
            edge_scores[f"{src_id}::{tgt_id}"] = float(score)

        return GATOutput(
            node_scores=node_scores,
            edge_scores=edge_scores,
            top_evidence_ids=top_evidence_ids
        )


_inference: GATInference | None = None


def get_gat_inference() -> GATInference:
    """Return a process-wide lazy GAT inference singleton."""
    global _inference
    if _inference is None:
        _inference = GATInference()
    return _inference


def reset_gat_inference() -> None:
    """Clear cached inference model (useful in tests)."""
    global _inference
    _inference = None
