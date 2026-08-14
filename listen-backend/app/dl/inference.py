"""Load a trained GAT checkpoint and run inference on evidence graphs."""

from __future__ import annotations

from pathlib import Path

import torch

from app import config
from app.dl.gat_model import (
    EvidenceGAT,
    evidence_graph_to_pyg,
    gat_output_from_logits,
)
from app.models.schemas import EvidenceGraph, GATOutput


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
                state = torch.load(self.checkpoint_path, map_location=self._device, weights_only=True)
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

    def run(self, graph: EvidenceGraph, top_k: int | None = None) -> GATOutput:
        """Score nodes and edges in an evidence graph.

        Args:
            graph: Built evidence graph for the current question.
            top_k: Override for number of top evidence segments.

        Returns:
            ``GATOutput`` with node/edge relevance scores.
        """
        if not graph.nodes:
            return GATOutput(node_scores={}, edge_scores={}, top_evidence_ids=[])

        data = evidence_graph_to_pyg(graph, device=self._device)
        with torch.no_grad():
            node_logits, edge_logits = self.model(data.x, data.edge_index, data.edge_attr)

        return gat_output_from_logits(
            segment_ids=data.segment_ids,
            node_logits=node_logits,
            edge_index=data.edge_index,
            edge_logits=edge_logits,
            top_k=top_k,
        )

    def run_without_node(self, graph: EvidenceGraph, node_id: str) -> GATOutput:
        """Run inference on a graph with one node (and its edges) removed.

        Used by the counterfactual XAI module.

        Args:
            graph: Original evidence graph.
            node_id: Segment id to remove.

        Returns:
            ``GATOutput`` for the modified graph.
        """
        filtered_nodes = [n for n in graph.nodes if n.segment_id != node_id]
        filtered_edges = [
            e
            for e in graph.edges
            if e.source != node_id and e.target != node_id
        ]
        modified = EvidenceGraph(nodes=filtered_nodes, edges=filtered_edges)
        return self.run(modified)


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
