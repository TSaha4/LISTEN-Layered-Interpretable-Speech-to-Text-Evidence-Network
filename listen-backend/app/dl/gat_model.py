"""Single-level Graph Attention Network for evidence ranking.

Design notes (vs. HGN/GATH on HotpotQA)
---------------------------------------
HGN and GATH operate on *hierarchical* graphs with multiple node types
(entity nodes, sentence nodes, document nodes) and multi-hop reasoning over
Wikipedia paragraphs.  LISTEN deliberately uses a **single-level** graph:

* One homogeneous node type: ASR transcript segments from real meeting audio.
* Node features: dense sentence embeddings (all-MiniLM-L6-v2), not hand-crafted
  lexical features or pre-trained QA representations.
* One GAT stack (configurable depth via ``GAT_NUM_LAYERS``) scores segment
  relevance jointly — there is no separate entity-level graph tier.
* Edge weights come from entity overlap, temporal adjacency, and semantic
  similarity among speech segments, not hyperlinks or co-reference chains.

This scoping keeps the model deployable on constrained hosts while preserving
the core graph-attention idea: neighbours influence a segment's evidence score.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from app import config
from app.models.schemas import EvidenceGraph, GATOutput


class EvidenceGAT(nn.Module):
    """Graph Attention Network for ranking evidence segments.

    Tensor shapes (batch size B = 1 at inference; N nodes, F_in input dim):
        x:          (N, F_in)  — node feature matrix (segment embeddings)
        edge_index: (2, E)     — COO format edge list
        edge_attr:  (E,)       — optional scalar edge weights (unused by GATConv
                                 but stored for edge-score readout)

    Forward returns:
        node_logits: (N, 1)    — raw relevance logits per segment
        edge_logits: (E, 1)    — pairwise edge importance logits
    """

    def __init__(
        self,
        in_dim: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
    ) -> None:
        """Initialise GAT layers; hyperparameters default to ``config`` values."""
        super().__init__()
        self.in_dim = in_dim or config.EMBEDDING_DIM
        self.hidden_dim = hidden_dim or config.GAT_HIDDEN_DIM
        self.num_heads = num_heads or config.GAT_NUM_HEADS
        self.num_layers = num_layers or config.GAT_NUM_LAYERS
        self.dropout = dropout if dropout is not None else config.GAT_DROPOUT

        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)

        self.gat_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            is_last = layer_idx == self.num_layers - 1
            if is_last:
                # Final layer: single head, fixed hidden_dim output.
                self.gat_layers.append(
                    GATConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        heads=1,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=True,
                    )
                )
            else:
                # Intermediate layers: multi-head with concatenation.
                self.gat_layers.append(
                    GATConv(
                        self.hidden_dim,
                        self.hidden_dim // self.num_heads,
                        heads=self.num_heads,
                        concat=True,
                        dropout=self.dropout,
                        add_self_loops=True,
                    )
                )

        self.node_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.edge_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run a forward pass over the evidence graph.

        Args:
            x: Node feature matrix of shape ``(N, in_dim)``.
            edge_index: Edge indices of shape ``(2, E)``.
            edge_attr: Optional edge weights of shape ``(E,)``.

        Returns:
            Tuple of ``(node_logits, edge_logits)`` with shapes ``(N, 1)``
            and ``(E, 1)`` respectively.
        """
        _ = edge_attr  # reserved; GATConv does not consume edge weights directly

        h = self.input_proj(x)
        for layer_idx, gat in enumerate(self.gat_layers):
            h = gat(h, edge_index)
            if layer_idx < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        node_logits = self.node_scorer(h)

        if edge_index.size(1) == 0:
            edge_logits = torch.zeros((0, 1), device=x.device, dtype=x.dtype)
        else:
            src, tgt = edge_index[0], edge_index[1]
            edge_feats = torch.cat([h[src], h[tgt]], dim=-1)
            edge_logits = self.edge_scorer(edge_feats)

        return node_logits, edge_logits


def evidence_graph_to_pyg(graph: EvidenceGraph, device: torch.device | None = None) -> Data:
    """Convert an ``EvidenceGraph`` schema object to a PyG ``Data`` batch.

    Args:
        graph: Evidence graph with node embeddings and edges.
        device: Target torch device; defaults to CPU.

    Returns:
        ``torch_geometric.data.Data`` with ``x``, ``edge_index``, ``edge_attr``,
        and ``segment_ids`` (Python list aligned with node rows).
    """
    device = device or torch.device("cpu")

    if not graph.nodes:
        x = torch.zeros((0, config.EMBEDDING_DIM), dtype=torch.float32, device=device)
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0,), dtype=torch.float32, device=device)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, segment_ids=[])

    segment_ids = [n.segment_id for n in graph.nodes]
    id_to_idx = {sid: i for i, sid in enumerate(segment_ids)}

    x = torch.tensor([n.embedding for n in graph.nodes], dtype=torch.float32, device=device)

    src_list: list[int] = []
    tgt_list: list[int] = []
    weights: list[float] = []
    for edge in graph.edges:
        if edge.source in id_to_idx and edge.target in id_to_idx:
            src_list.append(id_to_idx[edge.source])
            tgt_list.append(id_to_idx[edge.target])
            weights.append(edge.weight)
            # Undirected: add reverse edge for message passing
            src_list.append(id_to_idx[edge.target])
            tgt_list.append(id_to_idx[edge.source])
            weights.append(edge.weight)

    if src_list:
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long, device=device)
        edge_attr = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0,), dtype=torch.float32, device=device)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.segment_ids = segment_ids
    return data


def gat_output_from_logits(
    segment_ids: list[str],
    node_logits: Tensor,
    edge_index: Tensor,
    edge_logits: Tensor,
    top_k: int | None = None,
) -> GATOutput:
    """Convert raw model outputs into the ``GATOutput`` API contract.

    Args:
        segment_ids: Node ids aligned with rows of ``node_logits``.
        node_logits: Tensor of shape ``(N, 1)``.
        edge_index: Tensor of shape ``(2, E)`` (may include reverse edges).
        edge_logits: Tensor of shape ``(E, 1)``.
        top_k: Number of top evidence ids to return.

    Returns:
        ``GATOutput`` with sigmoid-normalised scores.
    """
    k = top_k or config.GAT_TOP_EVIDENCE_K
    node_probs = torch.sigmoid(node_logits.squeeze(-1)).detach().cpu().tolist()
    node_scores = {sid: float(score) for sid, score in zip(segment_ids, node_probs)}

    edge_scores: dict[str, float] = {}
    if edge_index.numel() > 0:
        edge_probs = torch.sigmoid(edge_logits.squeeze(-1)).detach().cpu().tolist()
        seen: set[str] = set()
        for idx in range(edge_index.size(1)):
            src_idx = int(edge_index[0, idx].item())
            tgt_idx = int(edge_index[1, idx].item())
            key = f"{segment_ids[src_idx]}_{segment_ids[tgt_idx]}"
            if key not in seen:
                seen.add(key)
                edge_scores[key] = float(edge_probs[idx])

    ranked = sorted(node_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_evidence_ids = [sid for sid, _ in ranked[:k]]

    return GATOutput(
        node_scores=node_scores,
        edge_scores=edge_scores,
        top_evidence_ids=top_evidence_ids,
    )


def build_model(**kwargs: Any) -> EvidenceGAT:
    """Factory helper returning an untrained ``EvidenceGAT``."""
    return EvidenceGAT(**kwargs)
