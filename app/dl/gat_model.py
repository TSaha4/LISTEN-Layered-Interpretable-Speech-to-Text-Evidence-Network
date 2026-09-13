"""Checkpoint-compatible graph-attention model and graph conversion helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from app import config
from app.models.schemas import EvidenceGraph, GATOutput


class EvidenceGAT(nn.Module):
    """Single-level GAT matching the trained QMSum checkpoint exactly."""

    def __init__(self, in_dim: int | None = None, hidden_dim: int | None = None, num_heads: int | None = None, num_layers: int | None = None, dropout: float | None = None) -> None:
        super().__init__()
        self.in_dim = in_dim or config.EMBEDDING_DIM
        self.hidden_dim = hidden_dim or config.GAT_HIDDEN_DIM
        self.num_heads = num_heads or config.GAT_NUM_HEADS
        self.num_layers = num_layers or config.GAT_NUM_LAYERS
        self.dropout = dropout if dropout is not None else config.GAT_DROPOUT
        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.gat_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            if layer_idx == self.num_layers - 1:
                self.gat_layers.append(GATConv(self.hidden_dim, self.hidden_dim, heads=1, concat=False, dropout=self.dropout, add_self_loops=True))
            else:
                self.gat_layers.append(GATConv(self.hidden_dim, self.hidden_dim // self.num_heads, heads=self.num_heads, concat=True, dropout=self.dropout, add_self_loops=True))
        self.node_scorer = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim // 2, 1))
        self.edge_scorer = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim // 2, 1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return raw node and edge relevance logits for a PyG graph."""
        _ = edge_attr
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
            edge_logits = self.edge_scorer(torch.cat([h[src], h[tgt]], dim=-1))
        return node_logits, edge_logits


def evidence_graph_to_pyg(graph: EvidenceGraph, device: torch.device | None = None) -> Data:
    """Convert an evidence graph schema object to a PyG graph."""
    device = device or torch.device("cpu")
    if not graph.nodes:
        return Data(x=torch.zeros((0, config.EMBEDDING_DIM), dtype=torch.float32, device=device), edge_index=torch.zeros((2, 0), dtype=torch.long, device=device), edge_attr=torch.zeros((0,), dtype=torch.float32, device=device), segment_ids=[])
    segment_ids = [node.segment_id for node in graph.nodes]
    id_to_idx = {segment_id: idx for idx, segment_id in enumerate(segment_ids)}
    x = torch.tensor([node.embedding for node in graph.nodes], dtype=torch.float32, device=device)
    src_list: list[int] = []
    tgt_list: list[int] = []
    weights: list[float] = []
    for edge in graph.edges:
        if edge.source in id_to_idx and edge.target in id_to_idx:
            src_list.extend([id_to_idx[edge.source], id_to_idx[edge.target]])
            tgt_list.extend([id_to_idx[edge.target], id_to_idx[edge.source]])
            weights.extend([edge.weight, edge.weight])
    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long, device=device) if src_list else torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.tensor(weights, dtype=torch.float32, device=device) if weights else torch.zeros((0,), dtype=torch.float32, device=device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.segment_ids = segment_ids
    return data


def gat_output_from_logits(segment_ids: list[str], node_logits: Tensor, edge_index: Tensor, edge_logits: Tensor, top_k: int | None = None) -> GATOutput:
    """Convert model logits into serialisable evidence scores."""
    k = top_k or config.GAT_TOP_EVIDENCE_K
    node_scores = {segment_id: float(score) for segment_id, score in zip(segment_ids, torch.sigmoid(node_logits.squeeze(-1)).detach().cpu().tolist())}
    edge_scores: dict[str, float] = {}
    if edge_index.numel() > 0:
        for idx, score in enumerate(torch.sigmoid(edge_logits.squeeze(-1)).detach().cpu().tolist()):
            source = segment_ids[int(edge_index[0, idx])]
            target = segment_ids[int(edge_index[1, idx])]
            edge_scores.setdefault(f"{source}::{target}", float(score))
    return GATOutput(node_scores=node_scores, edge_scores=edge_scores, top_evidence_ids=[segment_id for segment_id, _ in sorted(node_scores.items(), key=lambda item: item[1], reverse=True)[:k]])
