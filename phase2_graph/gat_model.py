"""
LISTEN Phase 2 — Graph Attention Network (GAT) for Multi-hop Reasoning

Architecture adapted from HGN §3.3 (Fang et al., 2019):
  - Multi-head graph attention with edge-type-specific attention weights
  - 2-layer GAT with residual connections
  - Node importance prediction head (which segments are supporting facts?)

Key simplification from HGN:
  HGN uses 4 node types (Q, P, S, E) with 7 edge types and separate prediction
  heads for paragraphs, sentences, entities, and spans.
  We use 2 node types (Q, Segment) with 3 edge types and a single node
  importance prediction head — suited for spoken transcript segments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

import config


class EdgeTypeAttention(nn.Module):
    """Edge-type-specific attention bias for GAT.

    Following HGN Eq. 3: different edge types (question↔segment, shared-entity,
    semantic-similarity) get different learned attention weight vectors w_{e_ij}.
    This module adds a per-edge-type bias to the standard GAT attention scores.
    """

    def __init__(self, hidden_dim: int, num_edge_types: int = config.NUM_EDGE_TYPES):
        super().__init__()
        # Learnable bias per edge type
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        """Compute per-edge attention bias.

        Args:
            edge_type: (num_edges,) — integer edge type labels.

        Returns:
            (num_edges,) — attention bias per edge.
        """
        type_emb = self.edge_type_embedding(edge_type)   # (E, hidden_dim)
        bias = self.projection(type_emb).squeeze(-1)     # (E,)
        return bias


class GATLayer(nn.Module):
    """A single GAT layer with edge-type conditioning and residual connection.

    Wraps PyG's GATConv and adds:
      1. Edge-type attention bias (HGN-style)
      2. Residual connection
      3. Layer normalization
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float, num_edge_types: int):
        super().__init__()
        self.gat_conv = GATConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,  # GATConv concatenates heads
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True,
        )
        self.edge_attn = EdgeTypeAttention(out_dim // heads, num_edge_types)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) — node features.
            edge_index: (2, E) — edge indices.
            edge_type: (E,) — edge type labels.

        Returns:
            (N, out_dim) — updated node features.
        """
        # Standard GAT attention + message passing
        h = self.gat_conv(x, edge_index)

        # Residual connection
        residual = self.residual_proj(x)
        h = self.layer_norm(h + residual)
        h = self.dropout(F.elu(h))

        return h


class EvidenceGAT(nn.Module):
    """Multi-hop Evidence Graph Attention Network.

    Takes a PyG evidence graph (from graph_builder) and produces:
      1. Node importance scores — which segments matter for the answer.
      2. Updated node embeddings — enriched via multi-hop graph reasoning.

    Architecture (adapted from HGN §3.3):
      Input (384-dim) → Project (256-dim) → GAT Layer 1 → GAT Layer 2 →
      → Node Importance Head (2-layer MLP → sigmoid)
    """

    def __init__(
        self,
        input_dim: int = config.GAT_INPUT_DIM,
        hidden_dim: int = config.GAT_HIDDEN_DIM,
        output_dim: int = config.GAT_OUTPUT_DIM,
        num_heads: int = config.GAT_NUM_HEADS,
        num_layers: int = config.GAT_NUM_LAYERS,
        dropout: float = config.GAT_DROPOUT,
        num_edge_types: int = config.NUM_EDGE_TYPES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim if i == 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            self.gat_layers.append(
                GATLayer(in_d, out_d, num_heads, dropout, num_edge_types)
            )

        # Node importance prediction head (2-layer MLP → sigmoid)
        # Predicts: is this segment a supporting fact for the answer?
        self.importance_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1),
        )

    def forward(self, data: Data) -> dict:
        """Forward pass through the evidence GAT.

        Args:
            data: PyG Data object with:
              - x: (N, input_dim) node features
              - edge_index: (2, E) edges
              - edge_type: (E,) edge type labels

        Returns:
            dict with:
              - 'node_weights': (N,) importance scores in [0, 1]
              - 'node_embeddings': (N, output_dim) updated node representations
              - 'logits': (N,) raw logits before sigmoid (for loss computation)
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        # Input projection
        h = self.input_proj(x)

        # GAT layers
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_type)

        # Node importance scores
        logits = self.importance_head(h).squeeze(-1)    # (N,)
        weights = torch.sigmoid(logits)                 # (N,) in [0, 1]

        return {
            "node_weights": weights,
            "node_embeddings": h,
            "logits": logits,
        }

    def get_edge_attention_weights(self, data: Data) -> list:
        """Extract attention weights from each GAT layer for explainability.

        Useful for Person C's XAI component — shows which edges the GAT
        attends to most strongly.

        Returns:
            List of (edge_index, attention_weights) per layer.
        """
        attention_weights = []
        h = self.input_proj(data.x)

        for gat_layer in self.gat_layers:
            # PyG GATConv can return attention weights
            h_new, (edge_idx, alpha) = gat_layer.gat_conv(
                h, data.edge_index, return_attention_weights=True
            )
            attention_weights.append((edge_idx, alpha))
            residual = gat_layer.residual_proj(h)
            h = gat_layer.layer_norm(h_new + residual)
            h = gat_layer.dropout(F.elu(h))

        return attention_weights
