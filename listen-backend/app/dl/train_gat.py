"""Train the Evidence GAT on QMSum query–evidence pairs.

Usage (from project root)::

    python -m app.dl.train_gat --data-dir data/qmsum --epochs 20

The script builds segment-level graphs from QMSum meeting transcripts,
labels nodes using gold ``relevant_text_span`` annotations, and checkpoints
the best validation model to ``config.CHECKPOINT_DIR``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app import config
from app.data.loaders import QMSumSample, load_qmsum_split
from app.data.preprocessing import qmsum_sample_to_segments
from app.dl.embeddings import SegmentIndex, embed_texts
from app.dl.gat_model import EvidenceGAT, evidence_graph_to_pyg
from app.dl.graph_builder import build_evidence_graph
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import SLPSegment


@dataclass
class TrainExample:
    """One training graph with node-level binary labels."""

    graph_data: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    labels: torch.Tensor
    segment_ids: list[str]


class QMSumGraphDataset(Dataset):
    """PyTorch dataset of QMSum evidence graphs."""

    def __init__(self, samples: list[QMSumSample]) -> None:
        self.examples: list[TrainExample] = []
        for sample in samples:
            example = self._build_example(sample)
            if example is not None:
                self.examples.append(example)

    def _build_example(self, sample: QMSumSample) -> TrainExample | None:
        segments = qmsum_sample_to_segments(sample)
        if len(segments) < 2:
            return None

        index = SegmentIndex()
        index.build(segments)
        segments_by_id = {s.segment_id: s for s in segments}

        retrieval = retrieve_candidates(sample.query, index, segments_by_id, top_k=config.RETRIEVAL_TOP_K)
        candidate_ids = [c.segment_id for c in retrieval.candidates]
        if not candidate_ids:
            return None

        graph = build_evidence_graph(candidate_ids, segments_by_id, index=index)
        if not graph.nodes:
            return None

        pyg = evidence_graph_to_pyg(graph)
        labels = self._node_labels(pyg.segment_ids, sample.gold_turn_indices, segments_by_id)

        return TrainExample(
            graph_data=pyg.x,
            edge_index=pyg.edge_index,
            edge_attr=pyg.edge_attr,
            labels=labels,
            segment_ids=pyg.segment_ids,
        )

    @staticmethod
    def _node_labels(
        segment_ids: list[str],
        gold_turn_indices: set[int],
        segments_by_id: dict[str, SLPSegment],
    ) -> torch.Tensor:
        """Mark nodes whose turn index appears in QMSum gold spans."""
        labels: list[float] = []
        for sid in segment_ids:
            seg = segments_by_id[sid]
            # segment_id format: {meeting_id}_turn_{idx}
            turn_idx = int(sid.rsplit("_", 1)[-1])
            labels.append(1.0 if turn_idx in gold_turn_indices else 0.0)
        return torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainExample:
        return self.examples[idx]


def collate_single(batch: list[TrainExample]) -> TrainExample:
    """Batch size is fixed to 1 because graphs vary in size."""
    return batch[0]


def train_epoch(
    model: EvidenceGAT,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; returns average loss."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        x = batch.graph_data.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        labels = batch.labels.to(device)

        optimiser.zero_grad()
        node_logits, _ = model(x, edge_index, edge_attr)
        loss = criterion(node_logits.squeeze(-1), labels)
        loss.backward()
        optimiser.step()

        total_loss += float(loss.item())
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(
    model: EvidenceGAT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, accuracy) on a held-out split."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch.graph_data.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        labels = batch.labels.to(device)

        node_logits, _ = model(x, edge_index, edge_attr)
        loss = criterion(node_logits.squeeze(-1), labels)
        total_loss += float(loss.item())

        preds = (torch.sigmoid(node_logits.squeeze(-1)) >= 0.5).float()
        correct += int((preds == labels).sum().item())
        total += labels.numel()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_qmsum_gat(
    data_dir: str | Path,
    epochs: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Full training loop with checkpointing and JSON results log.

    Args:
        data_dir: Directory containing QMSum ``train`` / ``val`` JSON files.
        epochs: Override for ``config.GAT_EPOCHS``.
        checkpoint_path: Where to save the best model weights.

    Returns:
        Dictionary with per-epoch metrics (also written to ``GAT_TRAIN_LOG``).
    """
    data_dir = Path(data_dir)
    epochs = epochs or config.GAT_EPOCHS
    checkpoint_path = Path(checkpoint_path or config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME)
    device = torch.device("cpu")

    train_samples = load_qmsum_split(data_dir, "train")
    val_samples = load_qmsum_split(data_dir, "val")

    train_ds = QMSumGraphDataset(train_samples)
    val_ds = QMSumGraphDataset(val_samples)

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training examples built from {data_dir}. "
            "Ensure QMSum JSON files are present (see README)."
        )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    model = EvidenceGAT().to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.GAT_LEARNING_RATE,
        weight_decay=config.GAT_WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch}/{epochs} — "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    log_path = config.CHECKPOINT_DIR / config.GAT_TRAIN_LOG
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main() -> None:
    """CLI entrypoint for GAT training."""
    parser = argparse.ArgumentParser(description="Train LISTEN Evidence GAT on QMSum")
    parser.add_argument("--data-dir", type=str, default=str(config.QMSUM_DIR))
    parser.add_argument("--epochs", type=int, default=config.GAT_EPOCHS)
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME))
    args = parser.parse_args()

    train_qmsum_gat(args.data_dir, epochs=args.epochs, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
