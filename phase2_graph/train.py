"""
LISTEN Phase 2 — GAT Training Script

Trains the EvidenceGAT model on:
  1. HotpotQA distractor set (base paper reproduction, text-only multi-hop QA)
  2. QMSum meeting dataset (domain fine-tuning for spoken meetings)

Loss: Binary cross-entropy on node importance (is this segment a supporting fact?)
Metrics: Precision, Recall, F1 at the segment level.

Usage:
    # Train on HotpotQA (pre-training)
    python phase2_graph/train.py --dataset hotpotqa --epochs 20

    # Fine-tune on QMSum
    python phase2_graph/train.py --dataset qmsum --epochs 10 --resume checkpoints/hotpotqa_best.pt

    # Smoke test
    python phase2_graph/train.py --dataset hotpotqa --epochs 2 --max_samples 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from phase2_graph.gat_model import EvidenceGAT
from phase2_graph.dataset import HotpotQAGraphDataset, QMSumGraphDataset


def compute_metrics(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute segment-level classification metrics.

    Args:
        preds: Predicted importance scores (0-1).
        labels: Ground-truth binary labels.
        threshold: Threshold for binary classification.

    Returns:
        Dict with precision, recall, f1, accuracy.
    """
    binary_preds = (preds >= threshold).astype(int)
    binary_labels = labels.astype(int)

    # Handle edge cases
    if binary_labels.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 1.0}

    return {
        "precision": precision_score(binary_labels, binary_preds, zero_division=0),
        "recall": recall_score(binary_labels, binary_preds, zero_division=0),
        "f1": f1_score(binary_labels, binary_preds, zero_division=0),
        "accuracy": (binary_preds == binary_labels).mean(),
    }


def train_epoch(
    model: EvidenceGAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Train for one epoch.

    Returns:
        Dict with average loss and metrics.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        output = model(batch)
        logits = output["logits"]

        # Skip question nodes (index 0 per graph) in loss computation
        # For batched graphs, we need to identify question nodes
        # In our setup, node 0 of each graph is the question node
        mask = _get_segment_mask(batch)

        if mask.sum() == 0:
            continue

        loss = criterion(logits[mask], batch.y[mask])
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Collect predictions for metrics
        with torch.no_grad():
            preds = torch.sigmoid(logits[mask]).cpu().numpy()
            labels = batch.y[mask].cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)

    if num_batches == 0:
        return {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / num_batches

    return metrics


@torch.no_grad()
def evaluate(
    model: EvidenceGAT,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate on validation set.

    Returns:
        Dict with average loss and metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        output = model(batch)
        logits = output["logits"]

        mask = _get_segment_mask(batch)
        if mask.sum() == 0:
            continue

        loss = criterion(logits[mask], batch.y[mask])
        total_loss += loss.item()
        num_batches += 1

        preds = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = batch.y[mask].cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)

    if num_batches == 0:
        return {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / num_batches

    return metrics


def _get_segment_mask(batch) -> torch.Tensor:
    """Create a boolean mask that excludes question nodes (node 0 of each graph).

    In a PyG batched graph, batch.batch maps each node to its graph index.
    The question node is the first node of each graph.
    """
    mask = torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.x.device)

    # Find the first node of each graph in the batch
    if hasattr(batch, "ptr"):
        # ptr[i] = start index of graph i's nodes
        for start_idx in batch.ptr[:-1]:
            mask[start_idx] = False
    else:
        # Fallback: use batch.batch to find graph boundaries
        for graph_idx in batch.batch.unique():
            graph_nodes = (batch.batch == graph_idx).nonzero(as_tuple=True)[0]
            mask[graph_nodes[0]] = False  # First node = question node

    return mask


def train(
    dataset_name: str = "hotpotqa",
    epochs: int = None,
    resume_from: str = None,
    max_samples: int = None,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    save_dir: str = str(config.CHECKPOINT_DIR),
):
    """Main training function.

    Args:
        dataset_name: 'hotpotqa' or 'qmsum'.
        epochs: Number of training epochs (defaults from config).
        resume_from: Path to checkpoint to resume from (for fine-tuning).
        max_samples: Limit dataset size for debugging.
        batch_size: Training batch size.
        lr: Learning rate.
        save_dir: Directory to save checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load dataset ──
    if dataset_name == "hotpotqa":
        if epochs is None:
            epochs = config.EPOCHS_HOTPOTQA
        dataset = HotpotQAGraphDataset(split="train", max_samples=max_samples)
    elif dataset_name == "qmsum":
        if epochs is None:
            epochs = config.EPOCHS_QMSUM
        dataset = QMSumGraphDataset(split="train", max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Check data paths.")
        return

    # Train/val split
    split_idx = int(len(dataset) * config.TRAIN_VAL_SPLIT)
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(dataset)))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ── Initialize model ──
    model = EvidenceGAT().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume from checkpoint (for fine-tuning)
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # ── Training setup ──
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ──
    best_val_f1 = 0.0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | "
              f"P: {train_metrics['precision']:.3f} | "
              f"R: {train_metrics['recall']:.3f} | "
              f"F1: {train_metrics['f1']:.3f}")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | "
              f"P: {val_metrics['precision']:.3f} | "
              f"R: {val_metrics['recall']:.3f} | "
              f"F1: {val_metrics['f1']:.3f}")

        scheduler.step()

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_path = save_path / f"{dataset_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_metrics": val_metrics,
            }, best_path)
            print(f"  [SUCCESS] Saved best model (F1={best_val_f1:.3f}) -> {best_path}")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            ckpt_path = save_path / f"{dataset_name}_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.3f}")
    print(f"Best checkpoint: {save_path / f'{dataset_name}_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EvidenceGAT")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "qmsum"], help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (debug)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)

    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        epochs=args.epochs,
        resume_from=args.resume,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        lr=args.lr,
    )
