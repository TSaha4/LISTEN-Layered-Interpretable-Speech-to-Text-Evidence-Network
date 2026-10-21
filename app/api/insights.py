"""Insights API endpoints providing real on-disk metrics for LISTEN."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from app import config

router = APIRouter(prefix="/insights", tags=["insights"])

TRAIN_LOG_PATH = config.CHECKPOINT_DIR / "gat_train_log.json"
PROCESSED_DIR = config.DATA_DIR / "processed"
ABLATION_PATH = PROCESSED_DIR / "ablation.json"
QMSUM_PRODUCT_DIR = config.DATA_DIR / "qmsum_product"


@router.get("/training")
async def get_training_insights() -> dict[str, Any]:
    """Return GAT training dynamics and model architecture metrics."""
    result: dict[str, Any] = {
        "available": False,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "best_val_f1": None,
        "best_val_accuracy": None,
        "total_parameters": None,
        "trainable_parameters": None,
        "architecture": {
            "input_dim": config.GAT_INPUT_DIM,
            "hidden_dim": config.GAT_HIDDEN_DIM,
            "output_dim": config.GAT_OUTPUT_DIM,
            "num_heads": config.GAT_NUM_HEADS,
            "num_layers": config.GAT_NUM_LAYERS,
            "dropout": config.GAT_DROPOUT,
            "num_edge_types": config.NUM_EDGE_TYPES,
        },
    }

    if TRAIN_LOG_PATH.is_file():
        try:
            with open(TRAIN_LOG_PATH, encoding="utf-8") as f:
                log_data = json.load(f)

            train_loss = log_data.get("train_loss", [])
            val_loss = log_data.get("val_loss", [])
            val_accuracy = log_data.get("val_accuracy", [])
            val_f1 = log_data.get("val_f1", [])

            n_epochs = max(len(train_loss), len(val_loss), len(val_accuracy), len(val_f1))
            epochs = list(range(1, n_epochs + 1))

            best_f1_obj = None
            if val_f1:
                max_f1 = max(val_f1)
                best_epoch = val_f1.index(max_f1) + 1
                best_f1_obj = {"value": float(max_f1), "epoch": best_epoch}

            best_acc_obj = None
            if val_accuracy:
                max_acc = max(val_accuracy)
                best_acc_epoch = val_accuracy.index(max_acc) + 1
                best_acc_obj = {"value": float(max_acc), "epoch": best_acc_epoch}

            result.update({
                "available": True,
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "best_val_f1": best_f1_obj,
                "best_val_accuracy": best_acc_obj,
            })
        except Exception:
            pass

    # Model parameter counts
    try:
        from app.dl.inference import get_gat_inference
        model = get_gat_inference().model
        result["total_parameters"] = sum(p.numel() for p in model.parameters())
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        result["total_parameters"] = None
        result["trainable_parameters"] = None

    return result


@router.get("/dataset")
async def get_dataset_insights() -> dict[str, Any]:
    """Return dataset metrics computed from data/processed and data/qmsum_product."""
    aligned_files = sorted(PROCESSED_DIR.glob("*_aligned_labels.json"))
    if not aligned_files:
        return {
            "available": False,
            "meetings": [],
            "totals": None,
        }

    meetings = []
    all_sims: list[float] = []
    all_entities: list[str] = []
    total_turns = 0
    total_kept = 0
    threshold = 0.4

    for align_file in aligned_files:
        meeting_id = align_file.stem.replace("_aligned_labels", "")
        seg_file = PROCESSED_DIR / f"{meeting_id}_segments.json"
        segs = []
        if seg_file.is_file():
            try:
                with open(seg_file, encoding="utf-8") as f:
                    segs = json.load(f)
            except Exception:
                segs = []

        duration = float(segs[-1]["end_time"]) if segs and "end_time" in segs[-1] else 0.0
        for s in segs:
            for ent in s.get("entities", []):
                all_entities.append(ent)

        try:
            with open(align_file, encoding="utf-8") as f:
                align_data = json.load(f)
        except Exception:
            align_data = {}

        threshold = align_data.get("confidence_threshold", threshold)
        t_count = int(align_data.get("turn_count", 0))
        k_count = int(align_data.get("kept_turn_count", len(align_data.get("matches", []))))
        total_turns += t_count
        total_kept += k_count

        matches = align_data.get("matches", [])
        for m in matches:
            if "semantic_similarity" in m:
                all_sims.append(float(m["semantic_similarity"]))

        queries = [
            {
                "query": q.get("query", ""),
                "gold_evidence_count": len(q.get("evidence_segment_ids", [])),
            }
            for q in align_data.get("queries", [])
        ]

        meetings.append({
            "id": meeting_id,
            "duration_s": duration,
            "segment_count": len(segs),
            "kept_alignment_count": k_count,
            "queries": queries,
        })

    percent_filtered = round(((total_turns - total_kept) / total_turns * 100), 2) if total_turns > 0 else 0.0
    mean_similarity = round(sum(all_sims) / len(all_sims), 4) if all_sims else None

    # Similarity histogram bins
    bins = [
        {"range": "0.40 - 0.50", "min": 0.4, "max": 0.5, "count": sum(1 for s in all_sims if 0.4 <= s < 0.5)},
        {"range": "0.50 - 0.60", "min": 0.5, "max": 0.6, "count": sum(1 for s in all_sims if 0.5 <= s < 0.6)},
        {"range": "0.60 - 0.70", "min": 0.6, "max": 0.7, "count": sum(1 for s in all_sims if 0.6 <= s < 0.7)},
        {"range": "0.70 - 0.80", "min": 0.7, "max": 0.8, "count": sum(1 for s in all_sims if 0.7 <= s < 0.8)},
        {"range": "0.80 - 0.90", "min": 0.8, "max": 0.9, "count": sum(1 for s in all_sims if 0.8 <= s < 0.9)},
        {"range": "0.90 - 1.00", "min": 0.9, "max": 1.01, "count": sum(1 for s in all_sims if 0.9 <= s <= 1.01)},
    ]

    top_entities = [{"entity": k, "count": v} for k, v in Counter(all_entities).most_common(10)]

    train_dir = QMSUM_PRODUCT_DIR / "train"
    val_dir = QMSUM_PRODUCT_DIR / "val"
    train_count = len(list(train_dir.glob("*.json"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.json"))) if val_dir.exists() else 0

    return {
        "available": True,
        "meetings": meetings,
        "totals": {
            "total_qmsum_turns": total_turns,
            "total_kept_turns": total_kept,
            "percent_filtered": percent_filtered,
            "mean_similarity": mean_similarity,
            "threshold_value": threshold,
            "similarity_histogram": bins,
            "top_entities": top_entities,
            "train_val_split": {"train": train_count, "val": val_count},
        },
    }


@router.get("/ablation")
async def get_ablation_insights() -> dict[str, Any]:
    """Serve real ablation results from data/processed/ablation.json."""
    if not ABLATION_PATH.is_file():
        return {
            "available": False,
            "detail": "Ablation results artifact not available on disk.",
        }

    try:
        with open(ABLATION_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "available": True,
            **data,
        }
    except Exception as exc:
        return {
            "available": False,
            "detail": f"Could not load ablation artifact: {exc}",
        }
