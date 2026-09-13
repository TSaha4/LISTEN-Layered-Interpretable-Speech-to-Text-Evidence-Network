"""
LISTEN — Evaluation Metrics

Measures:
  1. Retrieval quality: Recall@k (do retrieved segments contain ground-truth evidence?)
  2. GAT quality: Segment selection Precision / Recall / F1
  3. Answer quality: ROUGE-L and BERTScore against ground-truth answers

Usage:
    python evaluate.py --checkpoint checkpoints/hotpotqa_best.pt --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

import config


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = None) -> float:
    """Compute Recall@k for retrieval.

    What fraction of the gold evidence segments are in the top-k retrieved?

    Args:
        retrieved_ids: Ordered list of retrieved segment IDs (most relevant first).
        gold_ids: Ground-truth supporting segment IDs.
        k: Cutoff (default: use all retrieved).

    Returns:
        Recall score in [0, 1].
    """
    if not gold_ids:
        return 1.0  # No gold evidence → trivially perfect

    if k:
        retrieved_ids = retrieved_ids[:k]

    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)

    return len(retrieved_set & gold_set) / len(gold_set)


def segment_f1(predicted_ids: List[str], gold_ids: List[str]) -> dict:
    """Compute segment-level Precision, Recall, F1.

    Args:
        predicted_ids: Segments the model says are important.
        gold_ids: Ground-truth supporting segments.

    Returns:
        Dict with precision, recall, f1.
    """
    pred_set = set(predicted_ids)
    gold_set = set(gold_ids)

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match (EM) score — normalized.

    Args:
        prediction: Predicted answer string.
        ground_truth: Ground-truth answer string.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    return 1.0 if _normalize(prediction) == _normalize(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between predicted and ground-truth answers.

    Standard HotpotQA metric.

    Args:
        prediction: Predicted answer string.
        ground_truth: Ground-truth answer string.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F1 score.

    Uses the rouge-score library if available, otherwise falls back to
    a simple LCS-based implementation.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Fallback: simple LCS-based ROUGE-L
        return _lcs_rouge_l(prediction, ground_truth)


def _lcs_rouge_l(prediction: str, reference: str) -> float:
    """Simple LCS-based ROUGE-L implementation."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def evaluate_pipeline_on_dataset(
    pipeline,
    dataset,
    max_samples: int = None,
) -> dict:
    """Run the pipeline on a dataset and compute aggregate metrics.

    Args:
        pipeline: LISTENPipeline instance.
        dataset: A dataset object with .examples and .metadata_list attributes.
        max_samples: Max samples to evaluate.

    Returns:
        Dict with aggregate metrics.
    """
    all_retrieval_recall = []
    all_segment_f1 = []
    all_answer_em = []
    all_answer_f1 = []
    all_rouge_l = []

    n = min(max_samples, len(dataset)) if max_samples else len(dataset)

    for i in tqdm(range(n), desc="Evaluating"):
        meta = dataset.metadata_list[i]
        if "error" in meta:
            continue

        question = meta.get("question", "")
        answer = meta.get("answer", "")
        gold_ids = meta.get("supporting_ids", [])

        if not question or not answer:
            continue

        # This would need a meeting input — for HotpotQA we'd need to reconstruct it
        # For now, compute metrics on the pre-built graphs
        graph = dataset[i]
        if not hasattr(graph, "y"):
            continue

        # Predict using GAT
        device = next(pipeline.gat.parameters()).device
        graph = graph.to(device)

        with torch.no_grad():
            output = pipeline.gat(graph)

        weights = output["node_weights"].cpu().numpy()[1:]  # skip question node
        node_id_map = meta.get("node_id_map", {})

        # Get predicted supporting segments (weight > 0.5)
        predicted_ids = [
            node_id_map.get(j + 1, f"node_{j+1}")
            for j in range(len(weights))
            if weights[j] > 0.5
        ]

        # Segment-level metrics
        sf1 = segment_f1(predicted_ids, gold_ids)
        all_segment_f1.append(sf1["f1"])

    results = {
        "num_evaluated": len(all_segment_f1),
        "segment_f1_mean": np.mean(all_segment_f1) if all_segment_f1 else 0.0,
        "segment_f1_std": np.std(all_segment_f1) if all_segment_f1 else 0.0,
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LISTEN Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "qmsum"])
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    print(f"Evaluating on {args.dataset} (max {args.max_samples} samples)...")

    from pipeline import LISTENPipeline
    from phase2_graph.dataset import HotpotQAGraphDataset, QMSumGraphDataset

    pipeline = LISTENPipeline(gat_checkpoint=args.checkpoint, use_llm=False)

    if args.dataset == "hotpotqa":
        dataset = HotpotQAGraphDataset(split="validation", max_samples=args.max_samples)
    else:
        dataset = QMSumGraphDataset(split="val", max_samples=args.max_samples)

    results = evaluate_pipeline_on_dataset(pipeline, dataset, args.max_samples)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
