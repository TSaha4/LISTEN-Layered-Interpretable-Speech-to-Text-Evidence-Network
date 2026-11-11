"""Evaluate the uncertainty chain: does reliability gating help under noise?

For every noise condition produced by ``make_noisy_transcripts.py`` and every
labelled query, the script runs retrieval → graph → GAT and scores the top-k
evidence against the gold segment ids, with the reliability gate at several
``beta`` values (``beta = 0`` is the ungated baseline, i.e. the original
LISTEN). It also fits the answer-reliability calibration leave-one-query-out
and reports Expected Calibration Error before and after.

Everything is deterministic given the transcripts, so numbers are
reproducible. Output: data/processed/uncertainty_eval.json plus a Markdown
table on stdout.

Usage:
    LISTEN_DATA_DIR=../data python scripts/evaluate_uncertainty.py --meeting ES2002a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.dl.embeddings import SegmentIndex, embed_texts
from app.dl.gat_model import ReliabilityGate
from app.dl.graph_builder import EvidenceGraphBuilder
from app.dl.inference import GATInference
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import GATOutput, SLPSegment
from app.xai.reliability import (
    CalibrationParams,
    calibrate,
    evidence_agreement_component,
    evidence_reliability_component,
    retrieval_margin_component,
)

TOP_K = config.GAT_TOP_EVIDENCE_K


def prf(pred: list[str], gold: set[str]) -> tuple[float, float, float]:
    p = set(pred)
    if not p or not gold:
        return 0.0, 0.0, 0.0
    tp = len(p & gold)
    prec, rec = tp / len(p), tp / len(gold)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def ece(probs: list[float], labels: list[int], bins: int = 5) -> float:
    probs_a, labels_a = np.asarray(probs), np.asarray(labels, dtype=float)
    if len(probs_a) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs_a >= lo) & (probs_a < hi if hi < 1 else probs_a <= hi)
        if m.any():
            total += m.mean() * abs(probs_a[m].mean() - labels_a[m].mean())
    return float(total)


def run_condition(
    label: str,
    segments: list[SLPSegment],
    queries: list[dict],
    gat: GATInference,
    betas: list[float],
) -> dict:
    segments_by_id = {s.segment_id: s for s in segments}
    index = SegmentIndex()
    index.build(segments)
    builder = EvidenceGraphBuilder()

    per_query: list[dict] = []
    for q in queries:
        gold = set(q["evidence_segment_ids"])
        retrieval = retrieve_candidates(q["query"], index, segments_by_id)
        cand_ids = [c.segment_id for c in retrieval.candidates]
        segs = [segments_by_id[s] for s in cand_ids]
        emb = np.array([index.get_embedding(s) for s in cand_ids])
        q_emb = embed_texts([q["query"]])[0]
        data, meta = builder.build(q_emb, segs, emb)

        with torch.no_grad():
            node_logits, edge_logits = gat.model(data.x, data.edge_index, None)
            node_probs = torch.sigmoid(node_logits.squeeze(-1))
            edge_probs = torch.sigmoid(edge_logits.squeeze(-1))

        row = {
            "query": q["query"],
            "gold_count": len(gold),
            "retrieval_prf": prf(cand_ids[:TOP_K], gold),
            "gated_entity_edges": meta["gated_entity_edges"],
            "mean_reliability_gold": float(np.mean([meta["node_reliability"].get(g, 1.0) for g in gold if g in meta["node_reliability"]] or [float("nan")])),
            "mean_reliability_nongold": float(np.mean([r for sid, r in meta["node_reliability"].items() if sid not in gold and sid != "__question__"] or [float("nan")])),
            "beta": {},
        }
        for beta in betas:
            gate = ReliabilityGate(node_beta=beta, edge_beta=beta / 2)
            gated_nodes, _ = gate(node_probs, data.node_reliability, edge_probs, data.edge_confidence)
            scores = {meta["node_id_map"][i]: float(gated_nodes[i]) for i in range(data.num_nodes) if meta["node_id_map"][i] != "__question__"}
            top = [sid for sid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]]
            p, r, f = prf(top, gold)
            gat_out = GATOutput(
                node_scores=scores, edge_scores={}, top_evidence_ids=top,
                raw_node_scores={sid: float(node_probs[i]) for i, sid in meta["node_id_map"].items() if sid != "__question__"},
                node_reliability={sid: v for sid, v in meta["node_reliability"].items() if sid != "__question__"},
            )
            ctx_emb = np.array([index.get_embedding(s) for s in top])
            row["beta"][str(beta)] = {
                "prf": (p, r, f),
                "top": top,
                "components": {
                    "evidence_reliability": evidence_reliability_component(gat_out, segments_by_id, top),
                    "retrieval_margin": retrieval_margin_component(retrieval),
                    "evidence_agreement": evidence_agreement_component(ctx_emb),
                    "grounding_supported": 1.0,
                },
            }
        per_query.append(row)

    summary = {"condition": label, "n_queries": len(per_query), "mean_asr_confidence": float(np.mean([s.asr_confidence for s in segments]))}
    summary["retrieval_f1"] = float(np.mean([r["retrieval_prf"][2] for r in per_query]))
    summary["mean_reliability_gold"] = float(np.nanmean([r["mean_reliability_gold"] for r in per_query]))
    summary["mean_reliability_nongold"] = float(np.nanmean([r["mean_reliability_nongold"] for r in per_query]))
    summary["gated_entity_edges"] = int(sum(r["gated_entity_edges"] for r in per_query))
    summary["beta"] = {}
    for beta in betas:
        key = str(beta)
        vals = np.array([r["beta"][key]["prf"] for r in per_query])
        summary["beta"][key] = {"precision": float(vals[:, 0].mean()), "recall": float(vals[:, 1].mean()), "f1": float(vals[:, 2].mean())}
    return {"summary": summary, "per_query": per_query}


def calibration_report(results: list[dict], beta_key: str, correct_f1: float) -> dict:
    """Leave-one-query-out calibration of the answer-reliability score.

    The label is a proxy: an answer counts as correct when the gated top-k
    evidence reaches ``correct_f1`` against gold. No LLM is called.
    """
    rows, labels, groups = [], [], []
    for res in results:
        for r in res["per_query"]:
            b = r["beta"][beta_key]
            rows.append(b["components"])
            labels.append(1 if b["prf"][2] >= correct_f1 else 0)
            groups.append(r["query"])
    default = CalibrationParams()

    def score(p: CalibrationParams, c: dict) -> float:
        z = p.bias + p.w_evidence * c["evidence_reliability"] + p.w_margin * c["retrieval_margin"] + p.w_agreement * c["evidence_agreement"] + p.w_supported * c["grounding_supported"]
        return float(1 / (1 + np.exp(-z)))

    before = [score(default, c) for c in rows]
    after: list[float] = []
    for held in sorted(set(groups)):
        train_rows = [c for c, g in zip(rows, groups) if g != held]
        train_lab = [l for l, g in zip(labels, groups) if g != held]
        params = calibrate(train_rows, train_lab, epochs=800, lr=0.3)
        after.extend(score(params, c) for c, g in zip(rows, groups) if g == held)
    labels_held = [l for g_sorted in sorted(set(groups)) for l, g in zip(labels, groups) if g == g_sorted]
    return {
        "n": len(rows), "positive_rate": float(np.mean(labels)),
        "ece_default_weights": ece(before, labels),
        "ece_leave_one_out": ece(after, labels_held),
        "brier_default": float(np.mean((np.array(before) - np.array(labels)) ** 2)),
        "brier_leave_one_out": float(np.mean((np.array(after) - np.array(labels_held)) ** 2)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="ES2002a")
    ap.add_argument("--conditions", nargs="+", default=["clean", "snr10", "snr5", "snr0"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--correct-f1", type=float, default=0.3)
    args = ap.parse_args()

    processed = config.DATA_DIR / "processed"
    queries = json.loads((processed / f"{args.meeting}_aligned_labels.json").read_text(encoding="utf-8"))["queries"]
    gat = GATInference()

    results = []
    for cond in args.conditions:
        path = processed / f"{args.meeting}_segments_{cond}.json"
        if not path.is_file():
            print(f"skip {cond}: {path.name} missing")
            continue
        segments = [SLPSegment(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
        results.append(run_condition(cond, segments, queries, gat, args.betas))

    betas = [str(b) for b in args.betas]
    print(f"\n### Evidence F1@{TOP_K} on {args.meeting} ({len(queries)} labelled queries)\n")
    print("| condition | mean ASR conf | rel(gold) | rel(non-gold) | gated entity edges | retrieval-only | " + " | ".join(f"beta={b}" for b in betas) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(betas))
    for res in results:
        s = res["summary"]
        cells = [f"{s['beta'][b]['f1']:.3f}" for b in betas]
        print(f"| {s['condition']} | {s['mean_asr_confidence']:.3f} | {s['mean_reliability_gold']:.3f} | {s['mean_reliability_nongold']:.3f} | {s['gated_entity_edges']} | {s['retrieval_f1']:.3f} | " + " | ".join(cells) + " |")

    mean_cells = [f"{np.mean([r['summary']['beta'][b]['f1'] for r in results]):.3f}" for b in betas]
    print(f"| **mean** | | | | | {np.mean([r['summary']['retrieval_f1'] for r in results]):.3f} | " + " | ".join(mean_cells) + " |")

    best_beta = max(betas, key=lambda b: np.mean([r["summary"]["beta"][b]["f1"] for r in results]))
    calib = {b: calibration_report(results, b, args.correct_f1) for b in ("0.0", best_beta)} if "0.0" in betas else {}
    print(f"\nbest beta across conditions: {best_beta}")
    for b, c in calib.items():
        print(f"calibration (beta={b}): n={c['n']} pos={c['positive_rate']:.2f} ECE default={c['ece_default_weights']:.3f} → LOO-fit={c['ece_leave_one_out']:.3f}; Brier {c['brier_default']:.3f} → {c['brier_leave_one_out']:.3f}")

    out = processed / "uncertainty_eval.json"
    out.write_text(json.dumps({"meeting": args.meeting, "top_k": TOP_K, "best_beta": best_beta, "conditions": [r["summary"] for r in results], "calibration": calib, "per_query": [r["per_query"] for r in results]}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
