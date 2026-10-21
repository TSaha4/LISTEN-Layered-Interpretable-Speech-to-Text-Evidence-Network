"""LISTEN — Compute Real Ablation over Ground-Truth Aligned Meeting Queries.

Reads app code and models, does NOT modify any training or model logic.
Generates data/processed/ablation.json containing real hits and totals for:
  - Retrieval-only (Top-5)
  - Trained GAT (Top-5)
  - Untrained EvidenceGAT (seed 42) (Top-5)
  - LLM Context Set (hits and size)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.dl.embeddings import SegmentIndex, embed_texts
from app.dl.gat_model import EvidenceGAT
from app.dl.graph_builder import EvidenceGraphBuilder
from app.dl.inference import GATInference
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import SLPSegment
from app.slp.entity_extraction import enrich_segments_with_entities


def compute_ablation() -> dict:
    processed_dir = config.DATA_DIR / "processed"
    aligned_files = sorted(processed_dir.glob("*_aligned_labels.json"))

    if not aligned_files:
        print("No aligned labels found in data/processed/")
        return {}

    trained_checkpoint = config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME
    trained_gat = GATInference(checkpoint_path=trained_checkpoint)

    # Untrained model with seed 42
    torch.manual_seed(42)
    untrained_gat = EvidenceGAT()
    untrained_gat.eval()

    builder = EvidenceGraphBuilder()

    queries_result = []
    meetings_processed = []

    for align_file in aligned_files:
        meeting_id = align_file.stem.replace("_aligned_labels", "")
        seg_file = processed_dir / f"{meeting_id}_segments.json"
        if not seg_file.is_file():
            continue

        meetings_processed.append(meeting_id)

        with open(seg_file, encoding="utf-8") as f:
            raw_segs = json.load(f)
        segments = [SLPSegment(**row) for row in raw_segs]
        segments = enrich_segments_with_entities(segments)
        segments_by_id = {s.segment_id: s for s in segments}

        index = SegmentIndex()
        index.build(segments)

        with open(align_file, encoding="utf-8") as f:
            align_data = json.load(f)

        queries = align_data.get("queries", [])
        for idx, q_item in enumerate(queries):
            q_text = q_item.get("query", "")
            gold_ids = set(q_item.get("evidence_segment_ids", []))

            # (a) Retrieval only top-5
            retrieval = retrieve_candidates(q_text, index, segments_by_id)
            ret_cand_ids = [c.segment_id for c in retrieval.candidates]
            ret_top5 = ret_cand_ids[:5]
            hit_ret = len(set(ret_top5) & gold_ids)

            # Build graph using production EvidenceGraphBuilder
            q_emb = embed_texts([q_text])[0]
            cands = [segments_by_id[sid] for sid in ret_cand_ids]
            cand_embs = np.stack([index.get_embedding(sid) for sid in ret_cand_ids])
            data, meta = builder.build(q_emb, cands, cand_embs)

            # (b) Trained GAT ranking top-5
            trained_out = trained_gat.run(data, meta, top_k=5)
            gat_top5 = trained_out.top_evidence_ids[:5]
            hit_gat = len(set(gat_top5) & gold_ids)

            # (c) Untrained EvidenceGAT (seed 42) top-5
            with torch.no_grad():
                node_logits, _ = untrained_gat(data.x, data.edge_index, getattr(data, "edge_attr", None))
                scores = torch.sigmoid(node_logits.squeeze(-1)).cpu().numpy()
                node_id_map = meta["node_id_map"]
                seg_scores = [
                    (node_id_map[i], float(scores[i]))
                    for i in range(data.num_nodes)
                    if node_id_map[i] != "__question__"
                ]
                untrained_top5 = [sid for sid, _ in sorted(seg_scores, key=lambda x: x[1], reverse=True)[:5]]
            hit_untrained = len(set(untrained_top5) & gold_ids)

            # (d) LLM context set (GAT top-5 + raw retrieval top-5)
            llm_context_ids = list(dict.fromkeys(gat_top5 + ret_cand_ids[:5]))
            hit_llm = len(set(llm_context_ids) & gold_ids)

            queries_result.append({
                "meeting_id": meeting_id,
                "query_idx": len(queries_result) + 1,
                "query": q_text,
                "total_gold": len(gold_ids),
                "retrieval_hits": hit_ret,
                "trained_gat_hits": hit_gat,
                "untrained_gat_hits": hit_untrained,
                "llm_context_hits": hit_llm,
                "llm_context_size": len(llm_context_ids),
            })

    totals = {
        "total_gold": sum(r["total_gold"] for r in queries_result),
        "retrieval_hits": sum(r["retrieval_hits"] for r in queries_result),
        "trained_gat_hits": sum(r["trained_gat_hits"] for r in queries_result),
        "untrained_gat_hits": sum(r["untrained_gat_hits"] for r in queries_result),
        "llm_context_hits": sum(r["llm_context_hits"] for r in queries_result),
    }

    out_data = {
        "n_meetings": len(meetings_processed),
        "n_queries": len(queries_result),
        "meetings": meetings_processed,
        "totals": totals,
        "queries": queries_result,
    }

    out_file = processed_dir / "ablation.json"
    out_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"Saved ablation results to {out_file}")
    return out_data


if __name__ == "__main__":
    compute_ablation()
