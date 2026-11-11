"""Learn the ReliabilityGate exponents from labelled evidence under noise.

The pretrained GAT is frozen. Only ``beta_node`` / ``beta_edge`` are
optimised, with binary cross-entropy on gated node scores against the gold
evidence labels, pooled over every noise condition so the gate learns how
much to trust reliability across clean and noisy speech. The result is
written next to the GAT checkpoint as ``gate_state_dict`` so
``GATInference`` picks it up automatically.

Usage:
    LISTEN_DATA_DIR=../data python scripts/fit_reliability_gate.py --meeting ES2002a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.dl.embeddings import SegmentIndex, embed_texts
from app.dl.gat_model import ReliabilityGate
from app.dl.graph_builder import EvidenceGraphBuilder
from app.dl.inference import GATInference
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import SLPSegment


def build_examples(meeting: str, conditions: list[str], gat: GATInference) -> list[dict]:
    processed = config.DATA_DIR / "processed"
    queries = json.loads((processed / f"{meeting}_aligned_labels.json").read_text(encoding="utf-8"))["queries"]
    builder = EvidenceGraphBuilder()
    examples = []
    for cond in conditions:
        path = processed / f"{meeting}_segments_{cond}.json"
        if not path.is_file():
            continue
        segments = [SLPSegment(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
        by_id = {s.segment_id: s for s in segments}
        index = SegmentIndex()
        index.build(segments)
        for q in queries:
            gold = set(q["evidence_segment_ids"])
            retrieval = retrieve_candidates(q["query"], index, by_id)
            ids = [c.segment_id for c in retrieval.candidates]
            emb = np.array([index.get_embedding(s) for s in ids])
            data, meta = builder.build(embed_texts([q["query"]])[0], [by_id[s] for s in ids], emb)
            with torch.no_grad():
                node_logits, edge_logits = gat.model(data.x, data.edge_index, None)
            labels = torch.tensor([0.0] + [1.0 if s in gold else 0.0 for s in ids])
            examples.append({
                "node_probs": torch.sigmoid(node_logits.squeeze(-1)), "rel": data.node_reliability,
                "edge_probs": torch.sigmoid(edge_logits.squeeze(-1)), "edge_conf": data.edge_confidence,
                "labels": labels, "condition": cond, "query": q["query"],
            })
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="ES2002a")
    ap.add_argument("--conditions", nargs="+", default=["clean", "snr10", "snr5", "snr0"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--write", action="store_true", help="store gate_state_dict in the GAT checkpoint")
    args = ap.parse_args()

    gat = GATInference()
    examples = build_examples(args.meeting, args.conditions, gat)
    print(f"{len(examples)} (query, condition) graphs")

    gate = ReliabilityGate(node_beta=1.0, edge_beta=0.5)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    # Positives are rare (≈5 of 15 candidates); weight them so the gate is
    # not rewarded simply for pushing every score down.
    pos = sum(float(e["labels"][1:].sum()) for e in examples)
    neg = sum(float((1 - e["labels"][1:]).sum()) for e in examples)
    pos_weight = torch.tensor(neg / max(pos, 1.0))

    def loss_fn() -> torch.Tensor:
        total = 0.0
        for e in examples:
            gated, _ = gate(e["node_probs"], e["rel"], e["edge_probs"], e["edge_conf"])
            g = gated[1:].clamp(1e-5, 1 - 1e-5)
            total = total + F.binary_cross_entropy(g, e["labels"][1:], weight=torch.where(e["labels"][1:] > 0, pos_weight, torch.ones(1)))
        return total / len(examples)

    with torch.no_grad():
        print(f"loss at beta_node=1.0: {loss_fn():.4f}")
        gate.node_beta.fill_(0.0); gate.edge_beta.fill_(0.0)
        print(f"loss at beta=0 (ungated baseline): {loss_fn():.4f}")
        gate.node_beta.fill_(1.0); gate.edge_beta.fill_(0.5)

    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} loss {loss.item():.4f} beta_node {gate.node_beta.item():.3f} beta_edge {gate.edge_beta.item():.3f}")

    result = {"beta_node": gate.node_beta.item(), "beta_edge": gate.edge_beta.item(), "final_loss": loss.item(), "n_graphs": len(examples)}
    print(json.dumps(result, indent=1))
    (config.DATA_DIR / "processed" / "reliability_gate_fit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    if args.write:
        ckpt_path = config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            state = {"model_state_dict": state}
        state["gate_state_dict"] = gate.state_dict()
        backup = ckpt_path.with_suffix(".pre_gate.pt")
        if not backup.exists():
            ckpt_path.replace(backup)
        torch.save(state, ckpt_path)
        print(f"wrote gate into {ckpt_path} (backup at {backup.name})")


if __name__ == "__main__":
    main()
