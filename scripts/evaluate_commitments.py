"""Measure commitment-supersession accuracy on hand-labelled cases.

Each case in ``commitment_eval_cases.json`` is a chronological sequence of
statements about one topic. The script inserts them into a fresh
``CommitmentGraph`` (real sentence embeddings, so topic matching is tested
too) and checks (a) the relation assigned to the last statement and (b)
which statement ends up active. It runs twice: with the reliability margin
on (LISTEN-U) and with it off (plain "latest wins"), so the effect of the
uncertainty-aware rule is isolated.

Usage:
    LISTEN_DATA_DIR=../data python scripts/evaluate_commitments.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.commitments import graph as graph_mod
from app.commitments.graph import CommitmentGraph
from app.models.schemas import Commitment

CASES = Path(__file__).with_name("commitment_eval_cases.json")


def _commitment(s: dict, meeting: str) -> Commitment:
    return Commitment(
        commitment_id=s["id"], meeting_id=meeting, segment_id=s["id"], text=s["text"], kind="decision",
        owner=s.get("owner"), topic_entities=s.get("topic", []), values=s.get("values", []),
        polarity=s.get("polarity", 1), meeting_time=float(s["t"]) * 86400, offset_s=0.0,
        reliability=float(s["reliability"]),
    )


def run(reliability_margin: float) -> dict:
    """Return per-case results with the given supersession margin."""
    graph_mod.RELIABILITY_MARGIN = reliability_margin  # patched hook, see graph.py
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in cases:
            g = CommitmentGraph(path=Path(tmp) / f"{case['name']}.json")
            stmts = case["statements"]
            edges = []
            for s in stmts:
                edges.extend(g.add(_commitment(s, f"m{s['t']}")))
            last = stmts[-1]["id"]
            rel = next((e.relation for e in edges if e.source == last), None)
            active = sorted(c.commitment_id for c in g.nodes.values() if c.status == "active")
            exp_active = case["expected_active"]
            active_ok = (len(active) == len(stmts)) if exp_active == "both" else (active == [exp_active])
            rows.append({
                "case": case["name"], "expected_relation": case["expected_relation"], "relation": rel,
                "relation_ok": rel == case["expected_relation"], "expected_active": exp_active,
                "active": active, "active_ok": active_ok,
            })
    return {
        "margin": reliability_margin,
        "relation_accuracy": sum(r["relation_ok"] for r in rows) / len(rows),
        "active_accuracy": sum(r["active_ok"] for r in rows) / len(rows),
        "rows": rows,
    }


def main() -> None:
    results = {"listen_u": run(config.COMMITMENT_RELIABILITY_MARGIN), "latest_wins": run(float("inf"))}
    print("\n### Commitment supersession on hand-labelled cases\n")
    print("| variant | relation accuracy | active-decision accuracy |")
    print("|---|---|---|")
    for name, r in results.items():
        print(f"| {name} (margin={r['margin']}) | {r['relation_accuracy']:.2f} | {r['active_accuracy']:.2f} |")
    print("\nper case (LISTEN-U):")
    for r in results["listen_u"]["rows"]:
        flag = "ok " if r["relation_ok"] and r["active_ok"] else "FAIL"
        print(f"  {flag} {r['case']}: relation={r['relation']} (exp {r['expected_relation']}), active={r['active']} (exp {r['expected_active']})")
    out = config.DATA_DIR / "processed" / "commitment_eval.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
