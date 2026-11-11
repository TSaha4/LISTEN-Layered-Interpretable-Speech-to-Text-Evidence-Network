"""Aggregate per-meeting ``uncertainty_eval.json`` files into one Markdown table.

Reads every ``*uncertainty_eval.json`` under ``documents/results`` (the file
written by ``evaluate_uncertainty.py`` for each meeting, keyed by its
``meeting`` field) and reports, per meeting and condition, evidence F1@k for
the ungated model (beta = 0) and the learned gate (beta = 0.785), the
meeting-level mean, the query-pooled mean with a paired bootstrap 95 % CI
of the gain, and the pooled gold-vs-non-gold reliability gap.

Usage:
    python scripts/aggregate_uncertainty_eval.py [--results documents/results] [--out documents/results/multi_meeting_eval.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ["clean", "snr10", "snr5", "snr0"]
COND_LABEL = {"clean": "clean", "snr10": "10 dB", "snr5": "5 dB", "snr0": "0 dB"}
BASE, GATE = "0.0", "0.785"


def load(results_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*uncertainty_eval.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs[data["meeting"]] = data
    return runs


def per_query_f1(run: dict, cond: str, beta: str) -> list[float]:
    for summary, rows in zip(run["conditions"], run["per_query"]):
        if summary["condition"] == cond:
            return [r["beta"][beta]["prf"][2] for r in rows]
    return []


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pct(a: float, b: float) -> str:
    return "n/a" if a == 0 else f"{(b - a) / a:+.0%}"


def build_report(runs: dict[str, dict]) -> str:
    meetings = sorted(runs)
    k = next(iter(runs.values()))["top_k"]
    lines = [f"# Multi-meeting replication of the noise-robustness result", ""]
    lines.append(
        f"Evidence F1@{k} per AMI meeting and noise condition, reliability gate off (β = 0, original LISTEN) "
        f"vs the learned gate (β = {GATE}). Produced by `scripts/aggregate_uncertainty_eval.py` from the "
        f"per-meeting `*_uncertainty_eval.json` files in this folder."
    )
    lines.append("")

    # ---- per-meeting table
    header = "| meeting | queries | " + " | ".join(f"{COND_LABEL[c]} β=0 | {COND_LABEL[c]} β={GATE}" for c in CONDITIONS) + " |"
    lines += [header, "|---|---|" + "---|---|" * len(CONDITIONS)]
    for m in meetings:
        run = runs[m]
        cells = []
        for c in CONDITIONS:
            b = per_query_f1(run, c, BASE)
            g = per_query_f1(run, c, GATE)
            if not b:
                cells += ["–", "–"]
                continue
            mb, mg = float(np.mean(b)), float(np.mean(g))
            mark = "**" if mg > mb + 1e-9 else ""
            cells += [f"{mb:.3f}", f"{mark}{mg:.3f}{mark} ({pct(mb, mg)})"]
        n_q = run["conditions"][0]["n_queries"]
        lines.append(f"| {m} | {n_q} | " + " | ".join(cells) + " |")

    # ---- meeting-mean and pooled rows
    mean_cells, pooled_cells, ci_rows, helped = [], [], [], []
    for c in CONDITIONS:
        mb_list = [np.mean(per_query_f1(runs[m], c, BASE)) for m in meetings if per_query_f1(runs[m], c, BASE)]
        mg_list = [np.mean(per_query_f1(runs[m], c, GATE)) for m in meetings if per_query_f1(runs[m], c, GATE)]
        all_b = np.array([f for m in meetings for f in per_query_f1(runs[m], c, BASE)])
        all_g = np.array([f for m in meetings for f in per_query_f1(runs[m], c, GATE)])
        if len(all_b) == 0:
            mean_cells += ["–", "–"]
            pooled_cells += ["–", "–"]
            continue
        mb, mg = float(np.mean(mb_list)), float(np.mean(mg_list))
        mean_cells += [f"{mb:.3f}", f"{mg:.3f} ({pct(mb, mg)})"]
        pb, pg = float(all_b.mean()), float(all_g.mean())
        pooled_cells += [f"{pb:.3f}", f"{pg:.3f} ({pct(pb, pg)})"]
        diff = all_g - all_b
        lo, hi = bootstrap_ci(diff)
        n_up, n_down = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
        helped.append((c, int(sum(g > b + 1e-9 for b, g in zip(mb_list, mg_list))), int(sum(g < b - 1e-9 for b, g in zip(mb_list, mg_list))), len(mb_list)))
        ci_rows.append(f"| {COND_LABEL[c]} | {len(diff)} | {pb:.3f} | {pg:.3f} | {diff.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {n_up} / {n_down} / {len(diff) - n_up - n_down} |")
    lines.append(f"| **mean of meetings** | | " + " | ".join(mean_cells) + " |")
    lines.append(f"| **pooled over queries** | {sum(r['conditions'][0]['n_queries'] for r in runs.values())} | " + " | ".join(pooled_cells) + " |")
    lines.append("")
    lines.append("Bold = gate beats the ungated model for that meeting and condition. Percentages are relative to β = 0.")
    lines.append("")

    # ---- pooled gain with CI
    lines += [
        f"## Pooled gain of the gate (β = {GATE} − β = 0), paired over queries",
        "",
        "| condition | queries | F1 β=0 | F1 gate | mean gain | bootstrap 95 % CI | queries up / down / tied |",
        "|---|---|---|---|---|---|---|",
        *ci_rows,
        "",
        "Meetings where the gate helps / hurts / total: "
        + "; ".join(f"{COND_LABEL[c]}: {up} / {down} / {n}" for c, up, down, n in helped)
        + ".",
        "",
    ]

    # ---- all betas, pooled
    betas = list(next(iter(runs.values()))["conditions"][0]["beta"].keys())
    lines += [
        "## Pooled evidence F1 for every β evaluated",
        "",
        "| condition | queries | " + " | ".join(f"β={b}" for b in betas) + " |",
        "|---|---|" + "---|" * len(betas),
    ]
    for c in CONDITIONS:
        vals = {b: np.array([f for m in meetings for f in per_query_f1(runs[m], c, b)]) for b in betas}
        n = len(vals[betas[0]])
        if n == 0:
            continue
        best = max(betas, key=lambda b: vals[b].mean())
        cells = [f"**{vals[b].mean():.3f}**" if b == best else f"{vals[b].mean():.3f}" for b in betas]
        lines.append(f"| {COND_LABEL[c]} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "Bold = best β for that condition.", ""]

    # ---- reliability gap
    lines += [
        "## Reliability signal: gold vs non-gold candidate segments",
        "",
        "Mean node reliability of gold evidence segments vs the other retrieved candidates, per meeting (query means) and pooled.",
        "",
        "| meeting | " + " | ".join(f"{COND_LABEL[c]} gold / non-gold" for c in CONDITIONS) + " |",
        "|---|" + "---|" * len(CONDITIONS),
    ]
    pooled_gap = {c: ([], []) for c in CONDITIONS}
    for m in meetings:
        cells = []
        for c in CONDITIONS:
            s = next((s for s in runs[m]["conditions"] if s["condition"] == c), None)
            if s is None:
                cells.append("–")
                continue
            cells.append(f"{s['mean_reliability_gold']:.3f} / {s['mean_reliability_nongold']:.3f}")
            for rows in [r for sm, r in zip(runs[m]["conditions"], runs[m]["per_query"]) if sm["condition"] == c]:
                pooled_gap[c][0].extend(r["mean_reliability_gold"] for r in rows if not np.isnan(r["mean_reliability_gold"]))
                pooled_gap[c][1].extend(r["mean_reliability_nongold"] for r in rows if not np.isnan(r["mean_reliability_nongold"]))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    cells = []
    for c in CONDITIONS:
        g, n = pooled_gap[c]
        cells.append(f"{np.mean(g):.3f} / {np.mean(n):.3f} (gap {np.mean(g) - np.mean(n):+.3f})" if g else "–")
    lines.append("| **pooled** | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- ASR confidence context
    lines += ["## Mean word confidence per condition", "", "| meeting | " + " | ".join(COND_LABEL[c] for c in CONDITIONS) + " |", "|---|" + "---|" * len(CONDITIONS)]
    for m in meetings:
        cells = []
        for c in CONDITIONS:
            s = next((s for s in runs[m]["conditions"] if s["condition"] == c), None)
            cells.append(f"{s['mean_asr_confidence']:.3f}" if s else "–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=PROJECT_ROOT / "documents" / "results")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "documents" / "results" / "multi_meeting_eval.md")
    args = ap.parse_args()
    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no *uncertainty_eval.json under {args.results}")
    report = build_report(runs)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
