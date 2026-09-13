"""Narrate the real AMI/QMSum preprocessing path used by LISTEN.

Run from the repository root:
    .venv\\Scripts\\python scripts\\show_preprocessing.py

The script deliberately refuses to invent data.  It expects the real processed
segment and alignment artifacts produced by the dataset-preprocessing job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dl.embeddings import SegmentIndex
from app.dl.graph_builder import EvidenceGraphBuilder, build_evidence_graph
from app.models.schemas import SLPSegment

MEETING_ID = "ES2002a"
PROCESSED = ROOT / "data" / "processed" / f"{MEETING_ID}_segments.json"
ALIGNED = ROOT / "data" / "processed" / f"{MEETING_ID}_aligned_labels.json"
QMSUM_ROOTS = (ROOT / "data" / "qmsum_product", ROOT / "data" / "qmsum")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def locate_qmsum_entry() -> Path | None:
    for root in QMSUM_ROOTS:
        if root.exists():
            entry = next((path for path in root.rglob("*.json") if MEETING_ID.lower() in path.name.lower()), None)
            if entry is not None:
                return entry
    return None


def segment_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("segments", "asr_segments", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("The segments JSON has no recognised list of segments.")


def to_segment(row: dict[str, Any], index: int) -> SLPSegment:
    return SLPSegment(
        segment_id=str(row.get("segment_id", row.get("id", f"{MEETING_ID}_seg_{index:04d}"))),
        start_time=float(row.get("start_time", row.get("start", 0.0))),
        end_time=float(row.get("end_time", row.get("end", 0.0))),
        text=str(row.get("text", row.get("transcript", ""))),
        entities=list(row.get("entities", [])),
    )


def query_items(qmsum: Any) -> list[dict[str, Any]]:
    if not isinstance(qmsum, dict):
        return []
    return list(qmsum.get("specific_query_list", qmsum.get("general_query_list", [])))


def preview(value: Any, limit: int = 220) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text.replace("\n", " ")[:limit]


def main() -> int:
    qmsum_path = locate_qmsum_entry()
    missing = [path for path in (PROCESSED, ALIGNED) if not path.is_file()]
    if qmsum_path is None:
        missing.append(QMSUM_ROOTS[0] / f"**/{MEETING_ID}*.json")
    if missing:
        print("LISTEN preprocessing walkthrough cannot use synthetic fallback.")
        print("Missing real preprocessing artifact(s):")
        for path in missing:
            print(f"  - {path}")
        print("The raw AMI recording is present at data/audio/raw/ES2002a.wav, but its processed ASR/QMSum alignment files are not in this checkout.")
        return 2

    rows = segment_rows(load_json(PROCESSED))
    segments = [to_segment(row, index) for index, row in enumerate(rows)]
    qmsum = load_json(qmsum_path)
    queries = query_items(qmsum)
    if not segments or not queries:
        raise RuntimeError("Real segments or QMSum queries were empty.")
    question_item = queries[0]
    question = str(question_item.get("query", question_item.get("question", "What was discussed?")))

    print("=" * 72)
    print(f"1. REAL ASR + ENTITY OUTPUT - {PROCESSED.relative_to(ROOT)}")
    for segment in segments[:3]:
        print(f"  [{segment.start_time:.2f}s-{segment.end_time:.2f}s] {segment.segment_id}")
        print(f"    text: {segment.text}")
        print(f"    entities: {segment.entities}")

    print("\n2. REAL QMSum GOLD EVIDENCE")
    print(f"  source: {qmsum_path.relative_to(ROOT)}")
    print(f"  sample question: {question}")
    for item in queries[:2]:
        print(f"  query: {preview(item.get('query', item.get('question', '')))}")
        print(f"  gold turns/spans: {preview(item.get('relevant_text_span', item.get('relevant_turns', [])))}")

    print("\n3. QMSum -> ASR ALIGNMENT")
    alignment = load_json(ALIGNED)
    matches = alignment if isinstance(alignment, list) else alignment.get("matches", alignment.get("alignments", []))
    print(f"  source: {ALIGNED.relative_to(ROOT)}")
    for match in list(matches)[:2]:
        if isinstance(match, dict):
            print(f"  QMSum turn {match.get('turn_index')} -> {match.get('asr_segment_id')}")
            print(f"    ASR text: {preview(match.get('asr_text', ''))}")
            print(
                "    semantic_similarity="
                f"{match.get('semantic_similarity', match.get('similarity', 'not recorded'))}; "
                f"combined_score={match.get('combined_score', match.get('score', 'not recorded'))}"
            )
        else:
            print(f"  match: {preview(match)}")

    print("\n4. SEGMENTS -> 384-D EMBEDDINGS -> TOP-5 RETRIEVAL")
    index = SegmentIndex()
    index.build(segments)
    embeddings = index.get_all_embeddings()
    first_vector = next(iter(embeddings.values()))
    print(f"  embedding matrix: ({len(embeddings)}, {first_vector.shape[0]}) - each segment is a {first_vector.shape[0]}-dimensional vector")
    retrieved = index.search(question, top_k=5)
    retrieved_segments = [next(segment for segment in segments if segment.segment_id == segment_id) for segment_id, _ in retrieved]
    for rank, (segment_id, score) in enumerate(retrieved, start=1):
        segment = next(segment for segment in retrieved_segments if segment.segment_id == segment_id)
        print(f"  {rank}. {segment_id}  cosine={score:.4f}  {segment.text[:120]}")

    print("\n5. RETRIEVED SEGMENTS -> EVIDENCE GRAPH -> MODEL INPUT")
    from app.dl.embeddings import embed_texts
    question_vector = embed_texts([question])[0]
    segment_vectors = np.array([embeddings[segment.segment_id] for segment in retrieved_segments])
    graph, metadata = EvidenceGraphBuilder().build(question_vector, retrieved_segments, segment_vectors)
    weighted_graph = build_evidence_graph(
        [segment.segment_id for segment in retrieved_segments],
        {segment.segment_id: segment for segment in retrieved_segments},
        precomputed_embeddings=embeddings,
    )
    print(f"  nodes: {graph.num_nodes} (1 question + {len(retrieved_segments)} retrieved segments)")
    print(f"  directed edges: {graph.edge_index.shape[1]}")
    relations = metadata["edge_relations"]
    edge_order = sorted(range(len(relations)), key=lambda index: relations[index] == "question_link")
    for index in edge_order[:3]:
        relation = relations[index]
        source, target = graph.edge_index[:, index].tolist()
        source_id = metadata["node_id_map"][source]
        target_id = metadata["node_id_map"][target]
        weight = next((edge.weight for edge in weighted_graph.edges if {edge.source, edge.target} == {source_id, target_id}), None)
        if weight is not None:
            weight_text = f"combined_weight={weight:.4f}"
        elif source_id != "__question__" and target_id != "__question__":
            source_vector = embeddings[source_id]
            target_vector = embeddings[target_id]
            weight_text = f"cosine_similarity={float(np.dot(source_vector, target_vector)):.4f}"
        else:
            weight_text = "question-link (unweighted)"
        print(f"  edge {index + 1}: {source_id} -> {target_id}; type={relation}; weight={weight_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
