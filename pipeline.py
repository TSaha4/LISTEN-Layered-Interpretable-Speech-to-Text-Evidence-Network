"""
LISTEN — End-to-End DL Pipeline

Orchestrates: Person A's JSON → Phase 1 (Retrieval) → Phase 2 (GAT) → Phase 3 (LLM) → Output JSON

This is the main entry point for Person B's component.
Person C calls this pipeline and receives the output contract.

Usage:
    # As a module
    from pipeline import LISTENPipeline
    pipeline = LISTENPipeline(gat_checkpoint="checkpoints/hotpotqa_best.pt")
    result = pipeline.run(meeting_json, question="Why was the rubber casing chosen?")

    # From CLI
    python pipeline.py --input data/sample_meeting.json --question "What was decided?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config
from schemas import (
    MeetingInput,
    MeetingSegment,
    EvidenceNode,
    EvidenceEdge,
    EvidenceGraph,
    PipelineOutput,
    ModelMetadata,
)
from phase1_retrieval.bi_encoder import BiEncoder
from phase1_retrieval.retriever import SegmentRetriever
from phase2_graph.graph_builder import EvidenceGraphBuilder
from phase2_graph.gat_model import EvidenceGAT
from phase3_answer.generator import AnswerGenerator


class LISTENPipeline:
    """End-to-end pipeline: meeting JSON + question → answer + evidence graph.

    Phases:
      1. Bi-encoder retrieval: find top-k relevant segments
      2. Graph construction + GAT reasoning: build evidence graph, weight nodes
      3. LLM answer generation: synthesize answer from top-weighted segments
    """

    def __init__(
        self,
        gat_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        use_llm: bool = True,
    ):
        """
        Args:
            gat_checkpoint: Path to trained GAT checkpoint (.pt file).
                If None, uses randomly initialized GAT (for testing only).
            device: 'cuda' or 'cpu'. Auto-detected if None.
            use_llm: Whether to use Gemini for answer generation.
                If False, returns a structured summary instead.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Phase 1: Bi-encoder + retriever
        self.encoder = BiEncoder(device=device)
        self.retriever = SegmentRetriever(bi_encoder=self.encoder)

        # Phase 2: Graph builder + GAT
        self.graph_builder = EvidenceGraphBuilder()
        self.gat = EvidenceGAT().to(self.device)

        if gat_checkpoint and Path(gat_checkpoint).exists():
            print(f"Loading GAT checkpoint: {gat_checkpoint}")
            checkpoint = torch.load(gat_checkpoint, map_location=self.device)
            self.gat.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded (val F1={checkpoint.get('val_f1', 'N/A')})")
        else:
            print("WARNING: No GAT checkpoint loaded. Using random weights (for testing).")

        self.gat.eval()

        # Phase 3: LLM answer generator
        self.use_llm = use_llm
        self.generator = None
        if use_llm:
            try:
                self.generator = AnswerGenerator()
            except ValueError as e:
                print(f"WARNING: LLM not available ({e}). Will use fallback answers.")
                self.use_llm = False

    def run(
        self,
        meeting_input: dict | MeetingInput,
        question: str,
        top_k_retrieval: Optional[int] = None,
        top_k_answer: Optional[int] = None,
    ) -> PipelineOutput:
        """Run the full pipeline.

        Args:
            meeting_input: Either a raw dict (Person A's JSON) or a MeetingInput object.
            question: The user's question about the meeting.
            top_k_retrieval: Override default top-k for Phase 1 retrieval.
            top_k_answer: Override default top-k for Phase 3 answer generation.

        Returns:
            PipelineOutput with answer, evidence_graph, and top_segments.
        """
        # -- Parse input --
        if isinstance(meeting_input, dict):
            meeting = MeetingInput(**meeting_input)
        else:
            meeting = meeting_input

        print(f"\n{'-'*60}")
        print(f"Question: {question}")
        print(f"Meeting: {meeting.meeting_id} ({len(meeting.segments)} segments)")
        print(f"{'-'*60}")

        # -- Phase 1: Retrieval --
        print("Phase 1: Bi-encoder retrieval...")
        retrieved_segments, retrieval_scores, segment_embeddings = self.retriever.retrieve(
            question=question,
            segments=meeting.segments,
            top_k=top_k_retrieval,
            meeting_id=meeting.meeting_id,
        )
        print(f"  Retrieved {len(retrieved_segments)} segments (scores: "
              f"{retrieval_scores[0]:.3f} -> {retrieval_scores[-1]:.3f})")

        # -- Phase 2: Graph construction + GAT --
        print("Phase 2: Graph construction + GAT reasoning...")
        question_embedding = self.retriever.get_question_embedding(question)

        graph_data, graph_metadata = self.graph_builder.build(
            question_embedding=question_embedding,
            segments=retrieved_segments,
            segment_embeddings=segment_embeddings,
        )
        print(f"  Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges")
        print(f"  Edge types: {graph_metadata['num_question_edges']} question, "
              f"{graph_metadata['num_entity_edges']} entity, "
              f"{graph_metadata['num_similarity_edges']} similarity")

        # Run GAT inference
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            gat_output = self.gat(graph_data)

        node_weights = gat_output["node_weights"].cpu().numpy()
        # Skip question node (index 0), get segment weights
        segment_weights = node_weights[1:]  # (num_segments,)

        # -- Build evidence graph output --
        evidence_nodes = []
        for i, seg in enumerate(retrieved_segments):
            evidence_nodes.append(EvidenceNode(
                id=seg.segment_id,
                text=seg.text,
                speaker=seg.speaker,
                start_time=seg.start_time,
                end_time=seg.end_time,
                weight=float(segment_weights[i]),
                entities=seg.entities,
            ))

        # Sort by weight (descending) for answer generation
        evidence_nodes.sort(key=lambda n: n.weight, reverse=True)

        # Build evidence edges
        evidence_edges = []
        edge_index = graph_data.edge_index.cpu().numpy()
        edge_types = graph_data.edge_type.cpu().numpy()
        edge_relations = graph_metadata["edge_relations"]
        node_id_map = graph_metadata["node_id_map"]

        for e_idx in range(edge_index.shape[1]):
            src, tgt = edge_index[0, e_idx], edge_index[1, e_idx]
            # Skip edges involving the question node and duplicate bidirectional edges
            if src == 0 or tgt == 0:
                continue
            if src > tgt:  # Keep only one direction
                continue

            src_id = node_id_map.get(src, f"node_{src}")
            tgt_id = node_id_map.get(tgt, f"node_{tgt}")

            # Compute edge weight from endpoint node weights
            edge_weight = float((segment_weights[src - 1] + segment_weights[tgt - 1]) / 2)

            evidence_edges.append(EvidenceEdge(
                source=src_id,
                target=tgt_id,
                weight=edge_weight,
                relation=edge_relations[e_idx] if e_idx < len(edge_relations) else "unknown",
            ))

        evidence_graph = EvidenceGraph(nodes=evidence_nodes, edges=evidence_edges)

        # Top segments by weight
        top_segments = [node.id for node in evidence_nodes]

        # -- Phase 3: Answer generation --
        print("Phase 3: LLM answer generation...")
        if self.use_llm and self.generator:
            answer = self.generator.generate(
                question=question,
                evidence_nodes=evidence_nodes,
                top_k=top_k_answer,
            )
        else:
            # Fallback: concatenate top evidence
            top_k = top_k_answer or config.TOP_K_FOR_ANSWER
            top_texts = [f"[{n.speaker}]: {n.text}" for n in evidence_nodes[:top_k]]
            answer = (
                f"Based on the meeting evidence:\n" +
                "\n".join(f"  - {t}" for t in top_texts)
            )

        print(f"  Answer generated ({len(answer)} chars)")
        print(f"{'-'*60}")

        # -- Assemble output --
        retrieval_score_map = {
            seg.segment_id: float(score)
            for seg, score in zip(retrieved_segments, retrieval_scores)
        }

        return PipelineOutput(
            question=question,
            answer=answer,
            evidence_graph=evidence_graph,
            top_segments=top_segments,
            model_metadata=ModelMetadata(
                retrieval_scores=retrieval_score_map,
                num_segments_retrieved=len(retrieved_segments),
                num_segments_in_graph=graph_data.num_nodes - 1,
            ),
        )

    def run_counterfactual(
        self,
        meeting_input: dict | MeetingInput,
        question: str,
        remove_segment_ids: list[str],
        top_k_retrieval: Optional[int] = None,
        top_k_answer: Optional[int] = None,
    ) -> PipelineOutput:
        """Re-run pipeline with specific segments removed (for Person C's XAI counterfactual).

        This is the hook Person C calls when the user toggles off a segment in the UI.

        Args:
            meeting_input: Original meeting input.
            question: The user's question.
            remove_segment_ids: List of segment IDs to exclude.

        Returns:
            PipelineOutput with the counterfactual answer and graph.
        """
        if isinstance(meeting_input, dict):
            meeting = MeetingInput(**meeting_input)
        else:
            meeting = meeting_input

        # Filter out removed segments
        filtered_segments = [
            seg for seg in meeting.segments
            if seg.segment_id not in remove_segment_ids
        ]

        # Create modified meeting input
        modified_meeting = MeetingInput(
            meeting_id=meeting.meeting_id + "_counterfactual",
            segments=filtered_segments,
        )

        return self.run(modified_meeting, question, top_k_retrieval, top_k_answer)


# --------------------------------------------------
# CLI Entry Point
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LISTEN DL Pipeline")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to meeting JSON (Person A's output)")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask about the meeting")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to GAT checkpoint")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k segments to retrieve")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM, use fallback answer")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output JSON to this path")

    args = parser.parse_args()

    # Load meeting JSON
    with open(args.input, "r", encoding="utf-8") as f:
        meeting_json = json.load(f)

    # Run pipeline
    pipeline = LISTENPipeline(
        gat_checkpoint=args.checkpoint,
        use_llm=not args.no_llm,
    )
    result = pipeline.run(
        meeting_input=meeting_json,
        question=args.question,
        top_k_retrieval=args.top_k,
    )

    # Output
    output_dict = result.model_dump()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
        print(f"\nOutput saved to: {args.output}")
    else:
        print(f"\n{'='*60}")
        print(f"ANSWER: {result.answer}")
        print(f"\nEvidence nodes: {len(result.evidence_graph.nodes)}")
        print(f"Evidence edges: {len(result.evidence_graph.edges)}")
        print(f"Top segments: {result.top_segments[:5]}")


if __name__ == "__main__":
    main()
