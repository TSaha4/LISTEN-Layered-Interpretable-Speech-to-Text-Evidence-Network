"""
LISTEN Phase 2 — Dataset Loaders

Loads HotpotQA (for base paper reproduction) and QMSum (for meeting domain
fine-tuning) and converts them into PyG-compatible evidence graphs.

HotpotQA → used to pre-train the GAT on multi-hop text QA
QMSum → used to fine-tune on real meeting QA pairs
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

import config
from schemas import MeetingSegment
from phase1_retrieval.bi_encoder import BiEncoder
from phase2_graph.graph_builder import EvidenceGraphBuilder


# ──────────────────────────────────────────────────────────────
# HotpotQA Dataset
# ──────────────────────────────────────────────────────────────

class HotpotQAGraphDataset(Dataset):
    """Converts HotpotQA distractor examples into PyG evidence graphs.

    Each HotpotQA example has:
      - question: str
      - context: list of (title, sentences) pairs (10 paragraphs)
      - supporting_facts: list of (title, sentence_idx) pairs
      - answer: str

    We treat each sentence as a "segment" and build a graph using shared entities
    (via simple noun overlap) and cosine similarity. Ground-truth labels come
    from supporting_facts.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: str = str(config.HOTPOTQA_DIR),
    ):
        """
        Args:
            split: 'train' or 'validation' (HuggingFace dataset splits).
            max_samples: Limit number of examples (for debugging / smoke tests).
            cache_dir: Directory to cache the downloaded dataset.
        """
        self.split = split
        self.encoder = BiEncoder()
        self.graph_builder = EvidenceGraphBuilder()

        # Load HotpotQA via HuggingFace datasets
        from datasets import load_dataset
        print(f"Loading HotpotQA ({split})...")
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, cache_dir=cache_dir)

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.examples = ds
        print(f"Loaded {len(self.examples)} HotpotQA examples.")

        # Pre-build graphs (this takes a while but avoids repeated encoding)
        self.graphs: List[Data] = []
        self.metadata_list: List[dict] = []
        self._build_all_graphs()

    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction via capitalized noun phrases.

        For HotpotQA training we use a lightweight approach (no spaCy dependency
        for dataset preprocessing). Real meeting data from Person A will have
        proper NER entities.
        """
        # Extract capitalized multi-word phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Also extract quoted terms
        entities += re.findall(r'"([^"]+)"', text)
        # Deduplicate
        return list(set(e.strip() for e in entities if len(e) > 1))

    def _build_all_graphs(self):
        """Pre-build all evidence graphs from HotpotQA examples."""
        print("Building evidence graphs...")
        for idx in tqdm(range(len(self.examples)), desc="Building graphs"):
            example = self.examples[idx]

            try:
                graph, meta = self._example_to_graph(example)
                self.graphs.append(graph)
                self.metadata_list.append(meta)
            except Exception as e:
                # Skip problematic examples
                # Create a minimal placeholder graph
                x = torch.randn(2, config.EMBEDDING_DIM)
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                edge_type = torch.tensor([0, 0], dtype=torch.long)
                y = torch.tensor([0.0, 0.0], dtype=torch.float32)
                self.graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
                self.metadata_list.append({"error": str(e)})

    def _example_to_graph(self, example: dict) -> Tuple[Data, dict]:
        """Convert a single HotpotQA example to a PyG graph with labels.

        Maps HotpotQA's paragraph/sentence structure to our flat segment graph:
          - Each sentence becomes a MeetingSegment (segment_id = "title_sentN")
          - Supporting facts become positive labels
        """
        question = example["question"]
        context_titles = example["context"]["title"]
        context_sentences = example["context"]["sentences"]
        sup_titles = example["supporting_facts"]["title"]
        sup_sent_ids = example["supporting_facts"]["sent_id"]

        # Build supporting facts set: {(title, sent_idx)}
        supporting_set = set(zip(sup_titles, sup_sent_ids))

        # Convert sentences to MeetingSegment objects
        segments = []
        supporting_ids = []
        time_counter = 0.0

        for para_idx, (title, sentences) in enumerate(zip(context_titles, context_sentences)):
            for sent_idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                seg_id = f"{title}_s{sent_idx}"
                entities = self._extract_entities_simple(sentence)

                seg = MeetingSegment(
                    segment_id=seg_id,
                    text=sentence.strip(),
                    start_time=time_counter,
                    end_time=time_counter + 5.0,  # Dummy timestamps for text data
                    speaker=title,  # Use paragraph title as "speaker"
                    entities=entities,
                )
                segments.append(seg)

                if (title, sent_idx) in supporting_set:
                    supporting_ids.append(seg_id)

                time_counter += 5.0

        if not segments:
            raise ValueError("No segments extracted from example")

        # Limit number of segments per graph
        max_segs = config.MAX_SEGMENTS_PER_GRAPH - 1  # Reserve 1 for question node
        if len(segments) > max_segs:
            # Prioritize: keep all supporting segments + random non-supporting
            sup_segs = [s for s in segments if s.segment_id in supporting_ids]
            non_sup_segs = [s for s in segments if s.segment_id not in supporting_ids]
            np.random.shuffle(non_sup_segs)
            segments = sup_segs + non_sup_segs[:max_segs - len(sup_segs)]

        # Encode
        texts = [seg.text for seg in segments]
        seg_embeddings = self.encoder.encode(texts)
        q_embedding = self.encoder.encode_single(question)

        # Build graph with labels
        data, meta = self.graph_builder.build_with_labels(
            question_embedding=q_embedding,
            segments=segments,
            segment_embeddings=seg_embeddings,
            supporting_ids=supporting_ids,
        )

        meta["question"] = question
        meta["answer"] = example["answer"]
        meta["supporting_ids"] = supporting_ids

        return data, meta

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> Data:
        return self.graphs[idx]


# ──────────────────────────────────────────────────────────────
# QMSum Dataset
# ──────────────────────────────────────────────────────────────

class QMSumGraphDataset(Dataset):
    """Converts QMSum meeting QA examples into PyG evidence graphs.

    QMSum contains real meeting transcripts with query-summary pairs.
    Each query has annotated relevant turns from the transcript.

    Expected QMSum JSON format (per meeting):
    {
      "topic_list": [...],
      "general_query_list": [
        {"query": "...", "answer": "...", "relevant_text_span": [...]}
      ],
      "specific_query_list": [
        {"query": "...", "answer": "...", "relevant_text_span": [...]}
      ],
      "meeting_transcripts": [
        {"speaker": "...", "content": "..."}
      ]
    }
    """

    def __init__(
        self,
        data_dir: str = str(config.QMSUM_DIR),
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.encoder = BiEncoder()
        self.graph_builder = EvidenceGraphBuilder()

        # Load QMSum JSON files
        split_dir = Path(data_dir) / "data" / split
        self.graphs: List[Data] = []
        self.metadata_list: List[dict] = []

        if not split_dir.exists():
            print(f"QMSum directory not found: {split_dir}")
            print("Download QMSum from: https://github.com/Yale-LILY/QMSum")
            print(f"Place files in: {data_dir}/data/{{train,val,test}}/")
            return

        json_files = sorted(split_dir.glob("*.json"))
        if max_samples:
            json_files = json_files[:max_samples]

        print(f"Loading QMSum ({split}): {len(json_files)} meetings...")
        for jf in tqdm(json_files, desc="Processing meetings"):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    meeting = json.load(f)
                self._process_meeting(meeting)
            except Exception as e:
                print(f"Error processing {jf.name}: {e}")

        print(f"Built {len(self.graphs)} QMSum graphs.")

    def _process_meeting(self, meeting: dict):
        """Process a single QMSum meeting into multiple QA graphs."""
        transcripts = meeting.get("meeting_transcripts", [])
        if not transcripts:
            return

        # Convert transcript turns to segments
        segments = []
        for i, turn in enumerate(transcripts):
            seg = MeetingSegment(
                segment_id=f"turn_{i:04d}",
                text=turn["content"].strip(),
                start_time=float(i * 10),    # Approximate timestamps
                end_time=float((i + 1) * 10),
                speaker=turn.get("speaker", "UNKNOWN"),
                entities=[],  # QMSum doesn't have pre-extracted entities
            )
            segments.append(seg)

        # Encode all segments once per meeting
        texts = [seg.text for seg in segments]
        all_embeddings = self.encoder.encode(texts)

        # Process each query (both general and specific)
        queries = (
            meeting.get("general_query_list", []) +
            meeting.get("specific_query_list", [])
        )

        for query_item in queries:
            try:
                self._query_to_graph(query_item, segments, all_embeddings)
            except Exception:
                continue

    def _query_to_graph(
        self,
        query_item: dict,
        all_segments: List[MeetingSegment],
        all_embeddings: np.ndarray,
    ):
        """Convert a single QMSum query to a PyG evidence graph."""
        question = query_item["query"]
        answer = query_item.get("answer", "")
        relevant_spans = query_item.get("relevant_text_span", [])

        # Identify relevant turn indices from spans
        supporting_ids = set()
        for span_text in relevant_spans:
            # Match span text to transcript turns
            for seg in all_segments:
                if span_text.strip() in seg.text or seg.text in span_text.strip():
                    supporting_ids.add(seg.segment_id)

        # Retrieve top-k segments by similarity to question
        q_embedding = self.encoder.encode_single(question)
        similarities = all_embeddings @ q_embedding
        max_segs = config.MAX_SEGMENTS_PER_GRAPH - 1
        top_indices = np.argsort(similarities)[::-1][:max_segs]

        # Ensure supporting segments are included
        sup_indices = [
            i for i, seg in enumerate(all_segments)
            if seg.segment_id in supporting_ids
        ]
        combined_indices = list(set(list(top_indices) + sup_indices))[:max_segs]

        selected_segments = [all_segments[i] for i in combined_indices]
        selected_embeddings = all_embeddings[combined_indices]

        # Build graph with labels
        data, meta = self.graph_builder.build_with_labels(
            question_embedding=q_embedding,
            segments=selected_segments,
            segment_embeddings=selected_embeddings,
            supporting_ids=list(supporting_ids),
        )

        meta["question"] = question
        meta["answer"] = answer

        self.graphs.append(data)
        self.metadata_list.append(meta)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> Data:
        return self.graphs[idx]
