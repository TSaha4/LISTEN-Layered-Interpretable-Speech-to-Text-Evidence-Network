"""
LISTEN walkthrough: video/audio -> transcript -> text processing -> graph -> GAT.

By default, the script uses the checked-in ES2002a meeting artifacts so the
walkthrough is repeatable without rerunning ASR. Pass --video to extract audio
and transcribe a video live with the same functions used by the upload API.

Run from the project root:
    .venv/Scripts/python.exe scripts/show_preprocessing.py
    .venv/Scripts/python.exe scripts/show_preprocessing.py --video path/to/meeting.mp4

PNG diagrams and plots are written to data/visualizations/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.dl.embeddings import SegmentIndex, embed_texts
from app.dl.graph_builder import EvidenceGraphBuilder
from app.dl.inference import GATInference
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import SLPSegment
from app.slp.asr import transcribe_audio
from app.slp.audio_extraction import extract_audio_from_video, is_video_file
from app.slp.entity_extraction import (
    enrich_segments_with_entities,
    extract_entities_from_text,
)

AUDIO_PATH = PROJECT_ROOT / "data" / "audio" / "raw" / "ES2002a.wav"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "ES2002a_segments.json"
ALIGN_PATH = PROJECT_ROOT / "data" / "processed" / "ES2002a_aligned_labels.json"
TRAIN_LOG_PATH = PROJECT_ROOT / "data" / "checkpoints" / "gat_train_log.json"
CHECKPOINT_PATH = config.CHECKPOINT_DIR / config.GAT_CHECKPOINT_NAME
DEMO_QUERY_INDEX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional video file to run through ffmpeg, Whisper, and spaCy live.",
    )
    parser.add_argument(
        "--question",
        help="Question to retrieve evidence for; defaults to the sample QMSum query.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "visualizations",
        help="Directory for generated PNG plots (default: data/visualizations).",
    )
    return parser.parse_args()


def fmt_ts(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def short(text: str, limit: int = 76) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_code_map() -> None:
    print("\nWHAT CODE DOES EACH STEP (relative to project root)")
    pointers = [
        ("Video upload and processing order", "app/api/upload.py:51-98"),
        ("Video -> mono 16 kHz WAV with ffmpeg", "app/slp/audio_extraction.py:11-59"),
        ("Whisper transcription and timestamps", "app/slp/asr.py:38-76"),
        ("spaCy NER and duplicate removal", "app/slp/entity_extraction.py:17-70"),
        ("Embedding vectors and FAISS index", "app/dl/embeddings.py:37-109"),
        ("Question retrieval", "app/dl/retrieval.py:14-43"),
        ("Question/segment graph and three edge types", "app/dl/graph_builder.py:48-176"),
        ("Runtime GAT layers and scoring heads", "app/dl/gat_model.py:16-53"),
        ("Training datasets and graph labels", "phase2_graph/dataset.py:35-179,201-339"),
        ("Training loop and BCE-with-logits loss", "phase2_graph/train.py:71-115,203-309"),
        ("Runtime orchestration", "app/services/pipeline.py:24-100"),
    ]
    for description, location in pointers:
        print(f"  {description:<48} {location}")


def plot_pipeline(output_dir: Path) -> Path:
    stages = [
        ("VIDEO / AUDIO", "Uploaded recording\nMP4, MOV, WAV, …"),
        ("AUDIO PREP", "ffmpeg\nmono · PCM · 16 kHz"),
        ("ASR", "faster-whisper\ntext + segment times"),
        ("TEXT / NER", "spaCy entities\nper transcript segment"),
        ("EMBED + SEARCH", "MiniLM · 384-D\nFAISS top candidates"),
        ("EVIDENCE GRAPH", "question + segments\n3 relation types"),
        ("GAT", "attention message passing\nnode/edge scores"),
    ]
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.set_xlim(0, len(stages) * 2.5)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    for idx, (title, detail) in enumerate(stages):
        x = idx * 2.5 + 0.15
        box = FancyBboxPatch(
            (x, 0.9),
            2.0,
            1.35,
            boxstyle="round,pad=0.08",
            facecolor=("#dceeff" if idx < 4 else "#e4f4e8"),
            edgecolor="#315a75",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x + 1, 1.88, title, ha="center", va="center", weight="bold", fontsize=9)
        ax.text(x + 1, 1.38, detail, ha="center", va="center", fontsize=8)
        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 2.35, 1.58),
                xytext=(x + 2.04, 1.58),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#315a75"},
            )
    ax.text(
        8.75,
        0.42,
        "Training is separate: HotpotQA / QMSum examples → graph + supporting-fact labels → trained checkpoint",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    ax.set_title("LISTEN processing path (runtime)", fontsize=15, weight="bold", pad=12)
    fig.tight_layout()
    path = output_dir / "01_listen_pipeline.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_audio_and_preprocessing(
    audio_path: Path,
    segments: list[SLPSegment],
    candidate_ids: list[str],
    output_dir: Path,
) -> Path:
    info = sf.info(str(audio_path))
    # Plot at most the first ten minutes and decimate to keep the figure light.
    frames_to_read = min(info.frames, info.samplerate * 600)
    waveform, sample_rate = sf.read(
        str(audio_path), start=0, stop=frames_to_read, dtype="float32",
        always_2d=True,
    )
    mono = waveform.mean(axis=1)
    stride = max(1, math.ceil(len(mono) / 150_000))
    time_axis = np.arange(0, len(mono), stride) / sample_rate

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1.5, 1, 1.2]}
    )
    axes[0].plot(time_axis, mono[::stride], color="#356a8a", linewidth=0.45)
    axes[0].set_title("Audio waveform supplied to ASR (real audio; decimated for display)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (seconds)")

    candidate_set = set(candidate_ids)
    shown = [s for s in segments if s.start_time <= frames_to_read / sample_rate]
    for seg in shown:
        is_candidate = seg.segment_id in candidate_set
        axes[1].barh(
            0,
            max(0.08, seg.end_time - seg.start_time),
            left=seg.start_time,
            height=0.55,
            color="#e45756" if is_candidate else "#b9c8d3",
            alpha=0.85 if is_candidate else 0.55,
        )
    axes[1].set_xlim(0, max(frames_to_read / sample_rate, 1))
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_title("Whisper timestamped segments; red = retrieved for the question")

    entity_counts = Counter(
        entity for segment in segments for entity in segment.entities
    )
    top_entities = entity_counts.most_common(12)
    if top_entities:
        names, counts = zip(*top_entities)
        axes[2].barh(list(names)[::-1], list(counts)[::-1], color="#4c956c")
        axes[2].set_xlabel("Mentions in transcript")
        axes[2].set_title("Most frequent stored NER entities")
    else:
        axes[2].text(0.5, 0.5, "No entities found in these transcript segments", ha="center")
        axes[2].set_axis_off()

    fig.tight_layout()
    path = output_dir / "02_audio_transcript_ner.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gat_architecture(output_dir: Path) -> Path:
    blocks = [
        ("Node feature input", "Question + transcript segment\n384 values each"),
        ("Input projection", "Linear(384 → 128)"),
        ("GAT layer 1", "4 attention heads\n32 values/head → 128"),
        ("GAT layer 2", "1 attention head\n128 output values"),
        ("Prediction heads", "Node relevance score\nEdge relevance score"),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    for idx, (title, detail) in enumerate(blocks):
        x = idx * 3 + 0.25
        box = FancyBboxPatch(
            (x, 1.2),
            2.45,
            1.35,
            boxstyle="round,pad=0.08",
            facecolor="#edf2fb" if idx < 4 else "#e5f4e8",
            edgecolor="#315a75",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x + 1.225, 2.12, title, ha="center", weight="bold", fontsize=10)
        ax.text(x + 1.225, 1.58, detail, ha="center", va="center", fontsize=9)
        if idx < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(x + 2.82, 1.88),
                xytext=(x + 2.48, 1.88),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#315a75"},
            )
    ax.text(
        7.5,
        0.56,
        "Each GAT layer aggregates neighbor messages; learned attention weights determine their influence.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    ax.set_title("Serving EvidenceGAT architecture from app/dl/gat_model.py", fontsize=14, weight="bold")
    fig.tight_layout()
    path = output_dir / "03_gat_architecture.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_evidence_graph(data, metadata: dict, gat_output, output_dir: Path) -> Path:
    graph = nx.Graph()
    node_map = metadata["node_id_map"]
    scores = gat_output.node_scores
    top_rank = {
        segment_id: rank
        for rank, segment_id in enumerate(gat_output.top_evidence_ids, start=1)
    }
    for node_idx, segment_id in node_map.items():
        graph.add_node(node_idx, segment_id=segment_id)

    relation_names = {
        0: "Question link",
        1: "Shared entity",
        2: "Semantic similarity",
    }
    relation_colors = {
        0: "#7b8fa1",
        1: "#e09f3e",
        2: "#3a86a8",
    }
    segment_scores = [score for segment_id, score in scores.items()
                      if segment_id != "__question__"]
    min_score = min(segment_scores, default=0.0)
    score_range = max(segment_scores, default=1.0) - min_score
    edge_types = data.edge_type.detach().cpu().tolist()
    edge_index = data.edge_index.detach().cpu().tolist()
    for source, target, edge_type in zip(edge_index[0], edge_index[1], edge_types):
        if source == target:
            continue
        if not graph.has_edge(source, target):
            graph.add_edge(source, target, edge_type=int(edge_type))

    positions = nx.spring_layout(graph, seed=17, k=1.0)
    fig, ax = plt.subplots(figsize=(13, 9))
    node_colors = []
    labels = {}
    for node_idx, attrs in graph.nodes(data=True):
        segment_id = attrs["segment_id"]
        if segment_id == "__question__":
            node_colors.append("#ffd166")
            labels[node_idx] = "Q\nquestion"
        else:
            node_score = scores.get(segment_id, 0.0)
            normalized_score = (
                (node_score - min_score) / score_range if score_range else 0.5
            )
            node_colors.append(plt.cm.Reds(0.2 + 0.75 * normalized_score))
            rank = top_rank.get(segment_id)
            short_id = segment_id[-4:]
            rank_label = f"#{rank} {short_id}" if rank else short_id
            labels[node_idx] = f"{rank_label}\n{node_score:.2f}"

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color=[
            relation_colors[graph.edges[e].get("edge_type", 0)]
            for e in graph.edges
        ],
        width=1.5,
        alpha=0.7,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=1200,
        edgecolors="#333333",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=7)
    legend = [
        Line2D([0], [0], color=relation_colors[k], lw=2, label=v)
        for k, v in relation_names.items()
    ]
    legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#e45756",
                         markeredgecolor="#333333", label="Darker red = higher GAT score",
                         markersize=10))
    ax.legend(handles=legend, loc="best", fontsize=8)
    ax.set_title(
        "Retrieved evidence graph: edge colors show relation; labels show GAT rank/score",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    path = output_dir / "04_evidence_graph_gat_scores.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(train_log: dict, output_dir: Path) -> Path | None:
    train_loss = train_log.get("train_loss", [])
    val_loss = train_log.get("val_loss", [])
    val_f1 = train_log.get("val_f1", [])
    val_accuracy = train_log.get("val_accuracy", [])
    if not any((train_loss, val_loss, val_f1, val_accuracy)):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    if train_loss:
        axes[0].plot(range(1, len(train_loss) + 1), train_loss, label="Train loss")
    if val_loss:
        axes[0].plot(range(1, len(val_loss) + 1), val_loss, label="Validation loss")
    axes[0].set_title("Training objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy loss")
    axes[0].legend()

    if val_f1:
        axes[1].plot(range(1, len(val_f1) + 1), val_f1, label="Validation F1")
    if val_accuracy:
        axes[1].plot(range(1, len(val_accuracy) + 1), val_accuracy, label="Validation accuracy")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.suptitle("Recorded GAT training history", weight="bold")
    fig.tight_layout()
    path = output_dir / "05_gat_training_curves.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    started = time.time()
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("LISTEN - VIDEO / TRANSCRIPT / NER / GRAPH / GAT WALKTHROUGH")
    print_code_map()
    print("\n1. VIDEO TO TRANSCRIPT")

    if args.video is not None:
        video_path = args.video.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        if not is_video_file(video_path):
            raise ValueError(f"Unsupported video extension: {video_path.suffix}")
        audio_path = extract_audio_from_video(video_path)
        meeting_id = f"demo_{uuid.uuid4().hex[:8]}"
        segments = enrich_segments_with_entities(
            transcribe_audio(str(audio_path), meeting_id=meeting_id)
        )
        alignment = None
        question = args.question or "What are the main topics discussed?"
        print(f"  Video          : {video_path}")
        print(f"  Extracted audio: {audio_path}")
        print("  Live path      : ffmpeg -> faster-whisper -> spaCy NER")
    else:
        required = [AUDIO_PATH, SEGMENTS_PATH, ALIGN_PATH, CHECKPOINT_PATH]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required demo artifacts are missing:\n"
                + "\n".join(f"  - {path}" for path in missing)
                + "\nSupply --video to run ASR live (the GAT checkpoint is still required)."
            )
        audio_path = AUDIO_PATH
        segments = [
            SLPSegment(**row)
            for row in json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
        ]
        alignment = json.loads(ALIGN_PATH.read_text(encoding="utf-8"))
        if DEMO_QUERY_INDEX >= len(alignment["queries"]):
            raise IndexError(
                f"Demo query index {DEMO_QUERY_INDEX} is not available in {ALIGN_PATH}"
            )
        selected_query = alignment["queries"][DEMO_QUERY_INDEX]
        question = args.question or selected_query["query"]
        print(f"  Input audio    : {audio_path.relative_to(PROJECT_ROOT)}")
        print(f"  Transcript file: {SEGMENTS_PATH.relative_to(PROJECT_ROOT)}")
        print("  Mode           : reusing real stored ASR output; no ASR is rerun")

    if not segments:
        raise ValueError("ASR/transcript processing produced no transcript segments.")

    audio_info = sf.info(str(audio_path))
    print(f"  Audio duration: {audio_info.duration:.1f}s; "
          f"{audio_info.samplerate} Hz; {audio_info.channels} channel(s)")
    print(f"  Transcript segments: {len(segments)}")
    print("  ffmpeg output config: PCM s16le, 16 kHz, mono; video stream is discarded.")
    print("  Whisper config: model='base', CPU/int8, VAD enabled, segment timestamps.")
    print("  Example transcript segments:")
    for segment in segments[:3]:
        print(f"    [{fmt_ts(segment.start_time)}-{fmt_ts(segment.end_time)}] "
              f"{segment.segment_id}: {short(segment.text, 100)}")

    print("\n2. TEXT PREPROCESSING / NER")
    print("  Each ASR segment is whitespace-trimmed; empty ASR segments are skipped.")
    print("  spaCy en_core_web_sm extracts entities per segment; duplicate surface")
    print("  forms are removed case-insensitively. No stemming/stop-word removal is done.")
    for segment in segments[:5]:
        live_entities = extract_entities_from_text(segment.text)
        print(f"    {segment.segment_id}: {short(segment.text, 65)}")
        print(f"      entities: {live_entities}")

    print("\n3. EMBEDDINGS, RETRIEVAL, AND DATASET-TO-GRAPH")
    print(f"  Encoder: {config.EMBEDDING_MODEL}; dimension={config.EMBEDDING_DIM}; "
          "vectors L2-normalized.")
    index = SegmentIndex()
    index.build(segments)
    print(f"  Transcript index: {index.size} vectors in FAISS IndexFlatIP.")
    print(f"  Question: {question}")
    retrieval = retrieve_candidates(question, index, {s.segment_id: s for s in segments})
    if not retrieval.candidates:
        raise ValueError("No transcript candidates were returned for the question.")
    candidate_ids = [candidate.segment_id for candidate in retrieval.candidates]
    segments_by_id = {segment.segment_id: segment for segment in segments}
    candidates = [segments_by_id[sid] for sid in candidate_ids]
    question_embedding = embed_texts([question])[0]
    segment_embeddings = np.stack([index.get_embedding(sid) for sid in candidate_ids])
    print(f"  Retrieved {len(candidates)} candidates (configured top-k="
          f"{config.RETRIEVAL_TOP_K}); graph node 0 is the question.")
    for rank, candidate in enumerate(retrieval.candidates[:5], start=1):
        seg = segments_by_id[candidate.segment_id]
        print(f"    {rank:>2}. cosine={candidate.score:.3f} "
              f"{candidate.segment_id}: {short(seg.text, 72)}")

    builder = EvidenceGraphBuilder()
    data, metadata = builder.build(
        question_embedding=question_embedding,
        segments=candidates,
        segment_embeddings=segment_embeddings,
    )
    edge_counts = Counter(int(edge_type) for edge_type in data.edge_type.tolist())
    print(f"  PyG graph: x={tuple(data.x.shape)}, "
          f"edge_index={tuple(data.edge_index.shape)}")
    print("  Edge counts (directed): "
          f"question-segment={edge_counts[0]}, "
          f"shared-entity={edge_counts[1]}, "
          f"semantic-similarity={edge_counts[2]}.")
    print(f"  Graph labels: node_id_map={metadata['node_id_map']}")
    print("  Training note: phase2_graph/dataset.py builds one labeled graph per QA "
          "example; Data.y marks supporting evidence (1) vs other segments (0).")
    if alignment is not None:
        gold_ids = set(
            alignment["queries"][DEMO_QUERY_INDEX].get("evidence_segment_ids") or []
        )
        print(f"  This stored query has {len(gold_ids)} aligned supporting segment(s); "
              "runtime inference itself does not receive those labels.")
    else:
        gold_ids = set()
        print("  Live video has no dataset gold labels; graph construction is inference-only.")

    print("\n4. GAT MODEL AND EVIDENCE RANKING")
    print("  Serving model: app/dl/gat_model.py (not the separate phase2 training class).")
    print(f"  Architecture config: input={config.EMBEDDING_DIM}, "
          f"hidden={config.GAT_HIDDEN_DIM}, heads={config.GAT_NUM_HEADS}, "
          f"layers={config.GAT_NUM_LAYERS}.")
    gat = GATInference(checkpoint_path=CHECKPOINT_PATH)
    model_parameters = sum(parameter.numel() for parameter in gat.model.parameters())
    print(f"  Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)} "
          f"({model_parameters:,} parameters)")
    gat_output = gat.run(data, metadata)
    print("  Top evidence segments (sigmoid node relevance scores):")
    for rank, segment_id in enumerate(gat_output.top_evidence_ids, start=1):
        score = gat_output.node_scores[segment_id]
        marker = "GOLD" if segment_id in gold_ids else ""
        print(f"    {rank}. score={score:.4f} {marker:>4} {segment_id}: "
              f"{short(segments_by_id[segment_id].text, 74)}")

    print("\n5. PLOTS")
    plot_paths = [
        plot_pipeline(output_dir),
        plot_audio_and_preprocessing(audio_path, segments, candidate_ids, output_dir),
        plot_gat_architecture(output_dir),
        plot_evidence_graph(data, metadata, gat_output, output_dir),
    ]
    if TRAIN_LOG_PATH.is_file():
        training_plot = plot_training_curves(
            json.loads(TRAIN_LOG_PATH.read_text(encoding="utf-8")), output_dir
        )
        if training_plot is not None:
            plot_paths.append(training_plot)
    for path in plot_paths:
        print(f"  Saved: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")

    print("\nDATASET TRAINING FLOW")
    print("  HotpotQA: context sentences -> segment nodes -> supporting_facts labels.")
    print("  QMSum: meeting turns -> query-relevant candidate turns -> relevant-span labels.")
    print("  Both use phase2_graph/graph_builder.py::build_with_labels and are")
    print("  optimized with BCEWithLogitsLoss in phase2_graph/train.py.")
    print("\nWalkthrough complete in {:.1f}s.".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
