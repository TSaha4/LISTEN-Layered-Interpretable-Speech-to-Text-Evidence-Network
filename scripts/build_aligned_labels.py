"""Build ``<meeting>_segments.json`` and ``<meeting>_aligned_labels.json`` for an AMI meeting.

This reproduces the pipeline that produced the checked-in ES2002a artifacts:

1. ``data/audio/raw/<meeting>.wav`` is transcribed with the production ASR
   (``app.slp.asr.transcribe_audio``: faster-whisper, VAD, word timestamps)
   and run through spaCy NER (``enrich_segments_with_entities``).
2. Every QMSum transcript turn with at least ``--min-words`` alphabetic
   words (3; shorter turns are back-channels such as "Mm-hmm .") is embedded
   with the same MiniLM encoder the retriever uses and compared with the ASR
   segments whose relative position in the meeting lies within ``--window``
   (0.067, i.e. +-20 segments on a 300-segment meeting) of the turn's relative
   position.  The turn is matched to the candidate maximising

       combined = 0.85 * cosine(turn, segment)
               + 0.10 * (1 - |turn_pos - segment_pos|)      # relative position
               + 0.05 * speaker_score                       # always 0: ASR has no speakers

   and the match is kept only when the cosine similarity is at least
   ``--threshold`` (0.4, stored as ``confidence_threshold``).
3. Each QMSum ``specific_query_list`` entry has ``relevant_text_span``
   ranges of turn indices (inclusive).  The gold ``evidence_segment_ids`` of
   a query are the ASR segments matched to any turn inside those ranges.
   Queries whose spans contain no kept turn get no label and are reported.

Reproducibility: the original alignment script was not committed, so the
scoring above was reverse-engineered from ``ES2002a_aligned_labels.json``
(the stored ``semantic_similarity``, ``sequence_score`` and ``combined_score``
values are reproduced to 1e-7).  The candidate window and the length filter
are the best fit found: with ``--reuse-segments`` on ES2002a this script
recovers 114 of the 118 stored turn matches and adds 21, and the six gold
evidence sets agree with the stored ones at Jaccard 0.33-1.00 (mean 0.76).
The stored ES2002a files are left untouched; the new meetings are labelled
with this script.

QMSum source files are read from ``data/qmsum_product/train/<meeting>.json``
and fetched from the public Yale-LILY/QMSum repository (``data/ALL/train``,
falling back to ``val`` and ``test``) when missing.

Usage:
    LISTEN_DATA_DIR=../data python scripts/build_aligned_labels.py --meeting ES2002b ES2002c
    LISTEN_DATA_DIR=../data python scripts/build_aligned_labels.py --meeting ES2002a --reuse-segments --out-dir /tmp/check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.models.schemas import SLPSegment

QMSUM_RAW_BASE = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL"
QMSUM_SPLITS = ("train", "val", "test")

W_SEMANTIC, W_SEQUENCE, W_SPEAKER = 0.85, 0.10, 0.05
DEFAULT_THRESHOLD = 0.4
DEFAULT_MIN_WORDS = 3
DEFAULT_WINDOW = 0.067  # fraction of the meeting; +-20 segments on ES2002a

_TAG_RE = re.compile(r"\{[^}]*\}")  # QMSum markers such as {vocalsound}, {disfmarker}
_WORD_RE = re.compile(r"[A-Za-z']+")


def alpha_word_count(text: str) -> int:
    return len(_WORD_RE.findall(_TAG_RE.sub("", text)))


# --------------------------------------------------------------------------- QMSum


def fetch_qmsum(meeting: str, qmsum_dir: Path) -> Path:
    """Return the local QMSum JSON for ``meeting``, downloading it if needed."""
    qmsum_dir.mkdir(parents=True, exist_ok=True)
    target = qmsum_dir / f"{meeting}.json"
    if target.is_file():
        return target
    for split in QMSUM_SPLITS:
        url = f"{QMSUM_RAW_BASE}/{split}/{meeting}.json"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed public host
                payload = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        data = json.loads(payload.decode("utf-8"))
        if "meeting_transcripts" not in data or "specific_query_list" not in data:
            raise ValueError(f"{url} does not look like a QMSum meeting file")
        target.write_bytes(payload)
        print(f"  fetched QMSum {meeting} from split '{split}' -> {target}", flush=True)
        return target
    raise FileNotFoundError(f"{meeting}.json not found in any QMSum split under {QMSUM_RAW_BASE}")


# --------------------------------------------------------------------------- ASR


def transcribe_meeting(meeting: str, wav_path: Path) -> list[SLPSegment]:
    from app.slp.asr import transcribe_audio
    from app.slp.entity_extraction import enrich_segments_with_entities

    t0 = time.perf_counter()
    print(f"  transcribing {wav_path} (model={config.WHISPER_MODEL}, {config.WHISPER_DEVICE}/{config.WHISPER_COMPUTE_TYPE}) ...", flush=True)
    segments = transcribe_audio(str(wav_path), meeting_id=meeting)
    print(f"  {len(segments)} segments in {time.perf_counter() - t0:.0f}s; running NER ...", flush=True)
    segments = enrich_segments_with_entities(segments)
    return segments


def load_segments(path: Path) -> list[SLPSegment]:
    return [SLPSegment(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------- alignment


def align_turns(
    turns: list[dict],
    segments: list[SLPSegment],
    threshold: float,
    min_words: int = DEFAULT_MIN_WORDS,
    window: float = DEFAULT_WINDOW,
) -> list[dict]:
    """Match each QMSum turn to its best nearby ASR segment; keep matches above ``threshold``."""
    from app.dl.embeddings import embed_texts

    turn_texts = [t["content"] for t in turns]
    seg_texts = [s.text for s in segments]
    turn_emb = embed_texts(turn_texts)  # L2-normalised
    seg_emb = embed_texts(seg_texts)
    semantic = turn_emb @ seg_emb.T  # (T, S) cosine

    n_t, n_s = len(turns), len(segments)
    t_pos = np.arange(n_t) / max(n_t - 1, 1)
    s_pos = np.arange(n_s) / max(n_s - 1, 1)
    sequence = 1.0 - np.abs(t_pos[:, None] - s_pos[None, :])
    speaker = np.zeros_like(semantic)  # ASR output carries no speaker labels
    combined = W_SEMANTIC * semantic + W_SEQUENCE * sequence + W_SPEAKER * speaker

    matches: list[dict] = []
    for ti, turn in enumerate(turns):
        if alpha_word_count(turn["content"]) < min_words:
            continue
        candidates = np.flatnonzero(np.abs(s_pos - t_pos[ti]) <= window)
        if candidates.size == 0:
            continue
        si = int(candidates[np.argmax(combined[ti, candidates])])
        if semantic[ti, si] < threshold:
            continue
        seg = segments[si]
        matches.append(
            {
                "turn_index": ti,
                "qmsum_speaker": turn["speaker"],
                "qmsum_text": turn["content"],
                "asr_segment_id": seg.segment_id,
                "asr_start_time": seg.start_time,
                "asr_end_time": seg.end_time,
                "asr_text": seg.text,
                "semantic_similarity": float(semantic[ti, si]),
                "sequence_score": float(sequence[ti, si]),
                "speaker_score": float(speaker[ti, si]),
                "combined_score": float(combined[ti, si]),
            }
        )
    return matches


def label_queries(qmsum: dict, matches: list[dict], segments: list[SLPSegment]) -> tuple[list[dict], list[dict]]:
    """Turn ``relevant_text_span`` turn ranges into gold ASR segment ids."""
    order = {s.segment_id: i for i, s in enumerate(segments)}
    by_turn: dict[int, set[str]] = {}
    for m in matches:
        by_turn.setdefault(m["turn_index"], set()).add(m["asr_segment_id"])

    labelled, unlabelled = [], []
    for q in qmsum.get("specific_query_list", []):
        ids: set[str] = set()
        for span in q.get("relevant_text_span", []):
            lo, hi = int(span[0]), int(span[1])
            for t in range(lo, hi + 1):
                ids.update(by_turn.get(t, ()))
        row = {"query": q["query"], "answer": q["answer"], "evidence_segment_ids": sorted(ids, key=order.get)}
        if ids:
            labelled.append(row)
        else:
            unlabelled.append({**row, "relevant_text_span": q.get("relevant_text_span", [])})
    return labelled, unlabelled


# --------------------------------------------------------------------------- main


def build(meeting: str, args: argparse.Namespace) -> dict:
    processed = args.out_dir
    processed.mkdir(parents=True, exist_ok=True)
    qmsum_path = fetch_qmsum(meeting, args.qmsum_dir)
    qmsum = json.loads(qmsum_path.read_text(encoding="utf-8"))
    turns = qmsum["meeting_transcripts"]

    seg_path = processed / f"{meeting}_segments.json"
    existing = config.DATA_DIR / "processed" / f"{meeting}_segments.json"
    if args.reuse_segments and existing.is_file():
        print(f"  reusing {existing}", flush=True)
        segments = load_segments(existing)
    else:
        wav_path = config.AUDIO_DIR / "raw" / f"{meeting}.wav"
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        segments = transcribe_meeting(meeting, wav_path)
        seg_path.write_text(json.dumps([s.model_dump() for s in segments], indent=1), encoding="utf-8")
        print(f"  wrote {seg_path}", flush=True)
    if not segments:
        raise ValueError(f"{meeting}: ASR produced no segments")

    matches = align_turns(turns, segments, args.threshold, args.min_words, args.window)
    labelled, unlabelled = label_queries(qmsum, matches, segments)

    out = {
        "meeting_id": meeting,
        "confidence_threshold": args.threshold,
        "min_turn_words": args.min_words,
        "position_window": args.window,
        "segment_count": len(segments),
        "turn_count": len(turns),
        "kept_turn_count": len(matches),
        "matches": matches,
        "queries": labelled,
        "unlabelled_queries": unlabelled,
    }
    align_path = processed / f"{meeting}_aligned_labels.json"
    align_path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    sims = [m["semantic_similarity"] for m in matches]
    print(
        f"  {meeting}: {len(segments)} segments, {len(turns)} turns, {len(matches)} kept "
        f"({len(matches) / max(len(turns), 1):.0%}), mean similarity {np.mean(sims) if sims else 0:.3f}; "
        f"{len(labelled)} labelled queries (gold sizes {[len(q['evidence_segment_ids']) for q in labelled]}), "
        f"{len(unlabelled)} without evidence",
        flush=True,
    )
    print(f"  wrote {align_path}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meeting", nargs="+", required=True, help="AMI meeting id(s), e.g. ES2002b")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="minimum cosine similarity to keep a turn match")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS, help="skip QMSum turns with fewer alphabetic words")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW, help="candidate segments within this fraction of the meeting around the turn's relative position")
    ap.add_argument("--reuse-segments", action="store_true", help="use an existing <meeting>_segments.json instead of running ASR")
    ap.add_argument("--out-dir", type=Path, default=config.DATA_DIR / "processed", help="where to write the two JSON files")
    ap.add_argument("--qmsum-dir", type=Path, default=config.DATA_DIR / "qmsum_product" / "train", help="QMSum meeting JSON folder")
    args = ap.parse_args()

    for meeting in args.meeting:
        print(f"== {meeting}", flush=True)
        build(meeting, args)


if __name__ == "__main__":
    main()
