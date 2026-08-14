"""Dataset loaders for AMI (HuggingFace) and QMSum (GitHub JSON files)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app import config


@dataclass
class QMSumSample:
    """One QMSum specific query with gold evidence turn indices."""

    meeting_id: str
    query: str
    answer: str
    gold_turn_indices: set[int] = field(default_factory=set)
    meeting_transcripts: list[dict[str, str]] = field(default_factory=list)


def _parse_relevant_spans(spans: list[list[str]] | None) -> set[int]:
    """Expand QMSum ``relevant_text_span`` into a set of turn indices."""
    indices: set[int] = set()
    if not spans:
        return indices
    for span in spans:
        if len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        indices.update(range(start, end + 1))
    return indices


def _load_qmsum_json_files(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield raw QMSum meeting dicts from a directory of JSON files."""
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            yield json.load(fh)


def load_qmsum_split(data_dir: str | Path, split: str) -> list[QMSumSample]:
    """Load QMSum samples for a split (``train``, ``val``, or ``test``).

    Expects ``data_dir/{split}/*.json`` mirroring the Yale-LILY repo layout.
    Only ``specific_query_list`` entries (which include gold spans) are used
    for GAT supervision.

    Args:
        data_dir: Root QMSum data directory.
        split: One of ``train``, ``val``, ``test``.

    Returns:
        List of ``QMSumSample`` objects.
    """
    split_dir = Path(data_dir) / split
    samples: list[QMSumSample] = []

    for meeting_idx, meeting in enumerate(_load_qmsum_json_files(split_dir)):
        meeting_id = f"qmsum_{split}_{meeting_idx:04d}"
        transcripts = meeting.get("meeting_transcripts", [])

        for query_item in meeting.get("specific_query_list", []):
            gold = _parse_relevant_spans(query_item.get("relevant_text_span"))
            if not gold:
                continue
            samples.append(
                QMSumSample(
                    meeting_id=meeting_id,
                    query=query_item["query"],
                    answer=query_item["answer"],
                    gold_turn_indices=gold,
                    meeting_transcripts=transcripts,
                )
            )
    return samples


def load_ami_corpus(
    split: str = "train",
    max_meetings: int | None = None,
) -> list[dict[str, Any]]:
    """Load AMI meeting metadata via HuggingFace ``datasets``.

    Args:
        split: Dataset split passed to ``load_dataset``.
        max_meetings: Optional cap for development / low-memory hosts.

    Returns:
        List of AMI meeting records from the HF dataset.

    Raises:
        ImportError: If ``datasets`` is not installed.
        RuntimeError: If the dataset cannot be fetched.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to load AMI: pip install datasets") from exc

    try:
        ds = load_dataset("edinburghcstr/ami", split=split, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load AMI from HuggingFace. Check network access and "
            "that `edinburghcstr/ami` is available."
        ) from exc

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if max_meetings is not None and idx >= max_meetings:
            break
        records.append(dict(row))
    return records


def download_qmsum_instructions() -> str:
    """Return shell instructions for obtaining QMSum data."""
    return (
        "Clone https://github.com/Yale-LILY/QMSum and copy JSON files into:\n"
        f"  {config.QMSUM_DIR}/train/\n"
        f"  {config.QMSUM_DIR}/val/\n"
        f"  {config.QMSUM_DIR}/test/\n"
        "Use the domain folders (e.g. data/Academic/train/*.json)."
    )
