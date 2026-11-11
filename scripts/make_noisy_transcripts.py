"""Re-transcribe a labelled meeting under real acoustic noise.

For each noise condition the raw meeting audio is mixed with white noise at
a fixed SNR, then **each existing gold segment's time span** is transcribed
with word timestamps. Keeping the clean segmentation fixed means segment
ids (and therefore the aligned gold labels) stay valid, while the text and
the per-word probabilities come from genuinely noisy audio.

Output: data/processed/<meeting>_segments_<condition>.json, same schema as
the clean segments file plus the uncertainty fields.

Usage:
    LISTEN_DATA_DIR=../data python scripts/make_noisy_transcripts.py --meeting ES2002a --snr clean 10 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.models.schemas import SLPSegment, WordToken
from app.slp.asr import _get_whisper_model, _words_from_segment
from app.slp.entity_extraction import enrich_segments_with_entities
from app.slp.uncertainty import annotate_segment


def load_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        # Linear resample; adequate for noise-robustness experiments.
        n_out = int(len(pcm) * 16000 / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


def add_white_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    speech_rms = float(np.sqrt(np.mean(audio ** 2))) or 1e-6
    noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def transcribe_span(model, audio: np.ndarray, start: float, end: float) -> tuple[str, list[WordToken], float]:
    s, e = int(start * 16000), int(end * 16000)
    clip = audio[max(0, s):max(s + 1600, e)]
    segs, _ = model.transcribe(clip, language="en", word_timestamps=True, vad_filter=False, beam_size=1)
    text_parts, words, no_speech = [], [], 0.0
    for seg in segs:
        t = seg.text.strip()
        if t:
            text_parts.append(t)
        for w in _words_from_segment(seg):
            words.append(w.model_copy(update={"start": start + w.start, "end": start + w.end}))
        no_speech = max(no_speech, float(getattr(seg, "no_speech_prob", 0.0) or 0.0))
    return " ".join(text_parts), words, no_speech


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="ES2002a")
    ap.add_argument("--snr", nargs="+", default=["clean", "10", "0"], help="'clean' or SNR in dB")
    ap.add_argument("--limit", type=int, default=0, help="only first N segments (smoke test)")
    args = ap.parse_args()

    processed = config.DATA_DIR / "processed"
    seg_path = processed / f"{args.meeting}_segments.json"
    wav_path = config.AUDIO_DIR / "raw" / f"{args.meeting}.wav"
    gold = [SLPSegment(**row) for row in json.loads(seg_path.read_text(encoding="utf-8"))]
    if args.limit:
        gold = gold[: args.limit]

    print(f"loading {wav_path} ...", flush=True)
    clean = load_wav_mono16k(wav_path)
    model = _get_whisper_model()

    for cond in args.snr:
        audio = clean if cond == "clean" else add_white_noise(clean, float(cond))
        label = cond if cond == "clean" else f"snr{int(float(cond))}"
        out_path = processed / f"{args.meeting}_segments_{label}.json"
        t0 = time.perf_counter()
        rows: list[SLPSegment] = []
        for i, g in enumerate(gold):
            text, words, no_speech = transcribe_span(model, audio, g.start_time, g.end_time)
            seg = SLPSegment(
                segment_id=g.segment_id, start_time=g.start_time, end_time=g.end_time,
                text=text or "[inaudible]", entities=[], words=words, no_speech_prob=no_speech,
            )
            seg = annotate_segment(seg)
            if not text:
                # Nothing recognised: the span is evidence of nothing.
                seg = seg.model_copy(update={"asr_confidence": 0.05, "min_word_confidence": 0.05, "no_speech_prob": 1.0})
            rows.append(seg)
            if (i + 1) % 25 == 0:
                print(f"  [{label}] {i + 1}/{len(gold)} segments, {time.perf_counter() - t0:.0f}s", flush=True)
        rows = enrich_segments_with_entities(rows)
        out_path.write_text(json.dumps([r.model_dump() for r in rows], indent=1), encoding="utf-8")
        mean_conf = float(np.mean([r.asr_confidence for r in rows]))
        print(f"wrote {out_path.name}: {len(rows)} segments, mean ASR confidence {mean_conf:.3f}, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
