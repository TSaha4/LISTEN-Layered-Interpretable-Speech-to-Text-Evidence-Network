"""Extract mono PCM audio from video files using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from app import config


def extract_audio_from_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a video file to 16 kHz mono WAV suitable for Whisper.

    Args:
        video_path: Path to the input video file.
        output_path: Optional destination WAV path. When omitted, a file is
            created next to the video with a ``.wav`` suffix.

    Returns:
        Path to the extracted WAV audio file.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If ffmpeg exits with a non-zero status.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = config.AUDIO_DIR / f"{video_path.stem}.wav"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        config.FFMPEG_AUDIO_CODEC,
        "-ar",
        str(config.FFMPEG_SAMPLE_RATE),
        "-ac",
        "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    return output_path


def is_video_file(path: str | Path) -> bool:
    """Return True when the file extension is a supported video type."""
    return Path(path).suffix.lower() in config.ALLOWED_VIDEO_EXTENSIONS


def is_audio_file(path: str | Path) -> bool:
    """Return True when the file extension is a supported audio type."""
    return Path(path).suffix.lower() in config.ALLOWED_AUDIO_EXTENSIONS
