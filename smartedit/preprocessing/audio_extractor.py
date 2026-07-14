"""Audio extraction with ffmpeg and deterministic cache paths."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from smartedit.schemas import VideoMetadata

from .ffmpeg_utils import (
    FFmpegError,
    MediaValidationError,
    _run_command,
    cache_directory_for_video,
    inspect_video,
    validate_ffmpeg_tools,
    validate_video_path,
)

LOGGER = logging.getLogger(__name__)


class NoAudioStreamError(MediaValidationError):
    """Raised when audio extraction is requested for a silent video stream."""


def _usable_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 44
    except OSError:
        return False


def extract_audio(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    metadata: VideoMetadata | None = None,
    sample_rate: int = 16_000,
    channels: int = 1,
    force: bool = False,
    ffmpeg_path: str | Path | None = None,
    timeout_seconds: float = 600.0,
) -> Path:
    """Extract a cached PCM WAV at the requested sample rate and channel layout.

    ``output_path`` may be supplied directly.  Otherwise a stable filename in
    the source-specific ``cache_dir`` is used.  The source is never modified.
    """

    source = validate_video_path(video_path)
    if sample_rate < 8_000:
        raise ValueError("sample_rate must be at least 8000 Hz")
    if channels not in {1, 2}:
        raise ValueError("channels must be 1 (mono) or 2 (stereo)")

    video_metadata = metadata or inspect_video(source)
    if not video_metadata.has_audio:
        raise NoAudioStreamError(
            f"Video has no audio stream, so audio extraction was skipped: {source}"
        )

    if output_path is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required when output_path is omitted")
        artifact_dir = cache_directory_for_video(source, cache_dir)
        layout = "mono" if channels == 1 else "stereo"
        destination = artifact_dir / f"audio_{sample_rate}hz_{layout}.wav"
    else:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() != ".wav":
        raise ValueError("audio extraction output_path must use the .wav extension")

    if not force and _usable_file(destination):
        LOGGER.debug("Reusing cached extracted audio: %s", destination)
        return destination

    tools = validate_ffmpeg_tools(
        require_ffmpeg=True,
        require_ffprobe=False,
        ffmpeg_path=ffmpeg_path,
    )
    assert tools.ffmpeg is not None
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=".wav",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)

    command = [
        tools.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        LOGGER.info("Extracting audio at %d Hz: %s", sample_rate, destination)
        _run_command(
            command,
            operation=f"ffmpeg audio extraction from {source.name}",
            timeout_seconds=timeout_seconds,
        )
        if not _usable_file(temporary):
            raise FFmpegError("ffmpeg completed but produced an empty WAV file")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_audio_if_present(
    video_path: str | Path,
    *,
    metadata: VideoMetadata,
    **kwargs: Any,
) -> Path | None:
    """Return ``None`` for a genuinely silent video, otherwise extract audio."""

    if not metadata.has_audio:
        LOGGER.info("No audio stream detected; skipping audio extraction")
        return None
    return extract_audio(video_path, metadata=metadata, **kwargs)
