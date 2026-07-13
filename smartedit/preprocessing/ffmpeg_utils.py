"""Safe ffmpeg/ffprobe helpers used by the preprocessing pipeline.

All subprocesses are invoked with argument lists and ``shell=False``.  This
module deliberately does not install ffmpeg or download codecs; callers get a
specific, actionable exception when the local tools are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from smartedit.schemas import VideoMetadata

LOGGER = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})


class FFmpegError(RuntimeError):
    """Base class for local ffmpeg/ffprobe failures."""


class FFmpegNotFoundError(FFmpegError):
    """Raised when a required ffmpeg executable is not on PATH."""


class MediaValidationError(FFmpegError):
    """Raised when an input is missing, unsupported, or not decodable."""


@dataclass(frozen=True)
class FFmpegTools:
    """Resolved local tool paths."""

    ffmpeg: str | None = None
    ffprobe: str | None = None


def _resolve_executable(name: str, override: str | Path | None = None) -> str | None:
    candidate = str(override) if override is not None else None
    if candidate:
        expanded = str(Path(candidate).expanduser())
        if Path(expanded).is_file() and os.access(expanded, os.X_OK):
            return str(Path(expanded).resolve())
        located = shutil.which(expanded)
        if located:
            return located
        return None
    return shutil.which(name)


def validate_ffmpeg_tools(
    *,
    require_ffmpeg: bool = True,
    require_ffprobe: bool = True,
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
) -> FFmpegTools:
    """Resolve required binaries and raise a useful error if one is missing."""

    ffmpeg_override = ffmpeg_path or os.getenv("FFMPEG_BINARY")
    ffprobe_override = ffprobe_path or os.getenv("FFPROBE_BINARY")
    resolved_ffmpeg = _resolve_executable("ffmpeg", ffmpeg_override)
    resolved_ffprobe = _resolve_executable("ffprobe", ffprobe_override)

    missing: list[str] = []
    if require_ffmpeg and not resolved_ffmpeg:
        missing.append("ffmpeg")
    if require_ffprobe and not resolved_ffprobe:
        missing.append("ffprobe")
    if missing:
        names = " and ".join(missing)
        raise FFmpegNotFoundError(
            f"Required executable(s) {names} were not found. Install ffmpeg "
            "and ensure both ffmpeg and ffprobe are on PATH, or set "
            "FFMPEG_BINARY/FFPROBE_BINARY."
        )

    return FFmpegTools(ffmpeg=resolved_ffmpeg, ffprobe=resolved_ffprobe)


def validate_video_path(
    video_path: str | Path,
    *,
    allowed_extensions: frozenset[str] = SUPPORTED_VIDEO_EXTENSIONS,
) -> Path:
    """Validate a supported, non-empty local video path."""

    path = Path(video_path).expanduser()
    if not path.exists():
        raise MediaValidationError(f"Video file does not exist: {path}")
    if not path.is_file():
        raise MediaValidationError(f"Video path is not a regular file: {path}")
    if path.suffix.lower() not in allowed_extensions:
        supported = ", ".join(sorted(allowed_extensions))
        raise MediaValidationError(
            f"Unsupported video extension {path.suffix or '<none>'!r}. "
            f"Supported short-video containers are: {supported}."
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MediaValidationError(f"Cannot inspect video file {path}: {exc}") from exc
    if size <= 0:
        raise MediaValidationError(f"Video file is empty: {path}")
    if not os.access(path, os.R_OK):
        raise MediaValidationError(f"Video file is not readable: {path}")
    return path.resolve()


def _run_command(
    command: Sequence[str],
    *,
    operation: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{operation} timed out after {timeout_seconds:g} seconds") from exc
    except OSError as exc:
        raise FFmpegError(f"Unable to start {operation}: {exc}") from exc

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "no diagnostic output").strip()
        if len(details) > 2_000:
            details = details[-2_000:]
        raise FFmpegError(f"{operation} failed with exit code {completed.returncode}: {details}")
    return completed


def ffprobe_json(
    video_path: str | Path,
    *,
    ffprobe_path: str | Path | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Return raw ffprobe JSON for a validated local video."""

    path = validate_video_path(video_path)
    tools = validate_ffmpeg_tools(
        require_ffmpeg=False,
        require_ffprobe=True,
        ffprobe_path=ffprobe_path,
    )
    assert tools.ffprobe is not None
    command = [
        tools.ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(path),
    ]
    completed = _run_command(
        command,
        operation=f"ffprobe inspection of {path.name}",
        timeout_seconds=timeout_seconds,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaValidationError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MediaValidationError(f"ffprobe returned an unexpected result for {path}")
    return payload


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0 or not parsed < float("inf"):
        return None
    return parsed


def _parse_frame_rate(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0.0 else None


def _rotation_degrees(stream: dict[str, Any]) -> int:
    raw_rotation: Any = (stream.get("tags") or {}).get("rotate")
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, dict) and side_data.get("rotation") is not None:
            raw_rotation = side_data["rotation"]
            break
    try:
        rotation = int(round(float(raw_rotation))) % 360
    except (TypeError, ValueError):
        return 0
    return rotation


def _is_attached_picture(stream: dict[str, Any]) -> bool:
    disposition = stream.get("disposition")
    raw_value = disposition.get("attached_pic", 0) if isinstance(disposition, dict) else 0
    try:
        return int(raw_value) != 0
    except (TypeError, ValueError):
        return bool(raw_value)


def inspect_video(
    video_path: str | Path,
    *,
    ffprobe_path: str | Path | None = None,
    timeout_seconds: float = 30.0,
) -> VideoMetadata:
    """Inspect a video and normalize the metadata required by SmartEdit."""

    path = validate_video_path(video_path)
    payload = ffprobe_json(
        path,
        ffprobe_path=ffprobe_path,
        timeout_seconds=timeout_seconds,
    )
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaValidationError(f"ffprobe found no streams in {path}")

    video_streams = [
        stream
        for stream in streams
        if isinstance(stream, dict)
        and stream.get("codec_type") == "video"
        and not _is_attached_picture(stream)
    ]
    if not video_streams:
        raise MediaValidationError(
            f"No decodable video stream was found in {path}. The file may be "
            "audio-only, corrupt, or use an unsupported codec."
        )
    video_stream = video_streams[0]
    audio_streams = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    raw_format_info = payload.get("format")
    format_info: dict[str, Any] = raw_format_info if isinstance(raw_format_info, dict) else {}

    duration = _positive_float(video_stream.get("duration"))
    if duration is None:
        duration = _positive_float(format_info.get("duration"))

    fps = _parse_frame_rate(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_frame_rate(video_stream.get("r_frame_rate"))

    if duration is None and fps:
        frame_count = _positive_float(video_stream.get("nb_frames"))
        if frame_count:
            duration = frame_count / fps
    if fps is None and duration:
        frame_count = _positive_float(video_stream.get("nb_frames"))
        if frame_count:
            fps = frame_count / duration

    if duration is None:
        raise MediaValidationError(
            f"Could not determine video duration for {path}. ffprobe did not "
            "report a valid stream or container duration."
        )
    if fps is None:
        raise MediaValidationError(f"Could not determine a valid frame rate for {path}.")

    try:
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
    except (TypeError, ValueError) as exc:
        raise MediaValidationError(f"Invalid video resolution reported for {path}") from exc
    if width <= 0 or height <= 0:
        raise MediaValidationError(f"Could not determine a valid video resolution for {path}.")

    rotation = _rotation_degrees(video_stream)
    if rotation in {90, 270}:
        width, height = height, width

    LOGGER.debug(
        "Inspected %s: %.3fs, %.3f fps, %dx%d, audio=%s",
        path,
        duration,
        fps,
        width,
        height,
        bool(audio_streams),
    )
    return VideoMetadata(
        path=str(path),
        duration_seconds=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio=bool(audio_streams),
        format_name=str(format_info.get("format_name") or "") or None,
        video_codec=str(video_stream.get("codec_name") or "") or None,
        audio_codec=(
            str(audio_streams[0].get("codec_name") or "") or None if audio_streams else None
        ),
        rotation=rotation,
    )


def source_fingerprint(video_path: str | Path) -> str:
    """Return a cache key that changes when the source file changes."""

    path = validate_video_path(video_path)
    stat = path.stat()
    identity = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    return hashlib.sha256(identity).hexdigest()[:16]


def cache_directory_for_video(
    video_path: str | Path,
    cache_root: str | Path,
) -> Path:
    """Create and return a deterministic per-source artifact directory."""

    path = validate_video_path(video_path)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "video"
    artifact_dir = Path(cache_root).expanduser().resolve() / (
        f"{safe_stem}-{source_fingerprint(path)}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir

