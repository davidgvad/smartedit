"""Deterministic OpenCV frame sampling for multimodal model context."""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from smartedit.schemas import (
    SampledFrame,
    VideoMetadata,
)
from smartedit.schemas import SamplingReason as SchemaSamplingReason

from .ffmpeg_utils import cache_directory_for_video, inspect_video, validate_video_path

LOGGER = logging.getLogger(__name__)

SamplingReason = Literal["uniform", "shot_boundary"]


class FrameSamplingError(RuntimeError):
    """Raised when OpenCV cannot decode a requested frame."""


def _even_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [(length - 1) // 2]
    return sorted({int(round(position * (length - 1) / (count - 1))) for position in range(count)})


def _uniform_frame_indices(frame_count: int, count: int) -> list[int]:
    """Select endpoint-inclusive, deterministic indices without duplicates."""

    if frame_count <= 0 or count <= 0:
        return []
    if count >= frame_count:
        return list(range(frame_count))
    return _even_indices(frame_count, count)


def _validate_cut_timestamps(
    cut_timestamps: Iterable[float],
    duration_seconds: float,
) -> list[float]:
    normalized: list[float] = []
    for timestamp in cut_timestamps:
        try:
            value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid cut timestamp: {timestamp!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= duration_seconds:
            raise ValueError(
                f"Cut timestamp {value!r} is outside video duration [0, {duration_seconds:.6f}]"
            )
        normalized.append(value)
    return sorted(set(normalized))


def build_sampling_plan(
    *,
    duration_seconds: float,
    fps: float,
    max_frames: int,
    cut_timestamps: Iterable[float] = (),
    boundary_offset_seconds: float = 0.15,
) -> list[tuple[int, SamplingReason]]:
    """Build a deterministic frame-index plan with a strict total-frame cap.

    When cuts are available, approximately 60% of the budget preserves uniform
    temporal coverage and the remainder samples immediately before/after a
    representative set of boundaries.  A shot-boundary reason wins if a frame
    is selected by both strategies.
    """

    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be a finite positive number")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be a finite positive number")
    if max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    if boundary_offset_seconds < 0.0:
        raise ValueError("boundary_offset_seconds cannot be negative")

    cuts = _validate_cut_timestamps(cut_timestamps, duration_seconds)
    frame_count = max(1, int(math.ceil(duration_seconds * fps)))
    budget = min(max_frames, frame_count)

    if cuts and budget >= 2:
        uniform_budget = min(budget - 1, max(1, int(math.ceil(budget * 0.6))))
        boundary_budget = budget - uniform_budget
    else:
        uniform_budget = budget
        boundary_budget = 0

    planned: dict[int, SamplingReason] = {
        index: "uniform" for index in _uniform_frame_indices(frame_count, uniform_budget)
    }

    if boundary_budget:
        # Select cut groups evenly across the whole video.  Cycling offsets
        # across groups prevents the first cut from consuming the full budget.
        group_count = min(len(cuts), max(1, math.ceil(boundary_budget / 3)))
        representative_cuts = [cuts[index] for index in _even_indices(len(cuts), group_count)]
        boundary_candidates: list[int] = []
        offsets = (-boundary_offset_seconds, boundary_offset_seconds, 0.0)
        for offset in offsets:
            for cut in representative_cuts:
                candidate_time = min(max(cut + offset, 0.0), duration_seconds)
                candidate_index = min(
                    frame_count - 1,
                    max(0, int(round(candidate_time * fps))),
                )
                if candidate_index not in boundary_candidates:
                    boundary_candidates.append(candidate_index)
        for index in boundary_candidates[:boundary_budget]:
            planned[index] = "shot_boundary"

    # Collisions between uniform and boundary candidates can leave budget. Fill
    # gaps from a denser uniform grid without making the result nondeterministic.
    if len(planned) < budget:
        for index in _uniform_frame_indices(frame_count, min(frame_count, budget * 4)):
            planned.setdefault(index, "uniform")
            if len(planned) >= budget:
                break

    if len(planned) > budget:
        # Preserve all selected boundary frames, then uniformly thin ordinary
        # samples if boundary frames displaced rather than replaced a frame.
        boundary = sorted(index for index, reason in planned.items() if reason == "shot_boundary")
        ordinary = sorted(index for index, reason in planned.items() if reason == "uniform")
        keep_boundary = boundary[:budget]
        remaining = budget - len(keep_boundary)
        keep_ordinary = [ordinary[index] for index in _even_indices(len(ordinary), remaining)]
        planned = {
            **{index: "uniform" for index in keep_ordinary},
            **{index: "shot_boundary" for index in keep_boundary},
        }

    return sorted(planned.items())


def _load_opencv() -> object:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FrameSamplingError(
            "OpenCV is required for frame sampling. Install opencv-python-headless."
        ) from exc
    return cv2


def _write_jpeg_atomic(cv2: object, destination: Path, frame: object) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=".jpg",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)
    try:
        written = cv2.imwrite(  # type: ignore[attr-defined]
            str(temporary),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],  # type: ignore[attr-defined]
        )
        if not written or not temporary.is_file() or temporary.stat().st_size <= 0:
            raise FrameSamplingError(f"OpenCV failed to encode sampled frame {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def sample_frames(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    metadata: VideoMetadata | None = None,
    max_frames: int = 24,
    cut_timestamps: Iterable[float] = (),
    boundary_offset_seconds: float = 0.15,
    force: bool = False,
) -> list[SampledFrame]:
    """Decode uniformly distributed and boundary-adjacent JPEG frames."""

    source = validate_video_path(video_path)
    video_metadata = metadata or inspect_video(source)
    if output_dir is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required when output_dir is omitted")
        destination_dir = cache_directory_for_video(source, cache_dir) / "frames"
    else:
        destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    plan = build_sampling_plan(
        duration_seconds=video_metadata.duration_seconds,
        fps=video_metadata.fps,
        max_frames=max_frames,
        cut_timestamps=cut_timestamps,
        boundary_offset_seconds=boundary_offset_seconds,
    )
    cv2 = _load_opencv()
    capture = cv2.VideoCapture(str(source))  # type: ignore[attr-defined]
    if not capture.isOpened():
        capture.release()
        raise FrameSamplingError(
            f"OpenCV could not open the video stream in {source}. The codec may "
            "not be supported by this OpenCV build."
        )

    sampled: list[SampledFrame] = []
    previous_timestamp = -1.0
    try:
        LOGGER.info("Sampling %d model frames from %s", len(plan), source.name)
        for frame_index, reason in plan:
            estimated_timestamp = min(
                video_metadata.duration_seconds,
                max(0.0, frame_index / video_metadata.fps),
            )
            cached_candidates = sorted(
                path
                for path in destination_dir.glob(f"frame_{frame_index:09d}_*us.jpg")
                if path.is_file() and path.stat().st_size > 0
            )
            cached = not force and bool(cached_candidates)
            if not cached:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # type: ignore[attr-defined]
                decoded, frame = capture.read()
                if not decoded or frame is None:
                    raise FrameSamplingError(
                        f"OpenCV could not decode frame {frame_index} at "
                        f"{estimated_timestamp:.3f}s from {source}"
                    )
                reported_timestamp = (
                    float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1_000.0  # type: ignore[attr-defined]
                )
                timestamp = estimated_timestamp
                if (
                    math.isfinite(reported_timestamp)
                    and 0.0 <= reported_timestamp <= video_metadata.duration_seconds
                    and (frame_index == 0 or reported_timestamp > 0.0)
                    and reported_timestamp >= previous_timestamp
                    and abs(reported_timestamp - estimated_timestamp)
                    <= max(0.5, 3.0 / video_metadata.fps)
                ):
                    timestamp = reported_timestamp
                # Floor to the cache's microsecond precision so serialization can
                # never move a valid endpoint beyond the source duration.
                timestamp_us = int(math.floor(timestamp * 1_000_000.0 + 1e-9))
                timestamp = timestamp_us / 1_000_000.0
                destination = destination_dir / (
                    f"frame_{frame_index:09d}_{timestamp_us:015d}us.jpg"
                )
                _write_jpeg_atomic(cv2, destination, frame)
            else:
                destination = cached_candidates[0]
                try:
                    encoded_timestamp = destination.stem.rsplit("_", 1)[1]
                    timestamp = int(encoded_timestamp.removesuffix("us")) / 1_000_000.0
                except (IndexError, ValueError) as exc:
                    raise FrameSamplingError(
                        f"Malformed cached frame timestamp in {destination.name}"
                    ) from exc
                if not 0.0 <= timestamp <= video_metadata.duration_seconds:
                    raise FrameSamplingError(
                        f"Cached frame timestamp {timestamp} is outside the video duration"
                    )
                if timestamp < previous_timestamp:
                    raise FrameSamplingError(
                        "Cached sampled-frame timestamps are not monotonic; rerun with force=True"
                    )
            sampled.append(
                SampledFrame(
                    timestamp_seconds=timestamp,
                    path=str(destination),
                    frame_index=frame_index,
                    sampling_reason=SchemaSamplingReason(reason),
                )
            )
            previous_timestamp = timestamp
    finally:
        capture.release()

    return sampled
