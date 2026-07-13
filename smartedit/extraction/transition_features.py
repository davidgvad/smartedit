"""Deterministic shot statistics derived from TransNet-V2 predictions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from smartedit.models.transnet_adapter import TransNetPrediction, TransNetV2Adapter
from smartedit.schemas import (
    ShotInterval,
    TimestampedObservation,
    TransitionAnalysis,
    VideoMetadata,
    to_dict,
)


def _validate_inputs(
    *,
    duration_seconds: float,
    fps: float,
    cut_frames: Sequence[int],
    cut_timestamps: Sequence[float],
) -> None:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be a finite positive value")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be a finite positive value")
    if len(cut_frames) != len(cut_timestamps):
        raise ValueError("cut_frames and cut_timestamps must have equal lengths")
    if any(type(frame) is not int or frame < 0 for frame in cut_frames):
        raise ValueError("cut_frames must contain non-negative integers")
    if list(cut_frames) != sorted(cut_frames):
        raise ValueError("cut_frames must be sorted")
    if list(cut_timestamps) != sorted(cut_timestamps):
        raise ValueError("cut_timestamps must be sorted")
    for timestamp in cut_timestamps:
        if not math.isfinite(timestamp) or not 0.0 <= timestamp <= duration_seconds:
            raise ValueError(f"cut timestamp {timestamp!r} outside [0, {duration_seconds}]")


def _interior_unique_cuts(
    *,
    cut_frames: Sequence[int],
    cut_timestamps: Sequence[float],
    duration_seconds: float,
    frame_count: int,
) -> list[tuple[int, float]]:
    """Remove endpoints and exact duplicate frame boundaries without reordering."""

    pairs: list[tuple[int, float]] = []
    seen_frames: set[int] = set()
    for frame, timestamp in zip(cut_frames, cut_timestamps, strict=True):
        if frame <= 0 or frame >= frame_count:
            continue
        if timestamp <= 0.0 or timestamp >= duration_seconds:
            continue
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        pairs.append((frame, timestamp))
    return pairs


def build_transition_analysis(
    *,
    duration_seconds: float,
    fps: float,
    frame_count: int,
    cut_frames: Sequence[int],
    cut_timestamps: Sequence[float],
    model_name: str = "transnet_v2",
    evidence: Sequence[TimestampedObservation] = (),
    raw_output: dict[str, object] | None = None,
) -> TransitionAnalysis:
    """Calculate shots, cut rate, and population variance from detected cuts."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    _validate_inputs(
        duration_seconds=duration_seconds,
        fps=fps,
        cut_frames=cut_frames,
        cut_timestamps=cut_timestamps,
    )
    pairs = _interior_unique_cuts(
        cut_frames=cut_frames,
        cut_timestamps=cut_timestamps,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
    )
    normalized_frames = [frame for frame, _ in pairs]
    normalized_timestamps = [timestamp for _, timestamp in pairs]

    time_boundaries = [0.0, *normalized_timestamps, duration_seconds]
    frame_boundaries = [0, *normalized_frames, frame_count]
    shots: list[ShotInterval] = []
    durations: list[float] = []
    for index in range(len(time_boundaries) - 1):
        start = time_boundaries[index]
        end = time_boundaries[index + 1]
        duration = end - start
        if duration < 0.0:
            raise ValueError("cut timestamps would create a negative shot duration")
        start_frame = frame_boundaries[index]
        exclusive_end_frame = frame_boundaries[index + 1]
        end_frame = max(start_frame, exclusive_end_frame - 1)
        shots.append(
            ShotInterval(
                index=index,
                start=start,
                end=end,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        durations.append(duration)

    cut_count = len(normalized_timestamps)
    shot_count = len(shots)
    average_duration = statistics.fmean(durations)
    variance = statistics.pvariance(durations) if len(durations) > 1 else 0.0
    return TransitionAnalysis(
        shot_count=shot_count,
        cut_count=cut_count,
        cut_frames=normalized_frames,
        cut_timestamps=normalized_timestamps,
        shots=shots,
        average_shot_duration=average_duration,
        median_shot_duration=statistics.median(durations),
        minimum_shot_duration=min(durations),
        maximum_shot_duration=max(durations),
        cuts_per_minute=cut_count * 60.0 / duration_seconds,
        shot_duration_variance=variance,
        evidence=list(evidence),
        model_name=model_name,
        raw_output=raw_output,
    )


def transition_features_from_prediction(
    prediction: TransNetPrediction,
    metadata: VideoMetadata,
) -> TransitionAnalysis:
    """Convert raw TransNet probabilities/ranges into objective measurements."""

    prediction.validate_timestamps(metadata.duration_seconds)
    range_by_peak = {
        transition.peak_frame: transition for transition in prediction.transition_ranges
    }
    observations: list[TimestampedObservation] = []
    for frame, timestamp in zip(
        prediction.cut_frames,
        prediction.cut_timestamps,
        strict=True,
    ):
        transition = range_by_peak.get(frame)
        if transition is not None:
            description = (
                "TransNet-V2 detected a shot boundary "
                f"(peak probability {transition.peak_probability:.3f}, "
                f"frames {transition.start_frame}-{transition.end_frame})."
            )
        else:
            description = "TransNet-V2 detected a shot boundary."
        observations.append(
            TimestampedObservation(
                start=timestamp,
                end=timestamp,
                observation=description,
            )
        )
    if not observations:
        observations.append(
            TimestampedObservation(
                start=0.0,
                end=metadata.duration_seconds,
                observation=(
                    "No interior shot boundary exceeded the configured "
                    f"TransNet-V2 threshold of {prediction.threshold:.2f}."
                ),
            )
        )

    analysis = build_transition_analysis(
        duration_seconds=metadata.duration_seconds,
        fps=prediction.fps,
        frame_count=prediction.frame_count,
        cut_frames=prediction.cut_frames,
        cut_timestamps=prediction.cut_timestamps,
        model_name=prediction.model_name,
        evidence=observations,
        raw_output=to_dict(prediction),
    )
    analysis.validate_timestamps(metadata.duration_seconds)
    return analysis


def analyze_transitions(
    video_path: str,
    *,
    metadata: VideoMetadata,
    adapter: TransNetV2Adapter,
) -> TransitionAnalysis:
    """Run TransNet and convert its predictions into shot measurements."""

    prediction = adapter.analyze(video_path, metadata=metadata)
    return transition_features_from_prediction(prediction, metadata)
