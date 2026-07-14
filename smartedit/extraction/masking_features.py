"""Objective speech/music balance measurements from separated audio stems.

This module does not decide whether a video's audio editing is good or bad. It
measures the RMS loudness margin between a separated vocal stem and its
accompaniment during timestamped Whisper speech intervals. Fusion rules can
then interpret those measurements in context.

The two stems must remain on the same amplitude scale. In particular, do not
normalize the vocal and accompaniment files independently before calling this
module, because that would destroy their relative loudness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np

from smartedit.schemas import (
    NarrationAnalysis,
    SpeechMusicMaskingAnalysis,
    TimestampedObservation,
    TimestampedRange,
)

DEFAULT_WINDOW_SECONDS = 0.5
DEFAULT_HOP_SECONDS = 0.25
DEFAULT_SEVERE_MARGIN_DB = -3.0
DEFAULT_VOCAL_ACTIVITY_FLOOR_DBFS = -65.0
DEFAULT_MINIMUM_ANALYZED_SPEECH_SECONDS = 2.0
_DB_EPSILON = 1e-12


class SpeechMusicMaskingError(RuntimeError):
    """Raised when aligned stem audio cannot be measured reliably."""


def measure_speech_music_masking(
    vocals_path: str | Path,
    accompaniment_path: str | Path,
    narration: NarrationAnalysis | Mapping[str, Any] | Sequence[Any] | None,
    *,
    duration_seconds: float,
    model_name: str = "demucs",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    minimum_speech_overlap_ratio: float = 0.5,
    severe_margin_db: float = DEFAULT_SEVERE_MARGIN_DB,
    vocal_activity_floor_dbfs: float = DEFAULT_VOCAL_ACTIVITY_FLOOR_DBFS,
    minimum_analyzed_speech_seconds: float = DEFAULT_MINIMUM_ANALYZED_SPEECH_SECONDS,
    max_evidence_items: int = 5,
) -> SpeechMusicMaskingAnalysis:
    """Measure vocal-versus-accompaniment RMS balance during detected speech.

    Windows are aligned to the beginning of the audio, are 500 ms long by
    default, and advance by 250 ms. A window is included when at least half of
    it overlaps a validated Whisper speech interval. RMS is measured over both
    stereo channels together.

    A positive dB margin means the separated vocal is louder. A negative margin
    means the separated accompaniment is louder. This is an objective proxy,
    not proof that speech is or is not intelligible.
    """

    duration = _finite_positive(duration_seconds, "duration_seconds")
    window = _finite_positive(window_seconds, "window_seconds")
    hop = _finite_positive(hop_seconds, "hop_seconds")
    if hop > window:
        raise ValueError("hop_seconds cannot exceed window_seconds")
    overlap_ratio = _probability(minimum_speech_overlap_ratio, "minimum_speech_overlap_ratio")
    if overlap_ratio == 0.0:
        raise ValueError("minimum_speech_overlap_ratio must be greater than zero")
    if not math.isfinite(severe_margin_db) or severe_margin_db > 0.0:
        raise ValueError("severe_margin_db must be finite and no greater than zero")
    if not math.isfinite(vocal_activity_floor_dbfs):
        raise ValueError("vocal_activity_floor_dbfs must be finite")
    minimum_speech = _finite_nonnegative(
        minimum_analyzed_speech_seconds,
        "minimum_analyzed_speech_seconds",
    )
    if type(max_evidence_items) is not int or max_evidence_items < 0:
        raise ValueError("max_evidence_items must be a non-negative integer")

    vocals, vocal_sample_rate = _read_stereo_wav(vocals_path)
    accompaniment, accompaniment_sample_rate = _read_stereo_wav(accompaniment_path)
    if vocal_sample_rate != accompaniment_sample_rate:
        raise SpeechMusicMaskingError(
            "Vocal and accompaniment stems must have the same sample rate"
        )
    if vocals.shape != accompaniment.shape:
        raise SpeechMusicMaskingError(
            "Vocal and accompaniment stems must have identical frame and channel shapes"
        )
    if vocals.shape[1] != 2:
        raise SpeechMusicMaskingError(
            f"Expected stereo stems with two channels, found {vocals.shape[1]}"
        )

    sample_rate = vocal_sample_rate
    available_duration = vocals.shape[0] / sample_rate
    measured_duration = min(duration, available_duration)
    speech_intervals = _speech_intervals(narration, measured_duration)
    limitations = _limitations()
    common_raw: dict[str, Any] = {
        "vocals_path": str(Path(vocals_path).expanduser()),
        "accompaniment_path": str(Path(accompaniment_path).expanduser()),
        "available_audio_duration_seconds": round(available_duration, 6),
        "minimum_speech_overlap_ratio": overlap_ratio,
        "severe_margin_db": severe_margin_db,
        "vocal_activity_floor_dbfs": vocal_activity_floor_dbfs,
        "requires_shared_stem_amplitude_scale": True,
    }
    if not speech_intervals:
        return SpeechMusicMaskingAnalysis(
            status="no_speech_intervals",
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            model_name=model_name,
            window_seconds=window,
            hop_seconds=hop,
            limitations=limitations,
            raw_output={**common_raw, "validated_speech_intervals": []},
        )

    window_frames = max(1, round(window * sample_rate))
    hop_frames = max(1, round(hop * sample_rate))
    minimum_overlap_frames = max(1, math.ceil(window_frames * overlap_ratio))
    interval_frames = [
        (
            max(0, min(vocals.shape[0], math.floor(start * sample_rate))),
            max(0, min(vocals.shape[0], math.ceil(end * sample_rate))),
        )
        for start, end in speech_intervals
    ]

    candidates: list[dict[str, float | int]] = []
    final_start = vocals.shape[0] - window_frames
    if final_start >= 0:
        for start_frame in range(0, final_start + 1, hop_frames):
            end_frame = start_frame + window_frames
            speech_overlap = _overlap_frames(start_frame, end_frame, interval_frames)
            if speech_overlap < minimum_overlap_frames:
                continue
            vocal_rms = _rms(vocals[start_frame:end_frame])
            accompaniment_rms = _rms(accompaniment[start_frame:end_frame])
            vocal_dbfs = _to_dbfs(vocal_rms)
            candidates.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "vocal_rms": vocal_rms,
                    "accompaniment_rms": accompaniment_rms,
                    "vocal_dbfs": vocal_dbfs,
                    "accompaniment_dbfs": _to_dbfs(accompaniment_rms),
                    "margin_db": _amplitude_margin_db(vocal_rms, accompaniment_rms),
                }
            )

    valid_windows = [
        item for item in candidates if float(item["vocal_dbfs"]) >= vocal_activity_floor_dbfs
    ]
    discarded_count = len(candidates) - len(valid_windows)
    if not valid_windows:
        return SpeechMusicMaskingAnalysis(
            status="no_valid_windows",
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            model_name=model_name,
            window_seconds=window,
            hop_seconds=hop,
            discarded_quiet_window_count=discarded_count,
            limitations=limitations,
            raw_output={
                **common_raw,
                "validated_speech_intervals": _interval_dicts(speech_intervals),
                "candidate_speech_window_count": len(candidates),
                "reason": "No speech windows exceeded the vocal activity floor.",
            },
        )

    margins = np.asarray([float(item["margin_db"]) for item in valid_windows])
    vocal_rms_values = np.asarray([float(item["vocal_rms"]) for item in valid_windows])
    accompaniment_rms_values = np.asarray(
        [float(item["accompaniment_rms"]) for item in valid_windows]
    )
    vocal_rms_overall = float(np.sqrt(np.mean(np.square(vocal_rms_values))))
    accompaniment_rms_overall = float(np.sqrt(np.mean(np.square(accompaniment_rms_values))))
    median_margin = float(np.median(margins))
    p10_margin = float(np.quantile(margins, 0.10))
    accompaniment_dominant_ratio = float(np.mean(margins <= 0.0))
    severe_ratio = float(np.mean(margins <= severe_margin_db))

    valid_window_intervals = [
        (
            int(item["start_frame"]) / sample_rate,
            int(item["end_frame"]) / sample_rate,
        )
        for item in valid_windows
    ]
    measured_speech_intervals = _intersections(
        speech_intervals,
        _merge_intervals(valid_window_intervals),
    )
    analyzed_speech_seconds = sum(end - start for start, end in measured_speech_intervals)
    status = "ok" if analyzed_speech_seconds + 1e-9 >= minimum_speech else "insufficient_speech"
    confidence = _measurement_confidence(
        analyzed_speech_seconds,
        median_margin,
        accompaniment_dominant_ratio,
    )
    evidence = _severe_evidence(
        valid_windows,
        sample_rate,
        duration,
        severe_margin_db,
        max_evidence_items,
    )

    window_measurements = [
        {
            "start": _round_time(int(item["start_frame"]) / sample_rate, duration),
            "end": _round_time(int(item["end_frame"]) / sample_rate, duration),
            "vocal_rms_dbfs": round(float(item["vocal_dbfs"]), 4),
            "accompaniment_rms_dbfs": round(float(item["accompaniment_dbfs"]), 4),
            "voice_to_accompaniment_db": round(float(item["margin_db"]), 4),
        }
        for item in valid_windows
    ]
    result = SpeechMusicMaskingAnalysis(
        status=status,
        duration_seconds=duration,
        sample_rate_hz=sample_rate,
        model_name=model_name,
        window_seconds=window,
        hop_seconds=hop,
        speech_window_count=len(valid_windows),
        discarded_quiet_window_count=discarded_count,
        analyzed_speech_seconds=round(analyzed_speech_seconds, 3),
        vocal_rms_dbfs=round(_to_dbfs(vocal_rms_overall), 4),
        accompaniment_rms_dbfs=round(_to_dbfs(accompaniment_rms_overall), 4),
        voice_to_accompaniment_db_overall=round(
            _amplitude_margin_db(vocal_rms_overall, accompaniment_rms_overall), 4
        ),
        voice_to_accompaniment_db_median=round(median_margin, 4),
        voice_to_accompaniment_db_p10=round(p10_margin, 4),
        accompaniment_dominant_speech_ratio=round(accompaniment_dominant_ratio, 6),
        severe_accompaniment_dominant_speech_ratio=round(severe_ratio, 6),
        confidence=confidence,
        speech_timestamps_used=[
            TimestampedRange(
                start=_round_time(start, duration),
                end=_round_time(end, duration),
            )
            for start, end in measured_speech_intervals
        ],
        evidence=evidence,
        limitations=limitations,
        raw_output={
            **common_raw,
            "validated_speech_intervals": _interval_dicts(speech_intervals),
            "candidate_speech_window_count": len(candidates),
            "window_measurements": window_measurements,
        },
    )
    result.validate_timestamps(duration)
    return result


def _read_stereo_wav(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise SpeechMusicMaskingError(f"Separated stem WAV does not exist: {source}")
    if source.suffix.lower() != ".wav":
        raise SpeechMusicMaskingError(f"Separated stem must be a WAV file: {source}")
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        raise SpeechMusicMaskingError(f"Could not read separated stem {source}: {exc}") from exc
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise SpeechMusicMaskingError(f"Separated stem contains no audio frames: {source}")
    if not np.isfinite(values).all():
        raise SpeechMusicMaskingError(f"Separated stem contains non-finite samples: {source}")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise SpeechMusicMaskingError(f"Separated stem has an invalid sample rate: {source}")
    return values, sample_rate


def _speech_intervals(
    narration: NarrationAnalysis | Mapping[str, Any] | Sequence[Any] | None,
    duration: float,
) -> list[tuple[float, float]]:
    values: Any
    if isinstance(narration, NarrationAnalysis):
        values = narration.words if narration.words else narration.segments
    elif isinstance(narration, Mapping):
        words = narration.get("words")
        values = words if _is_sequence(words) and words else narration.get("segments", [])
    elif _is_sequence(narration):
        values = narration
    else:
        values = []

    intervals: list[tuple[float, float]] = []
    for item in values:
        interval = _coerce_interval(item)
        if interval is None:
            continue
        start, end = interval
        start = min(duration, max(0.0, start))
        end = min(duration, max(0.0, end))
        if end > start:
            intervals.append((start, end))
    return _merge_intervals(intervals, gap_tolerance=0.1)


def _coerce_interval(value: Any) -> tuple[float, float] | None:
    if isinstance(value, TimestampedRange):
        start_value, end_value = value.start, value.end
    elif isinstance(value, Mapping):
        timestamp = value.get("timestamp")
        if _is_sequence(timestamp) and len(timestamp) >= 2:
            start_value, end_value = timestamp[0], timestamp[1]
        else:
            start_value = value.get("start", value.get("start_seconds"))
            end_value = value.get("end", value.get("end_seconds"))
    elif _is_sequence(value) and len(value) >= 2:
        start_value, end_value = value[0], value[1]
    else:
        return None
    try:
        start, end = float(start_value), float(end_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return start, end


def _overlap_frames(
    start: int,
    end: int,
    intervals: Sequence[tuple[int, int]],
) -> int:
    return sum(max(0, min(end, right) - max(start, left)) for left, right in intervals)


def _rms(values: np.ndarray) -> float:
    # float64 accumulation avoids overflow and keeps deterministic precision.
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _to_dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(_DB_EPSILON, rms))


def _amplitude_margin_db(vocal_rms: float, accompaniment_rms: float) -> float:
    return 20.0 * math.log10(max(_DB_EPSILON, vocal_rms) / max(_DB_EPSILON, accompaniment_rms))


def _severe_evidence(
    windows: Sequence[Mapping[str, float | int]],
    sample_rate: int,
    duration: float,
    severe_margin_db: float,
    maximum_items: int,
) -> list[TimestampedObservation]:
    severe = [item for item in windows if float(item["margin_db"]) <= severe_margin_db]
    if not severe or maximum_items == 0:
        return []

    groups: list[list[Mapping[str, float | int]]] = []
    for item in severe:
        if groups and int(item["start_frame"]) <= int(groups[-1][-1]["end_frame"]):
            groups[-1].append(item)
        else:
            groups.append([item])

    ranked = sorted(groups, key=lambda group: min(float(item["margin_db"]) for item in group))
    selected = ranked[:maximum_items]
    observations: list[TimestampedObservation] = []
    for group in sorted(selected, key=lambda value: int(value[0]["start_frame"])):
        start = int(group[0]["start_frame"]) / sample_rate
        end = max(int(item["end_frame"]) for item in group) / sample_rate
        worst_margin = min(float(item["margin_db"]) for item in group)
        observations.append(
            TimestampedObservation(
                start=_round_time(start, duration),
                end=_round_time(end, duration),
                observation=(
                    "Separated accompaniment exceeded the vocal stem by up to "
                    f"{-worst_margin:.1f} dB across {len(group)} speech RMS window"
                    f"{'s' if len(group) != 1 else ''}."
                ),
            )
        )
    return observations


def _measurement_confidence(
    analyzed_speech_seconds: float,
    median_margin_db: float,
    accompaniment_dominant_ratio: float,
) -> float:
    if analyzed_speech_seconds < 2.0:
        confidence = 0.35
    elif analyzed_speech_seconds < 4.0:
        confidence = 0.55
    elif analyzed_speech_seconds < 8.0:
        confidence = 0.62
    else:
        confidence = 0.68
    if (
        analyzed_speech_seconds >= 2.0
        and median_margin_db <= -6.0
        and accompaniment_dominant_ratio >= 0.75
    ):
        confidence += 0.05
    return round(min(0.75, confidence), 6)


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    gap_tolerance: float = 0.0,
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap_tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _intersections(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            result.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return _merge_intervals(result)


def _interval_dicts(intervals: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"start": round(start, 6), "end": round(end, 6)} for start, end in intervals]


def _limitations() -> list[str]:
    return [
        "Separated stems are model estimates, not ground-truth source recordings.",
        "Full-band RMS balance is a proxy and does not directly measure speech "
        "intelligibility or frequency-specific masking.",
        "The stems must share one amplitude scale and must not be normalized independently.",
    ]


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _round_time(value: float, duration: float) -> float:
    return min(duration, max(0.0, round(float(value), 3)))


def _is_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
