"""Deterministic narration measurements derived from timestamped ASR output.

This module deliberately does not decide whether narration is *good* editing.
It only validates timestamps and measures speech coverage, speaking rate, and
silence. Contextual scoring belongs in the fusion layer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from smartedit.schemas import (
    NarrationAnalysis,
    SilentGap,
    TranscriptSegment,
    TranscriptWord,
)

_WORD_RE = re.compile(r"\b[\w'\N{RIGHT SINGLE QUOTATION MARK}-]+\b", re.UNICODE)


def extract_narration_features(
    raw_output: Mapping[str, Any] | Any,
    duration_seconds: float,
    *,
    model_name: str,
    long_silence_threshold_seconds: float = 2.0,
) -> NarrationAnalysis:
    """Validate ASR output and compute objective narration measurements.

    Invalid or non-finite timestamps are discarded. Slight endpoint overshoot
    from decoder rounding is capped to the media duration; negative timestamps
    are capped to zero. An interval whose end precedes its start is never
    repaired or invented and is therefore discarded.
    """

    duration = _finite_nonnegative(duration_seconds, "duration_seconds")
    threshold = _finite_nonnegative(
        long_silence_threshold_seconds, "long_silence_threshold_seconds"
    )
    data = _as_mapping(raw_output)

    segments = _validated_segments(data.get("segments", []), duration)
    words = _validated_words(data.get("words", []), duration)
    if not words:
        words = [word for segment in segments for word in segment.words]

    transcript = str(data.get("transcript", data.get("text", "")) or "").strip()
    if not transcript:
        transcript = " ".join(segment.text.strip() for segment in segments).strip()

    intervals = _merge_intervals(
        [(segment.start, segment.end) for segment in segments]
        or [(word.start, word.end) for word in words]
    )
    speech_duration = min(duration, sum(end - start for start, end in intervals))
    coverage = speech_duration / duration if duration > 0.0 else 0.0

    timestamped_word_count = len([word for word in words if word.text.strip()])
    transcript_word_count = len(_WORD_RE.findall(transcript))
    # Segment-level decoders may attach word timing to only part of a result.
    # Counting the larger complete source avoids under-reporting in that case.
    word_count = max(timestamped_word_count, transcript_word_count)
    words_per_minute = 60.0 * word_count / speech_duration if speech_duration > 0.0 else 0.0

    silent_gaps = [
        SilentGap(
            start=_round_time(start, duration),
            end=_round_time(end, duration),
        )
        for start, end in _complement(intervals, duration)
        if end - start + 1e-9 >= threshold
    ]
    detected_language = data.get("detected_language", data.get("language"))
    if detected_language is not None:
        detected_language = str(detected_language).strip() or None
    language_probability = data.get("language_probability")
    if language_probability is not None:
        try:
            language_probability = float(language_probability)
        except (TypeError, ValueError):
            language_probability = None
        if language_probability is not None and (
            not math.isfinite(language_probability) or not 0.0 <= language_probability <= 1.0
        ):
            language_probability = None

    return NarrationAnalysis(
        transcript=transcript,
        segments=segments,
        words=words,
        detected_language=detected_language,
        language_probability=language_probability,
        speech_duration=min(duration, round(speech_duration, 3)),
        speech_coverage=round(min(1.0, max(0.0, coverage)), 6),
        words_per_minute=round(max(0.0, words_per_minute), 3),
        long_silent_gaps=silent_gaps,
        narration_present=bool(transcript or intervals),
        model_name=model_name,
    )


def _validated_segments(values: Any, duration: float) -> list[TranscriptSegment]:
    if not _is_sequence(values):
        return []
    result: list[TranscriptSegment] = []
    for value in values:
        item = _as_mapping(value)
        interval = _validated_interval(item, duration)
        if interval is None:
            continue
        start, end = interval
        nested_words = _validated_words(item.get("words", []), duration)
        rounded_start = _round_time(start, duration)
        rounded_end = _round_time(end, duration)
        nested_words = [
            word
            for word in nested_words
            if word.start + 1e-6 >= rounded_start and word.end <= rounded_end + 1e-6
        ]
        result.append(
            TranscriptSegment(
                start=rounded_start,
                end=rounded_end,
                text=str(item.get("text", "") or "").strip(),
                words=nested_words,
            )
        )
    return sorted(result, key=lambda item: (item.start, item.end))


def _validated_words(values: Any, duration: float) -> list[TranscriptWord]:
    if not _is_sequence(values):
        return []
    result: list[TranscriptWord] = []
    for value in values:
        item = _as_mapping(value)
        interval = _validated_interval(item, duration)
        if interval is None:
            continue
        start, end = interval
        probability = item.get("probability")
        if probability is not None:
            try:
                probability = min(1.0, max(0.0, float(probability)))
            except (TypeError, ValueError):
                probability = None
        result.append(
            TranscriptWord(
                start=_round_time(start, duration),
                end=_round_time(end, duration),
                text=str(item.get("text", item.get("word", "")) or "").strip(),
                probability=probability,
            )
        )
    return sorted(result, key=lambda item: (item.start, item.end))


def _validated_interval(item: Mapping[str, Any], duration: float) -> tuple[float, float] | None:
    timestamp = item.get("timestamp")
    if _is_sequence(timestamp) and len(timestamp) >= 2:
        start_value, end_value = timestamp[0], timestamp[1]
    else:
        start_value = item.get("start", item.get("start_seconds"))
        end_value = item.get("end", item.get("end_seconds"))
    try:
        start, end = float(start_value), float(end_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        return None
    start = min(duration, max(0.0, start))
    end = min(duration, max(0.0, end))
    if end < start:
        return None
    return start, end


def _merge_intervals(
    intervals: Sequence[tuple[float, float]], *, gap_tolerance: float = 0.05
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


def _complement(
    intervals: Sequence[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration > cursor:
        gaps.append((cursor, duration))
    return gaps


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _is_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _round_time(value: float, duration: float) -> float:
    """Round to milliseconds, then cap to the exact media duration."""

    return min(duration, max(0.0, round(float(value), 3)))
