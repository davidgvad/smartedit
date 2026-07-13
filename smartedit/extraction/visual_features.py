"""Objective-context assembly for the visual-semantic adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def build_visual_context(
    *,
    duration_seconds: float,
    frame_samples: Sequence[Any],
    transition: Mapping[str, Any] | None,
    narration: Mapping[str, Any] | None,
    audio: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a concise, JSON-serializable evidence packet for Qwen3-VL."""

    transition = transition or {}
    narration = narration or {}
    audio = audio or {}
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_seconds must be finite and non-negative")
    return {
        "duration_seconds": duration,
        "frame_timestamps_seconds": [
            _validated_timestamp(_timestamp(sample), duration) for sample in frame_samples
        ],
        "cut_timestamps_seconds": _timestamps(
            transition.get("cut_timestamps", transition.get("cut_timestamps_seconds", [])),
            duration,
        ),
        "shot_statistics": {
            key: transition.get(key)
            for key in (
                "shot_count",
                "cut_count",
                "average_shot_duration",
                "median_shot_duration",
                "minimum_shot_duration",
                "maximum_shot_duration",
                "cuts_per_minute",
                "shot_duration_variance",
            )
            if transition.get(key) is not None
        },
        "transcript": str(narration.get("transcript", "")),
        "transcript_segments": [
            {
                "start": item.get("start", item.get("start_seconds")),
                "end": item.get("end", item.get("end_seconds")),
                "text": item.get("text", ""),
            }
            for item in narration.get("segments", [])
            if isinstance(item, Mapping)
        ],
        "speech_coverage": narration.get("speech_coverage", narration.get("speech_coverage_ratio")),
        "words_per_minute": narration.get("words_per_minute"),
        "audio_summary": _audio_summary(audio),
    }


def _timestamp(sample: Any) -> float:
    if isinstance(sample, Mapping):
        value = sample.get("timestamp_seconds", sample.get("timestamp", 0.0))
    else:
        value = getattr(sample, "timestamp_seconds", getattr(sample, "timestamp", 0.0))
    return float(value)


def _timestamps(values: Any, duration: float) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [_validated_timestamp(float(value), duration) for value in values]


def _validated_timestamp(value: float, duration: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= duration:
        raise ValueError(f"timestamp {value!r} is outside video duration [0, {duration}]")
    return value


def _audio_summary(audio: Mapping[str, Any]) -> dict[str, Any]:
    objective = audio.get("objective", audio.get("objective_features", {}))
    judgment = audio.get("judgment", audio.get("model_judgment", {}))
    if not isinstance(objective, Mapping):
        objective = {}
    if not isinstance(judgment, Mapping):
        judgment = {}
    fallback_used = bool(audio.get("fallback_used", audio.get("used_librosa_fallback", False)))
    raw_output = audio.get("raw_output", {})
    fallback_estimates: Mapping[str, Any] = {}
    if fallback_used and isinstance(raw_output, Mapping):
        possible_estimates = raw_output.get("fallback_estimates", {})
        if isinstance(possible_estimates, Mapping):
            fallback_estimates = possible_estimates
    return {
        "adapter": audio.get("adapter", audio.get("adapter_used", audio.get("backend"))),
        "fallback_used": fallback_used,
        "objective": {
            "rms_energy": objective.get("rms_energy", objective.get("rms_mean")),
            "estimated_tempo_bpm": objective.get("estimated_tempo_bpm"),
            "onset_strength": objective.get("onset_strength", objective.get("onset_strength_mean")),
            "spectral_centroid_hz": objective.get("spectral_centroid_hz"),
            "zero_crossing_rate": objective.get("zero_crossing_rate"),
            "harmonic_percussive_ratio": objective.get("harmonic_percussive_ratio"),
        },
        "model_judgment": {
            key: judgment.get(key)
            for key in (
                "background_music_present",
                "music_energy",
                "rhythmic_strength",
                "catchiness_confidence",
                "speech_music_interference",
                "environmental_sound",
                "audio_quality",
                "background_music_score",
                "catchy_music_score",
                "explanation",
            )
            if judgment.get(key) is not None
        },
        "fallback_estimates_not_model_equivalent": {
            key: fallback_estimates.get(key)
            for key in (
                "music_likelihood",
                "music_energy",
                "rhythmic_strength",
                "likely_silent",
                "confidence",
            )
            if fallback_estimates.get(key) is not None
        },
    }
