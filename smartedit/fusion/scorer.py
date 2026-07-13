"""Top-level Edit Signal fusion orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .rubrics import (
    RubricResult,
    score_background_music,
    score_catchy_music,
    score_length,
    score_narration,
    score_pace,
    score_semantic_signal,
    score_transitions,
)

SEMANTIC_SIGNALS = (
    "visual_variety",
    "text",
    "text_visibility",
    "effects",
    "story",
    "clear_start_middle_end",
    "consistent_theme",
)


def fuse_edit_signals(
    *,
    duration_seconds: float,
    has_audio: bool,
    transition: Mapping[str, Any] | None = None,
    narration: Mapping[str, Any] | None = None,
    audio: Mapping[str, Any] | None = None,
    qwen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse all stage outputs using explicit rubrics.

    The returned ``conflicts`` list is intended to be copied to report warnings,
    while every model's unmodified output remains in ``raw_model_outputs``.
    """

    category = extract_category(qwen)
    narration_for_pace = narration if has_audio else None
    results: dict[str, RubricResult] = {
        "length": score_length(duration_seconds, category),
        "pace": score_pace(duration_seconds, transition, narration_for_pace, qwen),
        "narration": score_narration(duration_seconds, narration, audio, qwen, category, has_audio),
        "background_music": score_background_music(duration_seconds, audio, has_audio),
        "catchy_music": score_catchy_music(duration_seconds, audio, has_audio),
        "transitions": score_transitions(duration_seconds, transition, qwen),
    }
    for signal in SEMANTIC_SIGNALS:
        results[signal] = score_semantic_signal(signal, qwen, duration_seconds)

    conflicts = [message for result in results.values() for message in result.conflicts]
    return {
        "signals": {name: result.as_dict() for name, result in results.items()},
        "category": category,
        "conflicts": conflicts,
    }


def extract_category(qwen: Mapping[str, Any] | None) -> dict[str, float]:
    qwen = qwen or {}
    raw = qwen.get("category", {}) if isinstance(qwen, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        name: _bounded(raw.get(name, 0.0)) for name in ("personal", "informational", "promotional")
    }


def _bounded(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError, OverflowError):
        return 0.0
