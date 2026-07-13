"""Small, deterministic confidence helpers used by the scoring rubrics."""

from __future__ import annotations


def clamp_confidence(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 3)


def penalize_conflict(confidence: float, conflicts: int = 1) -> float:
    """Apply a visible 20-point penalty per independent conflict."""

    return clamp_confidence(confidence - 0.2 * max(0, int(conflicts)))


def cap_fallback(confidence: float, cap: float = 0.45) -> float:
    """Prevent a handcrafted fallback from looking model-equivalent."""

    return clamp_confidence(min(confidence, cap))
