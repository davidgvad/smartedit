"""Audio-analysis orchestration with graceful, explicit partial failure."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from smartedit.models.audio_flamingo_adapter import AudioModelAdapter
from smartedit.models.librosa_fallback import (
    LibrosaFallbackAudioAdapter,
    build_librosa_fallback_analysis,
    compute_librosa_features,
)
from smartedit.schemas import AudioAnalysis, AudioModelJudgment, LibrosaFeatures

LOGGER = logging.getLogger(__name__)


def extract_audio_features(
    audio_path: str | Path | None,
    duration_seconds: float,
    *,
    audio_adapter: AudioModelAdapter | LibrosaFallbackAudioAdapter | None = None,
    use_librosa_fallback: bool = True,
) -> AudioAnalysis:
    """Combine objective audio features with a contextual model judgment.

    Librosa measurements are attempted independently of the audio-language model.
    If the model fails and fallback is enabled, the returned result has
    ``used_librosa_fallback=True`` and no ``AudioModelJudgment``. Its limited
    proxies live only in ``raw_output.fallback_estimates``.
    """

    duration = _finite_nonnegative(duration_seconds)
    if audio_path is None:
        return AudioAnalysis(
            objective=None,
            judgment=None,
            adapter_used="none",
            used_librosa_fallback=False,
            warnings=[],
            raw_output={"status": "skipped", "reason": "no audio stream"},
        )

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Extracted audio file does not exist: {path}")

    warnings: list[str] = []
    objective: LibrosaFeatures | None = None
    try:
        objective = compute_librosa_features(path)
    except Exception as exc:
        warning = f"Objective librosa measurements failed: {exc}"
        LOGGER.warning(warning)
        warnings.append(warning)

    if isinstance(audio_adapter, LibrosaFallbackAudioAdapter) or audio_adapter is None:
        if use_librosa_fallback:
            reason = (
                "No contextual audio model was configured."
                if audio_adapter is None
                else "The librosa fallback was selected explicitly."
            )
            return build_librosa_fallback_analysis(
                objective,
                reason=reason,
                additional_warnings=warnings,
            )
        return AudioAnalysis(
            objective=objective,
            judgment=None,
            adapter_used="none",
            used_librosa_fallback=False,
            warnings=warnings + ["No contextual audio model was configured."],
            raw_output={"status": "objective_only"},
        )

    adapter_name = str(getattr(audio_adapter, "name", audio_adapter.__class__.__name__))
    try:
        judgment_value = audio_adapter.analyze(path, duration_seconds=duration)
        judgment = _coerce_judgment(judgment_value)
    except Exception as exc:
        warning = f"{adapter_name} analysis failed: {exc}"
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception(warning)
        else:
            LOGGER.warning(warning)
        warnings.append(warning)
        if use_librosa_fallback:
            return build_librosa_fallback_analysis(
                objective,
                reason=(
                    f"The contextual audio backend {adapter_name!r} was unavailable; "
                    "fallback activated."
                ),
                additional_warnings=warnings,
                failed_contextual_model_output=_adapter_raw_output(audio_adapter),
            )
        return AudioAnalysis(
            objective=objective,
            judgment=None,
            adapter_used=adapter_name,
            used_librosa_fallback=False,
            warnings=warnings,
            raw_output=_adapter_raw_output(audio_adapter),
        )

    if judgment is None:
        warning = f"{adapter_name} returned no contextual audio judgment."
        warnings.append(warning)
        if use_librosa_fallback:
            return build_librosa_fallback_analysis(
                objective,
                reason=warning,
                additional_warnings=warnings,
            )
    return AudioAnalysis(
        objective=objective,
        judgment=judgment,
        adapter_used=adapter_name,
        used_librosa_fallback=False,
        warnings=warnings,
        raw_output=_adapter_raw_output(audio_adapter),
    )


def _coerce_judgment(value: Any) -> AudioModelJudgment | None:
    if value is None:
        return None
    if isinstance(value, AudioModelJudgment):
        return value
    raise TypeError("Audio adapter must return AudioModelJudgment or None")


def _adapter_raw_output(adapter: Any) -> dict[str, Any]:
    value = getattr(adapter, "last_raw_output", {})
    if isinstance(value, dict):
        return value
    return {"raw_output": str(value)}


def _finite_nonnegative(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("duration_seconds must be a finite non-negative number")
    return result
