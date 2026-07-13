"""Objective librosa measurements and an explicitly limited fallback adapter."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from smartedit.models.base import AdapterError, BaseModelAdapter
from smartedit.schemas import AudioAnalysis, LibrosaFeatures

LOGGER = logging.getLogger(__name__)


class LibrosaAnalysisError(AdapterError):
    """Raised when objective audio decoding or measurement fails."""


def compute_librosa_features(
    input_data: str | Path | None,
    *,
    target_sample_rate: int = 22_050,
) -> LibrosaFeatures | None:
    """Compute reproducible low-level audio measurements.

    These features do not identify semantic audio classes and must not be
    described as equivalent to Audio Flamingo judgments.
    """

    if input_data is None:
        return None
    audio_path = Path(input_data)
    if not audio_path.is_file():
        raise LibrosaAnalysisError(f"Audio file does not exist: {audio_path}")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise LibrosaAnalysisError(
            "Objective audio measurements require librosa and NumPy."
        ) from exc

    LOGGER.info("Computing objective librosa features for %s", audio_path.name)
    try:
        waveform, sample_rate = librosa.load(str(audio_path), sr=target_sample_rate, mono=True)
    except Exception as exc:
        raise LibrosaAnalysisError(f"Could not decode audio file: {audio_path}") from exc

    y = np.asarray(waveform, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    duration = float(y.size / sample_rate) if sample_rate > 0 else 0.0
    if y.size == 0:
        return _empty_features(duration, sample_rate)

    hop_length = 512
    frame_length = 2048
    try:
        rms = np.asarray(
            librosa.feature.rms(
                y=y,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )
        ).reshape(-1)
        onset_envelope = np.asarray(
            librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
        ).reshape(-1)
        centroid = np.asarray(
            librosa.feature.spectral_centroid(y=y, sr=sample_rate, hop_length=hop_length)
        ).reshape(-1)
        zcr = np.asarray(
            librosa.feature.zero_crossing_rate(
                y=y, frame_length=frame_length, hop_length=hop_length
            )
        ).reshape(-1)
        tempo = _estimate_tempo(librosa, onset_envelope, sample_rate, hop_length)
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_rms = _root_mean_square(np, harmonic)
        percussive_rms = _root_mean_square(np, percussive)
        if harmonic_rms <= 1e-10 and percussive_rms <= 1e-10:
            harmonic_percussive_ratio = 0.0
        else:
            harmonic_percussive_ratio = min(1_000_000.0, harmonic_rms / max(percussive_rms, 1e-10))
    except Exception as exc:
        raise LibrosaAnalysisError(
            f"librosa feature extraction failed for {audio_path.name}"
        ) from exc

    return LibrosaFeatures(
        rms_mean=_finite_stat(np, rms, "mean"),
        rms_std=_finite_stat(np, rms, "std"),
        estimated_tempo_bpm=max(0.0, _finite_float(tempo)),
        onset_strength_mean=_finite_stat(np, onset_envelope, "mean"),
        spectral_centroid_hz=max(0.0, _finite_stat(np, centroid, "mean")),
        zero_crossing_rate=max(0.0, _finite_stat(np, zcr, "mean")),
        harmonic_percussive_ratio=max(0.0, _finite_float(harmonic_percussive_ratio)),
        duration_seconds=round(max(0.0, duration), 3),
        sample_rate_hz=int(sample_rate),
    )


class LibrosaFallbackAudioAdapter(BaseModelAdapter[str | Path | None, AudioAnalysis]):
    """Fallback that exposes measurements without claiming semantic equivalence.

    The returned ``AudioAnalysis.judgment`` is always ``None``. Low-confidence
    proxy values are retained under ``raw_output.fallback_estimates`` solely so
    fusion code can make transparent neutral decisions. In particular, librosa
    cannot reliably distinguish background music from foreground music, speech,
    or environmental sound, nor judge compatibility or narration masking.
    """

    name = "librosa_fallback"
    adapter_name = "librosa_fallback"

    def __init__(
        self,
        model_name: str = "librosa_objective_features",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **_: Any,
    ) -> None:
        super().__init__(model_name, device=device, cache_dir=cache_dir)
        self.last_raw_output: dict[str, Any] = {}

    def _load(self) -> None:
        try:
            import librosa  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise LibrosaAnalysisError("The librosa fallback requires librosa and NumPy.") from exc

    def _analyze(self, input_data: str | Path | None, **kwargs: Any) -> AudioAnalysis:
        del kwargs  # Duration is measured from the extracted audio itself.
        objective = compute_librosa_features(input_data)
        result = build_librosa_fallback_analysis(objective)
        self.last_raw_output = dict(result.raw_output or {})
        return result


def build_librosa_fallback_analysis(
    objective: LibrosaFeatures | None,
    *,
    reason: str | None = None,
    additional_warnings: list[str] | None = None,
    failed_contextual_model_output: dict[str, Any] | None = None,
) -> AudioAnalysis:
    """Package existing measurements as a visibly limited fallback result."""

    limitations = [
        "background/foreground music separation not assessed",
        "catchiness not assessed",
        "music-video compatibility not assessed",
        "speech/music interference not assessed",
        "environmental sound semantics not assessed",
    ]
    estimates = _fallback_estimates(objective, limitations=limitations)
    warning = (
        "Audio Flamingo was not used. Librosa fallback values are objective "
        "signal measurements and limited proxies, not semantic audio judgments."
    )
    warnings = list(additional_warnings or [])
    if reason:
        warnings.append(reason)
    warnings.append(warning)
    warnings = list(dict.fromkeys(warnings))
    raw_output: dict[str, Any] = {
        "backend": "librosa_fallback",
        "fallback_estimates": estimates,
        "limitations": limitations,
    }
    if failed_contextual_model_output:
        raw_output["failed_contextual_model_output"] = failed_contextual_model_output
    return AudioAnalysis(
        objective=objective,
        judgment=None,
        adapter_used="librosa_fallback",
        used_librosa_fallback=True,
        warnings=warnings,
        raw_output=raw_output,
    )


def _fallback_estimates(
    features: LibrosaFeatures | None, *, limitations: list[str]
) -> dict[str, Any]:
    if features is None:
        return {
            "status": "objective_features_unavailable",
            "background_music_present": None,
            "music_likelihood": None,
            "music_energy": None,
            "rhythmic_strength": None,
            "confidence": 0.0,
            "limitations": limitations,
        }

    rms = float(features.rms_mean or 0.0)
    onset = float(features.onset_strength_mean or 0.0)
    tempo = float(features.estimated_tempo_bpm or 0.0)
    hp_ratio = float(features.harmonic_percussive_ratio or 0.0)
    if rms < 0.002:
        energy = "low"
    elif rms < 0.06:
        energy = "medium"
    else:
        energy = "high"

    # Onset strength is scale-dependent. This bounded transform is a proxy, not
    # a calibrated rhythmicity probability.
    rhythmic_proxy = 0.0 if onset <= 0.0 else onset / (onset + 2.0)
    harmonic_fraction = hp_ratio / (1.0 + hp_ratio) if hp_ratio >= 0.0 else 0.0
    active_audio = min(1.0, rms / 0.04)
    tempo_indicator = 1.0 if 40.0 <= tempo <= 240.0 else 0.0
    music_proxy = active_audio * (
        0.15 + 0.2 * rhythmic_proxy + 0.1 * harmonic_fraction + 0.1 * tempo_indicator
    )
    # Keep the proxy deliberately below a confident decision threshold.
    music_proxy = min(0.55, max(0.0, music_proxy))
    return {
        "status": "limited_objective_proxy",
        # Background-vs-foreground music cannot be established by these features.
        "background_music_present": None,
        "music_likelihood": round(music_proxy, 4),
        "music_energy": energy,
        "rhythmic_strength": round(rhythmic_proxy, 4),
        "confidence": round(min(0.4, 0.15 + 0.25 * active_audio), 4),
        "likely_silent": rms < 0.001,
        "limitations": limitations,
    }


def _estimate_tempo(librosa: Any, onset_envelope: Any, sample_rate: int, hop_length: int) -> float:
    if getattr(onset_envelope, "size", 0) < 2 or not bool(onset_envelope.any()):
        return 0.0
    kwargs = {
        "onset_envelope": onset_envelope,
        "sr": sample_rate,
        "hop_length": hop_length,
    }
    try:
        # librosa >= 0.10
        value = librosa.feature.tempo(**kwargs)
    except (AttributeError, TypeError):
        # librosa 0.9 compatibility
        value = librosa.beat.tempo(**kwargs)
    try:
        return float(value.reshape(-1)[0])
    except AttributeError:
        try:
            return float(value[0])
        except (IndexError, TypeError):
            return float(value)


def _empty_features(duration: float, sample_rate: int) -> LibrosaFeatures:
    return LibrosaFeatures(
        rms_mean=0.0,
        rms_std=0.0,
        estimated_tempo_bpm=0.0,
        onset_strength_mean=0.0,
        spectral_centroid_hz=0.0,
        zero_crossing_rate=0.0,
        harmonic_percussive_ratio=0.0,
        duration_seconds=round(max(0.0, duration), 3),
        sample_rate_hz=int(sample_rate),
    )


def _root_mean_square(np: Any, values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0


def _finite_stat(np: Any, values: Any, operation: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    value = np.mean(array) if operation == "mean" else np.std(array)
    return _finite_float(value)


def _finite_float(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else 0.0
