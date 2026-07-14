from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartedit.extraction import masking_features
from smartedit.schemas import (
    NarrationAnalysis,
    SpeechMusicMaskingAnalysis,
    TimestampedObservation,
    TranscriptSegment,
)


def _narration(start: float, end: float) -> NarrationAnalysis:
    return NarrationAnalysis(
        transcript="spoken words",
        segments=[TranscriptSegment(start=start, end=end, text="spoken words")],
        speech_duration=end - start,
        narration_present=True,
    )


def _mock_stems(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vocal_amplitude: float,
    accompaniment_amplitude: float,
    duration_seconds: float = 4.0,
    sample_rate: int = 1_000,
) -> None:
    shape = (round(duration_seconds * sample_rate), 2)
    stems = {
        "vocals.wav": np.full(shape, vocal_amplitude, dtype=np.float32),
        "accompaniment.wav": np.full(shape, accompaniment_amplitude, dtype=np.float32),
    }

    def read(path: str | Path) -> tuple[np.ndarray, int]:
        return stems[Path(path).name], sample_rate

    monkeypatch.setattr(masking_features, "_read_stereo_wav", read)


def test_accompaniment_louder_during_speech_produces_negative_margin_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_stems(
        monkeypatch,
        vocal_amplitude=0.1,
        accompaniment_amplitude=0.2,
    )

    result = masking_features.measure_speech_music_masking(
        "vocals.wav",
        "accompaniment.wav",
        _narration(0.0, 4.0),
        duration_seconds=4.0,
    )

    assert result.status == "ok"
    assert result.speech_window_count == 15
    assert result.analyzed_speech_seconds == pytest.approx(4.0)
    assert result.voice_to_accompaniment_db_median == pytest.approx(-6.0206)
    assert result.voice_to_accompaniment_db_p10 == pytest.approx(-6.0206)
    assert result.accompaniment_dominant_speech_ratio == 1.0
    assert result.severe_accompaniment_dominant_speech_ratio == 1.0
    assert len(result.evidence) == 1
    assert result.evidence[0].start == 0.0
    assert result.evidence[0].end == 4.0
    assert "6.0 dB" in result.evidence[0].observation


def test_louder_vocal_has_positive_margin_and_no_severe_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_stems(
        monkeypatch,
        vocal_amplitude=0.2,
        accompaniment_amplitude=0.1,
    )

    result = masking_features.measure_speech_music_masking(
        "vocals.wav",
        "accompaniment.wav",
        _narration(0.0, 4.0),
        duration_seconds=4.0,
    )

    assert result.voice_to_accompaniment_db_median == pytest.approx(6.0206)
    assert result.accompaniment_dominant_speech_ratio == 0.0
    assert result.severe_accompaniment_dominant_speech_ratio == 0.0
    assert result.evidence == []


def test_short_speech_is_measured_but_explicitly_marked_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_stems(
        monkeypatch,
        vocal_amplitude=0.1,
        accompaniment_amplitude=0.2,
    )

    result = masking_features.measure_speech_music_masking(
        "vocals.wav",
        "accompaniment.wav",
        _narration(0.0, 1.0),
        duration_seconds=4.0,
    )

    assert result.status == "insufficient_speech"
    assert result.analyzed_speech_seconds == pytest.approx(1.0)
    assert result.confidence == 0.35


def test_quiet_vocal_windows_are_not_treated_as_masking_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_stems(
        monkeypatch,
        vocal_amplitude=0.0001,
        accompaniment_amplitude=0.2,
    )

    result = masking_features.measure_speech_music_masking(
        "vocals.wav",
        "accompaniment.wav",
        _narration(0.0, 4.0),
        duration_seconds=4.0,
    )

    assert result.status == "no_valid_windows"
    assert result.speech_window_count == 0
    assert result.discarded_quiet_window_count == 15
    assert result.voice_to_accompaniment_db_median is None


def test_masking_schema_rejects_ratios_and_evidence_outside_media_limits() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        SpeechMusicMaskingAnalysis(
            status="no_valid_windows",
            duration_seconds=2.0,
            sample_rate_hz=44_100,
            accompaniment_dominant_speech_ratio=1.1,
        )

    with pytest.raises(ValueError, match="exceeds video duration"):
        SpeechMusicMaskingAnalysis(
            status="no_valid_windows",
            duration_seconds=2.0,
            sample_rate_hz=44_100,
            evidence=[
                TimestampedObservation(
                    start=1.5,
                    end=2.5,
                    observation="Out-of-range measurement.",
                )
            ],
        )


def test_stem_sample_rates_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.zeros((4_000, 2), dtype=np.float32)

    def read(path: str | Path) -> tuple[np.ndarray, int]:
        sample_rate = 1_000 if Path(path).name == "vocals.wav" else 2_000
        return audio, sample_rate

    monkeypatch.setattr(masking_features, "_read_stereo_wav", read)

    with pytest.raises(masking_features.SpeechMusicMaskingError, match="same sample rate"):
        masking_features.measure_speech_music_masking(
            "vocals.wav",
            "accompaniment.wav",
            _narration(0.0, 4.0),
            duration_seconds=4.0,
        )
