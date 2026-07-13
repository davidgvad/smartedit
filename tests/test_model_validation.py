from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from smartedit.extraction.narration_features import extract_narration_features
from smartedit.models.audio_flamingo_adapter import (
    AudioModelError,
    validate_audio_model_output,
)
from smartedit.models.whisper_adapter import WhisperAdapter


def _audio_payload(start: float, end: float) -> dict[str, Any]:
    return {
        "background_music_present": True,
        "music_energy": "medium",
        "rhythmic_strength": 0.6,
        "catchiness_confidence": 0.4,
        "speech_music_interference": "low",
        "audio_quality": "good",
        "background_music_score": 1,
        "catchy_music_score": 0,
        "evidence": [
            {
                "start": start,
                "end": end,
                "observation": "Music remains below the speech.",
            }
        ],
    }


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-0.0011, 0.5),
        (0.0, 10.0011),
        (10.0011, 10.0011),
    ],
)
def test_audio_evidence_rejects_grossly_out_of_range_timestamps(start: float, end: float) -> None:
    with pytest.raises(AudioModelError, match="within the audio duration"):
        validate_audio_model_output(_audio_payload(start, end), 10.0)


def test_audio_evidence_clamps_only_one_millisecond_rounding_error() -> None:
    result = validate_audio_model_output(_audio_payload(-0.001, 10.001), 10.0)

    assert result["evidence"][0]["start"] == 0.0
    assert result["evidence"][0]["end"] == 10.0


def test_audio_evidence_rounding_never_exceeds_exact_duration() -> None:
    duration = 1.0006
    result = validate_audio_model_output(_audio_payload(0.0, duration), duration)

    assert result["evidence"][0]["end"] == duration


class _ReturnLanguageMutatingPipeline:
    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    def __call__(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.received.append(input_data)
        input_data.pop("array")
        input_data.pop("sampling_rate")
        if kwargs.get("return_language"):
            raise TypeError("return_language is not supported")
        return {"text": "ok", "chunks": []}


def test_whisper_return_language_retry_uses_fresh_input_container() -> None:
    adapter = WhisperAdapter()
    pipeline = _ReturnLanguageMutatingPipeline()
    adapter._pipeline = pipeline
    waveform = object()
    pipeline_input = {"array": waveform, "sampling_rate": 16_000}

    result = adapter._run_pipeline(pipeline_input, return_timestamps="word")

    assert result["text"] == "ok"
    assert len(pipeline.received) == 2
    assert pipeline.received[0] is not pipeline.received[1]
    assert pipeline_input == {"array": waveform, "sampling_rate": 16_000}


class _WordTimestampMutatingPipeline:
    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self.waveforms: list[Any] = []

    def __call__(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.received.append(input_data)
        self.waveforms.append(input_data.pop("array"))
        input_data.pop("sampling_rate")
        if kwargs["return_timestamps"] == "word":
            raise RuntimeError("word timestamps unavailable")
        return {
            "text": "segment fallback",
            "chunks": [{"text": "segment fallback", "timestamp": (0.0, 1.0)}],
        }


def test_whisper_word_to_segment_retry_uses_fresh_input_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Waveform:
        size = 1

    waveform = _Waveform()
    monkeypatch.setattr(
        "smartedit.models.whisper_adapter._load_audio_mono_16khz",
        lambda _path: waveform,
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = WhisperAdapter()
    pipeline = _WordTimestampMutatingPipeline()
    adapter._pipeline = pipeline

    result = adapter._analyze(audio_path, duration_seconds=2.0)

    assert result.transcript == "segment fallback"
    assert len(pipeline.received) == 2
    assert pipeline.received[0] is not pipeline.received[1]
    assert pipeline.waveforms == [waveform, waveform]


def test_narration_rounding_never_exceeds_exact_duration() -> None:
    duration = 1.0006
    analysis = extract_narration_features(
        {
            "transcript": "hello",
            "segments": [
                {
                    "start": 0.0,
                    "end": duration,
                    "text": "hello",
                    "words": [{"start": 0.0, "end": duration, "text": "hello"}],
                }
            ],
        },
        duration,
        model_name="test-whisper",
    )
    silent = extract_narration_features(
        {},
        duration,
        model_name="test-whisper",
        long_silence_threshold_seconds=0.0,
    )

    assert analysis.segments[0].end == duration
    assert analysis.words[0].end == duration
    assert analysis.speech_duration <= duration
    assert silent.long_silent_gaps[0].end == duration
    analysis.validate_timestamps(duration)
    silent.validate_timestamps(duration)
