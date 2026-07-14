from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smartedit.extraction.audio_features import extract_audio_features
from smartedit.extraction.narration_features import extract_narration_features
from smartedit.models.audio_flamingo_adapter import (
    AudioFlamingoAdapter,
    AudioModelError,
    _dtype_for_device,
    _move_inputs,
    _parse_choice_answer,
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


def _scalar_audio_responses() -> list[str]:
    return [
        "YES",
        "HIGH",
        "HIGH",
        "HIGH",
        "LOW",
        "AUDIBLE",
        "GOOD",
        "SUPPORTIVE",
        "SUPPORTIVE",
    ]


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


def test_whisper_uses_single_item_batch_by_default() -> None:
    class _RecordingPipeline:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def __call__(self, _input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            self.kwargs = kwargs
            return {"text": "ok", "chunks": []}

    adapter = WhisperAdapter()
    pipeline = _RecordingPipeline()
    adapter._pipeline = pipeline

    adapter._run_pipeline({"array": object(), "sampling_rate": 16_000}, return_timestamps=True)

    assert pipeline.kwargs["batch_size"] == 1


class _DeviceRecordingTensor:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype
        self.calls: list[dict[str, Any]] = []

    def to(self, **kwargs: Any) -> _DeviceRecordingTensor:
        self.calls.append(kwargs)
        return self


def test_audio_flamingo_moves_inputs_without_casting_audio_dtype() -> None:
    audio_features = _DeviceRecordingTensor("float32")
    token_ids = _DeviceRecordingTensor("int64")

    moved = _move_inputs(
        {"input_features": audio_features, "input_ids": token_ids},
        "cuda",
    )

    assert moved["input_features"] is audio_features
    assert audio_features.dtype == "float32"
    assert audio_features.calls == [{"device": "cuda"}]
    assert token_ids.calls == [{"device": "cuda"}]


def test_audio_flamingo_uses_float32_model_precision() -> None:
    class _FakeTorch:
        float32 = object()

    torch = _FakeTorch()

    assert _dtype_for_device(torch, "cuda") is torch.float32
    assert _dtype_for_device(torch, "mps") is torch.float32
    assert _dtype_for_device(torch, "cpu") is torch.float32


class _GenerationTestProcessor:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": self.torch.tensor([[10, 11]]),
            "input_features": self.torch.tensor([[[0.0, 1.0]]]),
            "input_features_mask": self.torch.tensor([[1, 1]]),
        }

    def batch_decode(
        self, _token_ids: Any, *, skip_special_tokens: bool, **_kwargs: Any
    ) -> list[str]:
        return ["[END OF JSON]" if skip_special_tokens else "[END OF JSON]<|im_end|>"]


def test_audio_flamingo_generation_output_tracks_exact_prompt_prefix() -> None:
    torch = pytest.importorskip("torch")

    class _Model:
        def generate(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(sequences=torch.tensor([[10, 11, 12, 13]]))

    adapter = AudioFlamingoAdapter(device="cpu")
    adapter._model = _Model()
    adapter._processor = _GenerationTestProcessor(torch)
    adapter._torch = torch

    raw_text = adapter._generate_text([])

    assert raw_text == "[END OF JSON]"
    assert adapter._last_generation_diagnostics is not None
    assert adapter._last_generation_diagnostics["prompt_prefix_matches"] is True
    assert adapter._last_generation_diagnostics["continuation_token_count"] == 2
    assert (
        adapter._last_generation_diagnostics["requested_max_new_tokens"]
        == adapter.max_new_tokens
    )
    processor_inputs = adapter._last_generation_diagnostics["processor_inputs"]
    assert processor_inputs["input_features"]["shape"] == [1, 1, 2]
    assert processor_inputs["input_features"]["all_finite"] is True
    assert processor_inputs["input_features"]["nonzero_count"] == 1
    assert processor_inputs["input_features_mask"]["mask_sum"] == 2
    assert (
        adapter._last_generation_diagnostics["continuation_with_special_tokens"]
        == "[END OF JSON]<|im_end|>"
    )


def test_audio_flamingo_refuses_to_guess_continuation_on_prefix_mismatch() -> None:
    torch = pytest.importorskip("torch")

    class _Model:
        def generate(self, **_kwargs: Any) -> Any:
            return torch.tensor([[99, 11, 12]])

    adapter = AudioFlamingoAdapter(device="cpu")
    adapter._model = _Model()
    adapter._processor = _GenerationTestProcessor(torch)
    adapter._torch = torch

    with pytest.raises(AudioModelError, match="exact prompt token prefix"):
        adapter._generate_text([])

    assert adapter._last_generation_diagnostics is not None
    assert adapter._last_generation_diagnostics["prompt_prefix_matches"] is False


def test_audio_flamingo_scalar_questions_build_valid_judgment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    caption = (
        "Clear speech plays over an energetic beat with quiet street ambience; "
        "the music supports the delivery without masking it."
    )
    responses = iter([caption, *_scalar_audio_responses()])
    messages_seen: list[list[dict[str, Any]]] = []
    limits: list[int | None] = []

    def generate(
        messages: list[dict[str, Any]], *, max_new_tokens: int | None = None
    ) -> str:
        messages_seen.append(messages)
        limits.append(max_new_tokens)
        return next(responses)

    monkeypatch.setattr(adapter, "_generate_text", generate)

    result = adapter._analyze(audio_path, duration_seconds=5.0)

    assert result is not None
    assert result.background_music_present is True
    assert result.music_energy == "high"
    assert result.rhythmic_strength == pytest.approx(0.8)
    assert result.catchiness_confidence == pytest.approx(0.8)
    assert result.speech_music_interference == "low"
    assert result.environmental_sound == "audible; see the audio caption for context"
    assert result.audio_quality == "good"
    assert result.background_music_score == 1
    assert result.catchy_music_score == 1
    assert result.explanation == caption
    assert result.evidence == []
    assert len(messages_seen) == 10
    assert limits == [256] + [24] * 9
    assert adapter.last_raw_output["status"] == "ok"
    assert adapter.last_raw_output["selected_format"] == "scalar_questions"
    assert adapter.last_raw_output["scalar_retry_used"] is False
    assert len(adapter.last_raw_output["scalar_questions"]) == 9
    assert [item["attempt"] for item in adapter.last_raw_output["attempts"]] == list(
        range(1, 10)
    )
    assert all(
        item["format"] == "scalar_choice"
        and item["choice_attempt"] == 1
        and "parsed_choice" in item
        for item in adapter.last_raw_output["attempts"]
    )
    assert all(
        any(
            content.get("type") == "audio"
            for message in messages
            for content in message.get("content", [])
        )
        for messages in messages_seen[1:]
    )


def test_audio_flamingo_repairs_one_ambiguous_scalar_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    caption = "Speech plays over background music."
    responses = iter(
        [caption, "YES or NO", "YES", *_scalar_audio_responses()[1:]]
    )
    messages_seen: list[list[dict[str, Any]]] = []

    def generate(
        messages: list[dict[str, Any]], *, max_new_tokens: int | None = None
    ) -> str:
        assert max_new_tokens is not None
        messages_seen.append(messages)
        return next(responses)

    monkeypatch.setattr(adapter, "_generate_text", generate)

    result = adapter._analyze(audio_path, duration_seconds=5.0)

    assert result is not None
    assert result.background_music_present is True
    assert len(messages_seen) == 11
    assert adapter.last_raw_output["scalar_retry_used"] is True
    question = adapter.last_raw_output["scalar_questions"][
        "background_music_present"
    ]
    assert question["selected_choice"] == "YES"
    assert len(question["attempts"]) == 2
    assert "validation_error" in question["attempts"][0]
    assert question["attempts"][1]["parsed_choice"] == "YES"
    assert [item["format"] for item in adapter.last_raw_output["attempts"][:2]] == [
        "scalar_choice",
        "scalar_choice_repair",
    ]
    repair_prompt = messages_seen[2][-1]["content"][0]["text"]
    assert "exactly one word" in repair_prompt
    assert "YES, NO" in repair_prompt


def test_audio_flamingo_stops_after_invalid_scalar_repair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    responses = iter(["Speech and music are audible.", "maybe", "unclear"])
    calls = 0

    def generate(
        _messages: list[dict[str, Any]], *, max_new_tokens: int | None = None
    ) -> str:
        nonlocal calls
        assert max_new_tokens is not None
        calls += 1
        return next(responses)

    monkeypatch.setattr(adapter, "_generate_text", generate)

    with pytest.raises(AudioModelError, match="background_music_present"):
        adapter._analyze(audio_path, duration_seconds=5.0)

    assert calls == 3
    assert adapter.last_raw_output["status"] == "invalid_scalar_choice"
    assert adapter.last_raw_output["failed_field"] == "background_music_present"
    assert len(adapter.last_raw_output["attempts"]) == 2
    assert all(
        "validation_error" in item for item in adapter.last_raw_output["attempts"]
    )


def test_audio_flamingo_records_scalar_inference_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    calls = 0

    def generate(
        _messages: list[dict[str, Any]], *, max_new_tokens: int | None = None
    ) -> str:
        nonlocal calls
        assert max_new_tokens is not None
        calls += 1
        if calls == 1:
            return "Speech and music are audible."
        raise AudioModelError("simulated scalar inference failure")

    monkeypatch.setattr(adapter, "_generate_text", generate)

    with pytest.raises(AudioModelError, match="simulated scalar inference failure"):
        adapter._analyze(audio_path, duration_seconds=5.0)

    assert calls == 2
    assert adapter.last_raw_output["status"] == "scalar_question_inference_failed"
    assert adapter.last_raw_output["failed_field"] == "background_music_present"
    attempts = adapter.last_raw_output["scalar_questions"][
        "background_music_present"
    ]["attempts"]
    assert attempts[0]["inference_error"] == "simulated scalar inference failure"
    assert adapter.last_raw_output["attempts"][0]["inference_error"] == (
        "simulated scalar inference failure"
    )


def test_audio_flamingo_rejects_marker_as_natural_language_caption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    monkeypatch.setattr(
        adapter,
        "_generate_text",
        lambda _messages, **_kwargs: "[END OF JSON]",
    )

    with pytest.raises(AudioModelError, match="natural-language audio caption"):
        adapter._analyze(audio_path, duration_seconds=5.0)

    assert adapter.last_raw_output["status"] == "invalid_audio_caption"
    assert adapter.last_raw_output["audio_caption"] == "[END OF JSON]"
    assert adapter.last_raw_output["attempts"] == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HIGH", "HIGH"),
        (" high. ", "HIGH"),
        ("\nMEDIUM\n", "MEDIUM"),
    ],
)
def test_parse_choice_answer_accepts_one_exact_token(text: str, expected: str) -> None:
    assert _parse_choice_answer(text, ("LOW", "MEDIUM", "HIGH")) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "unknown",
        "low or medium",
        "not HIGH",
        "The answer is medium.",
        "I cannot say NO with confidence",
        "Y|ES",
    ],
)
def test_parse_choice_answer_rejects_ambiguous_or_extra_words(text: str) -> None:
    with pytest.raises(AudioModelError, match="exactly one of"):
        _parse_choice_answer(text, ("LOW", "MEDIUM", "HIGH", "YES", "NO"))


def test_audio_fallback_preserves_failed_contextual_model_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FailingAdapter:
        name = "audio_flamingo_3"
        last_raw_output = {
            "attempts": [{"attempt": 1, "raw_text": "plain response"}]
        }

        def analyze(self, _path: Path, **_kwargs: Any) -> None:
            raise AudioModelError("invalid structured output")

    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    monkeypatch.setattr(
        "smartedit.extraction.audio_features.compute_librosa_features",
        lambda _path: None,
    )
    adapter = _FailingAdapter()

    result = extract_audio_features(
        audio_path,
        5.0,
        audio_adapter=adapter,  # type: ignore[arg-type]
        use_librosa_fallback=True,
    )

    assert result.used_librosa_fallback is True
    assert result.raw_output is not None
    assert result.raw_output["failed_contextual_model_output"] == adapter.last_raw_output


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
