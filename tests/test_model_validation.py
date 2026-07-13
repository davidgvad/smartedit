from __future__ import annotations

import json
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
    _parse_tagged_audio_record,
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


def _tagged_audio_payload() -> str:
    return (
        "MUSIC_PRESENT=YES | MUSIC_ENERGY=HIGH | RHYTHMIC_STRENGTH=0.82 | "
        "CATCHINESS_CONFIDENCE=0.76 | SPEECH_MUSIC_INTERFERENCE=MEDIUM | "
        "AUDIO_QUALITY=ACCEPTABLE | BACKGROUND_MUSIC_SCORE=-1 | "
        "CATCHY_MUSIC_SCORE=1 | ENVIRONMENTAL_SOUND=quiet room tone"
    )


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


def test_audio_flamingo_retries_plain_text_once_and_preserves_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    valid_json = json.dumps(_audio_payload(0.0, 5.0))
    caption = "Speech overlaps with energetic background music."
    responses = iter([caption, "The music is energetic.", valid_json])
    messages_seen: list[list[dict[str, Any]]] = []

    def generate(messages: list[dict[str, Any]]) -> str:
        messages_seen.append(messages)
        return next(responses)

    monkeypatch.setattr(adapter, "_generate_text", generate)

    result = adapter._analyze(audio_path, duration_seconds=5.0)

    assert result is not None
    assert result.background_music_present is True
    assert len(messages_seen) == 3
    assert adapter.last_raw_output["audio_caption"] == caption
    assert adapter.last_raw_output["retry_used"] is True
    assert adapter.last_raw_output["selected_attempt"] == 2
    assert [item["raw_text"] for item in adapter.last_raw_output["attempts"]] == [
        "The music is energetic.",
        valid_json,
    ]
    retry_text = messages_seen[2][-1]["content"][0]["text"]
    assert "exactly one valid JSON object" in retry_text
    assert any(
        item.get("type") == "audio"
        for message in messages_seen[2]
        for item in message.get("content", [])
    )


def test_audio_flamingo_token_limited_json_skips_repair_and_uses_tagged_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    responses = iter(
        [
            "A voice performs over a hip-hop beat.",
            "A repeated non-JSON observation.",
            _tagged_audio_payload(),
        ]
    )
    calls = 0

    def generate(_messages: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        response = next(responses)
        if calls == 2:
            adapter._last_generation_diagnostics = {
                "continuation_token_count": adapter.max_new_tokens
            }
        return response

    monkeypatch.setattr(adapter, "_generate_text", generate)

    result = adapter._analyze(audio_path, duration_seconds=5.0)

    assert result is not None
    assert calls == 3
    assert adapter.last_raw_output["status"] == "ok"
    assert adapter.last_raw_output["json_status"] == "truncated_at_token_limit"
    assert adapter.last_raw_output["retry_used"] is False
    assert adapter.last_raw_output["attempts"][0]["hit_max_new_tokens"] is True
    assert [item["format"] for item in adapter.last_raw_output["attempts"]] == [
        "json",
        "tagged_record",
    ]


def test_audio_flamingo_json_repair_inference_failure_still_uses_tagged_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    calls = 0

    def generate(_messages: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "A speaker performs over background music."
        if calls == 2:
            return "not JSON"
        if calls == 3:
            raise AudioModelError("simulated long-context generation failure")
        return _tagged_audio_payload()

    monkeypatch.setattr(adapter, "_generate_text", generate)

    result = adapter._analyze(audio_path, duration_seconds=5.0)

    assert result is not None
    assert calls == 4
    assert adapter.last_raw_output["status"] == "ok"
    assert adapter.last_raw_output["json_status"] == "format_retry_inference_failed"
    assert "long-context" in adapter.last_raw_output["json_inference_error"]
    assert [item["format"] for item in adapter.last_raw_output["attempts"]] == [
        "json",
        "json_repair",
        "tagged_record",
    ]
    assert "long-context" in adapter.last_raw_output["attempts"][1]["inference_error"]
    assert adapter.last_raw_output["selected_format"] == "tagged_record_compatibility"


def test_audio_flamingo_stops_after_one_json_and_one_tagged_repair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    responses = iter(
        [
            "Speech and quiet background music overlap.",
            "first plain response",
            "second plain response",
            "invalid tagged response",
            "invalid tagged repair",
        ]
    )
    calls = 0

    def generate(_messages: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(adapter, "_generate_text", generate)

    with pytest.raises(AudioModelError, match="after one tagged repair"):
        adapter._analyze(audio_path, duration_seconds=5.0)

    assert calls == 5
    assert adapter.last_raw_output["status"] == "invalid_after_tagged_repair"
    assert [item["raw_text"] for item in adapter.last_raw_output["attempts"]] == [
        "first plain response",
        "second plain response",
        "invalid tagged response",
        "invalid tagged repair",
    ]
    assert all(
        "validation_error" in item for item in adapter.last_raw_output["attempts"]
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
    monkeypatch.setattr(adapter, "_generate_text", lambda _messages: "[END OF JSON]")

    with pytest.raises(AudioModelError, match="natural-language audio caption"):
        adapter._analyze(audio_path, duration_seconds=5.0)

    assert adapter.last_raw_output["status"] == "invalid_audio_caption"
    assert adapter.last_raw_output["audio_caption"] == "[END OF JSON]"
    assert adapter.last_raw_output["attempts"] == []


def test_audio_flamingo_marker_uses_tagged_compatibility_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    caption = "A speaker talks over rhythmic background music."
    responses = iter([caption, "[END OF JSON]", _tagged_audio_payload()])
    messages_seen: list[list[dict[str, Any]]] = []

    def generate(messages: list[dict[str, Any]]) -> str:
        messages_seen.append(messages)
        return next(responses)

    monkeypatch.setattr(adapter, "_generate_text", generate)

    result = adapter._analyze(audio_path, duration_seconds=5.0)

    assert result is not None
    assert result.background_music_present is True
    assert result.music_energy == "high"
    assert result.speech_music_interference == "medium"
    assert result.background_music_score == -1
    assert result.catchy_music_score == 1
    assert result.evidence == []
    assert len(messages_seen) == 3
    assert adapter.last_raw_output["audio_caption"] == caption
    assert adapter.last_raw_output["retry_used"] is False
    assert adapter.last_raw_output["tagged_fallback_used"] is True
    assert adapter.last_raw_output["json_marker_detected"] is True
    assert adapter.last_raw_output["selected_attempt"] == 2
    assert adapter.last_raw_output["selected_format"] == "tagged_record_compatibility"
    assert [item["format"] for item in adapter.last_raw_output["attempts"]] == [
        "json",
        "tagged_record",
    ]
    tagged_content = messages_seen[2][0]["content"]
    assert any(item.get("type") == "audio" for item in tagged_content)
    tagged_prompt = next(
        item["text"]
        for item in messages_seen[2][-1]["content"]
        if item.get("type") == "text"
    )
    assert "MUSIC_PRESENT" in tagged_prompt
    assert "ENVIRONMENTAL_SOUND" in tagged_prompt


def test_audio_flamingo_repairs_real_malformed_tagged_response_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    adapter = AudioFlamingoAdapter()
    adapter._model = object()
    adapter._processor = object()
    adapter._torch = object()
    malformed = (
        "MUSIC_PRESENT=NO | MUSIC_ENERGY=LOW | RHYTHMIC_STRENGTH=0.0 | "
        "CASTEITY_CONFIDENCE=0.0 | SPEECH_MUSIC_INTERFERENCE=LOW | "
        "AUDIO_QUALITY=GOOD | ENVIRONMENTAL_SOUND=none detected"
    )
    caption = "Speech and music are both audible."
    responses = iter([caption, "[END OF JSON]", malformed, _tagged_audio_payload()])
    messages_seen: list[list[dict[str, Any]]] = []

    def generate(messages: list[dict[str, Any]]) -> str:
        messages_seen.append(messages)
        return next(responses)

    monkeypatch.setattr(adapter, "_generate_text", generate)

    result = adapter._analyze(audio_path, duration_seconds=5.0)

    assert result is not None
    assert result.background_music_present is True
    assert adapter.last_raw_output["tagged_repair_used"] is True
    assert adapter.last_raw_output["selected_attempt"] == 3
    assert adapter.last_raw_output["selected_format"] == "tagged_record_repair"
    assert [item["format"] for item in adapter.last_raw_output["attempts"]] == [
        "json",
        "tagged_record",
        "tagged_record_repair",
    ]
    error = adapter.last_raw_output["attempts"][1]["validation_error"]
    assert "unknown fields: CASTEITY_CONFIDENCE" in error
    assert "missing fields: CATCHINESS_CONFIDENCE" in error
    assert "BACKGROUND_MUSIC_SCORE" in error
    assert "CATCHY_MUSIC_SCORE" in error
    repair_prompt = messages_seen[3][-1]["content"][0]["text"]
    assert "CASTEITY_CONFIDENCE" in repair_prompt
    assert "CATCHINESS_CONFIDENCE" in repair_prompt
    assert "BACKGROUND_MUSIC_SCORE" in repair_prompt
    assert "CATCHY_MUSIC_SCORE" in repair_prompt
    assert any(
        item.get("type") == "audio"
        for message in messages_seen[3]
        for item in message.get("content", [])
    )


def test_parse_tagged_audio_record_is_strict_and_adds_no_evidence() -> None:
    parsed = _parse_tagged_audio_record(_tagged_audio_payload())
    validated = validate_audio_model_output(parsed, 5.0)

    assert validated["background_music_present"] is True
    assert validated["rhythmic_strength"] == pytest.approx(0.82)
    assert validated["catchiness_confidence"] == pytest.approx(0.76)
    assert validated["environmental_sound"] == "quiet room tone"
    assert validated["evidence"] == []


@pytest.mark.parametrize(
    "text",
    [
        "MUSIC_PRESENT=YES",
        _tagged_audio_payload() + " | UNKNOWN=value",
        _tagged_audio_payload().replace(
            "MUSIC_ENERGY=HIGH", "MUSIC_ENERGY=HIGH | MUSIC_ENERGY=LOW"
        ),
        _tagged_audio_payload().replace("RHYTHMIC_STRENGTH=0.82", "RHYTHMIC_STRENGTH=nan"),
        _tagged_audio_payload().replace("CATCHY_MUSIC_SCORE=1", "CATCHY_MUSIC_SCORE=1.0"),
    ],
)
def test_parse_tagged_audio_record_rejects_ambiguous_values(text: str) -> None:
    with pytest.raises(AudioModelError):
        _parse_tagged_audio_record(text)


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
