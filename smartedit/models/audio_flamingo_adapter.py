"""Audio-language model adapter with an official Audio Flamingo 3 backend."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from smartedit.models.base import AdapterError, BaseModelAdapter
from smartedit.schemas import AudioModelJudgment, audio_judgment_from_dict

LOGGER = logging.getLogger(__name__)
_TIMESTAMP_TOLERANCE_SECONDS = 0.001


class AudioModelError(AdapterError):
    """Raised when an audio-language backend cannot load or infer safely."""


class AudioModelAdapter(BaseModelAdapter[str | Path | None, AudioModelJudgment | None]):
    """Replaceable interface for contextual audio-editing models.

    Objective signal processing is intentionally outside this interface. A
    backend implementing this class is expected to listen to audio and produce
    contextual judgments; the librosa fallback is explicitly not equivalent.
    """

    name = "audio_model"
    adapter_name = "audio_model"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        *,
        allow_download: bool = False,
    ) -> None:
        super().__init__(model_name, device=device, cache_dir=cache_dir)
        self.allow_download = bool(allow_download)
        self.last_raw_output: dict[str, Any] = {}

    def analyze(self, input_data: str | Path | None, **kwargs: Any) -> AudioModelJudgment | None:
        """Analyze an audio file without loading a model for a no-audio input."""

        if input_data is None:
            self.last_raw_output = {
                "status": "skipped",
                "reason": "video has no audio stream",
            }
            return None
        _finite_nonnegative(kwargs.get("duration_seconds"), "duration_seconds")
        audio_path = Path(input_data)
        if not audio_path.is_file():
            raise AudioModelError(f"Audio file does not exist: {audio_path}")
        return super().analyze(audio_path, **kwargs)


class AudioFlamingoAdapter(AudioModelAdapter):
    """Audio Flamingo 3 adapter using its official Transformers integration.

    ``nvidia/audio-flamingo-3-hf`` is roughly an 8B-parameter checkpoint and its
    license is not generally equivalent to a permissive software license. The
    adapter therefore never downloads it unless ``allow_download=True``.
    Compatible local or future Audio Flamingo 3 checkpoints can be selected via
    ``model_name``. Audio Flamingo Next can be added as another implementation of
    :class:`AudioModelAdapter` without changing feature or fusion code.
    """

    name = "audio_flamingo_3"
    adapter_name = "audio_flamingo_3"

    def __init__(
        self,
        model_name: str = "nvidia/audio-flamingo-3-hf",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        *,
        allow_download: bool = False,
        max_new_tokens: int = 768,
        prompt_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            model_name,
            device,
            cache_dir,
            allow_download=allow_download,
        )
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.prompt_path = Path(prompt_path) if prompt_path is not None else None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _load(self) -> None:
        # Use the explicit class rather than arbitrary remote modeling code from
        # a checkpoint repository.
        try:
            import torch
            from transformers import (
                AudioFlamingo3ForConditionalGeneration,
                AutoProcessor,
            )
        except (ImportError, AttributeError) as exc:
            raise AudioModelError(
                "Audio Flamingo 3 requires PyTorch and a Transformers release "
                "containing AudioFlamingo3ForConditionalGeneration."
            ) from exc

        dtype = _dtype_for_device(torch, self.device)
        local_only = not self.allow_download
        common: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": local_only,
        }
        LOGGER.info("Loading Audio Flamingo model %s on %s", self.model_name, self.device)
        try:
            processor = AutoProcessor.from_pretrained(self.model_name, **common)
            model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **common,
            )
            model.eval()
            model.to(self.device)
        except Exception as exc:
            hint = (
                "The checkpoint is not available locally. Explicitly enable model "
                "downloads only after reviewing the multi-gigabyte size and license."
                if local_only
                else "Check the model license, hardware memory, and Transformers version."
            )
            raise AudioModelError(
                f"Could not load Audio Flamingo checkpoint {self.model_name!r}. {hint}"
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model

    def _analyze(self, input_data: str | Path | None, **kwargs: Any) -> AudioModelJudgment | None:
        if input_data is None:  # guarded by AudioModelAdapter.analyze
            raise AudioModelError("Audio Flamingo requires an audio path")
        duration = _finite_nonnegative(kwargs.get("duration_seconds"), "duration_seconds")
        audio_path = Path(input_data)
        if not audio_path.is_file():
            raise AudioModelError(f"Audio file does not exist: {audio_path}")

        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        prompt = _load_prompt(kwargs.get("prompt_path") or self.prompt_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(audio_path.resolve())},
                ],
            }
        ]
        LOGGER.info("Analyzing %s with Audio Flamingo", audio_path.name)
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_inputs(inputs, self.device, self._model.dtype)
            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            input_length = int(inputs["input_ids"].shape[-1])
            generated_ids = output_ids[:, input_length:]
            raw_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise AudioModelError("Audio Flamingo inference failed") from exc

        parsed = _parse_json_object(raw_text)
        validated = validate_audio_model_output(parsed, duration)
        self.last_raw_output = {
            "model_name": self.model_name,
            "raw_text": raw_text,
            "parsed": validated,
        }
        return audio_judgment_from_dict(validated)

    def _unload(self) -> None:
        self._processor = None
        self._model = None
        self._torch = None


def validate_audio_model_output(
    value: Mapping[str, Any], duration_seconds: float
) -> dict[str, Any]:
    """Strictly validate model fields and cap valid evidence to media duration."""

    duration = _finite_nonnegative(duration_seconds, "duration_seconds")
    required = {
        "background_music_present",
        "music_energy",
        "rhythmic_strength",
        "catchiness_confidence",
        "speech_music_interference",
        "audio_quality",
        "background_music_score",
        "catchy_music_score",
        "evidence",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise AudioModelError(
            "Audio model output is missing required fields: " + ", ".join(missing)
        )
    if not isinstance(value["background_music_present"], bool):
        raise AudioModelError("background_music_present must be a JSON boolean")
    _enum(value["music_energy"], {"low", "medium", "high"}, "music_energy")
    _unit_number(value["rhythmic_strength"], "rhythmic_strength")
    _unit_number(value["catchiness_confidence"], "catchiness_confidence")
    _enum(
        value["speech_music_interference"],
        {"none", "low", "medium", "high"},
        "speech_music_interference",
    )
    environmental_sound = value.get("environmental_sound")
    if environmental_sound is not None and (
        not isinstance(environmental_sound, str) or not environmental_sound.strip()
    ):
        raise AudioModelError("environmental_sound must be a concise non-empty string")
    _enum(value["audio_quality"], {"poor", "acceptable", "good"}, "audio_quality")
    _score(value["background_music_score"], "background_music_score")
    _score(value["catchy_music_score"], "catchy_music_score")

    evidence = value["evidence"]
    if not isinstance(evidence, list):
        raise AudioModelError("evidence must be a JSON array")
    clean_evidence: list[dict[str, Any]] = []
    for index, observation in enumerate(evidence):
        if not isinstance(observation, Mapping):
            raise AudioModelError(f"evidence[{index}] must be a JSON object")
        start_value = observation.get("start")
        end_value = observation.get("end")
        if (
            isinstance(start_value, bool)
            or isinstance(end_value, bool)
            or not isinstance(start_value, (int, float))
            or not isinstance(end_value, (int, float))
        ):
            raise AudioModelError(f"evidence[{index}] has invalid timestamps")
        start = float(start_value)
        end = float(end_value)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise AudioModelError(f"evidence[{index}] has invalid timestamps")
        if (
            start < -_TIMESTAMP_TOLERANCE_SECONDS
            or end < -_TIMESTAMP_TOLERANCE_SECONDS
            or start > duration + _TIMESTAMP_TOLERANCE_SECONDS
            or end > duration + _TIMESTAMP_TOLERANCE_SECONDS
        ):
            raise AudioModelError(
                f"evidence[{index}] timestamps must be within the audio duration "
                f"(allowing at most {_TIMESTAMP_TOLERANCE_SECONDS:.3f}s decoder rounding)"
            )
        # Clamp only the tolerated decoder-rounding error. Grossly invalid model
        # evidence is rejected above instead of being made to look grounded.
        start = min(duration, max(0.0, start))
        end = min(duration, max(0.0, end))
        clean_evidence.append(
            {
                "start": _round_timestamp(start, duration),
                "end": _round_timestamp(end, duration),
                "observation": str(observation.get("observation", "") or "").strip(),
            }
        )

    cleaned = dict(value)
    cleaned["rhythmic_strength"] = float(value["rhythmic_strength"])
    cleaned["catchiness_confidence"] = float(value["catchiness_confidence"])
    cleaned["evidence"] = clean_evidence
    return cleaned


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise AudioModelError("Audio Flamingo did not return a JSON object")
    try:
        result = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AudioModelError(f"Audio Flamingo returned invalid JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise AudioModelError("Audio Flamingo JSON must be an object")
    return result


def _load_prompt(path: str | Path | None) -> str:
    prompt_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "prompts" / "audio_analysis.txt"
    )
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AudioModelError(f"Could not read audio prompt: {prompt_path}") from exc


def _move_inputs(inputs: Any, device: str, dtype: Any) -> Any:
    """Move tensors while preserving integer token IDs."""

    if not isinstance(inputs, Mapping):
        if hasattr(inputs, "items"):
            values = dict(inputs.items())
        else:
            raise AudioModelError("Audio processor returned an unsupported input batch")
    else:
        values = dict(inputs)
    for key, value in values.items():
        if not hasattr(value, "to"):
            continue
        is_floating = bool(hasattr(value, "is_floating_point") and value.is_floating_point())
        values[key] = value.to(device=device, dtype=dtype) if is_floating else value.to(device)
    return values


def _dtype_for_device(torch: Any, device: str) -> Any:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AudioModelError(f"{name} must be provided as a number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise AudioModelError(f"{name} must be a finite non-negative number")
    return number


def _round_timestamp(value: float, duration: float) -> float:
    """Round to milliseconds without crossing the media endpoint."""

    return min(duration, max(0.0, round(float(value), 3)))


def _enum(value: Any, allowed: set[str], name: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise AudioModelError(f"{name} must be one of: {', '.join(sorted(allowed))}")


def _unit_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioModelError(f"{name} must be a number from 0 to 1")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise AudioModelError(f"{name} must be a number from 0 to 1")


def _score(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {-1, 0, 1}:
        raise AudioModelError(f"{name} must be exactly -1, 0, or 1")
