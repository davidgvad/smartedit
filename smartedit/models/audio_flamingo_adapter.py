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
_TAGGED_AUDIO_FIELDS = (
    "MUSIC_PRESENT",
    "MUSIC_ENERGY",
    "RHYTHMIC_STRENGTH",
    "CATCHINESS_CONFIDENCE",
    "SPEECH_MUSIC_INTERFERENCE",
    "AUDIO_QUALITY",
    "BACKGROUND_MUSIC_SCORE",
    "CATCHY_MUSIC_SCORE",
    "ENVIRONMENTAL_SOUND",
)


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
        self._transformers_version: str | None = None
        self._last_generation_diagnostics: dict[str, Any] | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _load(self) -> None:
        # Use the explicit class rather than arbitrary remote modeling code from
        # a checkpoint repository.
        try:
            import torch
            import transformers
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
        LOGGER.info(
            "Loading Audio Flamingo model %s on %s with float32 precision",
            self.model_name,
            self.device,
        )
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
                f"Could not load Audio Flamingo checkpoint {self.model_name!r}. {hint} "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

        self._torch = torch
        self._transformers_version = str(transformers.__version__)
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
        prompt += (
            f"\n\nThe audio duration is exactly {duration:.3f} seconds. "
            "Use only timestamps within that duration. If reliable timestamped "
            "evidence is unavailable, return an empty evidence array."
        )
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
        self.last_raw_output = {
            "model_name": self.model_name,
            "transformers_version": self._transformers_version,
            "attempts": [],
            "retry_used": False,
            "tagged_fallback_used": False,
        }
        current_messages = messages
        for attempt_number in (1, 2):
            self._last_generation_diagnostics = None
            try:
                raw_text = self._generate_text(current_messages)
            except AudioModelError as exc:
                self.last_raw_output.update(
                    {
                        "status": (
                            "format_retry_inference_failed"
                            if attempt_number == 2
                            else "inference_failed"
                        ),
                        "inference_error": str(exc),
                    }
                )
                if self._last_generation_diagnostics is not None:
                    self.last_raw_output["failed_generation_diagnostics"] = (
                        self._last_generation_diagnostics
                    )
                raise
            attempt: dict[str, Any] = {
                "attempt": attempt_number,
                "format": "json" if attempt_number == 1 else "json_repair",
                "raw_text": raw_text,
            }
            if self._last_generation_diagnostics is not None:
                attempt["generation_diagnostics"] = self._last_generation_diagnostics
            self.last_raw_output["attempts"].append(attempt)
            self.last_raw_output["raw_text"] = raw_text
            try:
                parsed = _parse_json_object(raw_text)
                validated = validate_audio_model_output(parsed, duration)
            except AudioModelError as exc:
                attempt["validation_error"] = str(exc)
                if _is_json_end_marker(raw_text):
                    # Some Audio Flamingo checkpoints answer JSON-only prompts
                    # with this literal ordinary-text marker. Repeating the same
                    # request has proven unhelpful, so switch to an independently
                    # parseable record without treating the marker as evidence.
                    attempt["marker_only_response"] = True
                    self.last_raw_output["json_marker_detected"] = True
                    LOGGER.warning(
                        "Audio Flamingo generated only [END OF JSON]; switching "
                        "to the tagged compatibility format"
                    )
                    break
                if attempt_number == 2:
                    self.last_raw_output["json_status"] = "invalid_after_format_retry"
                    break
                LOGGER.warning(
                    "Audio Flamingo returned invalid structured output; retrying JSON "
                    "format once: %s",
                    exc,
                )
                self.last_raw_output["retry_used"] = True
                current_messages = _format_retry_messages(messages, raw_text, duration)
                continue

            self.last_raw_output.update(
                {
                    "status": "ok",
                    "parsed": validated,
                    "selected_attempt": attempt_number,
                    "selected_format": "json",
                }
            )
            return audio_judgment_from_dict(validated)

        # Audio Flamingo is a free-text audio-language model; the upstream model
        # does not guarantee JSON-constrained decoding. If both JSON prompting
        # routes fail, ask for a minimal tagged record and parse only exact,
        # enumerated fields. The final project report is still JSON, and this
        # compatibility path remains visible in raw_model_outputs.
        self.last_raw_output["tagged_fallback_used"] = True
        tagged_messages = _tagged_record_messages(audio_path, duration)
        tagged_attempt_number = len(self.last_raw_output["attempts"]) + 1
        self._last_generation_diagnostics = None
        try:
            raw_text = self._generate_text(tagged_messages)
        except AudioModelError as exc:
            self.last_raw_output.update(
                {
                    "status": "tagged_fallback_inference_failed",
                    "inference_error": str(exc),
                }
            )
            if self._last_generation_diagnostics is not None:
                self.last_raw_output["failed_generation_diagnostics"] = (
                    self._last_generation_diagnostics
                )
            raise

        attempt = {
            "attempt": tagged_attempt_number,
            "format": "tagged_record",
            "raw_text": raw_text,
        }
        if self._last_generation_diagnostics is not None:
            attempt["generation_diagnostics"] = self._last_generation_diagnostics
        self.last_raw_output["attempts"].append(attempt)
        self.last_raw_output["raw_text"] = raw_text
        try:
            parsed = _parse_tagged_audio_record(raw_text)
            validated = validate_audio_model_output(parsed, duration)
        except AudioModelError as exc:
            attempt["validation_error"] = str(exc)
            self.last_raw_output["status"] = "invalid_after_tagged_fallback"
            raise AudioModelError(
                "Audio Flamingo failed to return valid structured audio fields in "
                f"JSON or tagged compatibility format: {exc}"
            ) from exc

        self.last_raw_output.update(
            {
                "status": "ok",
                "parsed": validated,
                "selected_attempt": tagged_attempt_number,
                "selected_format": "tagged_record_compatibility",
            }
        )
        return audio_judgment_from_dict(validated)

    def _generate_text(self, messages: list[dict[str, Any]]) -> str:
        """Generate one response for an already prepared audio conversation."""

        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_inputs(inputs, self.device)
            with self._torch.inference_mode():
                generation_output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            output_ids = getattr(generation_output, "sequences", generation_output)
            input_ids = inputs.get("input_ids")
            if input_ids is None or not hasattr(input_ids, "shape"):
                raise AudioModelError("Audio Flamingo processor did not return input_ids")
            if not hasattr(output_ids, "shape") or len(output_ids.shape) != 2:
                raise AudioModelError(
                    "Audio Flamingo generate() returned an unsupported sequence container"
                )
            input_length = int(input_ids.shape[-1])
            input_shape = [int(size) for size in input_ids.shape]
            output_shape = [int(size) for size in output_ids.shape]
            shapes_allow_prefix = (
                len(input_shape) == 2
                and output_shape[0] == input_shape[0]
                and output_shape[1] >= input_length
            )
            prefix_matches = bool(
                shapes_allow_prefix
                and self._torch.equal(output_ids[:, :input_length], input_ids)
            )
            self._last_generation_diagnostics = {
                "input_ids_shape": input_shape,
                "output_ids_shape": output_shape,
                "prompt_prefix_matches": prefix_matches,
                "continuation_token_count": max(0, output_shape[1] - input_length),
            }
            if not prefix_matches:
                raise AudioModelError(
                    "Audio Flamingo generate() output did not begin with the exact "
                    f"prompt token prefix (input shape {input_shape}, output shape "
                    f"{output_shape}); refusing to guess where the continuation starts"
                )
            generated_ids = output_ids[:, input_length:]
            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not decoded:
                raise AudioModelError("Audio Flamingo decoded an empty response batch")
            try:
                debug_decoded = self._processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if debug_decoded:
                    self._last_generation_diagnostics[
                        "continuation_with_special_tokens"
                    ] = str(debug_decoded[0])
            except Exception as exc:  # diagnostics must not invalidate inference
                self._last_generation_diagnostics["special_token_decode_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            return str(decoded[0])
        except AudioModelError:
            raise
        except Exception as exc:
            raise AudioModelError(
                f"Audio Flamingo inference failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _unload(self) -> None:
        self._processor = None
        self._model = None
        self._torch = None
        self._last_generation_diagnostics = None


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
    decoder = json.JSONDecoder()
    errors: list[str] = []
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            result, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            errors.append(exc.msg)
            continue
        if isinstance(result, dict):
            return result
    if errors:
        raise AudioModelError(f"Audio Flamingo returned invalid JSON: {errors[0]}")
    raise AudioModelError("Audio Flamingo did not return a JSON object")


def _is_json_end_marker(text: str) -> bool:
    """Recognize only the exact literal response observed from the checkpoint."""

    return text.strip() == "[END OF JSON]"


def _tagged_record_messages(audio_path: Path, duration: float) -> list[dict[str, Any]]:
    """Build a fresh, minimal free-text request when JSON prompting fails."""

    prompt = (
        f"Listen to the attached {duration:.3f}-second audio and judge only its "
        "audio editing.\n"
        "Distinguish speech, music, environmental sound, and silence. Decide whether "
        "music supports the content or masks speech. Reply with exactly one "
        "pipe-separated record using every field below once and no commentary:\n\n"
        "MUSIC_PRESENT=YES | MUSIC_ENERGY=LOW | RHYTHMIC_STRENGTH=0.0 | "
        "CATCHINESS_CONFIDENCE=0.0 | SPEECH_MUSIC_INTERFERENCE=NONE | "
        "AUDIO_QUALITY=ACCEPTABLE | BACKGROUND_MUSIC_SCORE=0 | "
        "CATCHY_MUSIC_SCORE=0 | ENVIRONMENTAL_SOUND=none detected\n\n"
        "Allowed values:\n"
        "MUSIC_PRESENT is YES or NO.\n"
        "MUSIC_ENERGY is LOW, MEDIUM, or HIGH.\n"
        "RHYTHMIC_STRENGTH and CATCHINESS_CONFIDENCE are decimal numbers from 0 to 1.\n"
        "SPEECH_MUSIC_INTERFERENCE is NONE, LOW, MEDIUM, or HIGH.\n"
        "AUDIO_QUALITY is POOR, ACCEPTABLE, or GOOD.\n"
        "Each score is exactly -1, 0, or 1. A positive score means the choice "
        "supports the edit, zero means neutral or unnecessary, and negative means "
        "it harms the edit.\n"
        "ENVIRONMENTAL_SOUND is one short literal description.\n"
        "Do not include timestamps, explanations, headings, Markdown, or extra fields."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": str(audio_path.resolve())},
            ],
        }
    ]


def _parse_tagged_audio_record(text: str) -> dict[str, Any]:
    """Parse the deliberately small compatibility grammar without inference."""

    stripped = text.strip()
    if not stripped:
        raise AudioModelError("Audio Flamingo returned an empty tagged record")
    if "\n" in stripped and "|" in stripped:
        raise AudioModelError("Tagged audio record must use one delimiter style")
    delimiter = "|" if "|" in stripped else "\n"
    parts = stripped.split(delimiter)
    if any(not part.strip() for part in parts):
        raise AudioModelError("Tagged audio record contains an empty field")

    fields: dict[str, str] = {}
    allowed = set(_TAGGED_AUDIO_FIELDS)
    for part in parts:
        if part.count("=") != 1:
            raise AudioModelError(
                "Every tagged audio field must contain exactly one '=' separator"
            )
        key_text, value_text = part.split("=", 1)
        key = key_text.strip().upper()
        value = value_text.strip()
        if key not in allowed:
            raise AudioModelError(f"Unknown tagged audio field: {key or '<empty>'}")
        if key in fields:
            raise AudioModelError(f"Duplicate tagged audio field: {key}")
        if not value:
            raise AudioModelError(f"Tagged audio field {key} has an empty value")
        fields[key] = value

    missing = [key for key in _TAGGED_AUDIO_FIELDS if key not in fields]
    if missing:
        raise AudioModelError(
            "Tagged audio record is missing required fields: " + ", ".join(missing)
        )

    present_text = fields["MUSIC_PRESENT"].upper()
    if present_text not in {"YES", "NO"}:
        raise AudioModelError("MUSIC_PRESENT must be exactly YES or NO")

    score_values: dict[str, int] = {}
    for key in ("BACKGROUND_MUSIC_SCORE", "CATCHY_MUSIC_SCORE"):
        if fields[key] not in {"-1", "0", "1"}:
            raise AudioModelError(f"{key} must be exactly -1, 0, or 1")
        score_values[key] = int(fields[key])

    number_values: dict[str, float] = {}
    for key in ("RHYTHMIC_STRENGTH", "CATCHINESS_CONFIDENCE"):
        try:
            number = float(fields[key])
        except ValueError as exc:
            raise AudioModelError(f"{key} must be a number from 0 to 1") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise AudioModelError(f"{key} must be a number from 0 to 1")
        number_values[key] = number

    energy = fields["MUSIC_ENERGY"].lower()
    interference = fields["SPEECH_MUSIC_INTERFERENCE"].lower()
    quality = fields["AUDIO_QUALITY"].lower()
    _enum(energy, {"low", "medium", "high"}, "MUSIC_ENERGY")
    _enum(
        interference,
        {"none", "low", "medium", "high"},
        "SPEECH_MUSIC_INTERFERENCE",
    )
    _enum(quality, {"poor", "acceptable", "good"}, "AUDIO_QUALITY")

    return {
        "background_music_present": present_text == "YES",
        "music_energy": energy,
        "rhythmic_strength": number_values["RHYTHMIC_STRENGTH"],
        "catchiness_confidence": number_values["CATCHINESS_CONFIDENCE"],
        "speech_music_interference": interference,
        "environmental_sound": fields["ENVIRONMENTAL_SOUND"],
        "audio_quality": quality,
        "background_music_score": score_values["BACKGROUND_MUSIC_SCORE"],
        "catchy_music_score": score_values["CATCHY_MUSIC_SCORE"],
        # This compatibility grammar intentionally makes no timestamp claims.
        "evidence": [],
    }


def _format_retry_messages(
    original_messages: list[dict[str, Any]], raw_text: str, duration: float
) -> list[dict[str, Any]]:
    repair_prompt = (
        "Your previous response was not valid for the requested schema. Return "
        "exactly one valid JSON object with all fields from the original schema. "
        "Do not include Markdown, commentary, or reasoning outside the JSON. Do "
        "not add audio events or timestamps that you did not observe. If reliable "
        f"timestamps within 0.000 to {duration:.3f} seconds are unavailable, use "
        '"evidence": [].'
    )
    return [
        *original_messages,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": raw_text}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": repair_prompt}],
        },
    ]


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


def _move_inputs(inputs: Any, device: str) -> Any:
    """Move processor tensors to the model device without changing their dtype.

    Audio Flamingo's processor emits float32 audio features. Preserve those
    processor-selected dtypes so they match the float32 model used by this
    adapter.
    """

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
        values[key] = value.to(device=device)
    return values


def _dtype_for_device(torch: Any, _device: str) -> Any:
    # Audio Flamingo 3 currently mixes modules that must remain float32 with
    # modules that otherwise accept BF16. Loading the whole model in BF16 caused
    # alternating Float/BFloat16 failures in the audio encoder. Use one explicit
    # dtype for the beginner baseline; the 8B checkpoint therefore needs a
    # high-memory accelerator.
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
