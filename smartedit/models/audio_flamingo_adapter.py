"""Audio-language model adapter with an official Audio Flamingo 3 backend."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from smartedit.models.base import AdapterError, BaseModelAdapter
from smartedit.schemas import AudioModelJudgment, audio_judgment_from_dict

LOGGER = logging.getLogger(__name__)
_TIMESTAMP_TOLERANCE_SECONDS = 0.001
_AUDIO_CHOICE_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "field": "background_music_present",
        "question": (
            "Is edited music audible anywhere in the recording? Count a beat, "
            "instrumental, melody, bassline, background track under speech, or a "
            "foreground soundtrack as music."
        ),
        "choices": ("YES", "NO"),
        "values": {"YES": True, "NO": False},
    },
    {
        "field": "music_energy",
        "question": "What is the overall energy of the music?",
        "choices": ("LOW", "MEDIUM", "HIGH"),
        "values": {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"},
    },
    {
        "field": "rhythmic_strength",
        "question": "How strong and clearly defined is the musical beat or rhythm?",
        "choices": ("LOW", "MEDIUM", "HIGH"),
        "values": {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8},
    },
    {
        "field": "catchiness_confidence",
        "question": "How likely is the music to sound catchy or memorable?",
        "choices": ("LOW", "MEDIUM", "HIGH"),
        "values": {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8},
    },
    {
        "field": "speech_music_interference",
        "question": "How much does the music mask or compete with understandable speech?",
        "choices": ("NONE", "LOW", "MEDIUM", "HIGH"),
        "values": {"NONE": "none", "LOW": "low", "MEDIUM": "medium", "HIGH": "high"},
    },
    {
        "field": "environmental_sound",
        "question": (
            "Apart from speech and music, are environmental or ambient sounds "
            "clearly audible?"
        ),
        "choices": ("NONE", "AUDIBLE"),
        "values": {
            "NONE": "none detected",
            "AUDIBLE": "audible; see the audio caption for context",
        },
    },
    {
        "field": "audio_quality",
        "question": (
            "Considering clarity and mixing, what is the overall recording quality?"
        ),
        "choices": ("POOR", "ACCEPTABLE", "GOOD"),
        "values": {"POOR": "poor", "ACCEPTABLE": "acceptable", "GOOD": "good"},
    },
    {
        "field": "background_music_score",
        "question": (
            "Considering audio editing only, does the background music support mood "
            "or pacing without masking important speech? Music being absent is not "
            "automatically harmful."
        ),
        "choices": ("HARMFUL", "NEUTRAL", "SUPPORTIVE"),
        "values": {"HARMFUL": -1, "NEUTRAL": 0, "SUPPORTIVE": 1},
    },
    {
        "field": "catchy_music_score",
        "question": (
            "Does the music's memorable or rhythmic character help this audio edit? "
            "If catchiness is irrelevant or there is no music, choose NEUTRAL."
        ),
        "choices": ("HARMFUL", "NEUTRAL", "SUPPORTIVE"),
        "values": {"HARMFUL": -1, "NEUTRAL": 0, "SUPPORTIVE": 1},
    },
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

    The checkpoint first produces a natural-language audio caption. Independent
    single-choice questions then turn that perception into validated fields.

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

        caption_prompt = _load_prompt(kwargs.get("prompt_path") or self.prompt_path)
        caption_messages = _audio_caption_messages(
            audio_path,
            duration,
            caption_prompt,
        )
        LOGGER.info("Analyzing %s with Audio Flamingo", audio_path.name)
        self.last_raw_output = {
            "model_name": self.model_name,
            "transformers_version": self._transformers_version,
            "attempts": [],
            "scalar_questions": {},
            "scalar_retry_used": False,
        }

        self._last_generation_diagnostics = None
        try:
            raw_caption = self._generate_text(
                caption_messages,
                max_new_tokens=min(self.max_new_tokens, 256),
            )
        except AudioModelError as exc:
            self.last_raw_output.update(
                {
                    "status": "caption_inference_failed",
                    "inference_error": str(exc),
                }
            )
            if self._last_generation_diagnostics is not None:
                self.last_raw_output["caption_generation_diagnostics"] = (
                    self._last_generation_diagnostics
                )
            raise

        caption = raw_caption.strip()
        self.last_raw_output["audio_caption"] = raw_caption
        if self._last_generation_diagnostics is not None:
            self.last_raw_output["caption_generation_diagnostics"] = (
                self._last_generation_diagnostics
            )
        if not caption or caption == "[END OF JSON]":
            self.last_raw_output["status"] = "invalid_audio_caption"
            raise AudioModelError(
                "Audio Flamingo did not return a usable natural-language audio caption"
            )

        base_messages = [
            *caption_messages,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]
        parsed: dict[str, Any] = {}
        scalar_questions = self.last_raw_output["scalar_questions"]
        assert isinstance(scalar_questions, dict)

        for specification in _AUDIO_CHOICE_QUESTIONS:
            field = str(specification["field"])
            choices = tuple(str(choice) for choice in specification["choices"])
            question = (
                str(specification["question"])
                + "\nReply with only one of: "
                + ", ".join(choices)
                + "."
            )
            messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                },
            ]
            question_record: dict[str, Any] = {
                "question": question,
                "choices": list(choices),
                "attempts": [],
            }
            scalar_questions[field] = question_record

            for choice_attempt in (1, 2):
                self._last_generation_diagnostics = None
                try:
                    raw_text = self._generate_text(messages, max_new_tokens=24)
                except AudioModelError as exc:
                    failed_attempt: dict[str, Any] = {
                        "attempt": choice_attempt,
                        "inference_error": str(exc),
                    }
                    if self._last_generation_diagnostics is not None:
                        failed_attempt["generation_diagnostics"] = (
                            self._last_generation_diagnostics
                        )
                    question_record["attempts"].append(failed_attempt)
                    self.last_raw_output["attempts"].append(
                        {
                            **failed_attempt,
                            "attempt": len(self.last_raw_output["attempts"]) + 1,
                            "choice_attempt": choice_attempt,
                            "format": (
                                "scalar_choice"
                                if choice_attempt == 1
                                else "scalar_choice_repair"
                            ),
                            "field": field,
                        }
                    )
                    self.last_raw_output["status"] = "scalar_question_inference_failed"
                    self.last_raw_output["failed_field"] = field
                    self.last_raw_output["inference_error"] = str(exc)
                    raise

                attempt_record: dict[str, Any] = {
                    "attempt": choice_attempt,
                    "raw_text": raw_text,
                }
                if self._last_generation_diagnostics is not None:
                    attempt_record["generation_diagnostics"] = (
                        self._last_generation_diagnostics
                    )
                question_record["attempts"].append(attempt_record)
                self.last_raw_output["attempts"].append(
                    {
                        **attempt_record,
                        "attempt": len(self.last_raw_output["attempts"]) + 1,
                        "choice_attempt": choice_attempt,
                        "format": (
                            "scalar_choice"
                            if choice_attempt == 1
                            else "scalar_choice_repair"
                        ),
                        "field": field,
                    }
                )
                self.last_raw_output["raw_text"] = raw_text

                try:
                    choice = _parse_choice_answer(raw_text, choices)
                except AudioModelError as exc:
                    attempt_record["validation_error"] = str(exc)
                    self.last_raw_output["attempts"][-1]["validation_error"] = str(exc)
                    if choice_attempt == 2:
                        self.last_raw_output["status"] = "invalid_scalar_choice"
                        self.last_raw_output["failed_field"] = field
                        raise AudioModelError(
                            f"Audio Flamingo failed the scalar question {field!r} "
                            f"after one repair: {exc}"
                        ) from exc
                    self.last_raw_output["scalar_retry_used"] = True
                    messages = [
                        *messages,
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": raw_text}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "That answer could not be parsed. Reply with "
                                        "exactly one word from this list and nothing "
                                        "else: "
                                        + ", ".join(choices)
                                        + "."
                                    ),
                                }
                            ],
                        },
                    ]
                    continue

                attempt_record["parsed_choice"] = choice
                self.last_raw_output["attempts"][-1]["parsed_choice"] = choice
                question_record["selected_choice"] = choice
                values = specification["values"]
                assert isinstance(values, Mapping)
                parsed[field] = values[choice]
                break

        parsed["explanation"] = caption
        parsed["evidence"] = []
        validated = validate_audio_model_output(parsed, duration)
        self.last_raw_output.update(
            {
                "status": "ok",
                "parsed": validated,
                "selected_format": "scalar_questions",
            }
        )
        return audio_judgment_from_dict(validated)

    def _generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate one response for an already prepared audio conversation."""

        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        generation_limit = (
            self.max_new_tokens if max_new_tokens is None else max(1, int(max_new_tokens))
        )
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_inputs(inputs, self.device)
            self._last_generation_diagnostics = {
                "processor_inputs": _processor_input_diagnostics(inputs, self._torch),
                "requested_max_new_tokens": generation_limit,
            }
            with self._torch.inference_mode():
                generation_output = self._model.generate(
                    **inputs,
                    max_new_tokens=generation_limit,
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
            self._last_generation_diagnostics.update(
                {
                    "input_ids_shape": input_shape,
                    "output_ids_shape": output_shape,
                    "prompt_prefix_matches": prefix_matches,
                    "continuation_token_count": max(0, output_shape[1] - input_length),
                }
            )
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
        raise AudioModelError("background_music_present must be a boolean")
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
        raise AudioModelError("evidence must be a list")
    clean_evidence: list[dict[str, Any]] = []
    for index, observation in enumerate(evidence):
        if not isinstance(observation, Mapping):
            raise AudioModelError(f"evidence[{index}] must be an object")
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


def _audio_caption_messages(
    audio_path: Path,
    duration: float,
    prompt: str,
) -> list[dict[str, Any]]:
    """Ask AF3 for a grounded description before the scalar questions."""

    instruction = prompt.strip()
    if not instruction:
        raise AudioModelError("Audio analysis prompt is empty")
    instruction += (
        f"\n\nThe recording duration is exactly {duration:.3f} seconds. "
        "Do not infer visual information."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "audio", "path": str(audio_path.resolve())},
            ],
        }
    ]


def _parse_choice_answer(text: str, choices: tuple[str, ...]) -> str:
    """Accept exactly one allowed choice token and reject ambiguous prose."""

    allowed = tuple(choice.upper() for choice in choices)
    words = re.findall(r"[A-Z]+", text.strip().upper())
    if len(words) == 1 and words[0] in allowed:
        return words[0]
    expected = ", ".join(allowed)
    raise AudioModelError(
        f"Audio Flamingo must answer with exactly one of: {expected}"
    )


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


def _processor_input_diagnostics(inputs: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    """Record tensor health and shapes without storing model input values."""

    diagnostics: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        item: dict[str, Any] = {
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype),
        }
        if torch.is_floating_point(value):
            item["all_finite"] = bool(torch.isfinite(value).all().item())
            item["nonzero_count"] = int(torch.count_nonzero(value).item())
        if "mask" in key.lower():
            item["mask_sum"] = int(value.sum().item())
        diagnostics[str(key)] = item
    return diagnostics


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
