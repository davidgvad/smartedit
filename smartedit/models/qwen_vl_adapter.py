"""Qwen3-VL adapter for editing-only visual/semantic judgments."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

VISUAL_SIGNAL_KEYS = (
    "story",
    "clear_start_middle_end",
    "consistent_theme",
    "text",
    "text_visibility",
    "visual_variety",
    "effects",
    "pace",
)


class QwenAdapterError(RuntimeError):
    """Raised when Qwen cannot be loaded or produces unusable output."""


class QwenVLAdapter:
    """Thin, replaceable adapter around ``Qwen3VLForConditionalGeneration``.

    Large checkpoints are never fetched unless ``allow_download`` is true.
    Supplying a local directory as ``model_name`` works regardless of that flag.
    """

    name = "qwen3_vl"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        allow_download: bool = False,
        max_new_tokens: int = 1536,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.allow_download = allow_download
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self.last_raw_output: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise QwenAdapterError(
                "Qwen3-VL requires a Transformers release containing "
                "Qwen3VLForConditionalGeneration, plus PyTorch and Pillow."
            ) from exc

        local_only = not self.allow_download and not Path(self.model_name).exists()
        common: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": local_only,
        }
        dtype = self._dtype_for_device(torch, self.device)
        model_kwargs = {**common, "torch_dtype": dtype, "low_cpu_mem_usage": True}

        LOGGER.info("Loading Qwen3-VL model %s on %s", self.model_name, self.device)
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name, **common)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name, **model_kwargs
            )
            self._model.eval()
            self._model.to(self.device)
            self._torch = torch
        except Exception as exc:
            self._model = None
            self._processor = None
            hint = (
                "Model files were not found locally. Set "
                "SMARTEDIT_ALLOW_MODEL_DOWNLOADS=1 only after reviewing the "
                "checkpoint size and license."
                if local_only
                else "Review the model license, available memory, and Transformers version."
            )
            raise QwenAdapterError(f"Could not load {self.model_name}. {hint}") from exc

    def analyze(
        self,
        frame_samples: Sequence[Any],
        objective_context: Mapping[str, Any],
        duration_seconds: float,
        prompt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Analyze sampled frames and objective context, returning validated JSON."""

        if not frame_samples:
            raise QwenAdapterError("Qwen3-VL analysis requires at least one sampled frame.")
        prompt = self._load_prompt(prompt_path)
        context_text = json.dumps(objective_context, ensure_ascii=False, indent=2)
        content: list[dict[str, Any]] = []
        frame_manifest: list[dict[str, Any]] = []
        for sample in frame_samples:
            path, timestamp = _sample_path_and_timestamp(sample)
            if not path.is_file():
                raise QwenAdapterError(f"Sampled frame does not exist: {path}")
            if not math.isfinite(timestamp) or timestamp < 0.0 or timestamp > duration_seconds:
                raise QwenAdapterError(
                    f"Sampled frame timestamp {timestamp!r} is outside the video."
                )
            resolved_path = path.resolve()
            frame_manifest.append({"path": str(resolved_path), "timestamp_seconds": timestamp})
            # Transformers accepts an HTTP(S) URL or a plain local path here,
            # but not a file:// URI.
            content.append({"type": "image", "url": str(resolved_path)})
            content.append(
                {
                    "type": "text",
                    "text": f"Exact frame timestamp: {timestamp:.6f} seconds",
                }
            )
        content.append(
            {
                "type": "text",
                "text": f"{prompt}\n\nOBJECTIVE CONTEXT (JSON):\n{context_text}",
            }
        )
        messages = [{"role": "user", "content": content}]
        self.last_raw_output = {
            "model_name": self.model_name,
            "input_context": dict(objective_context),
            "sampled_frames": frame_manifest,
        }

        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        LOGGER.info("Running Qwen3-VL on %d sampled frames", len(frame_manifest))
        try:
            inputs = self._prepare_inputs(messages)
            # Qwen3-VL's processor can return token_type_ids, but the model's
            # generation interface does not accept them. This matches the
            # official Transformers Qwen3-VL inference example.
            inputs = {key: value for key, value in inputs.items() if key != "token_type_ids"}
            inputs = _move_batch(inputs, self.device)
            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            input_length = int(inputs["input_ids"].shape[1])
            generated = output_ids[:, input_length:]
            text = self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise QwenAdapterError(
                f"Qwen3-VL inference failed: {type(exc).__name__}: {exc}"
            ) from exc

        parsed = parse_json_object(text)
        validate_qwen_output(parsed, duration_seconds)
        self.last_raw_output.update({"raw_text": text, "parsed": parsed})
        return parsed

    def _prepare_inputs(self, messages: list[dict[str, Any]]) -> Mapping[str, Any]:
        assert self._processor is not None
        try:
            return self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise QwenAdapterError(
                "The Qwen processor could not prepare the sampled frame inputs."
            ) from exc

    @staticmethod
    def _dtype_for_device(torch: Any, device: str) -> Any:
        # MPS has broad fp16 support; CPU stays fp32. BF16 is preferred on CUDA
        # where available because Qwen checkpoints are commonly published in it.
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32

    def _load_prompt(self, prompt_path: str | Path | None) -> str:
        path = (
            Path(prompt_path)
            if prompt_path
            else Path(__file__).resolve().parents[1] / "prompts" / "qwen_edit_signals.txt"
        )
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise QwenAdapterError(f"Could not read Qwen prompt: {path}") from exc


def _move_batch(batch: Mapping[str, Any], device: str) -> Mapping[str, Any]:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _sample_path_and_timestamp(sample: Any) -> tuple[Path, float]:
    if isinstance(sample, Mapping):
        path = sample.get("path") or sample.get("file_path")
        timestamp = sample.get("timestamp_seconds", sample.get("timestamp"))
    else:
        path = getattr(sample, "path", getattr(sample, "file_path", None))
        timestamp = getattr(sample, "timestamp_seconds", getattr(sample, "timestamp", None))
    if path is None or timestamp is None:
        raise QwenAdapterError("Frame samples must contain path and timestamp_seconds.")
    return Path(path), float(timestamp)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract exactly one JSON object from a model response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise QwenAdapterError("Qwen3-VL did not return a JSON object.")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise QwenAdapterError(f"Qwen3-VL returned invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise QwenAdapterError("Qwen3-VL JSON must be an object.")
    return value


def validate_qwen_output(value: Mapping[str, Any], duration_seconds: float) -> None:
    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        raise QwenAdapterError("Video duration must be finite and non-negative.")
    missing = [key for key in (*VISUAL_SIGNAL_KEYS, "category") if key not in value]
    if missing:
        raise QwenAdapterError(f"Qwen3-VL output is missing keys: {', '.join(missing)}")
    for key in VISUAL_SIGNAL_KEYS:
        item = value[key]
        if not isinstance(item, Mapping):
            raise QwenAdapterError(f"Qwen field {key!r} must be an object.")
        if type(item.get("score")) is not int or item.get("score") not in (-1, 0, 1):
            raise QwenAdapterError(f"Qwen field {key!r} has an invalid score.")
        confidence = item.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise QwenAdapterError(f"Qwen field {key!r} has invalid confidence.")
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            raise QwenAdapterError(f"Qwen field {key!r} evidence must be a list.")
        for observation in evidence:
            if not isinstance(observation, Mapping):
                raise QwenAdapterError(f"Qwen field {key!r} has malformed evidence.")
            start_value = observation.get("start")
            end_value = observation.get("end")
            if (
                isinstance(start_value, bool)
                or isinstance(end_value, bool)
                or not isinstance(start_value, (int, float))
                or not isinstance(end_value, (int, float))
            ):
                raise QwenAdapterError(f"Qwen field {key!r} contains non-numeric timestamps.")
            start = float(start_value)
            end = float(end_value)
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end < start
                or start > duration_seconds + 1e-3
                or end > duration_seconds + 1e-3
            ):
                raise QwenAdapterError(f"Qwen field {key!r} contains an out-of-range timestamp.")
            if isinstance(observation, dict):
                observation["start"] = min(start, duration_seconds)
                observation["end"] = min(end, duration_seconds)
                if not str(observation.get("observation", "")).strip():
                    raise QwenAdapterError(f"Qwen field {key!r} contains empty evidence text.")
    category = value["category"]
    if not isinstance(category, Mapping):
        raise QwenAdapterError("Qwen category must be an object.")
    for key in ("personal", "informational", "promotional"):
        number = category.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or not 0.0 <= float(number) <= 1.0
        ):
            raise QwenAdapterError(f"Qwen category probability {key!r} is invalid.")
