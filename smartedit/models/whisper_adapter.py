"""Transformers adapter for timestamped Whisper transcription."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from smartedit.extraction.narration_features import extract_narration_features
from smartedit.models.base import AdapterError, BaseModelAdapter
from smartedit.schemas import NarrationAnalysis, to_dict

LOGGER = logging.getLogger(__name__)


class WhisperAdapterError(AdapterError):
    """Raised when Whisper is unavailable or transcription fails."""


class WhisperAdapter(BaseModelAdapter[str | Path | None, NarrationAnalysis]):
    """Run Whisper large-v3-turbo through ``transformers``.

    Model downloads are opt-in. With the default ``allow_download=False``,
    Hugging Face is queried only for files already present in the local cache (or
    ``model_name`` can point at a local checkpoint directory).
    """

    name = "whisper"
    adapter_name = "whisper"

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        *,
        allow_download: bool = False,
        chunk_length_seconds: float = 30.0,
        batch_size: int = 1,
    ) -> None:
        super().__init__(model_name, device=device, cache_dir=cache_dir)
        self.allow_download = bool(allow_download)
        self.chunk_length_seconds = float(chunk_length_seconds)
        if not math.isfinite(self.chunk_length_seconds) or self.chunk_length_seconds <= 0:
            raise ValueError("chunk_length_seconds must be a finite positive number")
        # This adapter sends one waveform to the Transformers ASR pipeline.
        # A batch size greater than one can fail while the pipeline unbatches
        # the final (single-item) audio chunk, so the safe default is one.
        self.batch_size = max(1, int(batch_size))
        self._model: Any | None = None
        self._processor: Any | None = None
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self.last_raw_output: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _load(self) -> None:
        """Load the configured checkpoint without an implicit network download."""

        try:
            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )
        except (ImportError, AttributeError) as exc:
            raise WhisperAdapterError(
                "Whisper requires PyTorch and Transformers with AutoModelForSpeechSeq2Seq support."
            ) from exc

        dtype = _dtype_for_device(torch, self.device)
        local_only = not self.allow_download
        common: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": local_only,
        }
        LOGGER.info("Loading Whisper model %s on %s", self.model_name, self.device)
        try:
            processor = AutoProcessor.from_pretrained(self.model_name, **common)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **common,
            )
            model.eval()
            model.to(self.device)
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=self.device,
            )
        except Exception as exc:
            hint = (
                "The checkpoint is not in the local Hugging Face cache. Enable "
                "model downloads explicitly only after reviewing its size and license."
                if local_only
                else "Check the checkpoint license, Transformers version, and available memory."
            )
            raise WhisperAdapterError(
                f"Could not load Whisper checkpoint {self.model_name!r}. {hint} "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model
        self._pipeline = asr_pipeline

    def analyze(
        self,
        input_data: str | Path | None,
        **kwargs: Any,
    ) -> NarrationAnalysis:
        """Transcribe audio and return validated, measured narration evidence.

        Required keyword argument: ``duration_seconds``. Supplying ``None`` for
        ``input_data`` returns an explicit empty result and does not load Whisper;
        this is the normal no-audio path.
        """

        if "duration_seconds" not in kwargs:
            raise WhisperAdapterError("duration_seconds is required for timestamp validation")
        try:
            duration_seconds = float(kwargs["duration_seconds"])
            silence_threshold = float(kwargs.get("long_silence_threshold_seconds", 2.0))
        except (TypeError, ValueError) as exc:
            raise WhisperAdapterError("duration and silence threshold must be numeric") from exc
        if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
            raise WhisperAdapterError("duration_seconds must be a finite non-negative number")
        if not math.isfinite(silence_threshold) or silence_threshold < 0.0:
            raise WhisperAdapterError(
                "long_silence_threshold_seconds must be finite and non-negative"
            )
        if input_data is None:
            self.last_raw_output = {
                "status": "skipped",
                "reason": "video has no audio stream",
            }
            return extract_narration_features(
                {},
                duration_seconds,
                model_name=self.model_name,
                long_silence_threshold_seconds=silence_threshold,
            )

        audio_path = Path(input_data)
        if not audio_path.is_file():
            raise WhisperAdapterError(f"Audio file does not exist: {audio_path}")

        return super().analyze(audio_path, **kwargs)

    def _analyze(
        self,
        input_data: str | Path | None,
        **kwargs: Any,
    ) -> NarrationAnalysis:
        """Run loaded Whisper resources for a concrete audio path."""

        if input_data is None:  # guarded by ``analyze``; retained for type safety
            raise WhisperAdapterError("Whisper requires an audio path")
        duration_seconds = float(kwargs["duration_seconds"])
        silence_threshold = float(kwargs.get("long_silence_threshold_seconds", 2.0))

        audio_path = Path(input_data)
        if not audio_path.is_file():
            raise WhisperAdapterError(f"Audio file does not exist: {audio_path}")

        waveform = _load_audio_mono_16khz(audio_path)
        if waveform.size == 0:
            self.last_raw_output = {"status": "empty_audio", "text": "", "chunks": []}
            return extract_narration_features(
                {},
                duration_seconds,
                model_name=self.model_name,
                long_silence_threshold_seconds=silence_threshold,
            )

        assert self._pipeline is not None
        LOGGER.info("Transcribing %s with Whisper", audio_path.name)
        pipeline_input = {"array": waveform, "sampling_rate": 16_000}

        word_timestamps = True
        try:
            output = self._run_pipeline(pipeline_input, return_timestamps="word")
        except Exception as word_error:
            LOGGER.warning(
                "Word timestamps were unavailable; retrying Whisper with segment timestamps: %s",
                word_error,
            )
            word_timestamps = False
            try:
                output = self._run_pipeline(pipeline_input, return_timestamps=True)
            except Exception as segment_error:
                raise WhisperAdapterError(
                    "Whisper transcription failed: "
                    f"{type(segment_error).__name__}: {segment_error}"
                ) from segment_error

        normalized = _normalize_pipeline_output(output, word_timestamps=word_timestamps)
        normalized["word_timestamps_supported"] = word_timestamps
        normalized["model_name"] = self.model_name
        analysis = extract_narration_features(
            normalized,
            duration_seconds,
            model_name=self.model_name,
            long_silence_threshold_seconds=silence_threshold,
        )
        original_output = _json_safe_value(output)
        self.last_raw_output = {
            "model_name": self.model_name,
            "word_timestamps_supported": word_timestamps,
            # Keep the exact JSON-compatible decoder payload for auditability as
            # a string. Structured timestamps below are the validated/capped
            # representation used by the pipeline.
            "pipeline_output_original_json": json.dumps(
                original_output, ensure_ascii=False, allow_nan=False
            ),
            "normalized": to_dict(analysis),
        }
        return analysis

    def _unload(self) -> None:
        self._pipeline = None
        self._processor = None
        self._model = None
        self._torch = None

    def _run_pipeline(
        self, pipeline_input: Mapping[str, Any], *, return_timestamps: str | bool
    ) -> Mapping[str, Any]:
        assert self._pipeline is not None
        pipeline = self._pipeline

        def invoke(*, include_language: bool) -> Any:
            call_kwargs: dict[str, Any] = {
                "return_timestamps": return_timestamps,
                "chunk_length_s": self.chunk_length_seconds,
                "batch_size": self.batch_size,
                "generate_kwargs": {"task": "transcribe"},
            }
            if include_language:
                call_kwargs["return_language"] = True
            # HF preprocessing may pop ``array``/``sampling_rate``. A shallow
            # copy preserves the large waveform while giving every attempt a
            # fresh mutable container.
            return pipeline(dict(pipeline_input), **call_kwargs)

        try:
            output = invoke(include_language=True)
        except (TypeError, ValueError) as exc:
            # ``return_language`` was introduced after timestamp support. Older
            # compatible Transformers releases can still provide the transcript.
            if "return_language" not in str(exc):
                raise
            output = invoke(include_language=False)
        if not isinstance(output, Mapping):
            raise WhisperAdapterError("Whisper returned an unexpected result type")
        return output


def _load_audio_mono_16khz(path: Path) -> Any:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise WhisperAdapterError("Loading audio for Whisper requires librosa and NumPy.") from exc
    try:
        waveform, _ = librosa.load(str(path), sr=16_000, mono=True)
    except Exception as exc:
        raise WhisperAdapterError(
            f"Could not decode audio file {path}: {type(exc).__name__}: {exc}"
        ) from exc
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return waveform


def _normalize_pipeline_output(
    output: Mapping[str, Any], *, word_timestamps: bool
) -> dict[str, Any]:
    transcript = str(output.get("text", "") or "").strip()
    chunks = output.get("chunks", [])
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes, bytearray)):
        chunks = []

    language = output.get("language")
    if language is None:
        languages = [
            str(chunk.get("language"))
            for chunk in chunks
            if isinstance(chunk, Mapping) and chunk.get("language")
        ]
        language = languages[0] if languages and len(set(languages)) == 1 else None

    if word_timestamps:
        raw_words = [_chunk_to_interval(chunk) for chunk in chunks]
        words: list[dict[str, Any]] = [word for word in raw_words if word is not None]
        segments = _group_words_into_segments(words)
    else:
        words = []
        raw_segments = [_chunk_to_interval(chunk) for chunk in chunks]
        segments = [segment for segment in raw_segments if segment is not None]

    return {
        "transcript": transcript,
        "segments": segments,
        "words": words,
        "detected_language": str(language) if language else None,
        "language_probability": output.get("language_probability"),
        # Preserve the decoded chunks for auditability; they contain only JSON-
        # compatible text/timestamp metadata from the pipeline.
        "chunks": [_json_safe_chunk(chunk) for chunk in chunks if isinstance(chunk, Mapping)],
    }


def _chunk_to_interval(chunk: Any) -> dict[str, Any] | None:
    if not isinstance(chunk, Mapping):
        return None
    timestamp = chunk.get("timestamp", chunk.get("timestamps"))
    if (
        not isinstance(timestamp, Sequence)
        or isinstance(timestamp, (str, bytes, bytearray))
        or len(timestamp) < 2
    ):
        return None
    start, end = timestamp[0], timestamp[1]
    if start is None or end is None:
        return None
    result: dict[str, Any] = {
        "start": start,
        "end": end,
        "text": str(chunk.get("text", chunk.get("word", "")) or ""),
    }
    if chunk.get("score") is not None:
        result["probability"] = chunk["score"]
    return result


def _group_words_into_segments(words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create auditable utterance segments when the pipeline emits words only."""

    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for word in words:
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            span = float(word["end"]) - float(current[0]["start"])
            if gap > 0.8 or span > 15.0:
                groups.append(current)
                current = []
        current.append(word)
        if str(word.get("text", "")).rstrip().endswith((".", "!", "?", "。", "！", "？")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    return [
        {
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": _join_word_tokens(group),
            "words": [dict(word) for word in group],
        }
        for group in groups
        if group
    ]


def _join_word_tokens(words: Sequence[Mapping[str, Any]]) -> str:
    # Whisper tokens commonly retain their leading spaces. If a tokenizer strips
    # them, joining with spaces still produces a readable audit transcript.
    tokens = [str(word.get("text", "")) for word in words]
    if any(token.startswith(" ") for token in tokens[1:]):
        return "".join(tokens).strip()
    return " ".join(token.strip() for token in tokens).strip()


def _json_safe_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("text", "word", "timestamp", "timestamps", "language", "score"):
        value = chunk.get(key)
        if value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        result[key] = value
    return result


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _dtype_for_device(torch: Any, device: str) -> Any:
    if device.startswith("cuda"):
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32
