"""End-to-end objective-evidence to Edit Signal extraction pipeline."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, cast

from smartedit.config import SmartEditConfig
from smartedit.extraction.audio_features import extract_audio_features
from smartedit.extraction.narration_features import extract_narration_features
from smartedit.extraction.transition_features import analyze_transitions
from smartedit.extraction.visual_features import build_visual_context
from smartedit.fusion.scorer import fuse_edit_signals
from smartedit.models.audio_flamingo_adapter import AudioFlamingoAdapter
from smartedit.models.qwen_vl_adapter import QwenVLAdapter
from smartedit.models.transnet_adapter import DeviceName, TransNetV2Adapter
from smartedit.models.whisper_adapter import WhisperAdapter
from smartedit.preprocessing.audio_extractor import extract_audio_if_present
from smartedit.preprocessing.ffmpeg_utils import (
    inspect_video,
)
from smartedit.preprocessing.frame_sampler import sample_frames
from smartedit.schemas import (
    AnalysisReport,
    AudioAnalysis,
    CategoryScores,
    EditSignalName,
    NarrationAnalysis,
    ObjectiveMeasurements,
    RawModelOutputs,
    TransitionAnalysis,
    edit_signal_from_dict,
    to_dict,
)

LOGGER = logging.getLogger(__name__)


class SmartEditPipeline:
    """Run all independent evidence stages and conservative rule-based fusion."""

    def __init__(self, config: SmartEditConfig | None = None) -> None:
        self.config = config or SmartEditConfig.from_env()
        if self.config.allow_model_downloads:
            LOGGER.warning(
                "Large checkpoint downloads are enabled explicitly; review model "
                "sizes, licenses, and available memory before inference."
            )

    def analyze(self, video_path: str | Path) -> AnalysisReport:
        """Analyze one local short video, preserving every successful stage."""

        warnings: list[str] = []
        LOGGER.info("[1/6] Inspecting video metadata")
        video = inspect_video(video_path)
        source = Path(video.path)
        device = self.config.resolved_device.value

        cache_root = self.config.cache_dir.expanduser().resolve()
        artifact_root = cache_root / "artifacts"
        model_cache = cache_root / "models"
        artifact_root.mkdir(parents=True, exist_ok=True)
        model_cache.mkdir(parents=True, exist_ok=True)
        audio_path: Path | None = None
        try:
            audio_path = extract_audio_if_present(
                source,
                metadata=video,
                cache_dir=artifact_root,
            )
        except Exception as exc:
            warning = f"Audio extraction failed; speech/audio stages were skipped: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)

        LOGGER.info("[2/6] Detecting shot boundaries")
        transition, transnet_raw = self._transition_stage(source, video, device, warnings)

        try:
            frames = sample_frames(
                source,
                cache_dir=artifact_root,
                metadata=video,
                max_frames=self.config.max_frames,
                cut_timestamps=transition.cut_timestamps if transition else (),
            )
        except Exception as exc:
            warning = f"Frame sampling failed; Qwen3-VL was skipped: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)
            frames = []

        LOGGER.info("[3/6] Transcribing narration")
        narration, whisper_raw = self._narration_stage(
            audio_path,
            video.duration_seconds,
            video.has_audio,
            model_cache,
            device,
            warnings,
        )
        _release_accelerator_memory()

        LOGGER.info("[4/6] Analyzing audio")
        audio = self._audio_stage(
            audio_path,
            video.duration_seconds,
            video.has_audio,
            model_cache,
            device,
            warnings,
        )
        audio_raw = {
            "adapter_used": audio.adapter_used,
            "used_librosa_fallback": audio.used_librosa_fallback,
            "warnings": audio.warnings,
            "backend_output": audio.raw_output or {},
        }
        warnings.extend(audio.warnings)
        _release_accelerator_memory()

        LOGGER.info("[5/6] Evaluating visual-semantic editing characteristics")
        qwen, qwen_raw = self._qwen_stage(
            frames=frames,
            duration_seconds=video.duration_seconds,
            transition=transition,
            narration=narration,
            audio=audio,
            model_cache=model_cache,
            device=device,
            warnings=warnings,
        )
        _release_accelerator_memory()

        LOGGER.info("[6/6] Fusing objective evidence and model judgments")
        transition_data = _model_dict(transition)
        # A zero-valued NarrationAnalysis for a genuinely silent video records an
        # objective zero, not a Whisper run. Keep it out of model provenance.
        narration_data = _model_dict(narration) if video.has_audio else {}
        audio_data = _model_dict(audio)
        qwen_data = _model_dict(qwen)
        fused = fuse_edit_signals(
            duration_seconds=video.duration_seconds,
            has_audio=video.has_audio,
            transition=transition_data,
            narration=narration_data,
            audio=audio_data,
            qwen=qwen_data,
        )
        warnings.extend(f"Evidence conflict: {item}" for item in fused["conflicts"])

        report = AnalysisReport(
            video=video,
            objective_measurements=_objective_measurements(
                transition=transition,
                narration=narration,
                audio=audio,
            ),
            signals={
                EditSignalName(name): edit_signal_from_dict(value)
                for name, value in fused["signals"].items()
            },
            category=CategoryScores(**fused["category"]),
            raw_model_outputs=RawModelOutputs(
                transnet_v2=transnet_raw,
                whisper=whisper_raw,
                audio_model=audio_raw,
                qwen3_vl=qwen_raw,
            ),
            warnings=_deduplicate(warnings),
        )
        return report

    def _transition_stage(
        self,
        source: Path,
        video: Any,
        device: str,
        warnings: list[str],
    ) -> tuple[TransitionAnalysis | None, dict[str, Any]]:
        adapter: TransNetV2Adapter | None = None
        try:
            adapter = TransNetV2Adapter(
                checkpoint_path=self.config.transnet_checkpoint,
                device=cast(DeviceName, device),
                threshold=0.5,
            )
            result = analyze_transitions(str(source), metadata=video, adapter=adapter)
            raw = result.raw_output or {}
            return result, raw
        except Exception as exc:
            warning = f"TransNet-V2 unavailable; transition measurements are missing: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)
            return None, {"status": "failed", "error": str(exc)}
        finally:
            if adapter is not None:
                adapter.unload()

    def _narration_stage(
        self,
        audio_path: Path | None,
        duration_seconds: float,
        has_audio_stream: bool,
        model_cache: Path,
        device: str,
        warnings: list[str],
    ) -> tuple[NarrationAnalysis | None, dict[str, Any]]:
        if audio_path is None:
            if has_audio_stream:
                return None, {
                    "status": "skipped",
                    "reason": "audio stream exists but extraction failed",
                }
            result = extract_narration_features(
                {}, duration_seconds, model_name=self.config.whisper_model
            )
            return result, {"status": "skipped", "reason": "no extracted audio"}

        adapter = WhisperAdapter(
            model_name=self.config.whisper_model,
            device=device,
            cache_dir=model_cache,
            allow_download=self.config.allow_model_downloads,
        )
        try:
            result = adapter.analyze(audio_path, duration_seconds=duration_seconds)
            raw = adapter.last_raw_output
            return result, raw
        except Exception as exc:
            warning = f"Whisper unavailable; narration measurements are missing: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)
            return None, {"status": "failed", "error": str(exc)}
        finally:
            del adapter

    def _audio_stage(
        self,
        audio_path: Path | None,
        duration_seconds: float,
        has_audio_stream: bool,
        model_cache: Path,
        device: str,
        warnings: list[str],
    ) -> AudioAnalysis:
        if audio_path is None and has_audio_stream:
            warning = "Audio stream exists, but extraction failed; audio content is unknown."
            return AudioAnalysis(
                adapter_used="none",
                warnings=[warning],
                raw_output={
                    "status": "skipped",
                    "reason": "audio stream exists but extraction failed",
                },
            )
        adapter = AudioFlamingoAdapter(
            model_name=self.config.audio_model,
            device=device,
            cache_dir=model_cache,
            allow_download=self.config.allow_model_downloads,
        )
        try:
            result = extract_audio_features(
                audio_path,
                duration_seconds,
                audio_adapter=adapter,
                use_librosa_fallback=True,
            )
            return result
        except Exception as exc:
            warning = f"Audio analysis failed completely: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)
            return AudioAnalysis(
                adapter_used="none",
                warnings=[warning],
                raw_output={"status": "failed", "error": str(exc)},
            )
        finally:
            del adapter

    def _qwen_stage(
        self,
        *,
        frames: list[Any],
        duration_seconds: float,
        transition: TransitionAnalysis | None,
        narration: NarrationAnalysis | None,
        audio: AudioAnalysis,
        model_cache: Path,
        device: str,
        warnings: list[str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if not frames:
            return None, {"status": "skipped", "reason": "no sampled frames"}
        context = build_visual_context(
            duration_seconds=duration_seconds,
            frame_samples=frames,
            transition=_model_dict(transition),
            narration=_model_dict(narration),
            audio=_model_dict(audio),
        )
        adapter = QwenVLAdapter(
            model_name=self.config.qwen_model,
            device=device,
            cache_dir=model_cache,
            allow_download=self.config.allow_model_downloads,
        )
        try:
            parsed = adapter.analyze(
                frames,
                context,
                duration_seconds,
            )
            raw = adapter.last_raw_output
            return parsed, raw
        except Exception as exc:
            warning = f"Qwen3-VL unavailable; visual-semantic signals are neutral/unknown: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)
            raw = dict(adapter.last_raw_output)
            raw.update({"status": "failed", "error": str(exc)})
            return None, raw
        finally:
            del adapter

def _objective_measurements(
    *,
    transition: TransitionAnalysis | None,
    narration: NarrationAnalysis | None,
    audio: AudioAnalysis,
) -> ObjectiveMeasurements:
    objective = audio.objective
    return ObjectiveMeasurements(
        shot_count=transition.shot_count if transition else None,
        cut_count=transition.cut_count if transition else None,
        cut_timestamps=transition.cut_timestamps if transition else [],
        average_shot_duration=transition.average_shot_duration if transition else None,
        median_shot_duration=transition.median_shot_duration if transition else None,
        minimum_shot_duration=transition.minimum_shot_duration if transition else None,
        maximum_shot_duration=transition.maximum_shot_duration if transition else None,
        cuts_per_minute=transition.cuts_per_minute if transition else None,
        shot_duration_variance=transition.shot_duration_variance if transition else None,
        speech_duration=narration.speech_duration if narration else None,
        speech_coverage=narration.speech_coverage if narration else None,
        words_per_minute=narration.words_per_minute if narration else None,
        long_silent_gaps=narration.long_silent_gaps if narration else [],
        estimated_tempo_bpm=objective.estimated_tempo_bpm if objective else None,
        rms_energy=objective.rms_mean if objective else None,
        onset_strength=objective.onset_strength_mean if objective else None,
        spectral_centroid_hz=objective.spectral_centroid_hz if objective else None,
        zero_crossing_rate=objective.zero_crossing_rate if objective else None,
        harmonic_percussive_ratio=(objective.harmonic_percussive_ratio if objective else None),
    )


def _model_dict(value: Any | None) -> dict[str, Any]:
    result = to_dict(value) if value is not None else {}
    return result if isinstance(result, dict) else {}


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except (ImportError, RuntimeError):
        return


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
