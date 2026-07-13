"""Simple data containers shared by the SmartEdit pipeline.

The machine-learning code produces these dataclasses.  Only model-generated
dictionaries need explicit parsing; normal internal code constructs the classes
directly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

Score = Literal[-1, 0, 1]
MusicEnergy = Literal["low", "medium", "high"]
InterferenceLevel = Literal["none", "low", "medium", "high"]
AudioQuality = Literal["poor", "acceptable", "good"]


def _check_score(value: int) -> None:
    if type(value) is not int or value not in {-1, 0, 1}:
        raise ValueError("score must be -1, 0, or 1")


def _check_probability(value: float, name: str = "confidence") -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between 0 and 1")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class TimestampedRange:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("timestamps must be finite")
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("timestamps must satisfy 0 <= start <= end")

    def validate_timestamps(self, duration_seconds: float) -> None:
        if self.end > duration_seconds + 1e-6:
            raise ValueError(f"timestamp {self.end} exceeds video duration {duration_seconds}")


@dataclass
class TimestampedObservation(TimestampedRange):
    observation: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.observation.strip():
            raise ValueError("observation cannot be empty")


@dataclass
class VideoMetadata:
    path: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    has_audio: bool
    format_name: str | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    rotation: int | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("video path cannot be empty")
        if self.duration_seconds <= 0 or self.fps <= 0:
            raise ValueError("video duration and fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video resolution must be positive")


class SamplingReason(StrEnum):
    UNIFORM = "uniform"
    SHOT_BOUNDARY = "shot_boundary"
    OTHER = "other"


@dataclass
class SampledFrame:
    timestamp_seconds: float
    path: str
    frame_index: int | None = None
    sampling_reason: SamplingReason = SamplingReason.UNIFORM

    def validate_timestamps(self, duration_seconds: float) -> None:
        if not 0.0 <= self.timestamp_seconds <= duration_seconds:
            raise ValueError(
                f"timestamp {self.timestamp_seconds} exceeds video duration {duration_seconds}"
            )


@dataclass
class ShotInterval(TimestampedRange):
    index: int
    start_frame: int | None = None
    end_frame: int | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame < self.start_frame
        ):
            raise ValueError("end_frame cannot precede start_frame")
        calculated = self.end - self.start
        if self.duration_seconds is None:
            self.duration_seconds = calculated


@dataclass
class TransitionAnalysis:
    shot_count: int
    cut_count: int
    average_shot_duration: float
    median_shot_duration: float
    minimum_shot_duration: float
    maximum_shot_duration: float
    cuts_per_minute: float
    shot_duration_variance: float
    cut_frames: list[int] = field(default_factory=list)
    cut_timestamps: list[float] = field(default_factory=list)
    shots: list[ShotInterval] = field(default_factory=list)
    evidence: list[TimestampedObservation] = field(default_factory=list)
    model_name: str = "transnet_v2"
    raw_output: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.cut_count != len(self.cut_timestamps):
            raise ValueError("cut_count must match cut_timestamps")
        if self.cut_frames and self.cut_count != len(self.cut_frames):
            raise ValueError("cut_count must match cut_frames")
        if self.shots and self.shot_count != len(self.shots):
            raise ValueError("shot_count must match shots")

    def validate_timestamps(self, duration_seconds: float) -> None:
        for timestamp in self.cut_timestamps:
            if not 0.0 <= timestamp <= duration_seconds:
                raise ValueError(f"cut timestamp {timestamp} exceeds video duration")
        for item in [*self.shots, *self.evidence]:
            item.validate_timestamps(duration_seconds)


@dataclass
class TranscriptWord(TimestampedRange):
    text: str
    probability: float | None = None


@dataclass
class TranscriptSegment(TimestampedRange):
    text: str
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass
class SilentGap(TimestampedRange):
    pass


@dataclass
class NarrationAnalysis:
    transcript: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    words: list[TranscriptWord] = field(default_factory=list)
    detected_language: str | None = None
    language_probability: float | None = None
    speech_duration: float = 0.0
    speech_coverage: float = 0.0
    words_per_minute: float = 0.0
    long_silent_gaps: list[SilentGap] = field(default_factory=list)
    narration_present: bool = False
    model_name: str = "openai/whisper-large-v3-turbo"
    raw_output: dict[str, Any] | None = None

    def validate_timestamps(self, duration_seconds: float) -> None:
        for item in [*self.segments, *self.words, *self.long_silent_gaps]:
            item.validate_timestamps(duration_seconds)


@dataclass
class LibrosaFeatures:
    rms_mean: float | None = None
    rms_std: float | None = None
    estimated_tempo_bpm: float | None = None
    onset_strength_mean: float | None = None
    spectral_centroid_hz: float | None = None
    zero_crossing_rate: float | None = None
    harmonic_percussive_ratio: float | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None


@dataclass
class AudioModelJudgment:
    background_music_present: bool | None = None
    music_energy: MusicEnergy | None = None
    rhythmic_strength: float | None = None
    catchiness_confidence: float | None = None
    speech_music_interference: InterferenceLevel | None = None
    environmental_sound: str | None = None
    audio_quality: AudioQuality | None = None
    background_music_score: Score | None = None
    catchy_music_score: Score | None = None
    explanation: str | None = None
    evidence: list[TimestampedObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.background_music_score is not None:
            _check_score(self.background_music_score)
        if self.catchy_music_score is not None:
            _check_score(self.catchy_music_score)
        if self.rhythmic_strength is not None:
            _check_probability(self.rhythmic_strength, "rhythmic_strength")
        if self.catchiness_confidence is not None:
            _check_probability(self.catchiness_confidence, "catchiness_confidence")


@dataclass
class AudioAnalysis:
    objective: LibrosaFeatures | None = None
    judgment: AudioModelJudgment | None = None
    adapter_used: str | None = None
    used_librosa_fallback: bool = False
    warnings: list[str] = field(default_factory=list)
    raw_output: dict[str, Any] | None = None


@dataclass
class CategoryScores:
    personal: float = 0.0
    informational: float = 0.0
    promotional: float = 0.0

    def __post_init__(self) -> None:
        _check_probability(self.personal, "personal")
        _check_probability(self.informational, "informational")
        _check_probability(self.promotional, "promotional")


class EditSignalName(StrEnum):
    LENGTH = "length"
    PACE = "pace"
    VISUAL_VARIETY = "visual_variety"
    TEXT = "text"
    TEXT_VISIBILITY = "text_visibility"
    NARRATION = "narration"
    BACKGROUND_MUSIC = "background_music"
    CATCHY_MUSIC = "catchy_music"
    TRANSITIONS = "transitions"
    EFFECTS = "effects"
    STORY = "story"
    CLEAR_START_MIDDLE_END = "clear_start_middle_end"
    CONSISTENT_THEME = "consistent_theme"


@dataclass
class EditSignal:
    score: Score
    confidence: float
    explanation: str
    evidence: list[TimestampedObservation] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_score(self.score)
        _check_probability(self.confidence)


@dataclass
class ObjectiveMeasurements:
    shot_count: int | None = None
    cut_count: int | None = None
    cut_timestamps: list[float] = field(default_factory=list)
    average_shot_duration: float | None = None
    median_shot_duration: float | None = None
    minimum_shot_duration: float | None = None
    maximum_shot_duration: float | None = None
    cuts_per_minute: float | None = None
    shot_duration_variance: float | None = None
    speech_duration: float | None = None
    speech_coverage: float | None = None
    words_per_minute: float | None = None
    long_silent_gaps: list[SilentGap] = field(default_factory=list)
    estimated_tempo_bpm: float | None = None
    rms_energy: float | None = None
    onset_strength: float | None = None
    spectral_centroid_hz: float | None = None
    zero_crossing_rate: float | None = None
    harmonic_percussive_ratio: float | None = None


@dataclass
class RawModelOutputs:
    transnet_v2: dict[str, Any] = field(default_factory=dict)
    whisper: dict[str, Any] = field(default_factory=dict)
    audio_model: dict[str, Any] = field(default_factory=dict)
    qwen3_vl: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    video: VideoMetadata
    objective_measurements: ObjectiveMeasurements
    signals: dict[EditSignalName, EditSignal]
    category: CategoryScores
    raw_model_outputs: RawModelOutputs = field(default_factory=RawModelOutputs)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        missing = set(EditSignalName) - set(self.signals)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"missing signals: {names}")
        duration = self.video.duration_seconds
        for timestamp in self.objective_measurements.cut_timestamps:
            if not 0.0 <= timestamp <= duration:
                raise ValueError(f"cut timestamp {timestamp} exceeds video duration {duration}")
        for signal in self.signals.values():
            for observation in signal.evidence:
                observation.validate_timestamps(duration)


def observation_from_dict(value: Mapping[str, Any]) -> TimestampedObservation:
    """Convert one already-validated model evidence item."""

    return TimestampedObservation(
        start=float(value["start"]),
        end=float(value["end"]),
        observation=str(value["observation"]),
    )


def audio_judgment_from_dict(value: Mapping[str, Any]) -> AudioModelJudgment:
    """Convert Audio Flamingo's validated JSON dictionary."""

    return AudioModelJudgment(
        background_music_present=value.get("background_music_present"),
        music_energy=value.get("music_energy"),
        rhythmic_strength=value.get("rhythmic_strength"),
        catchiness_confidence=value.get("catchiness_confidence"),
        speech_music_interference=value.get("speech_music_interference"),
        environmental_sound=value.get("environmental_sound"),
        audio_quality=value.get("audio_quality"),
        background_music_score=value.get("background_music_score"),
        catchy_music_score=value.get("catchy_music_score"),
        explanation=value.get("explanation"),
        evidence=[observation_from_dict(item) for item in value.get("evidence", [])],
    )


def edit_signal_from_dict(value: Mapping[str, Any]) -> EditSignal:
    return EditSignal(
        score=value["score"],
        confidence=float(value["confidence"]),
        explanation=str(value["explanation"]),
        evidence=[observation_from_dict(item) for item in value.get("evidence", [])],
        sources=[str(item) for item in value.get("sources", [])],
        conflicts=[str(item) for item in value.get("conflicts", [])],
    )


def to_dict(value: Any, *, exclude_none: bool = True) -> Any:
    """Recursively turn dataclasses into ordinary JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        result = {}
        for item in fields(value):
            nested = getattr(value, item.name)
            if exclude_none and nested is None:
                continue
            result[item.name] = to_dict(nested, exclude_none=exclude_none)
        return result
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, StrEnum) else key): to_dict(
                nested, exclude_none=exclude_none
            )
            for key, nested in value.items()
            if not (exclude_none and nested is None)
        }
    if isinstance(value, (list, tuple)):
        return [to_dict(item, exclude_none=exclude_none) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def to_json(value: Any, *, indent: int = 2) -> str:
    return json.dumps(to_dict(value), indent=indent, ensure_ascii=False, allow_nan=False)
