"""Readable, deterministic Edit Signal scoring rules.

These rules intentionally stay conservative. Objective measurements can flag
clear problems, but they cannot turn an element into a positive editing choice
without contextual semantic evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from .confidence import cap_fallback, clamp_confidence, penalize_conflict

VALID_SCORES = {-1, 0, 1}


@dataclass(frozen=True)
class RubricResult:
    score: int
    confidence: float
    explanation: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.score) is not int or self.score not in VALID_SCORES:
            raise ValueError(f"Invalid Edit Signal score: {self.score}")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(f"Invalid confidence: {self.confidence}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "confidence": clamp_confidence(self.confidence),
            "explanation": self.explanation,
            "evidence": self.evidence,
            "sources": _deduplicate(self.sources),
            "conflicts": self.conflicts,
        }


def neutral_unavailable(signal: str) -> RubricResult:
    explanation = (
        f"{signal.replace('_', ' ').title()} could not be assessed from available evidence."
    )
    return RubricResult(
        score=0,
        confidence=0.0,
        explanation=explanation,
        sources=[],
    )


def score_semantic_signal(
    signal: str, qwen: Mapping[str, Any] | None, duration: float
) -> RubricResult:
    item = _mapping(qwen).get(signal)
    if not isinstance(item, Mapping) or not _is_valid_score(item.get("score")):
        return neutral_unavailable(signal)
    confidence = _probability(item.get("confidence"), default=0.0)
    evidence = _safe_evidence(item.get("evidence", []), duration)
    return RubricResult(
        score=int(item["score"]),
        confidence=confidence,
        explanation=str(item.get("explanation", item.get("reason", "Qwen3-VL judgment."))),
        evidence=evidence,
        sources=["qwen3_vl"],
    )


def score_length(duration: float, category: Mapping[str, Any] | None) -> RubricResult:
    """Judge duration only against broad category-specific short-video ranges."""

    probabilities = {
        name: _probability(_mapping(category).get(name), 0.0)
        for name in ("personal", "informational", "promotional")
    }
    likely = [name for name, probability in probabilities.items() if probability >= 0.35]
    evidence = [_global_evidence(duration, f"Video duration is {duration:.2f} seconds.")]
    if not likely:
        return RubricResult(
            0,
            0.25,
            "Duration is measurable, but category confidence is too low for a "
            "contextual length judgment.",
            evidence,
            ["ffprobe"],
        )

    positive_ranges = {
        "personal": (5.0, 90.0),
        "informational": (15.0, 120.0),
        "promotional": (6.0, 60.0),
    }
    harmful_ranges = {
        "personal": (0.0, 180.0),
        "informational": (6.0, 240.0),
        "promotional": (3.0, 120.0),
    }
    category_names = ", ".join(likely)
    confidence = clamp_confidence(0.45 + 0.25 * max(probabilities.values()))
    if any(positive_ranges[name][0] <= duration <= positive_ranges[name][1] for name in likely):
        return RubricResult(
            1,
            confidence,
            "The duration falls within the broad short-form range for the likely "
            f"{category_names} purpose.",
            evidence,
            ["ffprobe", "qwen3_vl"],
        )
    if all(not (harmful_ranges[name][0] <= duration <= harmful_ranges[name][1]) for name in likely):
        return RubricResult(
            -1,
            confidence,
            f"The duration is an extreme outlier for the likely {category_names} purpose.",
            evidence,
            ["ffprobe", "qwen3_vl"],
        )
    return RubricResult(
        0,
        clamp_confidence(confidence - 0.1),
        f"The duration is plausible for {category_names}, but objective length "
        "alone is inconclusive.",
        evidence,
        ["ffprobe", "qwen3_vl"],
    )


def score_pace(
    duration: float,
    transition: Mapping[str, Any] | None,
    narration: Mapping[str, Any] | None,
    qwen: Mapping[str, Any] | None,
) -> RubricResult:
    transition = _mapping(transition)
    narration = _mapping(narration)
    semantic = score_semantic_signal("pace", qwen, duration)
    wpm = _number(narration.get("words_per_minute"), 0.0)
    coverage = _number(
        narration.get("speech_coverage", narration.get("speech_coverage_ratio")), 0.0
    )
    cuts_per_minute = _number(transition.get("cuts_per_minute"), 0.0)
    average_shot = _number(transition.get("average_shot_duration"), duration)
    shot_count = int(_number(transition.get("shot_count"), 0.0))

    warnings: list[str] = []
    if coverage >= 0.2 and wpm > 215:
        warnings.append(f"speaking rate is very fast ({wpm:.0f} WPM)")
    if coverage >= 0.25 and 0 < wpm < 65:
        warnings.append(f"speaking rate is unusually slow ({wpm:.0f} WPM)")
    if shot_count >= 6 and cuts_per_minute > 20:
        warnings.append(f"cut rate is very high ({cuts_per_minute:.1f}/minute)")
    if duration >= 30 and shot_count <= 2 and average_shot > 15:
        warnings.append(f"visual holds are long (average {average_shot:.1f} seconds)")

    sources = list(semantic.sources)
    if transition:
        sources.append("transnet_v2")
    if narration:
        sources.append("whisper")
    evidence = list(semantic.evidence)
    if transition:
        evidence.append(
            _global_evidence(
                duration,
                f"{shot_count} shots; {cuts_per_minute:.1f} cuts/minute; "
                f"average shot {average_shot:.2f} seconds.",
            )
        )
    if coverage > 0:
        evidence.append(
            _global_evidence(
                duration, f"Speech covers {coverage:.0%} at approximately {wpm:.0f} WPM."
            )
        )

    if semantic.confidence > 0.5:
        if semantic.score == 1 and len(warnings) >= 2:
            conflict = (
                "Contextual Qwen pace judgment conflicts with multiple objective pace warnings."
            )
            return RubricResult(
                0,
                penalize_conflict(semantic.confidence, 1),
                "Evidence conflicts: Qwen judged the pace supportive, while "
                f"{'; '.join(warnings)}.",
                evidence,
                sources,
                [conflict],
            )
        if semantic.score == 1 and warnings:
            conflict = "Contextual Qwen pace judgment conflicts with an objective pace warning."
            return RubricResult(
                1,
                penalize_conflict(semantic.confidence, 1),
                f"Qwen found the contextual pace effective, although {warnings[0]}.",
                evidence,
                sources,
                [conflict],
            )
        if semantic.score == 0 and len(warnings) >= 1:
            return RubricResult(
                -1,
                0.58,
                "Multiple objective indicators suggest poorly balanced pace: "
                f"{'; '.join(warnings)}.",
                evidence,
                sources,
            )
        return RubricResult(
            semantic.score,
            semantic.confidence,
            semantic.explanation,
            evidence,
            sources,
        )

    if len(warnings) >= 2:
        return RubricResult(
            -1,
            0.7,
            f"Multiple objective indicators suggest poorly balanced pace: {'; '.join(warnings)}.",
            evidence,
            sources,
        )
    return RubricResult(
        0,
        0.5 if transition or narration else 0.0,
        "Objective timing evidence alone does not establish whether the pace fits the content.",
        evidence,
        sources,
    )


def score_narration(
    duration: float,
    narration: Mapping[str, Any] | None,
    audio: Mapping[str, Any] | None,
    qwen: Mapping[str, Any] | None,
    category: Mapping[str, Any] | None,
    has_audio: bool,
) -> RubricResult:
    narration = _mapping(narration)
    category = _mapping(category)
    qwen = _mapping(qwen)
    informational = _probability(category.get("informational"), 0.0)
    text_score = _nested_score(qwen, "text")
    story_score = _nested_score(qwen, "story")
    pace_score = _nested_score(qwen, "pace")
    clearly_needed = (
        informational >= 0.75 and story_score == -1 and text_score != 1 and duration >= 8
    )
    if not has_audio:
        evidence = [_global_evidence(duration, "ffprobe found no audio stream in the video.")]
        if clearly_needed:
            return RubricResult(
                -1,
                0.7,
                "The video has no audio track, and strong contextual evidence "
                "indicates that spoken or written explanation is clearly needed.",
                evidence,
                ["ffprobe", "qwen3_vl"],
            )
        return RubricResult(
            0,
            0.95,
            "The video has no audio track, but the evidence does not show that "
            "narration is required.",
            evidence,
            ["ffprobe"],
        )
    if not narration:
        return neutral_unavailable("narration")
    audio_judgment = _audio_judgment(audio)
    semantic_audio_available = bool(audio_judgment) and not _is_audio_fallback(audio)
    coverage = _number(
        narration.get("speech_coverage", narration.get("speech_coverage_ratio")), 0.0
    )
    wpm = _number(narration.get("words_per_minute"), 0.0)
    speech_duration = _number(narration.get("speech_duration"), coverage * duration)
    evidence = (
        [
            _global_evidence(
                duration,
                f"Detected speech duration {speech_duration:.2f}s ({coverage:.0%} coverage), "
                f"approximately {wpm:.0f} WPM.",
            )
        ]
        if narration
        else []
    )
    sources = ["whisper"] if narration else (["ffprobe"] if not has_audio else [])
    if coverage < 0.02:
        if clearly_needed:
            return RubricResult(
                -1,
                0.56,
                "No narration was detected, and strong contextual evidence indicates "
                "the informational video remains unclear without text or spoken "
                "explanation.",
                evidence,
                sources + ["qwen3_vl"],
            )
        return RubricResult(
            0,
            0.72 if narration or not has_audio else 0.35,
            "No narration was detected, but the evidence does not show that narration is required.",
            evidence,
            sources,
        )

    interference = str(audio_judgment.get("speech_music_interference", "unknown")).lower()
    quality = str(audio_judgment.get("audio_quality", "unknown")).lower()
    if wpm > 190 or interference == "high" or quality == "poor":
        reasons = []
        if wpm > 190:
            reasons.append(f"speaking rate is very fast ({wpm:.0f} WPM)")
        if interference == "high" or interference == "medium":
            reasons.append("music masks speech")
        if quality == "poor":
            reasons.append("audio quality causes comprehension risk")
        if semantic_audio_available:
            sources.append("audio_flamingo_3")
        return RubricResult(
            -1,
            0.72,
            "Narration needs improvement because " + "; ".join(reasons) + ".",
            evidence,
            sources,
        )

    semantic_support = story_score == 1 and pace_score == 1 
    workable_rate = 85 <= wpm <= 180
    safe_audio = not semantic_audio_available or (
        interference in {"none", "low"} and quality != "poor"
    )
    if coverage >= 0.12 and workable_rate and safe_audio and semantic_support:
        confidence = 0.68 if semantic_audio_available else 0.56
        if semantic_audio_available:
            sources.append("audio_flamingo_3")
        if semantic_audio_available:
            explanation = (
                "Narration is sustained and delivered at a workable rate without "
                "evidence of serious masking; visual-semantic context supports its "
                "usefulness or timing."
            )
        else:
            explanation = (
                "Timestamped narration is sustained and delivered at a workable "
                "rate, and visual-semantic context supports its usefulness or "
                "timing; semantic audio masking was not fully assessed."
            )
        return RubricResult(
            1,
            confidence,
            explanation,
            evidence,
            sources + ["qwen3_vl"],
        )
    return RubricResult(
        0,
        0.5,
        "Narration is present but available evidence does not show a clear positive "
        "or harmful editing effect.",
        evidence,
        sources,
    )


def score_background_music(
    duration: float,
    audio: Mapping[str, Any] | None,
    has_audio: bool,
) -> RubricResult:
    if not has_audio:
        return RubricResult(
            0,
            0.9,
            "The video has no audio track; absence of music is not automatically harmful.",
            [],
            ["ffprobe"],
        )
    judgment = _audio_judgment(audio)
    if not judgment:
        return neutral_unavailable("background_music")
    fallback = _is_audio_fallback(audio)
    presence_value = judgment.get("background_music_present")
    presence = bool(presence_value)
    interference = str(judgment.get("speech_music_interference", "none")).lower()
    explicit_score = judgment.get("background_music_score")
    score = int(explicit_score) if _is_valid_score(explicit_score) else 0
    if fallback and presence_value is None:
        score = 0
        explanation = (
            "Librosa measurements cannot reliably establish whether music is "
            "background, foreground, speech, or environmental audio."
        )
    elif not presence:
        if score == -1 and judgment.get("evidence"):
            explanation = (
                "No background music was detected, and the audio model explicitly "
                "judged that its absence weakens the audible content. Visual fit "
                "was not independently confirmed."
            )
        else:
            score = 0
            explanation = (
                "No background music was detected, the evidence does not establish "
                "that music is needed."
            )
    elif interference == "high":
        score = -1
        explanation = "Background music is present and is judged to mask narration strongly."
    elif score == 1:
        explanation = (
            "Background music is judged to support the content without serious speech masking."
        )
    elif score == -1:
        explanation = "Background music is judged mismatched, distracting, or poorly balanced."
    else:
        explanation = (
            "Background music is present but has no clearly positive or harmful editing effect."
        )
    confidence = _probability(judgment.get("confidence"), 0.72 if not fallback else 0.4)
    if fallback:
        confidence = cap_fallback(confidence)
        explanation += (
            " This is a librosa-feature fallback judgment, not Audio Flamingo equivalence."
        )
    evidence = _safe_evidence(judgment.get("evidence", []), duration)
    sources = ["librosa_fallback" if fallback else "audio_flamingo_3"]
    return RubricResult(
        score,
        confidence,
        explanation,
        evidence,
        sources,
    )


def score_catchy_music(
    duration: float, audio: Mapping[str, Any] | None, has_audio: bool
) -> RubricResult:
    if not has_audio:
        return RubricResult(
            0,
            0.9,
            "No audio track is present; catchy music is not inherently required.",
            [],
            ["ffprobe"],
        )
    judgment = _audio_judgment(audio)
    if not judgment:
        return neutral_unavailable("catchy_music")
    fallback = _is_audio_fallback(audio)
    presence_value = judgment.get("background_music_present")
    if fallback and presence_value is None:
        score = 0
        explanation = (
            "Librosa rhythm features cannot determine subjective musical catchiness "
            "or even separate music from other audio."
        )
    elif not bool(presence_value):
        score = 0
        explanation = "No music was detected; catchiness is irrelevant rather than harmful."
    else:
        explicit = judgment.get("catchy_music_score")
        score = int(explicit) if _is_valid_score(explicit) else 0
        catchiness = _probability(judgment.get("catchiness_confidence"), 0.0)
        if not _is_valid_score(explicit) and catchiness >= 0.7:
            score = 1
        explanation = {
            1: "The music has a strong, memorable rhythmic character that supports the edit.",
            0: "Music is present, but catchiness has little established editing impact.",
            -1: "The music choice is judged to work against the edit's intended feel.",
        }[score]
    confidence = _probability(judgment.get("confidence"), 0.65 if not fallback else 0.35)
    if fallback:
        confidence = cap_fallback(confidence, 0.4)
        explanation += (
            " Librosa rhythm features cannot establish subjective catchiness "
            "equivalently to an audio-language model."
        )
    return RubricResult(
        score,
        confidence,
        explanation,
        _safe_evidence(judgment.get("evidence", []), duration),
        ["librosa_fallback" if fallback else "audio_flamingo_3"],
    )


def score_transitions(
    duration: float,
    transition: Mapping[str, Any] | None,
    qwen: Mapping[str, Any] | None,
) -> RubricResult:
    transition = _mapping(transition)
    effects = score_semantic_signal("effects", qwen, duration)
    text = " ".join(
        [effects.explanation] + [str(item.get("observation", "")) for item in effects.evidence]
    ).lower()
    explicitly_visual = any(
        term in text for term in ("transition", "dissolve", "wipe", "fade", "cut")
    )
    sources = (["transnet_v2"] if transition else []) + (
        ["qwen3_vl"] if effects.confidence > 0 else []
    )
    evidence: list[dict[str, Any]] = []
    if transition:
        cut_count = int(_number(transition.get("cut_count"), 0.0))
        evidence.append(
            _global_evidence(
                duration,
                f"TransNet-V2 detected {cut_count} cuts between shots.",
            )
        )
    if explicitly_visual and effects.score != 0:
        return RubricResult(
            effects.score,
            effects.confidence,
            effects.explanation,
            effects.evidence + evidence,
            sources,
        )
    return RubricResult(
        0,
        0.45 if transition and effects.confidence > 0 else (0.3 if transition else 0.0),
        "Cut boundaries were measured, but no reliable evidence shows that "
        "transition styling helps or harms, simple cuts may be sufficient.",
        evidence,
        sources,
    )


def _audio_judgment(audio: Mapping[str, Any] | None) -> Mapping[str, Any]:
    audio = _mapping(audio)
    for key in ("judgment", "model_judgment", "analysis"):
        value = audio.get(key)
        if isinstance(value, Mapping):
            return value
    # Some adapters return the constrained JSON directly.
    if "background_music_present" in audio:
        return audio
    raw_output = audio.get("raw_output")
    if isinstance(raw_output, Mapping):
        estimates = raw_output.get("fallback_estimates")
        if isinstance(estimates, Mapping):
            return estimates
    return {}


def _is_audio_fallback(audio: Mapping[str, Any] | None) -> bool:
    audio = _mapping(audio)
    explicitly_used = bool(audio.get("fallback_used", audio.get("used_librosa_fallback", False)))
    adapter = audio.get("adapter", audio.get("adapter_used", audio.get("backend", "")))
    return explicitly_used or "fallback" in str(adapter).lower()


def _nested_score(value: Mapping[str, Any], key: str) -> int | None:
    item = value.get(key)
    if isinstance(item, Mapping) and _is_valid_score(item.get("score")):
        return int(item["score"])
    return None


def _safe_evidence(value: Any, duration: float) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        start_value = item.get("start", item.get("start_seconds", 0.0))
        end_value = item.get("end", item.get("end_seconds", start_value))
        if (
            isinstance(start_value, bool)
            or isinstance(end_value, bool)
            or not isinstance(start_value, (int, float))
            or not isinstance(end_value, (int, float))
        ):
            continue
        start = float(start_value)
        end = float(end_value)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
            or end > duration + 1e-3
        ):
            continue
        rounded_start = min(duration, round(start, 3))
        rounded_end = min(duration, round(end, 3))
        observation = str(item.get("observation", item.get("text", ""))).strip()
        if not observation:
            continue
        result.append(
            {
                "start": rounded_start,
                "end": max(rounded_start, rounded_end),
                "observation": observation,
            }
        )
    return result


def _global_evidence(duration: float, observation: str) -> dict[str, Any]:
    valid_duration = max(0.0, duration)
    return {
        "start": 0.0,
        "end": min(valid_duration, round(valid_duration, 3)),
        "observation": observation,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _probability(value: Any, default: float = 0.0) -> float:
    return clamp_confidence(_number(value, default))


def _is_valid_score(value: Any) -> TypeGuard[int]:
    return type(value) is int and value in VALID_SCORES


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
