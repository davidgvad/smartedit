from __future__ import annotations

import json

import pytest

from smartedit.schemas import (
    AnalysisReport,
    AudioModelJudgment,
    CategoryScores,
    EditSignal,
    EditSignalName,
    ObjectiveMeasurements,
    RawModelOutputs,
    TimestampedObservation,
    VideoMetadata,
    to_dict,
)


def judgment(score: int = 0, confidence: float = 0.5) -> EditSignal:
    return EditSignal(
        score=score,
        confidence=confidence,
        explanation="Editing-neutral evidence.",
        evidence=[TimestampedObservation(start=0.0, end=0.5, observation="Observed frame")],
    )


def complete_signals() -> dict[EditSignalName, EditSignal]:
    return {name: judgment() for name in EditSignalName}


def video() -> VideoMetadata:
    return VideoMetadata(
        path="clip.mp4",
        duration_seconds=5.0,
        fps=29.97,
        width=1080,
        height=1920,
        has_audio=True,
    )


@pytest.mark.parametrize("invalid_score", [-2, 2, 1.0, "1", True, None])
def test_score_is_exact_integer_literal(invalid_score: object) -> None:
    with pytest.raises(ValueError):
        EditSignal(
            score=invalid_score,  # type: ignore[arg-type]
            confidence=0.5,
            explanation="invalid",
        )


@pytest.mark.parametrize("valid_score", [-1, 0, 1])
def test_valid_scores(valid_score: int) -> None:
    assert judgment(valid_score).score == valid_score


@pytest.mark.parametrize("invalid_confidence", [-0.01, 1.01])
def test_confidence_bounds(invalid_confidence: float) -> None:
    with pytest.raises(ValueError):
        EditSignal(
            score=0,
            confidence=invalid_confidence,
            explanation="invalid",
        )


@pytest.mark.parametrize("invalid_confidence", [True, "0.7"])
def test_confidence_rejects_booleans_and_numeric_strings(
    invalid_confidence: object,
) -> None:
    with pytest.raises(ValueError):
        EditSignal(
            score=0,
            confidence=invalid_confidence,  # type: ignore[arg-type]
            explanation="invalid",
        )


def test_audio_scores_are_strict_when_present() -> None:
    assert AudioModelJudgment(background_music_score=-1).background_music_score == -1
    assert AudioModelJudgment().background_music_score is None
    with pytest.raises(ValueError):
        AudioModelJudgment(catchy_music_score=0.0)  # type: ignore[arg-type]


def test_category_scores_are_independent() -> None:
    scores = CategoryScores(personal=0.7, informational=0.8, promotional=0.4)
    assert scores.personal + scores.informational + scores.promotional > 1.0


@pytest.mark.parametrize("invalid_confidence", [True, "0.7"])
def test_category_confidence_rejects_booleans_and_numeric_strings(
    invalid_confidence: object,
) -> None:
    with pytest.raises(ValueError):
        CategoryScores(personal=invalid_confidence)  # type: ignore[arg-type]


def test_final_report_requires_every_signal() -> None:
    signals = complete_signals()
    signals.pop(EditSignalName.LENGTH)
    with pytest.raises(ValueError, match="missing signals: length"):
        AnalysisReport(
            video=video(),
            objective_measurements=ObjectiveMeasurements(),
            signals=signals,
            category=CategoryScores(),
        )


def test_final_report_serializes_to_json_shape() -> None:
    report = AnalysisReport(
        video=video(),
        objective_measurements=ObjectiveMeasurements(
            shot_count=3,
            cut_count=2,
            cut_timestamps=[1.0, 3.0],
            average_shot_duration=5.0 / 3.0,
            speech_coverage=0.6,
            words_per_minute=142.0,
            estimated_tempo_bpm=118.0,
        ),
        signals=complete_signals(),
        category=CategoryScores(personal=0.1, informational=0.9, promotional=0.2),
        raw_model_outputs=RawModelOutputs(
            transnet_v2={"cuts": [1.0, 3.0]},
            whisper={"text": "example"},
        ),
        warnings=["Audio Flamingo unavailable; objective fallback used."],
    )

    payload = to_dict(report)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["video"]["path"] == "clip.mp4"
    assert decoded["signals"]["pace"]["score"] == 0
    assert decoded["category"]["informational"] == pytest.approx(0.9)
    assert decoded["raw_model_outputs"]["transnet_v2"]["cuts"] == [1.0, 3.0]
