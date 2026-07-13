from __future__ import annotations

import pytest

from smartedit.schemas import (
    AnalysisReport,
    CategoryScores,
    EditSignal,
    EditSignalName,
    ObjectiveMeasurements,
    SampledFrame,
    TimestampedObservation,
    TranscriptSegment,
    TranscriptWord,
    VideoMetadata,
)


def complete_signals(*, evidence_end: float = 1.0) -> dict[EditSignalName, EditSignal]:
    return {
        name: EditSignal(
            score=0,
            confidence=0.0,
            explanation="No reliable assessment available.",
            evidence=[
                TimestampedObservation(
                    start=0.0,
                    end=evidence_end,
                    observation="Timestamped evidence",
                )
            ],
            sources=[],
        )
        for name in EditSignalName
    }


def metadata(duration: float = 4.0) -> VideoMetadata:
    return VideoMetadata(
        path="clip.mp4",
        duration_seconds=duration,
        fps=30.0,
        width=640,
        height=360,
        has_audio=False,
    )


def test_timestamp_range_rejects_negative_start() -> None:
    with pytest.raises(ValueError):
        TimestampedObservation(start=-0.1, end=1.0, observation="invalid")


def test_timestamp_range_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="0 <= start <= end"):
        TimestampedObservation(start=2.0, end=1.0, observation="invalid")


def test_explicit_validator_accepts_nested_timestamps() -> None:
    segment = TranscriptSegment(
        start=0.5,
        end=2.0,
        text="hello world",
        words=[
            TranscriptWord(start=0.5, end=1.0, text="hello"),
            TranscriptWord(start=1.2, end=2.0, text="world"),
        ],
    )
    segment.validate_timestamps(2.0)


def test_explicit_validator_rejects_sampled_frame_after_duration() -> None:
    frame = SampledFrame(timestamp_seconds=4.01, path="frame.jpg")
    with pytest.raises(ValueError, match="exceeds video duration"):
        frame.validate_timestamps(4.0)


def test_final_report_automatically_checks_objective_timestamps() -> None:
    with pytest.raises(ValueError, match="exceeds video duration"):
        AnalysisReport(
            video=metadata(),
            objective_measurements=ObjectiveMeasurements(cut_timestamps=[4.5]),
            signals=complete_signals(),
            category=CategoryScores(),
        )


def test_final_report_automatically_checks_signal_evidence() -> None:
    with pytest.raises(ValueError, match="exceeds video duration"):
        AnalysisReport(
            video=metadata(),
            objective_measurements=ObjectiveMeasurements(),
            signals=complete_signals(evidence_end=4.5),
            category=CategoryScores(),
        )


def test_timestamp_at_exact_duration_is_valid() -> None:
    report = AnalysisReport(
        video=metadata(),
        objective_measurements=ObjectiveMeasurements(cut_timestamps=[4.0]),
        signals=complete_signals(evidence_end=4.0),
        category=CategoryScores(),
    )
    assert report.objective_measurements.cut_timestamps == [4.0]
