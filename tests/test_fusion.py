from smartedit.fusion.scorer import fuse_edit_signals
from smartedit.schemas import (
    AnalysisReport,
    CategoryScores,
    EditSignalName,
    NarrationAnalysis,
    ObjectiveMeasurements,
    VideoMetadata,
    edit_signal_from_dict,
    to_dict,
)


def _qwen_signal(score: int, confidence: float = 0.8) -> dict:
    return {
        "score": score,
        "confidence": confidence,
        "explanation": "Test judgment.",
        "evidence": [],
    }


def test_fusion_returns_every_required_signal() -> None:
    qwen = {
        key: _qwen_signal(0)
        for key in (
            "story",
            "clear_start_middle_end",
            "consistent_theme",
            "text",
            "text_visibility",
            "visual_variety",
            "effects",
            "pace",
        )
    }
    qwen["category"] = {
        "personal": 0.1,
        "informational": 0.8,
        "promotional": 0.2,
    }
    fused = fuse_edit_signals(
        duration_seconds=30.0,
        has_audio=False,
        transition={"shot_count": 3, "cut_count": 2, "cuts_per_minute": 4.0},
        qwen=qwen,
    )
    assert set(fused["signals"]) == {
        "length",
        "pace",
        "visual_variety",
        "text",
        "text_visibility",
        "narration",
        "background_music",
        "catchy_music",
        "transitions",
        "effects",
        "story",
        "clear_start_middle_end",
        "consistent_theme",
    }
    assert fused["category"]["informational"] == 0.8


def test_conflicts_are_exported() -> None:
    fused = fuse_edit_signals(
        duration_seconds=30.0,
        has_audio=True,
        transition={
            "shot_count": 31,
            "cut_count": 30,
            "cuts_per_minute": 60,
            "average_shot_duration": 0.97,
        },
        narration={"speech_coverage": 0.8, "words_per_minute": 240},
        qwen={"pace": _qwen_signal(1, 0.9)},
    )
    assert fused["conflicts"]
    assert fused["signals"]["pace"]["confidence"] < 0.9


def test_non_millisecond_duration_fuses_into_a_valid_report() -> None:
    duration = 1.23456
    fused = fuse_edit_signals(duration_seconds=duration, has_audio=False)

    report = AnalysisReport(
        video=VideoMetadata(
            path="clip.mp4",
            duration_seconds=duration,
            fps=30.0,
            width=640,
            height=360,
            has_audio=False,
        ),
        objective_measurements=ObjectiveMeasurements(),
        signals={
            EditSignalName(name): edit_signal_from_dict(value)
            for name, value in fused["signals"].items()
        },
        category=CategoryScores(**fused["category"]),
    )

    assert all(
        observation.end <= duration
        for signal in report.signals.values()
        for observation in signal.evidence
    )


def test_no_audio_empty_narration_does_not_claim_whisper_provenance() -> None:
    narration = to_dict(NarrationAnalysis(model_name="openai/whisper-large-v3-turbo"))

    fused = fuse_edit_signals(
        duration_seconds=12.0,
        has_audio=False,
        narration=narration,
    )

    assert "whisper" not in fused["signals"]["narration"]["sources"]
    assert "whisper" not in fused["signals"]["pace"]["sources"]
