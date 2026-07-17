from __future__ import annotations

from pathlib import Path

import pytest

from smartedit.ui_presenter import (
    category_rows,
    evidence_rows,
    format_confidence,
    grouped_signals,
    model_status_rows,
    persist_upload,
    safe_upload_filename,
    score_metadata,
    summarize_signals,
    validate_upload,
)


def test_upload_name_is_reduced_to_a_safe_basename() -> None:
    assert safe_upload_filename("../../My weird video.MP4") == "My_weird_video.mp4"
    assert safe_upload_filename(r"C:\Users\me\clip.webm") == "clip.webm"


def test_upload_validation_rejects_empty_and_unsupported_files() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_upload("clip.mp4", 0)
    with pytest.raises(ValueError, match="MP4"):
        validate_upload("notes.txt", 10)


def test_persist_upload_reuses_a_content_addressed_path(tmp_path: Path) -> None:
    data = b"small test video bytes"
    first = persist_upload("clip.mp4", data, tmp_path)
    second = persist_upload("clip.mp4", data, tmp_path)

    assert first == second
    assert first.read_bytes() == data
    assert first.parent.parent.name == "uploads"


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (1, "Supports the edit"),
        (0, "Neutral / not necessary"),
        (-1, "Needs improvement"),
        (None, "Unavailable / unknown"),
    ],
)
def test_score_metadata(score: int | None, label: str) -> None:
    assert score_metadata(score)["label"] == label


def test_confidence_formatting_keeps_zero_distinct_from_missing() -> None:
    assert format_confidence(0.0) == "0%"
    assert format_confidence(0.824) == "82%"
    assert format_confidence(None) == "Unavailable"


def test_summary_and_groups_use_canonical_signal_order() -> None:
    report = {
        "signals": {
            "pace": {"score": -1, "confidence": 0.8},
            "length": {"score": 1, "confidence": 0.8},
            "story": {"score": 0, "confidence": 0.8},
        }
    }

    assert summarize_signals(report) == {
        "positive": 1,
        "neutral": 1,
        "needs_improvement": 1,
        "unavailable": 0,
    }
    groups = grouped_signals(report)
    assert groups[0][0] == "Timing"
    assert [name for name, _ in groups[0][1]] == ["length", "pace"]


def test_zero_confidence_signal_is_summarized_as_unknown() -> None:
    report = {"signals": {"story": {"score": 0, "confidence": 0.0}}}

    assert summarize_signals(report) == {
        "positive": 0,
        "neutral": 0,
        "needs_improvement": 0,
        "unavailable": 1,
    }
    assert score_metadata(0, 0.0)["label"] == "Unavailable / unknown"


def test_categories_remain_independent_and_are_not_normalized() -> None:
    rows = category_rows({"category": {"personal": 0.8, "informational": 0.7, "promotional": 0.1}})

    assert [row["confidence"] for row in rows] == [0.8, 0.7, 0.1]
    assert sum(row["confidence"] for row in rows) == pytest.approx(1.6)


def test_evidence_is_flattened_without_losing_timestamps() -> None:
    report = {
        "signals": {
            "pace": {
                "sources": ["whisper", "transnet_v2"],
                "evidence": [{"start": 1.25, "end": 2.5, "observation": "A rushed cut."}],
            }
        }
    }

    assert evidence_rows(report) == [
        {
            "Signal": "Pace",
            "Start (s)": 1.25,
            "End (s)": 2.5,
            "Observation": "A rushed cut.",
            "Sources": "whisper, transnet_v2",
        }
    ]


def test_model_status_distinguishes_failure_skip_fallback_and_success() -> None:
    report = {
        "raw_model_outputs": {
            "transnet_v2": {"status": "failed", "error": "checkpoint missing"},
            "whisper": {"status": "skipped", "reason": "no audio"},
            "audio_model": {
                "adapter_used": "librosa_fallback",
                "used_librosa_fallback": True,
            },
            "qwen3_vl": {"parsed": {"story": {"score": 0}}},
        }
    }

    statuses = {row["Model"]: row["Status"] for row in model_status_rows(report)}
    assert statuses == {
        "TransNet-V2": "Failed",
        "Whisper": "Skipped",
        "Audio model": "Limited fallback",
        "Qwen3-VL": "Complete",
    }


def test_audio_model_status_reports_a_silent_video_as_skipped() -> None:
    report = {
        "raw_model_outputs": {
            "audio_model": {
                "adapter_used": "none",
                "used_librosa_fallback": False,
                "backend_output": {"status": "skipped", "reason": "no audio stream"},
            }
        }
    }

    audio_row = model_status_rows(report)[2]
    assert audio_row["Status"] == "Skipped"
    assert audio_row["Details"] == "no audio stream"
