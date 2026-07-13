from __future__ import annotations

import pytest

from smartedit.models.qwen_vl_adapter import (
    VISUAL_SIGNAL_KEYS,
    QwenAdapterError,
    validate_qwen_output,
)


def qwen_payload(*, evidence_end: float) -> dict:
    payload = {
        key: {
            "score": 0,
            "confidence": 0.5,
            "explanation": "Editing-neutral evidence.",
            "evidence": [],
        }
        for key in VISUAL_SIGNAL_KEYS
    }
    payload["story"]["evidence"] = [
        {
            "start": 9.0,
            "end": evidence_end,
            "observation": "The final frame completes the sequence.",
        }
    ]
    payload["category"] = {
        "personal": 0.1,
        "informational": 0.8,
        "promotional": 0.0,
        "explanation": "The video explains a process.",
    }
    return payload


def test_qwen_tiny_endpoint_overshoot_is_normalized() -> None:
    duration = 10.0
    payload = qwen_payload(evidence_end=duration + 0.0005)

    validate_qwen_output(payload, duration)

    assert payload["story"]["evidence"][0]["end"] == duration


def test_qwen_gross_endpoint_overshoot_is_rejected() -> None:
    duration = 10.0
    payload = qwen_payload(evidence_end=duration + 0.01)

    with pytest.raises(QwenAdapterError, match="out-of-range timestamp"):
        validate_qwen_output(payload, duration)
