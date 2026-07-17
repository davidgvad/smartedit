"""Small, model-free helpers used by the Streamlit interface.

Keeping these functions separate makes the UI easy to test without loading
PyTorch, Transformers, or Streamlit.
"""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})
DEFAULT_MAX_UPLOAD_BYTES = 500 * 1024 * 1024

SIGNAL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Timing", ("length", "pace")),
    (
        "Visual editing",
        ("visual_variety", "text", "text_visibility", "transitions", "effects"),
    ),
    ("Audio and delivery", ("narration", "background_music", "catchy_music")),
    (
        "Structure",
        ("story", "clear_start_middle_end", "consistent_theme"),
    ),
)

_SCORE_METADATA = {
    1: {"label": "Supports the edit", "icon": "🟢", "summary_key": "positive"},
    0: {"label": "Neutral / not necessary", "icon": "⚪", "summary_key": "neutral"},
    -1: {
        "label": "Needs improvement",
        "icon": "🔴",
        "summary_key": "needs_improvement",
    },
}


def safe_upload_filename(filename: str) -> str:
    """Return a basename containing only simple, portable characters."""

    basename = str(filename).replace("\\", "/").rsplit("/", 1)[-1].strip()
    path = Path(basename)
    suffix = path.suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return f"{stem or 'video'}{suffix}"


def validate_upload(
    filename: str,
    size_bytes: int,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> None:
    """Reject empty, oversized, or unsupported browser uploads."""

    suffix = Path(safe_upload_filename(filename)).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Upload an MP4, MOV, or WebM video ({supported}).")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("The uploaded video is empty.")
    if size_bytes > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(f"The uploaded video exceeds the {limit_mb:g} MB UI limit.")


def persist_upload(filename: str, data: bytes, cache_dir: str | Path) -> Path:
    """Save uploaded bytes once at a deterministic, content-addressed path."""

    validate_upload(filename, len(data))
    digest = hashlib.sha256(data).hexdigest()
    destination = (
        Path(cache_dir).expanduser().resolve() / "uploads" / digest / safe_upload_filename(filename)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == len(data):
        return destination

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def humanize_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def score_metadata(score: Any, confidence: Any = None) -> dict[str, str]:
    """Return the visible label and icon for one valid Edit Signal score."""

    if type(score) is not int or score not in _SCORE_METADATA:
        return _unavailable_score_metadata()
    if confidence is not None and _probability_or_none(confidence) == 0.0:
        return _unavailable_score_metadata()
    return dict(_SCORE_METADATA[score])


def format_confidence(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Unavailable"
    number = float(value)
    if not math.isfinite(number):
        return "Unavailable"
    return f"{max(0.0, min(1.0, number)):.0%}"


def summarize_signals(report_or_signals: Mapping[str, Any]) -> dict[str, int]:
    signals = _signals_from(report_or_signals)
    summary = {
        "positive": 0,
        "neutral": 0,
        "needs_improvement": 0,
        "unavailable": 0,
    }
    for value in signals.values():
        if not isinstance(value, Mapping):
            continue
        metadata = score_metadata(value.get("score"), value.get("confidence"))
        key = metadata["summary_key"]
        if key in summary:
            summary[key] += 1
    return summary


def grouped_signals(
    report_or_signals: Mapping[str, Any],
) -> list[tuple[str, list[tuple[str, Mapping[str, Any]]]]]:
    """Return signals in the fixed, human-friendly UI order."""

    signals = _signals_from(report_or_signals)
    result: list[tuple[str, list[tuple[str, Mapping[str, Any]]]]] = []
    for group_name, names in SIGNAL_GROUPS:
        items = [
            (name, value) for name in names if isinstance((value := signals.get(name)), Mapping)
        ]
        result.append((group_name, items))
    return result


def category_rows(report_or_category: Mapping[str, Any]) -> list[dict[str, Any]]:
    category = report_or_category.get("category", report_or_category)
    if not isinstance(category, Mapping):
        category = {}
    rows = []
    for name in ("personal", "informational", "promotional"):
        value = _probability(category.get(name))
        rows.append(
            {
                "category": name,
                "label": humanize_name(name),
                "confidence": value,
                "confidence_label": format_confidence(value),
            }
        )
    return rows


def evidence_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_name, signal in _signals_from(report).items():
        if not isinstance(signal, Mapping):
            continue
        sources = signal.get("sources", [])
        source_text = ", ".join(str(item) for item in sources) if isinstance(sources, list) else ""
        evidence = signal.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    "Signal": humanize_name(str(signal_name)),
                    "Start (s)": item.get("start"),
                    "End (s)": item.get("end"),
                    "Observation": str(item.get("observation", "")),
                    "Sources": source_text,
                }
            )
    return rows


def model_status_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = report.get("raw_model_outputs", {})
    if not isinstance(raw, Mapping):
        raw = {}

    rows = [
        _ordinary_model_status("TransNet-V2", raw.get("transnet_v2")),
        _ordinary_model_status("Whisper", raw.get("whisper")),
        _audio_model_status(raw.get("audio_model")),
        _ordinary_model_status("Qwen3-VL", raw.get("qwen3_vl")),
    ]
    return rows


def _signals_from(value: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = value.get("signals")
    return signals if isinstance(signals, Mapping) else value


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _probability_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return max(0.0, min(1.0, number))


def _unavailable_score_metadata() -> dict[str, str]:
    return {
        "label": "Unavailable / unknown",
        "icon": "⚠️",
        "summary_key": "unavailable",
    }


def _ordinary_model_status(name: str, value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        return {"Model": name, "Status": "Unavailable", "Details": "No output recorded."}
    raw_status = str(value.get("status", "")).lower()
    if raw_status == "failed":
        return {
            "Model": name,
            "Status": "Failed",
            "Details": str(value.get("error", "The model stage failed.")),
        }
    if raw_status == "skipped":
        return {
            "Model": name,
            "Status": "Skipped",
            "Details": str(value.get("reason", "The model stage was skipped.")),
        }
    return {"Model": name, "Status": "Complete", "Details": "Output is available."}


def _audio_model_status(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        return {
            "Model": "Audio model",
            "Status": "Unavailable",
            "Details": "No output recorded.",
        }
    adapter = str(value.get("adapter_used", "audio model"))
    fallback = bool(value.get("used_librosa_fallback", False)) or "fallback" in adapter.lower()
    if fallback:
        return {
            "Model": "Audio model",
            "Status": "Limited fallback",
            "Details": (
                "librosa objective features were used; this is not Audio Flamingo equivalence."
            ),
        }
    backend = value.get("backend_output", {})
    if isinstance(backend, Mapping):
        backend_status = str(backend.get("status", "")).lower()
        if backend_status == "failed":
            return {
                "Model": "Audio model",
                "Status": "Failed",
                "Details": str(backend.get("error", "The contextual audio model failed.")),
            }
        if backend_status == "skipped":
            return {
                "Model": "Audio model",
                "Status": "Skipped",
                "Details": str(backend.get("reason", "The audio stage was skipped.")),
            }
        if backend_status == "objective_only":
            return {
                "Model": "Audio model",
                "Status": "Objective only",
                "Details": "No contextual audio-model judgment is available.",
            }
    return {
        "Model": "Audio model",
        "Status": "Complete",
        "Details": f"{adapter} output is available.",
    }


__all__ = [
    "DEFAULT_MAX_UPLOAD_BYTES",
    "SIGNAL_GROUPS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "category_rows",
    "evidence_rows",
    "format_confidence",
    "grouped_signals",
    "humanize_name",
    "model_status_rows",
    "persist_upload",
    "safe_upload_filename",
    "score_metadata",
    "summarize_signals",
    "validate_upload",
]
