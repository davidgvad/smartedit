"""Streamlit interface for the SmartEdit Edit Signal pipeline.

Run from the repository root with:

    python -m streamlit run smartedit/ui.py
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st

from smartedit.config import SmartEditConfig
from smartedit.pipeline import SmartEditPipeline
from smartedit.preprocessing.ffmpeg_utils import inspect_video
from smartedit.schemas import to_dict, to_json
from smartedit.ui_presenter import (
    category_rows,
    evidence_rows,
    format_confidence,
    grouped_signals,
    humanize_name,
    model_status_rows,
    persist_upload,
    safe_upload_filename,
    score_metadata,
    summarize_signals,
    validate_upload,
)

LOGGER = logging.getLogger(__name__)

_SOURCE_LABELS = {
    "ffprobe": "ffprobe",
    "transnet_v2": "TransNet-V2",
    "whisper": "Whisper",
    "audio_flamingo_3": "Audio Flamingo 3",
    "librosa_fallback": "librosa fallback",
    "qwen3_vl": "Qwen3-VL",
}

_MEASUREMENTS: tuple[tuple[str, str, str], ...] = (
    ("shot_count", "Shot count", "integer"),
    ("cut_count", "Cut count", "integer"),
    ("cut_timestamps", "Cut timestamps", "seconds_list"),
    ("average_shot_duration", "Average shot duration", "seconds"),
    ("median_shot_duration", "Median shot duration", "seconds"),
    ("minimum_shot_duration", "Minimum shot duration", "seconds"),
    ("maximum_shot_duration", "Maximum shot duration", "seconds"),
    ("cuts_per_minute", "Cuts per minute", "decimal"),
    ("shot_duration_variance", "Shot-duration variance", "decimal"),
    ("speech_duration", "Speech duration", "seconds"),
    ("speech_coverage", "Speech coverage", "percent"),
    ("words_per_minute", "Speaking rate", "wpm"),
    ("long_silent_gaps", "Long silent gaps", "ranges"),
    ("estimated_tempo_bpm", "Estimated tempo", "bpm"),
    ("rms_energy", "Mean RMS energy", "small_decimal"),
    ("onset_strength", "Mean onset strength", "decimal"),
    ("spectral_centroid_hz", "Spectral centroid", "hz"),
    ("zero_crossing_rate", "Zero-crossing rate", "small_decimal"),
    ("harmonic_percussive_ratio", "Harmonic/percussive ratio", "decimal"),
)


def main() -> None:
    st.set_page_config(
        page_title="SmartEdit — Edit Signal Analyzer",
        page_icon="✦",
        layout="wide",
    )

    defaults = _default_config()
    settings = _render_sidebar(defaults)

    st.title("SmartEdit — Edit Signal Analyzer")
    st.write(
        "Extract objective editing evidence and contextual Edit Signal scores from one short video."
    )
    st.caption(
        "This interface covers Edit Signal extraction only. It does not predict "
        "engagement or generate recommendations."
    )

    source_path, source_name, preview = _select_video_source(settings["cache_dir"])
    metadata: dict[str, Any] | None = None
    if source_path is not None:
        try:
            stat = source_path.stat()
            metadata = _inspect_for_ui(
                str(source_path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
            )
        except Exception as exc:
            st.error(f"This video could not be validated: {exc}")
        else:
            _render_video_preview(preview, metadata)

    try:
        config = _build_config(settings)
    except Exception as exc:
        config = None
        st.error(f"The analysis settings are invalid: {exc}")

    st.subheader("3. Run analysis")
    ready = source_path is not None and metadata is not None and config is not None
    analyze_clicked = st.button(
        "Analyze video",
        type="primary",
        disabled=not ready,
        width="stretch",
    )
    if not ready:
        st.caption("Choose a valid video to enable analysis.")

    current_key = (
        _analysis_key(source_path, config)
        if source_path is not None and config is not None
        else None
    )
    if analyze_clicked and source_path is not None and config is not None:
        _run_analysis(source_path, source_name or source_path.name, config, current_key)

    stored_report = st.session_state.get("smartedit_report")
    stored_key = st.session_state.get("smartedit_analysis_key")
    if isinstance(stored_report, Mapping) and stored_key == current_key:
        _render_report(
            dict(stored_report),
            str(st.session_state.get("smartedit_report_json", "{}")),
            str(st.session_state.get("smartedit_source_name", "video")),
        )
    elif isinstance(stored_report, Mapping) and source_path is not None:
        st.info("The selected video or settings changed. Press **Analyze video** for a new report.")


def _default_config() -> SmartEditConfig:
    try:
        return SmartEditConfig.from_env()
    except Exception as exc:
        st.warning(
            "Some SMARTEDIT_* environment settings were invalid, so safe UI "
            f"defaults are being used: {exc}"
        )
        return SmartEditConfig(allow_model_downloads=False)


def _render_sidebar(defaults: SmartEditConfig) -> dict[str, Any]:
    with st.sidebar:
        st.header("Analysis settings")
        device_options = ["auto", "cuda", "mps", "cpu"]
        default_device = str(defaults.device)
        device = st.selectbox(
            "Device",
            device_options,
            index=(device_options.index(default_device) if default_device in device_options else 0),
            help="Auto chooses CUDA, then Apple MPS, then CPU.",
        )
        max_frames = int(
            st.number_input(
                "Maximum Qwen frames",
                min_value=1,
                max_value=128,
                value=int(defaults.max_frames),
                step=1,
                help="This limits sampled visual frames, not the video duration.",
            )
        )
        allow_downloads = st.checkbox(
            "Allow large model downloads",
            value=bool(defaults.allow_model_downloads),
            help="Keep this off when checkpoints are already cached or downloads are not allowed.",
        )
        if allow_downloads:
            st.warning(
                "Qwen3-VL, Whisper, and Audio Flamingo checkpoints may download "
                "many gigabytes. Check model licenses and free disk/GPU memory."
            )

        with st.expander("Advanced model settings"):
            qwen_model = st.text_input("Qwen model", value=defaults.qwen_model)
            whisper_model = st.text_input("Whisper model", value=defaults.whisper_model)
            audio_model = st.text_input("Audio model", value=defaults.audio_model)
            transnet_checkpoint = st.text_input(
                "TransNet-V2 checkpoint",
                value=(
                    str(defaults.transnet_checkpoint)
                    if defaults.transnet_checkpoint is not None
                    else ""
                ),
                help="Path to the converted TransNet-V2 .pth checkpoint.",
            )
            cache_dir = st.text_input("Cache directory", value=str(defaults.cache_dir))
            debug = st.checkbox("Debug logging", value=bool(defaults.debug))

    return {
        "device": device,
        "max_frames": max_frames,
        "allow_model_downloads": allow_downloads,
        "qwen_model": qwen_model.strip(),
        "whisper_model": whisper_model.strip(),
        "audio_model": audio_model.strip(),
        "transnet_checkpoint": transnet_checkpoint.strip(),
        "cache_dir": cache_dir.strip(),
        "debug": debug,
    }


def _select_video_source(
    cache_dir: str,
) -> tuple[Path | None, str | None, bytes | str | None]:
    st.subheader("1. Choose a video")
    mode = st.radio(
        "Video source",
        ("Upload from this browser", "Use a path on the server"),
        horizontal=True,
    )

    if mode == "Upload from this browser":
        uploaded = st.file_uploader(
            "Upload an MP4, MOV, or WebM video",
            type=["mp4", "mov", "webm"],
            accept_multiple_files=False,
        )
        if uploaded is None:
            return None, None, None
        try:
            validate_upload(uploaded.name, int(uploaded.size))
            data = uploaded.getvalue()
            path = persist_upload(uploaded.name, data, cache_dir)
        except Exception as exc:
            st.error(f"The upload could not be prepared: {exc}")
            return None, None, None
        return path, uploaded.name, data

    entered = st.text_input(
        "Absolute video path on the server",
        placeholder="/home/user/videos/example.mp4",
    ).strip()
    if not entered:
        return None, None, None
    path = Path(entered).expanduser()
    return path, path.name, str(path)


@st.cache_data(show_spinner=False)
def _inspect_for_ui(path: str, size_bytes: int, modified_ns: int) -> dict[str, Any]:
    # Size and mtime are cache-key inputs, so replacing a file at the same path
    # causes ffprobe to run again.
    del size_bytes, modified_ns
    result = to_dict(inspect_video(path))
    return result if isinstance(result, dict) else {}


def _render_video_preview(preview: bytes | str | None, metadata: Mapping[str, Any]) -> None:
    st.subheader("2. Check the video")
    if preview is not None:
        st.video(preview)

    columns = st.columns(4)
    columns[0].metric("Duration", f"{float(metadata['duration_seconds']):.2f} s")
    columns[1].metric("Resolution", f"{metadata['width']} × {metadata['height']}")
    columns[2].metric("Frame rate", f"{float(metadata['fps']):.2f} FPS")
    columns[3].metric("Audio stream", "Present" if metadata.get("has_audio") else "None")


def _build_config(settings: Mapping[str, Any]) -> SmartEditConfig:
    checkpoint_text = str(settings["transnet_checkpoint"])
    overrides = {
        "device": settings["device"],
        "qwen_model": settings["qwen_model"],
        "whisper_model": settings["whisper_model"],
        "audio_model": settings["audio_model"],
        "transnet_checkpoint": Path(checkpoint_text) if checkpoint_text else None,
        "cache_dir": Path(str(settings["cache_dir"])),
        "max_frames": settings["max_frames"],
        "debug": settings["debug"],
        "allow_model_downloads": settings["allow_model_downloads"],
    }
    return SmartEditConfig.from_env(overrides)


def _analysis_key(
    source_path: Path | None,
    config: SmartEditConfig | None,
) -> tuple[str, ...] | None:
    if source_path is None or config is None:
        return None
    try:
        stat = source_path.expanduser().stat()
        source_version = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        source_version = "unavailable"
    return (
        str(source_path.expanduser().resolve()),
        source_version,
        str(config.device),
        config.qwen_model,
        config.whisper_model,
        config.audio_model,
        str(config.transnet_checkpoint or ""),
        str(config.cache_dir),
        str(config.max_frames),
        str(config.allow_model_downloads),
    )


def _run_analysis(
    source_path: Path,
    source_name: str,
    config: SmartEditConfig,
    analysis_key: tuple[str, ...] | None,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    progress = st.progress(0.0, text="Preparing analysis")
    stage_text = st.empty()

    def update_progress(stage: int, total: int, message: str) -> None:
        completed_before_this_stage = max(0.0, (stage - 1) / total)
        progress.progress(
            completed_before_this_stage,
            text=f"Stage {stage}/{total}: {message}",
        )
        stage_text.caption("Expensive model stages can take several minutes on their first run.")

    try:
        report = SmartEditPipeline(config).analyze(
            source_path,
            progress_callback=update_progress,
        )
    except Exception as exc:
        progress.empty()
        stage_text.empty()
        st.error(f"Analysis stopped before a report could be produced: {exc}")
        if config.debug:
            st.exception(exc)
        LOGGER.exception("SmartEdit UI analysis failed")
        return

    progress.progress(1.0, text="Analysis complete")
    stage_text.success("The report is ready below.")
    data = to_dict(report)
    st.session_state["smartedit_report"] = data
    st.session_state["smartedit_report_json"] = to_json(report)
    st.session_state["smartedit_analysis_key"] = analysis_key
    st.session_state["smartedit_source_name"] = source_name


def _render_report(report: dict[str, Any], serialized: str, source_name: str) -> None:
    st.divider()
    st.header("Analysis results")

    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        st.warning(
            f"The pipeline returned a partial report with {len(warnings)} warning(s). "
            "Completed model outputs are still preserved."
        )
        with st.expander("Show pipeline warnings", expanded=True):
            for warning in warnings:
                st.markdown(f"- {warning}")
    else:
        st.success("All configured analysis stages completed without warnings.")

    _render_summary(report)
    signal_tab, evidence_tab, model_tab, raw_tab = st.tabs(
        ("Edit Signals", "Evidence", "Model status", "Raw JSON")
    )
    with signal_tab:
        _render_signal_groups(report)
    with evidence_tab:
        _render_evidence(report)
    with model_tab:
        _render_model_status(report)
    with raw_tab:
        _render_raw_json(report, serialized, source_name)


def _render_summary(report: Mapping[str, Any]) -> None:
    st.subheader("Overview")
    summary = summarize_signals(report)
    columns = st.columns(4)
    columns[0].metric("Supports the edit (+1)", summary["positive"])
    columns[1].metric("Neutral / not necessary (0)", summary["neutral"])
    columns[2].metric("Needs improvement (-1)", summary["needs_improvement"])
    columns[3].metric("Unavailable / unknown", summary["unavailable"])

    st.markdown("#### Video category")
    category_columns = st.columns(3)
    for column, row in zip(category_columns, category_rows(report), strict=True):
        with column:
            st.metric(row["label"], row["confidence_label"])
            st.progress(float(row["confidence"]))
    st.caption(
        "Category values are independent confidences and do not need to add up to 100%. "
        "No overall quality score is calculated."
    )


def _render_signal_groups(report: Mapping[str, Any]) -> None:
    st.caption(
        "Confidence describes the model/rubric certainty for that signal; it is not "
        "a calibrated probability that the score is correct."
    )
    for group_name, signals in grouped_signals(report):
        st.subheader(group_name)
        columns = st.columns(2)
        for index, (name, signal) in enumerate(signals):
            with columns[index % 2]:
                _render_signal_card(name, signal)


def _render_signal_card(name: str, signal: Mapping[str, Any]) -> None:
    score = signal.get("score")
    metadata = score_metadata(score, signal.get("confidence"))
    confidence = _probability(signal.get("confidence"))
    with st.container(border=True):
        st.markdown(f"#### {humanize_name(name)}")
        numeric_score = f"{score:+d}" if type(score) is int and score in {-1, 0, 1} else "?"
        st.markdown(f"{metadata['icon']} **{metadata['label']}** (`{numeric_score}`)")
        st.progress(confidence, text=f"Confidence: {format_confidence(signal.get('confidence'))}")
        explanation = str(signal.get("explanation", "No explanation is available.")).strip()
        st.write(explanation or "No explanation is available.")

        sources = signal.get("sources", [])
        if isinstance(sources, list) and sources:
            labels = [_SOURCE_LABELS.get(str(item), humanize_name(str(item))) for item in sources]
            st.caption("Sources: " + " · ".join(labels))
        else:
            st.caption("Sources: unavailable")

        conflicts = signal.get("conflicts", [])
        if isinstance(conflicts, list):
            for conflict in conflicts:
                st.warning(f"Evidence conflict: {conflict}")

        evidence = signal.get("evidence", [])
        if isinstance(evidence, list) and evidence:
            with st.expander(f"Timestamped evidence ({len(evidence)})"):
                for item in evidence:
                    if not isinstance(item, Mapping):
                        continue
                    start = _number(item.get("start"), 0.0)
                    end = _number(item.get("end"), start)
                    observation = str(item.get("observation", ""))
                    st.markdown(f"- `{start:.2f}s–{end:.2f}s` — {observation}")
        else:
            st.caption("No timestamped evidence was returned for this signal.")


def _render_evidence(report: Mapping[str, Any]) -> None:
    st.subheader("Objective measurements")
    objective = report.get("objective_measurements", {})
    if not isinstance(objective, Mapping):
        objective = {}
    measurement_rows = [
        {"Measurement": label, "Value": _format_measurement(objective.get(key), kind)}
        for key, label, kind in _MEASUREMENTS
    ]
    st.dataframe(measurement_rows, hide_index=True, width="stretch")

    st.subheader("Timestamped signal evidence")
    rows = evidence_rows(report)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No timestamped Edit Signal evidence is available in this report.")


def _render_model_status(report: Mapping[str, Any]) -> None:
    st.subheader("Model and fallback status")
    st.dataframe(model_status_rows(report), hide_index=True, width="stretch")
    st.caption(
        "A completed pipeline may still contain failed or skipped individual stages. "
        "The librosa fallback is a limited objective proxy, not Audio Flamingo equivalence."
    )


def _render_raw_json(report: Mapping[str, Any], serialized: str, source_name: str) -> None:
    safe_stem = Path(safe_upload_filename(source_name)).stem or "smartedit"
    st.download_button(
        "Download JSON report",
        data=serialized + "\n",
        file_name=f"{safe_stem}_smartedit.json",
        mime="application/json",
        type="primary",
    )
    st.caption(
        "The raw report can contain local cache paths and full model outputs. "
        "Review it before sharing outside your environment."
    )
    if st.checkbox("Render the full JSON in the browser"):
        st.json(report, expanded=False)


def _format_measurement(value: Any, kind: str) -> str:
    if value is None:
        return "Unavailable"
    if kind == "integer":
        return str(int(_number(value, 0.0)))
    if kind == "seconds":
        return f"{_number(value, 0.0):.2f} s"
    if kind == "percent":
        return f"{_probability(value):.1%}"
    if kind == "wpm":
        return f"{_number(value, 0.0):.1f} WPM"
    if kind == "bpm":
        return f"{_number(value, 0.0):.1f} BPM"
    if kind == "hz":
        return f"{_number(value, 0.0):.1f} Hz"
    if kind == "small_decimal":
        return f"{_number(value, 0.0):.4f}"
    if kind == "decimal":
        return f"{_number(value, 0.0):.2f}"
    if kind == "seconds_list":
        if not isinstance(value, list):
            return "Unavailable"
        return ", ".join(f"{_number(item, 0.0):.2f} s" for item in value) or "None"
    if kind == "ranges":
        if not isinstance(value, list):
            return "Unavailable"
        ranges = []
        for item in value:
            if isinstance(item, Mapping):
                start = _number(item.get("start"), 0.0)
                end = _number(item.get("end"), start)
                ranges.append(f"{start:.2f}–{end:.2f} s")
        return ", ".join(ranges) or "None"
    return json.dumps(value, ensure_ascii=False)


def _probability(value: Any) -> float:
    return max(0.0, min(1.0, _number(value, 0.0)))


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


if __name__ == "__main__":
    main()
