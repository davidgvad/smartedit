"""Command-line interface for SmartEdit Edit Signal extraction."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from smartedit.config import SmartEditConfig
from smartedit.pipeline import SmartEditPipeline
from smartedit.schemas import to_json

LOGGER = logging.getLogger("smartedit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartedit",
        description="Extract objective evidence and Edit Signal scores from a short video.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    analyze = subcommands.add_parser("analyze", help="analyze one local video")
    analyze.add_argument("video", type=Path, help="input MP4, MOV, or WebM file")
    analyze.add_argument("--output", "-o", type=Path, required=True, help="result JSON path")
    analyze.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    analyze.add_argument("--qwen-model", default=None, help="Qwen3-VL model id or path")
    analyze.add_argument("--whisper-model", default=None, help="Whisper model id or local path")
    analyze.add_argument(
        "--audio-model", default=None, help="Audio Flamingo model id or local path"
    )
    analyze.add_argument("--cache-dir", type=Path, default=None)
    analyze.add_argument("--max-frames", type=int, default=None)
    analyze.add_argument("--debug", action="store_true", default=None)
    analyze.set_defaults(handler=_run_analyze)
    return parser


def _run_analyze(arguments: argparse.Namespace) -> int:
    overrides = {
        key: value
        for key, value in {
            "device": arguments.device,
            "qwen_model": arguments.qwen_model,
            "whisper_model": arguments.whisper_model,
            "audio_model": arguments.audio_model,
            "cache_dir": arguments.cache_dir,
            "max_frames": arguments.max_frames,
            "debug": arguments.debug,
        }.items()
        if value is not None
    }
    config = SmartEditConfig.from_env(overrides)
    _configure_logging(config.debug)
    if config.allow_model_downloads:
        LOGGER.warning(
            "Large model downloads are explicitly enabled. Qwen3-VL, Whisper "
            "large-v3-turbo, and Audio Flamingo 3 may require many gigabytes; "
            "review their licenses and available memory before continuing."
        )
    LOGGER.info("Using device: %s", config.resolved_device.value)
    report = SmartEditPipeline(config).analyze(arguments.video)
    _write_report(arguments.output, to_json(report))
    LOGGER.info("Analysis complete: %s", arguments.output.expanduser())
    return 0


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if not debug:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def _write_report(destination: Path, serialized: str) -> None:
    path = destination.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except KeyboardInterrupt:
        LOGGER.error("Analysis interrupted.")
        return 130
    except Exception as exc:
        debug = bool(getattr(arguments, "debug", False))
        if debug:
            LOGGER.exception("Analysis failed: %s", exc)
        else:
            LOGGER.error("Analysis failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
