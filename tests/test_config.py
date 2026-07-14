from smartedit.cli import build_parser
from smartedit.config import SmartEditConfig


def test_demucs_masking_is_opt_in_by_default() -> None:
    config = SmartEditConfig()
    assert config.enable_demucs is False
    assert config.demucs_model == "HDEMUCS_HIGH_MUSDB_PLUS"


def test_demucs_settings_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SMARTEDIT_ENABLE_DEMUCS", "true")
    monkeypatch.setenv("SMARTEDIT_DEMUCS_MODEL", "HDEMUCS_HIGH_MUSDB_PLUS")

    config = SmartEditConfig.from_env()

    assert config.enable_demucs is True
    assert config.demucs_model == "HDEMUCS_HIGH_MUSDB_PLUS"


def test_cli_exposes_demucs_opt_in() -> None:
    arguments = build_parser().parse_args(
        [
            "analyze",
            "clip.mp4",
            "--output",
            "result.json",
            "--enable-demucs",
        ]
    )
    assert arguments.enable_demucs is True
