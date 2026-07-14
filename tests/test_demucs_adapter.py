from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartedit.models.demucs_adapter import (
    DemucsAdapter,
    _chunk_starts,
    _find_bundle_checkpoint,
)


def test_chunk_starts_are_deterministic_and_cover_the_tail() -> None:
    assert _chunk_starts(30, 10, 2) == [0, 8, 16, 20]
    assert _chunk_starts(8, 10, 2) == [0]


@pytest.mark.parametrize(
    ("segment", "overlap"),
    [(0.0, 0.0), (10.0, -1.0), (10.0, 10.0), (10.0, 11.0)],
)
def test_invalid_chunk_settings_are_rejected(segment: float, overlap: float) -> None:
    with pytest.raises(ValueError):
        DemucsAdapter(segment_seconds=segment, overlap_seconds=overlap)


def test_cached_stems_are_reused_without_loading_optional_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"not decoded because the cache is valid")
    adapter = DemucsAdapter()
    cache_dir = tmp_path / "stems" / "HDEMUCS_HIGH_MUSDB_PLUS"
    cache_dir.mkdir(parents=True)
    vocals = cache_dir / "vocals.wav"
    accompaniment = cache_dir / "accompaniment.wav"
    vocals.write_bytes(b"cached vocal stem")
    accompaniment.write_bytes(b"cached accompaniment stem")
    stat = source.stat()
    manifest = {
        "version": 1,
        "input": {
            "path": str(source.resolve()),
            "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
        },
        "model_name": "HDEMUCS_HIGH_MUSDB_PLUS",
        "sample_rate_hz": 44_100,
        "segment_seconds": 10.0,
        "overlap_seconds": 1.0,
        "common_gain_preserved": True,
        "common_dc_offset_removed": True,
        "independent_stem_normalization": False,
    }
    (cache_dir / "separation.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = adapter.separate(source, output_dir=tmp_path / "stems")

    assert result.cache_reused is True
    assert result.vocals_path == vocals
    assert result.accompaniment_path == accompaniment
    assert result.raw_output["cache_reused"] is True


def test_checkpoint_lookup_uses_torchaudio_asset_path(tmp_path: Path) -> None:
    class Bundle:
        _path = "models/hdemucs_high_trained.pt"

    checkpoint = tmp_path / "torchaudio" / Bundle._path
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert _find_bundle_checkpoint(Bundle(), tmp_path) == checkpoint
