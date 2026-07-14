"""Optional Hybrid Demucs source separation through the official TorchAudio bundle.

This adapter separates a full-band stereo mixture into a vocal stem and one
combined accompaniment stem.  The stems retain one shared gain transform: they
are never normalized independently, because their relative level is the useful
measurement for speech/music masking.
"""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartedit.models.base import AdapterError, BaseModelAdapter

LOGGER = logging.getLogger(__name__)
DEFAULT_DEMUCS_BUNDLE = "HDEMUCS_HIGH_MUSDB_PLUS"
_EXPECTED_CHANNELS = 2
_MANIFEST_NAME = "separation.json"


class DemucsAdapterError(AdapterError):
    """Hybrid Demucs could not be loaded or could not separate the audio."""


@dataclass(frozen=True)
class SeparatedStems:
    """Cached paths and provenance for one source-separation result."""

    vocals_path: Path
    accompaniment_path: Path
    sample_rate_hz: int
    model_name: str
    cache_reused: bool
    raw_output: dict[str, Any]


class DemucsAdapter(BaseModelAdapter[Path, SeparatedStems]):
    """Separate vocals with TorchAudio's Hybrid Demucs bundle.

    ``torchaudio`` is imported only when a non-cached separation is needed. Its
    major/minor version must match PyTorch's major/minor version.  The model
    checkpoint is not downloaded unless ``allow_download`` is explicitly true.
    """

    adapter_name = "hybrid_demucs_torchaudio"

    def __init__(
        self,
        model_name: str = DEFAULT_DEMUCS_BUNDLE,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        allow_download: bool = False,
        segment_seconds: float = 10.0,
        overlap_seconds: float = 1.0,
    ) -> None:
        super().__init__(model_name, device=device, cache_dir=cache_dir)
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not math.isfinite(segment_seconds) or segment_seconds <= 0:
            raise ValueError("segment_seconds must be a positive finite number")
        if (
            not math.isfinite(overlap_seconds)
            or overlap_seconds < 0
            or overlap_seconds >= segment_seconds
        ):
            raise ValueError(
                "overlap_seconds must be finite, nonnegative, and smaller than segment_seconds"
            )
        self.allow_download = bool(allow_download)
        self.segment_seconds = float(segment_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self._torch: Any | None = None
        self._torchaudio: Any | None = None
        self._soundfile: Any | None = None
        self._bundle: Any | None = None
        self._model: Any | None = None
        self._sources: tuple[str, ...] = ()
        self._sample_rate_hz: int | None = None
        self._checkpoint_path: Path | None = None

    def separate(
        self,
        audio_path: str | Path,
        *,
        output_dir: str | Path,
    ) -> SeparatedStems:
        """Separate a 44.1 kHz stereo WAV and cache float32 WAV stems."""

        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise DemucsAdapterError(f"Audio file does not exist: {source}")
        destination = Path(output_dir).expanduser().resolve() / _safe_name(self.model_name)
        destination.mkdir(parents=True, exist_ok=True)

        cached = _read_cached_result(
            source,
            destination,
            model_name=self.model_name,
            segment_seconds=self.segment_seconds,
            overlap_seconds=self.overlap_seconds,
        )
        if cached is not None:
            LOGGER.info("Reusing cached Hybrid Demucs stems from %s", destination)
            return cached

        return super().analyze(source, output_dir=destination)

    def _load(self) -> None:
        try:
            import soundfile
            import torch
            import torchaudio
        except (ImportError, OSError) as exc:
            raise DemucsAdapterError(
                "Hybrid Demucs separation requires soundfile and a TorchAudio build "
                "that exactly matches the installed PyTorch release."
            ) from exc

        torch_version = _major_minor(torch.__version__)
        torchaudio_version = _major_minor(torchaudio.__version__)
        if torch_version != torchaudio_version:
            raise DemucsAdapterError(
                "PyTorch and TorchAudio major/minor versions must match for Hybrid "
                f"Demucs (torch={torch.__version__}, "
                f"torchaudio={torchaudio.__version__})."
            )

        pipelines = getattr(torchaudio, "pipelines", None)
        bundle = getattr(pipelines, self.model_name, None) if pipelines is not None else None
        if bundle is None:
            raise DemucsAdapterError(
                f"TorchAudio does not provide the source-separation bundle "
                f"{self.model_name!r}. Use {DEFAULT_DEMUCS_BUNDLE!r} with a "
                "compatible TorchAudio release."
            )

        previous_hub_dir = Path(torch.hub.get_dir())
        if self.cache_dir is not None:
            hub_dir = self.cache_dir.resolve() / "torch_hub"
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))
        else:
            hub_dir = previous_hub_dir

        try:
            checkpoint = _find_bundle_checkpoint(bundle, hub_dir)
            if checkpoint is None and not self.allow_download:
                bundle_path = str(getattr(bundle, "_path", "the bundle checkpoint"))
                raise DemucsAdapterError(
                    "The Hybrid Demucs checkpoint is not cached, and model downloads "
                    "are disabled. Set SMARTEDIT_ALLOW_MODEL_DOWNLOADS=1 only after "
                    f"reviewing the checkpoint size and license. Expected asset: "
                    f"{bundle_path!r} under {hub_dir / 'torchaudio'}."
                )

            LOGGER.info(
                "Loading TorchAudio Hybrid Demucs bundle %s on %s",
                self.model_name,
                self.device,
            )
            try:
                model = bundle.get_model()
            except Exception as exc:
                action = (
                    "Check network access, disk space, and the model license."
                    if self.allow_download
                    else "The locally cached checkpoint may be incomplete or incompatible."
                )
                raise DemucsAdapterError(
                    f"Could not load TorchAudio bundle {self.model_name!r}. {action} "
                    f"Underlying error: {type(exc).__name__}: {exc}"
                ) from exc
            checkpoint = checkpoint or _find_bundle_checkpoint(bundle, hub_dir)
        finally:
            if self.cache_dir is not None:
                torch.hub.set_dir(str(previous_hub_dir))

        try:
            model.eval()
            model.to(self.device)
        except Exception as exc:
            raise DemucsAdapterError(
                f"Could not move Hybrid Demucs to device {self.device!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        sources = tuple(str(item) for item in getattr(model, "sources", ()))
        if "vocals" not in sources or len(sources) < 2:
            raise DemucsAdapterError(
                "The selected TorchAudio bundle does not expose the expected vocals "
                "and accompaniment source names."
            )
        sample_rate = int(getattr(bundle, "sample_rate", 0))
        if sample_rate <= 0:
            raise DemucsAdapterError("The TorchAudio bundle has no valid sample rate")

        self._torch = torch
        self._torchaudio = torchaudio
        self._soundfile = soundfile
        self._bundle = bundle
        self._model = model
        self._sources = sources
        self._sample_rate_hz = sample_rate
        self._checkpoint_path = checkpoint

    def _analyze(self, input_data: Path, **kwargs: Any) -> SeparatedStems:
        output_dir_value = kwargs.get("output_dir")
        if output_dir_value is None:
            raise DemucsAdapterError("output_dir is required for source separation")
        output_dir = Path(output_dir_value)

        assert self._torch is not None
        assert self._soundfile is not None
        assert self._model is not None
        assert self._sample_rate_hz is not None

        try:
            waveform, sample_rate = self._soundfile.read(
                str(input_data),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise DemucsAdapterError(
                f"Could not read source-separation input {input_data}: {type(exc).__name__}: {exc}"
            ) from exc
        if int(sample_rate) != self._sample_rate_hz:
            raise DemucsAdapterError(
                f"Hybrid Demucs expects {self._sample_rate_hz} Hz audio, but "
                f"{input_data.name} is {sample_rate} Hz. Extract a 44.1 kHz model input."
            )
        if waveform.ndim != 2 or waveform.shape[1] != _EXPECTED_CHANNELS:
            channels = waveform.shape[1] if waveform.ndim == 2 else 0
            raise DemucsAdapterError(
                f"Hybrid Demucs expects stereo audio, but {input_data.name} has "
                f"{channels} channel(s). Extract a stereo model input."
            )
        if waveform.shape[0] == 0:
            raise DemucsAdapterError("Hybrid Demucs cannot separate an empty audio file")

        torch = self._torch
        mixture = torch.from_numpy(waveform.T.copy()).to(dtype=torch.float32)
        reference = mixture.mean(dim=0)
        common_mean = reference.mean()
        common_std = reference.std()
        if not bool(torch.isfinite(common_std)) or float(common_std) <= 1e-8:
            raise DemucsAdapterError(
                "Hybrid Demucs cannot separate audio with no measurable signal variation"
            )
        normalized = (mixture - common_mean) / (common_std + 1e-8)

        LOGGER.info(
            "Separating vocals with %s (%s, %.2fs chunks, %.2fs overlap)",
            self.model_name,
            self.device,
            self.segment_seconds,
            self.overlap_seconds,
        )
        try:
            separated = _separate_overlap_add(
                self._model,
                normalized,
                sample_rate_hz=self._sample_rate_hz,
                segment_seconds=self.segment_seconds,
                overlap_seconds=self.overlap_seconds,
                device=self.device,
                torch=torch,
            )
        except Exception as exc:
            if isinstance(exc, DemucsAdapterError):
                raise
            raise DemucsAdapterError(
                f"Hybrid Demucs inference failed: {type(exc).__name__}: {exc}"
            ) from exc

        # Restore the one common scale for every source.  The mixture DC offset
        # stays removed: adding it to every stem would multiply that offset when
        # non-vocal stems are summed into accompaniment and bias RMS comparisons.
        # Do not peak-scale, clamp, or normalize stems independently.
        separated = separated * common_std.cpu()
        if separated.ndim != 4 or separated.shape[0] != 1:
            raise DemucsAdapterError(
                f"Hybrid Demucs returned an unexpected tensor shape {tuple(separated.shape)}"
            )
        if separated.shape[1] != len(self._sources):
            raise DemucsAdapterError(
                "Hybrid Demucs returned a different source count than its source labels"
            )
        if not bool(torch.isfinite(separated).all()):
            raise DemucsAdapterError("Hybrid Demucs returned non-finite audio samples")

        vocal_index = self._sources.index("vocals")
        vocals = separated[0, vocal_index]
        accompaniment_indices = [
            index for index, name in enumerate(self._sources) if name != "vocals"
        ]
        accompaniment = separated[0, accompaniment_indices].sum(dim=0)
        vocals_path = output_dir / "vocals.wav"
        accompaniment_path = output_dir / "accompaniment.wav"
        manifest_path = output_dir / _MANIFEST_NAME

        _write_float_wav_atomic(
            vocals_path,
            vocals.transpose(0, 1).contiguous().numpy(),
            self._sample_rate_hz,
            self._soundfile,
        )
        _write_float_wav_atomic(
            accompaniment_path,
            accompaniment.transpose(0, 1).contiguous().numpy(),
            self._sample_rate_hz,
            self._soundfile,
        )

        manifest = {
            "version": 1,
            "input": _source_identity(input_data),
            "model_name": self.model_name,
            "model_family": "Hybrid Demucs",
            "adapter": self.adapter_name,
            "sample_rate_hz": self._sample_rate_hz,
            "sources": list(self._sources),
            "vocal_source": "vocals",
            "accompaniment_sources": [name for name in self._sources if name != "vocals"],
            "segment_seconds": self.segment_seconds,
            "overlap_seconds": self.overlap_seconds,
            "checkpoint_path": (
                str(self._checkpoint_path) if self._checkpoint_path is not None else None
            ),
            "torch_version": str(self._torch.__version__),
            "torchaudio_version": str(self._torchaudio.__version__),
            "common_gain_preserved": True,
            "common_dc_offset_removed": True,
            "independent_stem_normalization": False,
            "vocals_path": str(vocals_path),
            "accompaniment_path": str(accompaniment_path),
        }
        _write_json_atomic(manifest_path, manifest)
        raw_output = {"status": "ok", "cache_reused": False, **manifest}
        return SeparatedStems(
            vocals_path=vocals_path,
            accompaniment_path=accompaniment_path,
            sample_rate_hz=self._sample_rate_hz,
            model_name=self.model_name,
            cache_reused=False,
            raw_output=raw_output,
        )

    def _unload(self) -> None:
        model = self._model
        torch = self._torch
        if model is not None:
            with suppress(Exception):
                model.to("cpu")
        self._model = None
        self._bundle = None
        self._sources = ()
        self._sample_rate_hz = None
        self._checkpoint_path = None
        self._soundfile = None
        self._torchaudio = None
        self._torch = None
        if torch is not None and self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _separate_overlap_add(
    model: Any,
    mixture: Any,
    *,
    sample_rate_hz: int,
    segment_seconds: float,
    overlap_seconds: float,
    device: str,
    torch: Any,
) -> Any:
    """Run deterministic weighted overlap-add over ``[channels, frames]`` audio."""

    if mixture.ndim != 2:
        raise DemucsAdapterError("Demucs mixture must have [channels, frames] shape")
    total_frames = int(mixture.shape[-1])
    segment_frames = max(1, int(round(segment_seconds * sample_rate_hz)))
    overlap_frames = max(0, int(round(overlap_seconds * sample_rate_hz)))
    starts = _chunk_starts(total_frames, segment_frames, overlap_frames)
    accumulator: Any | None = None
    weight_sum = torch.zeros(total_frames, dtype=torch.float32)

    for index, start in enumerate(starts):
        end = min(total_frames, start + segment_frames)
        chunk = mixture[:, start:end].unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(chunk)
        output = output.detach().to(device="cpu", dtype=torch.float32)
        if output.ndim != 4 or output.shape[0] != 1:
            raise DemucsAdapterError(
                f"Hybrid Demucs chunk returned unexpected shape {tuple(output.shape)}"
            )
        if output.shape[-1] != end - start:
            raise DemucsAdapterError(
                "Hybrid Demucs changed the chunk length unexpectedly during inference"
            )
        if accumulator is None:
            accumulator = torch.zeros(
                (1, output.shape[1], output.shape[2], total_frames),
                dtype=torch.float32,
            )

        previous_end = min(total_frames, starts[index - 1] + segment_frames) if index > 0 else start
        next_start = starts[index + 1] if index + 1 < len(starts) else end
        weights = _chunk_weights(
            end - start,
            left_overlap=max(0, previous_end - start),
            right_overlap=max(0, end - next_start),
            torch=torch,
        )
        accumulator[..., start:end] += output * weights.view(1, 1, 1, -1)
        weight_sum[start:end] += weights

    if accumulator is None or bool((weight_sum <= 0).any()):
        raise DemucsAdapterError("Hybrid Demucs overlap-add left uncovered audio frames")
    return accumulator / weight_sum.view(1, 1, 1, -1)


def _chunk_starts(total_frames: int, segment_frames: int, overlap_frames: int) -> list[int]:
    if total_frames <= 0 or segment_frames <= 0:
        raise ValueError("total_frames and segment_frames must be positive")
    if overlap_frames < 0 or overlap_frames >= segment_frames:
        raise ValueError("overlap_frames must be between zero and segment_frames")
    if total_frames <= segment_frames:
        return [0]
    hop = segment_frames - overlap_frames
    starts = list(range(0, total_frames - segment_frames + 1, hop))
    final_start = total_frames - segment_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _chunk_weights(
    length: int,
    *,
    left_overlap: int,
    right_overlap: int,
    torch: Any,
) -> Any:
    weights = torch.ones(length, dtype=torch.float32)
    if left_overlap:
        ramp = torch.arange(1, left_overlap + 1, dtype=torch.float32) / (left_overlap + 1)
        weights[:left_overlap] = ramp
    if right_overlap:
        ramp = torch.arange(right_overlap, 0, -1, dtype=torch.float32) / (right_overlap + 1)
        weights[-right_overlap:] = torch.minimum(weights[-right_overlap:], ramp)
    return weights


def _find_bundle_checkpoint(bundle: Any, hub_dir: Path) -> Path | None:
    asset = getattr(bundle, "_path", None)
    if not isinstance(asset, str) or not asset.strip():
        return None
    asset_path = Path(asset)
    candidates = (
        hub_dir / "torchaudio" / asset_path,
        hub_dir / asset_path,
        hub_dir / "checkpoints" / asset_path.name,
    )
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def _read_cached_result(
    source: Path,
    output_dir: Path,
    *,
    model_name: str,
    segment_seconds: float,
    overlap_seconds: float,
) -> SeparatedStems | None:
    manifest_path = output_dir / _MANIFEST_NAME
    vocals_path = output_dir / "vocals.wav"
    accompaniment_path = output_dir / "accompaniment.wav"
    if not all(_usable_file(path) for path in (manifest_path, vocals_path, accompaniment_path)):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    expected = {
        "input": _source_identity(source),
        "model_name": model_name,
        "segment_seconds": segment_seconds,
        "overlap_seconds": overlap_seconds,
        "common_gain_preserved": True,
        "common_dc_offset_removed": True,
        "independent_stem_normalization": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    sample_rate = manifest.get("sample_rate_hz")
    if type(sample_rate) is not int or sample_rate <= 0:
        return None
    raw_output = {"status": "ok", "cache_reused": True, **manifest}
    return SeparatedStems(
        vocals_path=vocals_path,
        accompaniment_path=accompaniment_path,
        sample_rate_hz=sample_rate,
        model_name=model_name,
        cache_reused=True,
        raw_output=raw_output,
    )


def _source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def _write_float_wav_atomic(path: Path, audio: Any, sample_rate: int, soundfile: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=".wav",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        soundfile.write(
            str(temporary),
            audio,
            sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        if not _usable_file(temporary):
            raise DemucsAdapterError(f"Writing {path.name} produced an empty WAV file")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=".json",
            dir=path.parent,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _usable_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if match is None:
        raise DemucsAdapterError(f"Could not parse package version {version!r}")
    return int(match.group(1)), int(match.group(2))


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "hybrid_demucs"


__all__ = [
    "DEFAULT_DEMUCS_BUNDLE",
    "DemucsAdapter",
    "DemucsAdapterError",
    "SeparatedStems",
]
