"""Small runtime configuration for the SmartEdit pipeline."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class DeviceUnavailableError(RuntimeError):
    pass


class ConfigurationError(ValueError):
    pass


def _torch_capabilities() -> tuple[bool, bool]:
    try:
        import torch
    except (ImportError, OSError):
        return False, False

    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    return cuda, mps


def resolve_device(requested: Device | str = Device.AUTO) -> Device:
    """Choose CUDA, Apple MPS, or CPU."""

    try:
        device = requested if isinstance(requested, Device) else Device(requested)
    except ValueError as exc:
        raise ConfigurationError("device must be auto, cpu, cuda, or mps") from exc

    cuda_available, mps_available = _torch_capabilities()
    if device is Device.AUTO:
        if cuda_available:
            return Device.CUDA
        if mps_available:
            return Device.MPS
        return Device.CPU
    if device is Device.CUDA and not cuda_available:
        raise DeviceUnavailableError("CUDA was requested but is not available")
    if device is Device.MPS and not mps_available:
        raise DeviceUnavailableError("MPS was requested but is not available")
    return device


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _downloads_default() -> bool:
    value = os.getenv("SMARTEDIT_ALLOW_MODEL_DOWNLOADS")
    return _parse_bool("SMARTEDIT_ALLOW_MODEL_DOWNLOADS", value) if value else False


@dataclass
class SmartEditConfig:
    """Settings needed by the preprocessing and model stages."""

    device: Device | str = Device.AUTO
    qwen_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    whisper_model: str = "openai/whisper-large-v3-turbo"
    audio_model: str = "nvidia/audio-flamingo-3-hf"
    transnet_checkpoint: Path | None = None
    cache_dir: Path = Path(".smartedit-cache")
    max_frames: int = 24
    debug: bool = False
    allow_model_downloads: bool = field(default_factory=_downloads_default)

    def __post_init__(self) -> None:
        try:
            self.device = self.device if isinstance(self.device, Device) else Device(self.device)
        except ValueError as exc:
            raise ConfigurationError("device must be auto, cpu, cuda, or mps") from exc
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.transnet_checkpoint = (
            Path(self.transnet_checkpoint).expanduser() if self.transnet_checkpoint else None
        )
        if self.max_frames <= 0:
            raise ConfigurationError("max_frames must be positive")
        if not self.qwen_model or not self.whisper_model or not self.audio_model:
            raise ConfigurationError("model names cannot be empty")

    @property
    def resolved_device(self) -> Device:
        return resolve_device(self.device)

    @classmethod
    def from_env(cls, overrides: Mapping[str, Any] | None = None) -> SmartEditConfig:
        """Read optional SMARTEDIT_* values, then apply CLI overrides."""

        values: dict[str, Any] = {}
        text_fields = {
            "device": "SMARTEDIT_DEVICE",
            "qwen_model": "SMARTEDIT_QWEN_MODEL",
            "whisper_model": "SMARTEDIT_WHISPER_MODEL",
            "audio_model": "SMARTEDIT_AUDIO_MODEL",
            "transnet_checkpoint": "SMARTEDIT_TRANSNET_CHECKPOINT",
            "cache_dir": "SMARTEDIT_CACHE_DIR",
        }
        for field_name, env_name in text_fields.items():
            value = os.getenv(env_name)
            if value is not None:
                values[field_name] = value

        max_frames = os.getenv("SMARTEDIT_MAX_FRAMES")
        if max_frames is not None:
            try:
                values["max_frames"] = int(max_frames)
            except ValueError as exc:
                raise ConfigurationError("SMARTEDIT_MAX_FRAMES must be an integer") from exc

        for field_name, env_name in {
            "debug": "SMARTEDIT_DEBUG",
            "allow_model_downloads": "SMARTEDIT_ALLOW_MODEL_DOWNLOADS",
        }.items():
            value = os.getenv(env_name)
            if value is not None:
                values[field_name] = _parse_bool(env_name, value)

        if overrides:
            values.update(overrides)
        return cls(**values)
