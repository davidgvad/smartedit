"""Minimal shared interface for heavyweight model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AdapterError(RuntimeError):
    """A model could not be loaded or could not analyze its input."""


class BaseModelAdapter(ABC, Generic[InputT, OutputT]):
    """Load a model once, then reuse it for analysis."""

    adapter_name = "model"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def analyze(self, input_data: InputT, **kwargs: Any) -> OutputT:
        self.load()
        return self._analyze(input_data, **kwargs)

    def unload(self) -> None:
        self._unload()
        self._loaded = False

    @abstractmethod
    def _load(self) -> None:
        pass

    @abstractmethod
    def _analyze(self, input_data: InputT, **kwargs: Any) -> OutputT:
        pass

    def _unload(self) -> None:
        pass
