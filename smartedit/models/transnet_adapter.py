"""Isolated TransNet-V2 PyTorch integration.

The adapter expects the ``transnetv2_pytorch`` inference package and a local,
converted ``.pth`` checkpoint.  It never downloads weights and never runs a
randomly initialized network as if it were a valid detector.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from smartedit.preprocessing.ffmpeg_utils import inspect_video, validate_video_path
from smartedit.schemas import VideoMetadata

LOGGER = logging.getLogger(__name__)

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


class TransNetError(RuntimeError):
    """Base class for TransNet-V2 adapter failures."""


class TransNetUnavailableError(TransNetError):
    """Raised when code, weights, or requested hardware are unavailable."""


class TransNetInferenceError(TransNetError):
    """Raised when decoding or model inference fails."""


@dataclass
class TransitionRange:
    """A contiguous run of frames over the transition threshold."""

    start_frame: int
    end_frame: int
    peak_frame: int
    start_timestamp: float
    end_timestamp: float
    peak_probability: float

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame cannot precede start_frame")
        if not self.start_frame <= self.peak_frame <= self.end_frame:
            raise ValueError("peak_frame must lie inside the transition range")
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("end_timestamp cannot precede start_timestamp")


@dataclass
class TransNetPrediction:
    """Raw, inspectable output from a successful TransNet-V2 pass."""

    checkpoint_path: str
    device: str
    threshold: float
    frame_count: int
    fps: float
    model_name: str = "transnet_v2"
    cut_frames: list[int] = field(default_factory=list)
    cut_timestamps: list[float] = field(default_factory=list)
    transition_ranges: list[TransitionRange] = field(default_factory=list)
    single_frame_probabilities: list[float] = field(default_factory=list)
    all_frame_probabilities: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.cut_frames) != len(self.cut_timestamps):
            raise ValueError("cut_frames and cut_timestamps must have equal lengths")
        for name, probabilities in (
            ("single_frame_probabilities", self.single_frame_probabilities),
            ("all_frame_probabilities", self.all_frame_probabilities),
        ):
            if probabilities and len(probabilities) != self.frame_count:
                raise ValueError(f"{name} length must equal frame_count when retained")

    def validate_timestamps(self, duration_seconds: float) -> None:
        """Reject any model-derived timestamp outside the source duration."""

        for timestamp in self.cut_timestamps:
            if not 0.0 <= timestamp <= duration_seconds:
                raise ValueError(f"TransNet timestamp {timestamp} outside [0, {duration_seconds}]")
        for transition in self.transition_ranges:
            if not (
                0.0 <= transition.start_timestamp <= transition.end_timestamp <= duration_seconds
            ):
                raise ValueError(
                    "TransNet transition range outside source duration: "
                    f"{transition.start_timestamp}-{transition.end_timestamp}"
                )


def _resolve_device(torch: Any, requested: DeviceName) -> str:
    if requested == "auto":
        if bool(torch.cuda.is_available()):
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"
    if requested == "cuda" and not bool(torch.cuda.is_available()):
        raise TransNetUnavailableError(
            "CUDA was requested for TransNet-V2, but PyTorch reports no CUDA device."
        )
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise TransNetUnavailableError(
                "MPS was requested for TransNet-V2, but PyTorch reports no MPS device."
            )
    return requested


def _transition_runs(
    probabilities: Sequence[float],
    threshold: float,
) -> list[tuple[int, int, int, float]]:
    """Return inclusive (start, end, peak, peak_probability) transition runs."""

    runs: list[tuple[int, int, int, float]] = []
    start: int | None = None
    for index, probability in enumerate(probabilities):
        active = probability > threshold
        if active and start is None:
            start = index
        at_end = index == len(probabilities) - 1
        if start is not None and ((not active) or at_end):
            end = index if active and at_end else index - 1
            peak = max(range(start, end + 1), key=lambda item: probabilities[item])
            runs.append((start, end, peak, float(probabilities[peak])))
            start = None
    return runs


class TransNetV2Adapter:
    """Load a local TransNet-V2 checkpoint and detect boundaries in a video."""

    model_name = "transnet_v2"
    adapter_name = "transnet_v2"
    input_height = 27
    input_width = 48
    context_frames = 25
    core_frames = 50
    window_frames = 100

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: DeviceName = "auto",
        threshold: float = 0.5,
        batch_size: int = 4,
        retain_probabilities: bool = True,
        maximum_decoded_frames: int = 108_000,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if maximum_decoded_frames <= 0:
            raise ValueError("maximum_decoded_frames must be positive")
        normalized_device = str(device)
        if normalized_device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        self.requested_checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        self.requested_device = cast(DeviceName, normalized_device)
        self.threshold = threshold
        self.batch_size = batch_size
        self.retain_probabilities = retain_probabilities
        self.maximum_decoded_frames = maximum_decoded_frames

        self._torch: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._checkpoint: Path | None = None

    def _find_checkpoint(self) -> Path:
        path = self.requested_checkpoint_path
        if path is None:
            raise TransNetUnavailableError(
                "Set SMARTEDIT_TRANSNET_CHECKPOINT to the converted TransNet-V2 .pth file."
            )
        if not path.is_file() or path.stat().st_size == 0:
            raise TransNetUnavailableError(f"TransNet-V2 checkpoint is missing: {path}")
        return path.resolve()

    def load(self) -> None:
        """Load model resources once."""

        if self._model is not None:
            return
        self._load()

    def _load(self) -> None:
        """Load package, architecture, and trusted local state dict once."""

        if self._model is not None:
            return
        checkpoint_path = self._find_checkpoint()
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise TransNetUnavailableError(
                "PyTorch is required for TransNet-V2 but is not installed."
            ) from exc
        try:
            package = importlib.import_module("transnetv2_pytorch")
            model_class = package.TransNetV2
        except TransNetError:
            raise
        except Exception as exc:
            raise TransNetUnavailableError(
                "Could not import transnetv2_pytorch.TransNetV2. Add the official "
                "TransNetV2/inference-pytorch directory to PYTHONPATH."
            ) from exc

        device = _resolve_device(torch, self.requested_device)
        LOGGER.info(
            "Loading TransNet-V2 from local checkpoint %s on %s",
            checkpoint_path,
            device,
        )
        try:
            model = model_class()
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise TransNetUnavailableError("TransNet checkpoint must be a state dictionary")
            model.load_state_dict(checkpoint, strict=True)
            model.to(device)
            model.eval()
        except TransNetError:
            raise
        except Exception as exc:
            raise TransNetUnavailableError(
                f"Unable to load TransNet-V2 checkpoint {checkpoint_path}: {exc}"
            ) from exc

        self._torch = torch
        self._model = model
        self._device = device
        self._checkpoint = checkpoint_path

    def unload(self) -> None:
        """Release model references; useful between large multimodal stages."""

        self._model = None
        self._torch = None
        self._device = None
        self._checkpoint = None

    def _decode_video(
        self,
        video_path: Path,
        metadata: VideoMetadata,
    ) -> tuple[Any, list[float]]:
        try:
            cv2 = importlib.import_module("cv2")
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise TransNetUnavailableError(
                "TransNet-V2 decoding requires numpy and opencv-python-headless."
            ) from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise TransNetInferenceError(f"OpenCV could not open {video_path}")

        frames: list[Any] = []
        timestamps: list[float] = []
        try:
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break
                index = len(frames)
                if index >= self.maximum_decoded_frames:
                    raise TransNetInferenceError(
                        f"Video exceeds the configured TransNet-V2 decode limit of "
                        f"{self.maximum_decoded_frames} frames. Increase "
                        "maximum_decoded_frames explicitly if this is intentional."
                    )
                resized = cv2.resize(
                    frame,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_AREA,
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb)

                fallback_timestamp = index / metadata.fps
                reported_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1_000.0
                timestamp = fallback_timestamp
                if (
                    math.isfinite(reported_timestamp)
                    and 0.0 <= reported_timestamp <= metadata.duration_seconds
                    and (not timestamps or reported_timestamp > timestamps[-1])
                    and abs(reported_timestamp - fallback_timestamp) <= max(0.5, 3.0 / metadata.fps)
                ):
                    timestamp = reported_timestamp
                timestamp = min(metadata.duration_seconds, max(0.0, timestamp))
                timestamps.append(timestamp)
        finally:
            capture.release()

        if not frames:
            raise TransNetInferenceError(f"No video frames could be decoded from {video_path}")
        stacked = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
        return stacked, timestamps

    def _windows(self, frames: Any) -> list[tuple[int, int, Any]]:
        """Build official-style 25/50/25 context windows with edge padding."""

        np = importlib.import_module("numpy")
        windows: list[tuple[int, int, Any]] = []
        frame_count = int(frames.shape[0])
        for core_start in range(0, frame_count, self.core_frames):
            core_length = min(self.core_frames, frame_count - core_start)
            indices = np.arange(
                core_start - self.context_frames,
                core_start - self.context_frames + self.window_frames,
            )
            indices = np.clip(indices, 0, frame_count - 1)
            windows.append((core_start, core_length, frames[indices]))
        return windows

    def _infer_probabilities(self, frames: Any) -> tuple[list[float], list[float]]:
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None
        np = importlib.import_module("numpy")
        windows = self._windows(frames)
        single_output: list[float] = []
        all_output: list[float] = []

        try:
            with self._torch.inference_mode():
                for batch_start in range(0, len(windows), self.batch_size):
                    batch_windows = windows[batch_start : batch_start + self.batch_size]
                    batch_array = np.stack([item[2] for item in batch_windows], axis=0)
                    # Official PyTorch inference accepts uint8 RGB [B,T,27,48,3].
                    batch_tensor = self._torch.from_numpy(batch_array).to(self._device)
                    outputs = self._model(batch_tensor)
                    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                        single_logits, all_logits = outputs[0], outputs[1]
                    elif isinstance(outputs, dict):
                        single_logits = outputs.get("single_frame_pred")
                        all_logits = outputs.get("all_frame_pred")
                        if single_logits is None or all_logits is None:
                            raise ValueError("model output dictionary is missing prediction keys")
                    else:
                        raise ValueError("expected (single_frame_pred, all_frame_pred) from model")
                    if isinstance(all_logits, dict):
                        for key in ("many_hot", "all_frame_pred", "logits"):
                            candidate = all_logits.get(key)
                            if candidate is not None:
                                all_logits = candidate
                                break
                        else:
                            raise ValueError("all_frame_pred mapping has no many_hot/logits tensor")
                    single_batch = self._torch.sigmoid(single_logits.float()).detach().cpu().numpy()
                    all_batch = self._torch.sigmoid(all_logits.float()).detach().cpu().numpy()
                    single_batch = np.asarray(single_batch).reshape(len(batch_windows), -1)
                    all_batch = np.asarray(all_batch).reshape(len(batch_windows), -1)

                    for batch_index, (_, core_length, _) in enumerate(batch_windows):
                        core_slice = slice(
                            self.context_frames,
                            self.context_frames + core_length,
                        )
                        single_output.extend(single_batch[batch_index, core_slice].tolist())
                        all_output.extend(all_batch[batch_index, core_slice].tolist())
        except Exception as exc:
            raise TransNetInferenceError(f"TransNet-V2 inference failed: {exc}") from exc

        expected = int(frames.shape[0])
        if len(single_output) != expected or len(all_output) != expected:
            raise TransNetInferenceError(
                "TransNet-V2 returned an unexpected temporal output length: "
                f"single={len(single_output)}, all={len(all_output)}, expected={expected}"
            )
        return single_output, all_output

    def _analyze(
        self,
        video_path: str | Path,
        metadata: VideoMetadata | None = None,
    ) -> TransNetPrediction:
        """Run full-video boundary detection and return raw probabilities/cuts."""

        source = validate_video_path(video_path)
        video_metadata = metadata or inspect_video(source)
        assert self._checkpoint is not None
        assert self._device is not None

        LOGGER.info("Decoding video for TransNet-V2: %s", source.name)
        frames, timestamps = self._decode_video(source, video_metadata)
        LOGGER.info("Running TransNet-V2 over %d frames", int(frames.shape[0]))
        single, all_frames = self._infer_probabilities(frames)
        runs = _transition_runs(single, self.threshold)

        transitions: list[TransitionRange] = []
        cut_frames: list[int] = []
        cut_timestamps: list[float] = []
        last_index = len(single) - 1
        for start, end, peak, probability in runs:
            transitions.append(
                TransitionRange(
                    start_frame=start,
                    end_frame=end,
                    peak_frame=peak,
                    start_timestamp=timestamps[start],
                    end_timestamp=timestamps[end],
                    peak_probability=probability,
                )
            )
            # Runs touching a physical endpoint are not boundaries between two
            # observable shots, though they remain in raw transition evidence.
            if start > 0 and end < last_index and 0 < peak < last_index:
                cut_frames.append(peak)
                cut_timestamps.append(timestamps[peak])

        prediction = TransNetPrediction(
            checkpoint_path=str(self._checkpoint),
            device=self._device,
            threshold=self.threshold,
            frame_count=int(frames.shape[0]),
            fps=video_metadata.fps,
            cut_frames=cut_frames,
            cut_timestamps=cut_timestamps,
            transition_ranges=transitions,
            single_frame_probabilities=(
                [round(float(value), 7) for value in single] if self.retain_probabilities else []
            ),
            all_frame_probabilities=(
                [round(float(value), 7) for value in all_frames]
                if self.retain_probabilities
                else []
            ),
        )
        prediction.validate_timestamps(video_metadata.duration_seconds)
        return prediction

    def analyze(
        self,
        video_path: str | Path,
        metadata: VideoMetadata | None = None,
    ) -> TransNetPrediction:
        """Load lazily and run inference."""

        self.load()
        return self._analyze(video_path, metadata=metadata)
