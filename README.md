# SmartEdit Edit Signal extraction

This repository implements the first stage of the SmartEdit workflow:

```text
short video -> objective evidence -> scored Edit Signals
```

It does **not** implement engagement prediction, a Random Forest, SHAP,
recommendation generation, or automatic edit decisions. Model-specific code is
isolated so Qwen3-VL, TransNet-V2, Whisper, and Audio Flamingo can be replaced or
benchmarked independently.

## What it produces

Every non-category signal contains a score (`-1`, `0`, or `1`), confidence,
explanation, timestamped evidence, sources, and any evidence conflicts:

- `length`
- `pace`
- `visual_variety`
- `text`
- `text_visibility`
- `narration`
- `background_music`
- `catchy_music`
- `transitions`
- `effects`
- `story`
- `clear_start_middle_end`
- `consistent_theme`

Category confidences are independent values for `personal`, `informational`, and
`promotional`; they do not have to sum to one. This allows mixed-purpose videos.

Absence is not automatically negative. For example, no narration in a visual
montage stays neutral unless strong, independent context says the video is an
unclear informational piece that lacks both useful text and speech.

## Architecture

| Stage | Implementation | Output type |
|---|---|---|
| Metadata and artifacts | ffprobe, ffmpeg, OpenCV | objective metadata, WAV, timestamped frames |
| Shot boundaries | official TransNet-V2 PyTorch inference | cut frames/times, shot intervals and descriptive statistics |
| Narration | Whisper large-v3-turbo through Transformers | transcript, language, segments, words, coverage, WPM, silent gaps |
| Audio | Audio Flamingo 3 adapter plus independent librosa measurements | semantic judgment separated from objective features |
| Visual semantics | Qwen3-VL with sampled frames and exact evidence context | strict editing-only JSON judgments and multi-label category |
| Fusion | deterministic Python rubrics | final Edit Signals and explicit conflicts |

The pipeline lives in `smartedit/pipeline.py`. Adapters under
`smartedit/models/` do not know the fusion rules. Deterministic measurements
under `smartedit/extraction/` do not decide whether an editing choice is good.
That separation is deliberate.

## Requirements

- Python 3.11 or newer
- `ffmpeg` and `ffprobe` on `PATH`
- Enough local disk space for cached artifacts and any selected checkpoints
- PyTorch-compatible hardware for the model paths you enable

Install ffmpeg first. Examples:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg
```

Create an environment and install the package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Large-model download policy

SmartEdit passes `local_files_only=True` by default. It does not silently fetch
Qwen3-VL, Whisper large-v3-turbo, or Audio Flamingo 3. Point the CLI at local
model directories, pre-populate the Hugging Face cache, or deliberately opt in:

```bash
export SMARTEDIT_ALLOW_MODEL_DOWNLOADS=1
```

The CLI prints a warning before analysis when downloads are enabled. Review
checkpoint sizes, terms, and hardware requirements first. The default model ids
are:

```text
Qwen/Qwen3-VL-4B-Instruct
openai/whisper-large-v3-turbo
nvidia/audio-flamingo-3-hf
```

## TransNet-V2 setup

TransNet-V2 is also local-only. The official PyTorch inference code expects the
`TransNetV2` class from `inference-pytorch/transnetv2_pytorch.py` and a converted
`transnetv2-pytorch-weights.pth` state dict. Follow the
[official TransNet-V2 inference instructions](https://github.com/soCzech/TransNetV2/tree/master/inference-pytorch),
then put the official inference file on Python's import path and configure the
checkpoint:

```bash
export PYTHONPATH=/absolute/path/to/TransNetV2/inference-pytorch:$PYTHONPATH
export SMARTEDIT_TRANSNET_CHECKPOINT=/absolute/path/to/transnetv2-pytorch-weights.pth
```

The upstream weight-conversion environment is older than this package's runtime
environment, so conversion may be easiest in a separate environment. SmartEdit
never runs a randomly initialized TransNet model and never labels an OpenCV
heuristic as TransNet-V2.

## CLI

The requested command is:

```bash
python -m smartedit.cli analyze path/to/video.mp4 --output result.json
```

Equivalent installed command:

```bash
smartedit analyze path/to/video.mp4 --output result.json
```

All primary options:

```bash
python -m smartedit.cli analyze clip.mov \
  --output result.json \
  --device auto \
  --qwen-model Qwen/Qwen3-VL-4B-Instruct \
  --whisper-model openai/whisper-large-v3-turbo \
  --audio-model nvidia/audio-flamingo-3-hf \
  --cache-dir .smartedit-cache \
  --max-frames 24 \
  --debug
```

Use an explicit local model directory in any model option to avoid Hub lookup.
`--device auto` chooses CUDA, then Apple MPS, then CPU. An explicitly requested
unavailable accelerator produces a useful error instead of silently moving the
model.

See [`.env.example`](.env.example) for optional environment settings and
[`examples/sample_output.json`](examples/sample_output.json) for an illustrative
result. No sample video is bundled.

## Pipeline details

### 1. Video preprocessing

The input must be a readable, non-empty `.mp4`, `.mov`, or `.webm` containing a
decodable video stream. ffprobe supplies duration, average frame rate,
orientation-aware resolution, codecs, and audio availability. ffmpeg extracts a
mono 16 kHz PCM WAV. OpenCV samples deterministic, endpoint-inclusive frames.
When TransNet cuts are available, part of the fixed frame budget is allocated to
frames immediately around representative boundaries while uniform coverage is
retained.

### 2. Transition measurements

TransNet receives RGB `uint8` tensors with shape `[B, T, 27, 48, 3]`. Inference
uses overlapping 25/50/25 context windows. Contiguous probabilities over the
configured threshold become one boundary candidate at the peak frame. The
deterministic extractor calculates:

- shots and cuts
- cut frames and source timestamps
- average, median, minimum, and maximum shot duration
- cuts per minute
- population variance of shot durations

Cut frequency alone never earns or loses a pace point.

### 3. Narration measurements

Whisper first requests word timestamps and retries with segment timestamps when
the installed Transformers backend does not support word timing. Overlapping
speech intervals are merged before calculating coverage. Speaking rate uses
detected speech time, not total video duration. Silent gaps default to intervals
of at least two seconds.

### 4. Audio analysis and the librosa fallback

Audio Flamingo receives only audio plus the constrained prompt in
`smartedit/prompts/audio_analysis.txt`. It cannot infer visual compatibility from
frames it was not given; compatibility is considered later alongside Qwen
context.

Librosa features are calculated independently even when Audio Flamingo works:

- RMS mean and standard deviation
- tempo estimate
- mean onset strength
- mean spectral centroid
- zero-crossing rate
- harmonic/percussive energy ratio

If Audio Flamingo cannot load or infer, `LibrosaFallbackAudioAdapter` is used.
Its `judgment` is deliberately `null`; limited energy/rhythm/music-likelihood
proxies live under `raw_output.fallback_estimates`, are labeled as non-equivalent,
and are confidence-capped. Librosa cannot reliably distinguish background from
foreground music, decide catchiness, understand environmental sounds, judge
music/visual fit, or measure semantic speech masking.

### 5. Qwen3-VL context

Qwen receives ordered sampled frames, an adjacent exact timestamp for every
frame, transcript segments, shot statistics, speech coverage/WPM, and an audio
summary that distinguishes model judgments from fallback estimates. The prompt
in `smartedit/prompts/qwen_edit_signals.txt` requires JSON only and explicitly
forbids judging personal interest in the topic or inventing evidence.

The adapter validates every score, confidence, category value, and evidence
timestamp before the output reaches fusion. Malformed output becomes a warning;
it is never repaired into invented evidence.

### 6. Fusion

`smartedit/fusion/rubrics.py` contains readable thresholds and decisions. Qwen
provides contextual visual judgments; objective evidence can support them, flag
clear risks, or create a conflict. Examples:

- No narration is neutral unless strong informational/story/text evidence says
  spoken or written explanation is clearly needed.
- One extreme cut-rate indicator cannot make pace negative. Multiple independent
  pace warnings can.
- A positive Qwen pace judgment conflicting with fast speech and a very high cut
  rate is reduced to neutral and confidence is penalized.
- Transition style remains neutral when TransNet measured boundaries but Qwen did
  not explicitly observe helpful or harmful transition styling.
- Text and text visibility are fused as separate signals.

The thresholds are intentionally centralized and unit tested so they can later
be calibrated against VidES or human annotations.

## Caching and partial failure

Extracted audio and sampled frames are stored under the configured cache root in
source-fingerprinted directories. Hugging Face model files use the model cache.
Model-analysis results are deliberately recomputed on each run; this keeps the
baseline easy to follow and avoids a generic object cache/deserialization layer.

Heavy models are run sequentially and accelerator caches are released between
stages. If one stage fails:

- completed stages remain intact;
- the failed raw-output entry contains a status/error where available;
- dependent measurements stay `null` or neutral/unknown;
- a warning explains the missing evidence;
- no placeholder prediction is invented.

Invalid input metadata and an unwritable output destination remain fatal because
a trustworthy report cannot be produced without them.

## Output data

Public structures are ordinary standard-library dataclasses in
`smartedit/schemas.py`; Pydantic is not used. Qwen's manually validated output
stays an ordinary dictionary for fusion, while one explicit function converts
Audio Flamingo output into a dataclass. Another small helper serializes the final
report to JSON.

Checks remain where bad data can enter the algorithm: model-generated scores and
confidences, evidence timestamps, media metadata, and final signal completeness.
Objective measurements remain separate from subjective/model judgments.

## Tests

```bash
python -m pytest -q
```

The deterministic test suite covers:

- scores, confidences, and complete signal sets
- structured timestamp bounds
- absence-is-not-negative rules
- contextual length and narration decisions
- pace conflicts and confidence penalties
- Audio Flamingo versus librosa fallback provenance
- complete fusion output

Full model-integration tests require local checkpoints and are intentionally not
part of the deterministic unit suite.

## Model licenses and hardware notes

The source code in this repository is MIT licensed. Model weights and upstream
code have their own terms:

- [Qwen3-VL model cards](https://huggingface.co/Qwen) specify the license for
  each selected checkpoint. Do not assume every Qwen variant has identical
  terms.
- [Whisper](https://github.com/openai/whisper) and its selected checkpoint have
  separate upstream terms.
- [TransNet-V2](https://github.com/soCzech/TransNetV2) publishes its code under
  the license in that repository; verify any separately obtained checkpoint.
- [Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3-hf) has
  model-specific NVIDIA/non-commercial and incorporated-model terms. Review them
  before downloading or using outputs commercially.

Qwen3-VL 4B, Whisper large-v3-turbo, and Audio Flamingo 3 are substantial
models. CUDA is the most practical full-stack target. Apple MPS is supported on
a best-effort basis where PyTorch and each operation allow it. CPU execution is
valid but can be extremely slow; Audio Flamingo will commonly fall back to
librosa on modest hardware. Quantization is not enabled automatically because it
changes dependencies, accuracy, and device support.

The baseline Audio Flamingo adapter deliberately loads the model in float32.
Mixed BF16/float32 execution can fail inside the current audio encoder, while an
8B float32 model needs roughly 32 GB for weights alone. A high-memory GPU such as
an L40S 48 GB or larger is therefore recommended for this adapter.

## Limitations

- Sampled frames can miss brief text, effects, or transitions between samples.
- Container-reported average FPS is exact for ordinary constant-frame-rate
  media; unusual variable-frame-rate files depend on decoder-reported PTS and
  should be checked carefully.
- ASR timestamps and transcripts can be wrong for music, accents, overlapping
  speakers, or noisy recordings.
- Audio-language and vision-language models may still produce plausible but
  incorrect judgments; manual boundary checks prevent malformed evidence, not
  semantic error.
- Category-specific duration ranges and other rubric thresholds are transparent
  engineering defaults, not claims from the SmartEdit authors.
- There is no engagement outcome, causality claim, or recommendation layer here.

### What remains uncertain without access to VidES

- the exact prompts used by the authors;
- the exact feature and scoring thresholds;
- the exact calibration of `-1/0/1` labels;
- agreement with the authors' human annotations.

Those uncertainties are why raw model outputs, objective measurements, model
provenance, conflicts, and explicit rubric functions are retained.
