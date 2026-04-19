# Known Workarounds & Temporary Limitations

This file tracks hacks, workarounds, and known issues that are expected to be resolved upstream.
When revisiting this project, check each item below to see if the underlying issue has been fixed.

**Last reviewed:** 2026-04-18

---

## 1. NeMo version mismatch (parakeet-unified-en-0.6b requires unreleased NeMo)

**Status:** ⏳ Waiting for NeMo 2.7.3 on PyPI

**Problem:** The `nvidia/parakeet-unified-en-0.6b` model requires `att_chunk_context_size` in
`ConformerEncoder`, which was added after NeMo 2.7.2 (the latest PyPI release as of 2026-04-18).
The model card says "Runtime Engine: NeMo 2.7.3" but 2.7.3 is not yet published.

**Current workaround:** After every `uv sync`, manually upgrade NeMo from git main:
```bash
uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@main" --no-deps
```
All scripts use `uv run --no-sync` to prevent uv from reverting NeMo back to 2.7.2.

**How to check if resolved:**
```bash
uv pip install --dry-run "nemo_toolkit[asr]>=2.7.3"
```
If it resolves, update `pyproject.toml` to `nemo-toolkit[asr]>=2.7.3` and remove the `--no-sync`
workaround from README and all documentation.

**Files affected:**
- `pyproject.toml` — version constraint
- `README.md` — setup instructions mentioning the workaround
- All usage examples reference `uv run --no-sync`

---

## 2. NeMo missing `validation_ds` config for parakeet-unified model

**Status:** ⏳ Likely fixed in NeMo 2.7.3+

**Problem:** The parakeet-unified-en-0.6b model config has `validation_ds = None`, causing
`AttributeError: 'NoneType' object has no attribute 'get'` when calling `model.transcribe()`.

**Current workaround:** In `dictate.py` and `asr_server.py`, we manually set:
```python
from omegaconf import OmegaConf
if model.cfg.get("validation_ds", None) is None:
    model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})
```

**How to check if resolved:** Load the model with the released NeMo version and call
`model.transcribe(["test.wav"])` without the workaround. If it works, remove the fix.

**Files affected:**
- `dictate.py` — `load_model()` function
- `asr_server.py` — `load_model()` function

---

## 3. llama.cpp audio encoder CPU bottleneck (~2.1s per request)

**Status:** ⏳ Upstream limitation in llama.cpp's `mtmd-audio.cpp`

**Problem:** The mel spectrogram computation in llama.cpp's audio pipeline runs entirely on CPU
using a manual Cooley-Tukey FFT implementation (`log_mel_spectrogram_worker_thread()` in
`tools/mtmd/mtmd-audio.cpp`). This takes ~2100ms for 10 seconds of 16kHz audio, regardless of
`-ngl 99`, CUDA backend, or audio format.

For comparison, NeMo uses `torch.stft()` → cuFFT on GPU, achieving ~100ms for the same task.

**Benchmark data (RTX 3060 Ti, Ryzen 5 3600):**
- Time to first token (streaming): 2119ms
- Server-reported LLM prompt eval: 164ms
- Server-reported generation: 149ms
- Hidden audio preprocessing: ~2000ms (mel spectrogram + encoder)
- `-ngl 99` with CUDA backend does NOT affect audio preprocessing time
- Pre-resampling to 16kHz mono does NOT help

**Root cause:** `tools/mtmd/mtmd-audio.cpp` lines in `log_mel_spectrogram_worker_thread()` — pure
C++ CPU FFT with no GPU acceleration. The `--mmproj-offload` flag only offloads the encoder
forward pass, not the mel spectrogram computation.

**How to check if resolved:**
```bash
# Start llama-server with a Qwen3-ASR model
llama-server -hf ggml-org/Qwen3-ASR-0.6B-GGUF -ngl 99

# Benchmark time-to-first-token with streaming
# If TTFT drops below ~500ms for 10s audio, the FFT has been GPU-accelerated
```

**Related upstream issues:**
- https://github.com/ggml-org/llama.cpp/issues/16885 (closed as stale — suggested integrating
  whisper.cpp for audio processing)
- Audio support is marked as "highly experimental" in llama.cpp docs

**Files affected:**
- `asr_server_llama.py` — the bridge server (functionally correct, just slow due to upstream)

---

## 4. PyTorch CUDA wheels lack upload dates (uv `exclude-newer` conflict)

**Status:** ⏳ PyTorch / uv upstream issue

**Problem:** The global `exclude-newer = "7 days"` in `~\AppData\Roaming\uv\uv.toml` filters out
all PyTorch CUDA wheels from `download.pytorch.org` because they have no upload date metadata.

**Current workaround:** In `pyproject.toml`:
```toml
[tool.uv]
exclude-newer-package = { torch = false }

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

**How to check if resolved:** Remove `exclude-newer-package` and run `uv lock`. If it resolves
torch from the CUDA index without errors, the upload dates have been fixed.

**Files affected:**
- `pyproject.toml` — `[tool.uv]` section
