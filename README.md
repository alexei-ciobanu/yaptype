# yaptype

Push-to-talk voice dictation powered by NVIDIA NeMo [Parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b).

Hold a key → speak → release → transcription is pasted into the active window.

## Project structure

```
yaptype/
├── .python-version       # Python 3.10
├── pyproject.toml        # Dependencies & uv config (torch CUDA, NeMo)
├── uv.lock               # Lockfile
├── asr_server.py         # ASR server (loads model once, stays hot in VRAM)
├── dictate_client.py     # Push-to-talk client (connects to server, starts instantly)
├── dictate.py            # Standalone push-to-talk (no server needed, slower startup)
├── transcribe.py         # Batch file transcription script
└── README.md
```

## Setup

```bash
# 1. Install dependencies (torch CUDA 12.8 + NeMo)
uv sync

# 2. Upgrade NeMo to main branch (required for parakeet-unified-en-0.6b)
#    NeMo 2.7.3+ needed but not yet on PyPI as of 2026-04-18.
uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@main" --no-deps
```

> **Note:** Re-run step 2 after any `uv sync`. Once NeMo 2.7.3 is on PyPI, step 2 won't be needed.

## Usage

### Client-server mode (recommended)

Start the server once — it loads the model and keeps it hot in VRAM:

```bash
# Terminal 1: start server (loads model ~20s, then stays ready)
uv run --no-sync python asr_server.py
```

Then launch the client whenever you need dictation — it starts instantly:

```bash
# Terminal 2: start client (connects to server, no model loading)
uv run --no-sync python dictate_client.py
```

### Standalone mode

If you don't want to run a separate server:

```bash
uv run --no-sync python dictate.py
```

### Options

```bash
# Custom hotkey (default: Right Ctrl)
--key scroll_lock
--key f13

# Specific microphone (list with: python -m sounddevice)
--device 5

# Custom server address (client only)
--host 127.0.0.1 --port 9876
```

### Batch file transcription

```bash
# Edit transcribe.py to point at your audio files, then:
uv run --no-sync python transcribe.py
```

## Performance (RTX 3060 Ti, 8GB VRAM)

| Metric | Value |
|---|---|
| Model load (server) | ~20s (one-time) |
| Client startup | **instant** |
| Inference (1-3s clips) | 0.12–0.19s |
| Inference (15s clip) | 0.35s |
| Realtime factor | 9–40x realtime |

## Key config notes

- **`exclude-newer-package = { torch = false }`** — Exempts torch from global `exclude-newer` date filter (PyTorch CUDA wheels lack upload dates)
- **`pytorch-cu128` index** — Routes torch to NVIDIA's CUDA 12.8 wheel index
- **`uv run --no-sync`** — Prevents uv from reverting manually-installed NeMo

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

## Dependencies

- **Python** 3.10
- **PyTorch** 2.11.0+cu128
- **NeMo** 2.8.0rc0 (from main branch)
- **Model:** nvidia/parakeet-unified-en-0.6b (~1.2 GB, cached in `~/.cache/huggingface/`)
