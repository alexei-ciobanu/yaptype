# Potential Enhancements

Ideas for future development, roughly ordered by impact/effort.

## Short-term

### Streaming transcription (NeMo Parakeet)
Parakeet-unified-en-0.6b natively supports streaming with ~160ms latency. Instead of
batch-transcribing after key release, partial transcriptions could appear as you speak.
NeMo provides chunked/buffered RNNT decoding — see
[speech_to_text_streaming_infer_rnnt.py](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py).

### Punctuation, capitalization, and inverse text normalization
NeMo includes PnC (Punctuation and Capitalization) and ITN (Inverse Text Normalization) models
that can be chained after ASR for cleaner output — proper casing, punctuation, and expanding
"twenty three" → "23", etc.

### System tray icon (Windows)
A proper Windows system tray app showing server status, with right-click menu for start/stop/quit.
Libraries: `pystray` + `Pillow`. Could replace `yapctl.py` for daily use.

### Auto-start server on login
Windows Task Scheduler to launch the ASR server on boot so it's always ready.

## Medium-term

### Voice commands
Detect phrases like "new line", "delete that", "period", "select all" and map them to keyboard
actions instead of literal text insertion.

### Per-app profiles
Different behavior in different apps — e.g., Markdown formatting in VS Code, plain text in
Notepad, code-aware in terminals.

### Audio preprocessing
Noise reduction or Voice Activity Detection (VAD) to auto-trim silence at the start/end of
recordings for cleaner transcription.

### Multiple languages
NeMo has multilingual Parakeet models. Could add a `--lang` flag or auto-detect language.

## Architecture

### Named pipe / Unix socket instead of TCP
Lower overhead, no port conflicts, no firewall issues. On Windows, use named pipes.

### WebSocket API
Enable browser extensions to use the ASR server directly — dictation in any web app.

### Package as a proper CLI tool
Publish to PyPI so `uv tool install yaptype` makes `yapctl` globally available without needing
to be in the project directory.

### Client-side preprocessing
Move audio encoding (mono conversion, resampling) to the client side to reduce data sent over
the wire and offload work from the server.
