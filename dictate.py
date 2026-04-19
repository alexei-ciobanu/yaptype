"""
Push-to-talk voice dictation using NVIDIA Parakeet-unified-en-0.6b.

Hold a key (default: Right Ctrl) to record from your microphone.
Release to transcribe and paste into the currently active text area.

Usage:
    uv run --no-sync python dictate.py
    uv run --no-sync python dictate.py --key scroll_lock
    uv run --no-sync python dictate.py --key f13 --device 1
"""

# Suppress all stderr noise before any imports (OneLogger, triton, etc.)
import os
import sys

_original_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import argparse
import contextlib
import logging
import threading
import time
import warnings

import numpy as np
import pyperclip
import sounddevice as sd
import soundfile as sf
import torch

# Restore stderr now that noisy imports are done
sys.stderr.close()
sys.stderr = _original_stderr

# Suppress Python-level warnings
warnings.filterwarnings("ignore")
os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
for logger_name in ["nemo_logger", "nemo", "pytorch_lightning", "lhotse", "wandb"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Push-to-talk voice dictation")
    parser.add_argument(
        "--key",
        default="ctrl_r",
        help="Hold this key to record (default: ctrl_r). Examples: scroll_lock, f13, ctrl_r, alt_r",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (see `python -m sounddevice` for list). Default: system default.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate in Hz (default: 16000)",
    )
    return parser.parse_args()


def load_model():
    """Load the Parakeet ASR model onto GPU."""

    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading Parakeet-unified-en-0.6b model...")
    t0 = time.time()

    # Suppress all NeMo/Lhotse logging during model load
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    import nemo.utils

    nemo.utils.logging.setLevel(logging.CRITICAL)

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-unified-en-0.6b")
    sys.stderr.close()
    sys.stderr = old_stderr

    # Fix for missing validation_ds config
    if model.cfg.get("validation_ds", None) is None:
        model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Warmup: run a dummy transcription to JIT-compile CUDA kernels
    print("Warming up CUDA kernels...", end="", flush=True)
    dummy = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    transcribe_audio(model, dummy, 16000)
    print(" done")

    # Keep NeMo quiet during runtime too
    nemo.utils.logging.setLevel(logging.CRITICAL)
    return model


def transcribe_audio(model, audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcribe audio numpy array. Returns transcription text."""
    import tempfile

    # Convert stereo to mono if needed
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Write to a temporary WAV file (NeMo expects file paths)
    tmp_path = os.path.join(tempfile.gettempdir(), "dictation_audio.wav")
    sf.write(tmp_path, audio_data, sample_rate)

    # Suppress NeMo/Lhotse stderr warnings during transcription
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        result = model.transcribe([tmp_path], verbose=False)
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

    # Extract text
    if isinstance(result, list):
        if hasattr(result[0], "text"):
            return result[0].text
        return result[0]
    return str(result)


def paste_text(text: str):
    """Paste text into the currently active window via clipboard + Ctrl+V."""
    import pyautogui

    old_clipboard = ""
    with contextlib.suppress(Exception):
        old_clipboard = pyperclip.paste()

    pyperclip.copy(text)
    time.sleep(0.05)  # Small delay to ensure clipboard is set
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)

    # Restore old clipboard after a short delay
    def restore():
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            pyperclip.copy(old_clipboard)

    threading.Thread(target=restore, daemon=True).start()


def resolve_key(key_name: str):
    """Resolve a key name string to a pynput Key or KeyCode."""
    from pynput.keyboard import Key, KeyCode

    # Map friendly names to pynput Key attributes
    key_map = {
        "ctrl_r": Key.ctrl_r,
        "ctrl_l": Key.ctrl_l,
        "alt_r": Key.alt_r,
        "alt_l": Key.alt_l,
        "shift_r": Key.shift_r,
        "shift_l": Key.shift_l,
        "scroll_lock": Key.scroll_lock,
        "pause": Key.pause,
        "insert": Key.insert,
        "caps_lock": Key.caps_lock,
        "num_lock": Key.num_lock,
        "print_screen": Key.print_screen,
    }

    lower = key_name.lower()
    if lower in key_map:
        return key_map[lower]

    # F-keys (f1-f24)
    if lower.startswith("f") and lower[1:].isdigit():
        fnum = int(lower[1:])
        try:
            return getattr(Key, f"f{fnum}")
        except AttributeError:
            pass

    # Single character
    if len(key_name) == 1:
        return KeyCode.from_char(key_name)

    raise ValueError(
        f"Unknown key: '{key_name}'. "
        f"Valid keys: {', '.join(sorted(key_map.keys()))}, f1-f24, or a single character."
    )


def main():
    args = parse_args()
    target_key = resolve_key(args.key)

    # Load model first (takes ~20s)
    model = load_model()

    # Recording state
    recording = False
    audio_chunks: list[np.ndarray] = []
    stream = None
    lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"  [!] Audio: {status}", file=sys.stderr)
        audio_chunks.append(indata.copy())

    def start_recording():
        nonlocal recording, stream, audio_chunks
        with lock:
            if recording:
                return
            recording = True
            audio_chunks = []

        print("  [REC] Recording...", end="", flush=True)
        stream = sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            device=args.device,
            callback=audio_callback,
        )
        stream.start()

    def stop_recording_and_transcribe():
        nonlocal recording, stream
        with lock:
            if not recording:
                return
            recording = False

        if stream is not None:
            stream.stop()
            stream.close()
            stream = None

        if not audio_chunks:
            print(" (no audio captured)")
            return

        audio = np.concatenate(audio_chunks, axis=0)
        duration = len(audio) / args.sample_rate
        print(f" {duration:.1f}s captured")

        if duration < 0.3:
            print("  [skip] Too short, skipping")
            return

        print("  [>>] Transcribing...", end="", flush=True)
        t0 = time.time()
        text = transcribe_audio(model, audio, args.sample_rate)
        elapsed = time.time() - t0
        print(f" done in {elapsed:.2f}s")

        if text.strip():
            print(f'  [text] "{text}"')
            paste_text(text + " ")
            print("  [ok] Pasted!")
        else:
            print("  (empty transcription)")

    # Set up global hotkey listener
    from pynput.keyboard import Listener

    def on_press(key):
        if key == target_key:
            start_recording()

    def on_release(key):
        if key == target_key:
            # Run transcription in a thread to not block the listener
            threading.Thread(target=stop_recording_and_transcribe, daemon=True).start()

    key_display = args.key.replace("_", " ").title()
    print(f"\n{'=' * 50}")
    print("  Push-to-talk dictation ready!")
    print(f"  Hold [{key_display}] to record, release to transcribe")
    print(f"  Microphone: {sd.query_devices(args.device, 'input')['name']}")
    print("  Press Ctrl+C in this terminal to quit")
    print(f"{'=' * 50}\n")

    with Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            # Poll instead of join() so the main thread can receive KeyboardInterrupt
            while listener.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nBye!")


if __name__ == "__main__":
    main()
