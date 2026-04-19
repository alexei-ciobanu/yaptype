"""
Push-to-talk dictation client. Connects to asr_server.py for fast transcription.

Starts instantly (no model loading). Hold a key to record, release to transcribe.

Usage:
    uv run --no-sync python dictate_client.py
    uv run --no-sync python dictate_client.py --key scroll_lock
    uv run --no-sync python dictate_client.py --port 9876
"""

import argparse
import contextlib
import socket
import struct
import sys
import threading
import time

import numpy as np
import pyperclip
import sounddevice as sd

DEFAULT_PORT = 9876


def parse_args():
    parser = argparse.ArgumentParser(description="Push-to-talk dictation client")
    parser.add_argument(
        "--key",
        default="ctrl_r",
        help="Hold this key to record (default: ctrl_r). Examples: scroll_lock, f13, ctrl_r, alt_r",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index. Default: system default.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate (default: 16000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    return parser.parse_args()


def send_audio_to_server(host: str, port: int, audio_data: np.ndarray, sample_rate: int) -> str:
    """Send audio to the ASR server and get transcription back."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    try:
        sock.connect((host, port))

        # Send: [4 bytes: sample_rate][4 bytes: audio_length][audio_bytes]
        audio_bytes = audio_data.astype(np.float32).tobytes()
        sock.sendall(struct.pack("!I", sample_rate))
        sock.sendall(struct.pack("!I", len(audio_bytes)))
        sock.sendall(audio_bytes)

        # Receive: [4 bytes: text_length][text_bytes]
        text_len_data = b""
        while len(text_len_data) < 4:
            chunk = sock.recv(4 - len(text_len_data))
            if not chunk:
                raise ConnectionError("Server closed connection")
            text_len_data += chunk

        text_len = struct.unpack("!I", text_len_data)[0]
        text_data = b""
        while len(text_data) < text_len:
            chunk = sock.recv(text_len - len(text_data))
            if not chunk:
                raise ConnectionError("Server closed connection")
            text_data += chunk

        return text_data.decode("utf-8")
    finally:
        sock.close()


def paste_text(text: str):
    """Paste text into the currently active window via clipboard + Ctrl+V."""
    import pyautogui

    old_clipboard = ""
    with contextlib.suppress(Exception):
        old_clipboard = pyperclip.paste()

    pyperclip.copy(text)
    time.sleep(0.05)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)

    def restore():
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            pyperclip.copy(old_clipboard)

    threading.Thread(target=restore, daemon=True).start()


def resolve_key(key_name: str):
    """Resolve a key name string to a pynput Key or KeyCode."""
    from pynput.keyboard import Key, KeyCode

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

    if lower.startswith("f") and lower[1:].isdigit():
        fnum = int(lower[1:])
        try:
            return getattr(Key, f"f{fnum}")
        except AttributeError:
            pass

    if len(key_name) == 1:
        return KeyCode.from_char(key_name)

    raise ValueError(f"Unknown key: '{key_name}'")


def main():
    args = parse_args()
    target_key = resolve_key(args.key)

    # Check server connectivity
    print(f"Connecting to ASR server at {args.host}:{args.port}...", end="", flush=True)
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(3)
        test_sock.connect((args.host, args.port))
        test_sock.close()
        print(" connected!")
    except (TimeoutError, ConnectionRefusedError):
        print(f"\n\nError: Cannot connect to ASR server at {args.host}:{args.port}")
        print("Start the server first with: uv run --no-sync python asr_server.py")
        sys.exit(1)

    # Recording state
    recording = False
    audio_chunks: list[np.ndarray] = []
    stream = None
    lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
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

        audio = np.concatenate(audio_chunks, axis=0).flatten()
        duration = len(audio) / args.sample_rate
        print(f" {duration:.1f}s captured")

        if duration < 0.3:
            print("  [skip] Too short, skipping")
            return

        print("  [>>] Transcribing...", end="", flush=True)
        t0 = time.time()
        try:
            text = send_audio_to_server(args.host, args.port, audio, args.sample_rate)
        except Exception as e:
            print(f" error: {e}")
            return
        elapsed = time.time() - t0
        print(f" done in {elapsed:.2f}s")

        if text.strip():
            print(f'  [text] "{text}"')
            paste_text(text + " ")
            print("  [ok] Pasted!")
        else:
            print("  (empty transcription)")

    from pynput.keyboard import Listener

    def on_press(key):
        if key == target_key:
            start_recording()

    def on_release(key):
        if key == target_key:
            threading.Thread(target=stop_recording_and_transcribe, daemon=True).start()

    key_display = args.key.replace("_", " ").title()
    print(f"\n{'=' * 50}")
    print("  Push-to-talk dictation ready! (client mode)")
    print(f"  Hold [{key_display}] to record, release to transcribe")
    print(f"  Microphone: {sd.query_devices(args.device, 'input')['name']}")
    print(f"  Server: {args.host}:{args.port}")
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
