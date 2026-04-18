"""
ASR server: loads Parakeet model once and serves transcription requests.

Start this once and leave it running. The client (dictate_client.py) connects to it.

Usage:
    uv run --no-sync python asr_server.py
    uv run --no-sync python asr_server.py --port 9876
"""

# Suppress all stderr noise before any imports
import os
import sys

_original_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import argparse
import logging
import socket
import struct
import tempfile
import time
import warnings

import numpy as np
import soundfile as sf
import torch

sys.stderr.close()
sys.stderr = _original_stderr

warnings.filterwarnings("ignore")
os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
for name in ["nemo_logger", "nemo", "pytorch_lightning", "lhotse", "wandb"]:
    logging.getLogger(name).setLevel(logging.ERROR)


DEFAULT_PORT = 9876


def load_model():
    """Load the Parakeet ASR model onto GPU."""
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading Parakeet-unified-en-0.6b model...")
    t0 = time.time()

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    import nemo.utils

    nemo.utils.logging.setLevel(logging.CRITICAL)

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-unified-en-0.6b")
    sys.stderr.close()
    sys.stderr = old_stderr

    if model.cfg.get("validation_ds", None) is None:
        model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Warmup
    print("Warming up CUDA kernels...", end="", flush=True)
    transcribe(model, np.zeros(16000, dtype=np.float32), 16000)
    print(" done")

    nemo.utils.logging.setLevel(logging.CRITICAL)
    return model


def transcribe(model, audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcribe audio numpy array."""
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    tmp_path = os.path.join(tempfile.gettempdir(), "asr_server_audio.wav")
    sf.write(tmp_path, audio_data, sample_rate)

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        result = model.transcribe([tmp_path], verbose=False)
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

    if isinstance(result, list):
        if hasattr(result[0], "text"):
            return result[0].text
        return result[0]
    return str(result)


def recv_all(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes."""
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Client disconnected")
        data += chunk
    return data


def handle_client(conn: socket.socket, addr, model):
    """Handle a single client connection."""
    try:
        # Protocol: [4 bytes: sample_rate][4 bytes: audio_length][audio_bytes as float32]
        header = recv_all(conn, 8)
        sample_rate = struct.unpack("!I", header[:4])[0]
        audio_len = struct.unpack("!I", header[4:8])[0]

        audio_bytes = recv_all(conn, audio_len)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

        duration = len(audio_data) / sample_rate
        print(f"  [{addr[1]}] Received {duration:.1f}s audio @ {sample_rate}Hz")

        t0 = time.time()
        text = transcribe(model, audio_data, sample_rate)
        elapsed = time.time() - t0

        print(f'  [{addr[1]}] Transcribed in {elapsed:.2f}s: "{text}"')

        # Send back: [4 bytes: text_length][text_bytes as utf-8]
        text_bytes = text.encode("utf-8")
        conn.sendall(struct.pack("!I", len(text_bytes)))
        conn.sendall(text_bytes)

    except Exception as e:
        print(f"  [{addr[1]}] Error: {e}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="ASR transcription server")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    model = load_model()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)

    print(f"\nASR server listening on {args.host}:{args.port}")
    print("Waiting for clients... (Ctrl+C to quit)\n")

    try:
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr, model)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.close()


if __name__ == "__main__":
    main()
