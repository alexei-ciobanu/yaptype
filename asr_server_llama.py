"""
ASR server using llama.cpp backend (Qwen3-ASR via llama-server).

Expects llama-server running with a Qwen3-ASR model, e.g.:
    llama-server -hf ggml-org/Qwen3-ASR-0.6B-GGUF

This server provides the same TCP protocol as asr_server.py (NeMo backend),
so dictate_client.py works with either backend.

Usage:
    uv run --no-sync python asr_server_llama.py
    uv run --no-sync python asr_server_llama.py --llama-url http://localhost:8080
"""

import argparse
import base64
import io
import json
import socket
import struct
import time
import urllib.request

import numpy as np
import soundfile as sf

DEFAULT_PORT = 9876
DEFAULT_LLAMA_URL = "http://localhost:8080"


def transcribe(audio_data: np.ndarray, sample_rate: int, llama_url: str) -> str:
    """Transcribe audio via llama-server's chat completions API."""
    # Convert to mono if needed
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Encode as WAV in memory
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = json.dumps(
        {
            "model": "qwen3-asr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        }
                    ],
                }
            ],
        }
    ).encode()

    req = urllib.request.Request(
        f"{llama_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())
    text = result["choices"][0]["message"]["content"]

    # Qwen3-ASR returns "language English<asr_text>..." — extract just the text
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]

    return text.strip()


def recv_all(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes."""
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Client disconnected")
        data += chunk
    return data


def handle_client(conn: socket.socket, addr: tuple, llama_url: str):
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
        text = transcribe(audio_data, sample_rate, llama_url)
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
    parser = argparse.ArgumentParser(description="ASR server (llama.cpp backend)")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument(
        "--llama-url",
        default=DEFAULT_LLAMA_URL,
        help=f"llama-server URL (default: {DEFAULT_LLAMA_URL})",
    )
    args = parser.parse_args()

    # Verify llama-server is reachable
    print(f"Checking llama-server at {args.llama_url}...", end="", flush=True)
    try:
        resp = urllib.request.urlopen(f"{args.llama_url}/v1/models", timeout=5)
        models = json.loads(resp.read())
        model_name = models["data"][0]["id"] if models.get("data") else "unknown"
        print(f" OK ({model_name})")
    except Exception as e:
        print(f" FAILED: {e}")
        print("Make sure llama-server is running: llama-server -hf ggml-org/Qwen3-ASR-0.6B-GGUF")
        return

    # Warmup
    print("Warming up...", end="", flush=True)
    dummy = np.zeros(16000, dtype=np.float32)
    transcribe(dummy, 16000, args.llama_url)
    print(" done")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)

    print(f"\nASR server (llama.cpp) listening on {args.host}:{args.port}")
    print("Waiting for clients... (Ctrl+C to quit)\n")

    try:
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr, args.llama_url)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.close()


if __name__ == "__main__":
    main()
