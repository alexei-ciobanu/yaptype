"""Transcribe audio files using Parakeet-unified-en-0.6b."""

import os
import tempfile
import time

import soundfile as sf
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

print("Loading NeMo ASR model...")
t0 = time.time()
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-unified-en-0.6b")

# Fix: validation_ds is None in the model config
if model.cfg.get("validation_ds", None) is None:
    model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})

if torch.cuda.is_available():
    model = model.cuda()
model.eval()
print(f"Model loaded in {time.time() - t0:.1f}s")


def prepare_audio(filepath: str) -> str:
    """Ensure audio is mono 16kHz WAV. Returns path to a usable file."""
    data, sr = sf.read(filepath)
    needs_conversion = False

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)
        needs_conversion = True

    # Resample to 16kHz if needed
    if sr != 16000:
        import resampy

        data = resampy.resample(data, sr, 16000)
        sr = 16000
        needs_conversion = True

    if needs_conversion:
        tmp = os.path.join(tempfile.gettempdir(), f"nemo_mono_{os.path.basename(filepath)}")
        sf.write(tmp, data, sr)
        return tmp
    return filepath


# Transcribe the recordings
audio_files = [
    r"C:\Users\alexe\recordings\untitled.wav",
    r"C:\Users\alexe\recordings\voice-message.wav",
]

for audio_file in audio_files:
    print(f"\n{'=' * 60}")
    print(f"File: {os.path.basename(audio_file)}")
    print(f"{'=' * 60}")

    # Prepare audio (mono, 16kHz)
    prepped = prepare_audio(audio_file)
    info = sf.info(prepped)
    duration = info.frames / info.samplerate
    print(f"Duration: {duration:.1f}s | {info.samplerate}Hz | {info.channels}ch")

    t0 = time.time()
    result = model.transcribe([prepped])
    elapsed = time.time() - t0

    # Extract text from result
    if isinstance(result, list):
        text = result[0].text if hasattr(result[0], "text") else result[0]
    else:
        text = str(result)

    print(f"Inference time: {elapsed:.2f}s ({duration / elapsed:.1f}x realtime)")
    print(f"Transcription: {text}")
