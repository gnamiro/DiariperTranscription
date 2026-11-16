# --- transcribe.py ---
"""
Step 2: Transcribe and align audio using WhisperX.
Saves results to cache so this step is skipped on re-runs.
Run independently:  python transcribe.py
"""

import gc
import whisperx
from config import (
    AUDIO_FILE, DEVICE, COMPUTE_TYPE, BATCH_SIZE,
    WHISPER_MODEL, CACHE_TRANSCRIPTION, CACHE_ALIGNED
)
from utils import save_json, load_json, cache_exists


def transcribe(audio_file=AUDIO_FILE, force=False):
    """
    Transcribe audio with WhisperX.
    Returns raw result dict and detected language.
    Uses cache if available.
    """
    if cache_exists(CACHE_TRANSCRIPTION) and not force:
        print("[Step 2a] Loading transcription from cache...")
        return load_json(CACHE_TRANSCRIPTION)

    print(f"[Step 2a] Transcribing '{audio_file}'...")
    model  = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
    audio  = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)

    print(f"[Step 2a] Detected language: {result['language']}")
    print(f"[Step 2a] Segments: {len(result['segments'])}")

    # Cache: store segments + language (audio array is not serializable)
    cache_data = {
        "language": result["language"],
        "segments": result["segments"]
    }
    save_json(cache_data, CACHE_TRANSCRIPTION)

    del model
    gc.collect()
    print("[Step 2a] Done.")
    return cache_data


def align(audio_file=AUDIO_FILE, force=False):
    """
    Align transcription to get word-level timestamps.
    Returns aligned result dict.
    Uses cache if available.
    """
    if cache_exists(CACHE_ALIGNED) and not force:
        print("[Step 2b] Loading alignment from cache...")
        return load_json(CACHE_ALIGNED)

    transcription = transcribe(audio_file, force=False)

    print("[Step 2b] Aligning...")
    audio    = whisperx.load_audio(audio_file)
    model_a, metadata = whisperx.load_align_model(
        language_code=transcription["language"],
        device=DEVICE
    )
    result = whisperx.align(
        transcription["segments"],
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )

    cache_data = {
        "language": transcription["language"],
        "segments": result["segments"]
    }
    save_json(cache_data, CACHE_ALIGNED)

    del model_a
    gc.collect()
    print("[Step 2b] Done.")
    return cache_data


if __name__ == "__main__":
    align()