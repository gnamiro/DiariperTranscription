# --- diarize.py ---
"""
Step 3: Diarize audio and assign speakers to segments.
Saves results to cache so this step is skipped on re-runs.
Run independently:  python diarize.py
"""


import gc
import pandas as pd
import whisperx
from config import (
    AUDIO_FILE, DEVICE, HF_TOKEN,
    CACHE_ALIGNED, CACHE_DIARIZED, CACHE_MERGED
)
from utils import (
    save_json, load_json, cache_exists, seconds_to_hms,
    _list_to_dataframe, _dataframe_to_list
)

try:
    from whisperx import DiarizationPipeline
except ImportError:
    from whisperx.diarize import DiarizationPipeline

import torch
import torchaudio
from pyannote.audio import Model, Inference
import tempfile
import os 

def diarize(audio_file=AUDIO_FILE, force=False):
    """
    Run diarization on audio.
    Returns diarization as a pandas DataFrame (what assign_word_speakers expects).
    Caches as a list of dicts for serialization.
    """
    if cache_exists(CACHE_DIARIZED) and not force:
        print("[Step 3a] Loading diarization from cache...")
        segments_list = load_json(CACHE_DIARIZED)
        return _list_to_dataframe(segments_list)
 
    print("[Step 3a] Running diarization...")
    diarize_model    = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    audio            = whisperx.load_audio(audio_file)
    diarize_segments = diarize_model(audio)   # returns a DataFrame

    if not isinstance(diarize_segments, pd.DataFrame):
        raise TypeError(
            f"Expected DataFrame from DiarizationPipeline, got {type(diarize_segments)}. "
            "Check your WhisperX version."
        )
 
    segments_list = _dataframe_to_list(diarize_segments)
    save_json(segments_list, CACHE_DIARIZED)
 
    del diarize_model
    gc.collect()
 
    unique_speakers = set(s["speaker"] for s in segments_list)
    print(f"[Step 3a] Found {len(unique_speakers)} speaker(s): {unique_speakers}")
    print("[Step 3a] Done.")
    return diarize_segments   # return the live DataFrame


 
def merge_speakers(audio_file=AUDIO_FILE, force=False):
    """
    Assign speakers to WhisperX aligned segments using assign_word_speakers.
    Returns merged segment list with hh:mm:ss timestamps.
    Uses cache if available.
    """
    if cache_exists(CACHE_MERGED) and not force:
        print("[Step 3b] Loading merged segments from cache...")
        return load_json(CACHE_MERGED)
 
    # Ensure aligned transcription exists
    if not cache_exists(CACHE_ALIGNED):
        raise FileNotFoundError(
            f"Aligned transcription not found at '{CACHE_ALIGNED}'. "
            "Run step2_transcribe.py first."
        )
    aligned = load_json(CACHE_ALIGNED)
 
    # Get diarization DataFrame (from cache or fresh run)
    diarize_df = diarize(audio_file, force=force)

    diarize_df = diarize_df.copy()
    print(diarize_df.head())
    print("[Step 3b] Assigning speakers to segments...")
    result = whisperx.assign_word_speakers(diarize_df, aligned)
 
    # Build merged segments with hh:mm:ss and Unknown fallback
    merged     = []
    last_known = "SPEAKER_00"
    for seg in result["segments"]:
        speaker = seg.get("speaker", "Unknown")
        if speaker != "Unknown":
            last_known = speaker
        else:
            speaker = last_known
 
        merged.append({
            "speaker": speaker,
            "start":   seconds_to_hms(seg["start"]),
            "end":     seconds_to_hms(seg["end"]),
            "text":    seg["text"].strip()
        })
 
    save_json(merged, CACHE_MERGED)
    print(f"[Step 3b] Merged {len(merged)} segments.")
    print("[Step 3b] Done.")
    return merged
 
 
if __name__ == "__main__":
    merge_speakers()