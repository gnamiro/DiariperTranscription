# --- diarize.py ---
"""
Step 3: Diarize audio and assign speakers to segments.
Uses DiariZen (BUT-FIT/diarizen-wavlm-large-s80-md) instead of pyannote via WhisperX.
Saves results to cache so this step is skipped on re-runs.
Run independently:  python diarize.py
"""

import gc
import tempfile
import os

import pandas as pd
import numpy as np
import whisperx

from config import (
    AUDIO_FILE, DEVICE, HF_TOKEN,
    CACHE_ALIGNED, CACHE_DIARIZED, CACHE_MERGED
)
from utils import (
    save_json, load_json, cache_exists, seconds_to_hms,
    _list_to_dataframe, _dataframe_to_list
)

import subprocess
import shutil

DIARIZEN_PYTHON = shutil.which("python") or "python"  # override below

DIARIZEN_PYTHON = os.path.expanduser("~/miniconda3/envs/diarizen_env/bin/python")
## On Windows:
# DIARIZEN_PYTHON = os.path.expanduser("~/miniconda3/envs/diarizen_env/python.exe")

def diarize(audio_file=AUDIO_FILE, force=False):
    if cache_exists(CACHE_DIARIZED) and not force:
        print("[Step 3a] Loading diarization from cache...")
        return _list_to_dataframe(load_json(CACHE_DIARIZED))

    # Convert to wav if needed (DiariZen needs a wav file path)
    audio_path = audio_file
    tmp_wav = None
    if not audio_file.lower().endswith(".wav"):
        import torchaudio, tempfile
        waveform, sample_rate = torchaudio.load(audio_file)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp_wav.name, waveform, sample_rate)
        audio_path = tmp_wav.name
        print(f"[Step 3a] Converted to temporary WAV: {audio_path}")

    print("[Step 3a] Running DiariZen via subprocess...")
    try:
        subprocess.run(
            [
                DIARIZEN_PYTHON, "diarize_worker.py",
                "--audio", audio_path,
                "--out",   CACHE_DIARIZED,
                "--token", HF_TOKEN or ""
            ],
            check=True
        )
    finally:
        if tmp_wav is not None:
            os.unlink(tmp_wav.name)

    segments_list = load_json(CACHE_DIARIZED)
    unique_speakers = set(s["speaker"] for s in segments_list)
    print(f"[Step 3a] Found {len(unique_speakers)} speaker(s): {unique_speakers}")
    print("[Step 3a] Done.")
    return _list_to_dataframe(segments_list)

def assign_word_speakers(diarize_df, aligned, fill_nearest=True):
    """
    Assign speakers to each segment and word in the aligned transcription.

    Strategy: for each segment (and each word within it), compute how much
    it overlaps with every diarization speaker interval, then pick the speaker
    with the greatest total overlap. Optionally, if there's no overlap at all,
    fall back to the nearest speaker by gap distance.

    Args:
        diarize_df   : pd.DataFrame with columns [start, end, speaker]
                       (one row per diarization interval from DiariZen/pyannote)
        aligned      : dict with key "segments", each segment having:
                         - start, end  (float seconds)
                         - text        (str)
                         - words       (list of {word, start, end, score})
        fill_nearest : if True, segments/words with no speaker overlap get
                       assigned to the temporally closest speaker interval

    Returns:
        dict {"segments": [...]}, same structure as `aligned` but with
        "speaker" keys added to each segment and each word that has timing.
    """
    def _best_speaker(start, end):
        """
        Given a time span [start, end], find the diarization speaker with
        the most overlap. Returns None if there is no overlap and
        fill_nearest is False.
        """
        # Overlap = min(ends) - max(starts), clamped to 0
        intersection = (
            np.minimum(diarize_df["end"], end)
            - np.maximum(diarize_df["start"], start)
        )

        overlapping = diarize_df[intersection > 0]

        if len(overlapping) > 0:
            # Sum overlap per speaker, pick the one with the most
            return (
                overlapping
                .assign(intersection=intersection[intersection > 0])
                .groupby("speaker")["intersection"]
                .sum()
                .idxmax()
            )

        if fill_nearest:
            # No overlap — fall back to the interval whose gap to [start,end]
            # is smallest. Negative intersection = gap magnitude.
            return diarize_df.loc[intersection.idxmax(), "speaker"]

        return None

    segments = aligned["segments"]

    for seg in segments:
        # --- Assign speaker to the whole segment ---
        speaker = int(_best_speaker(seg["start"], seg["end"]))
        if speaker is not None:
            seg["speaker"] = speaker

        # --- Assign speaker to each individual word ---
        for word in seg.get("words", []):
            # Some words (e.g. punctuation-only tokens) may lack timing
            if "start" not in word or "end" not in word:
                continue
            word_speaker = int(_best_speaker(word["start"], word["end"]))
            if word_speaker is not None:
                word["speaker"] = word_speaker

    return {"segments": segments}

def merge_speakers(audio_file=AUDIO_FILE, force=False):
    """
    Assign speakers to WhisperX aligned segments using assign_word_speakers.
    Returns merged segment list with hh:mm:ss timestamps.
    Uses cache if available.
    """
    if cache_exists(CACHE_MERGED) and not force:
        print("[Step 3b] Loading merged segments from cache...")
        return load_json(CACHE_MERGED)

    if not cache_exists(CACHE_ALIGNED):
        raise FileNotFoundError(
            f"Aligned transcription not found at '{CACHE_ALIGNED}'. "
            "Run step2_transcribe.py first."
        )
    aligned = load_json(CACHE_ALIGNED)

    diarize_df = diarize(audio_file, force=force)
    diarize_df = diarize_df.copy()
    # print(diarize_df.head(5))

    print("[Step 3b] Assigning speakers to segments...")
    result = assign_word_speakers(diarize_df, aligned)

    print("[Step 3b] Speakers are assigned to the segments")
    print(result['segments'])
    merged = []
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

    
    print(f"[Step 3b] Merged {len(merged)} segments.")
    save_json(merged, CACHE_MERGED)
    print("[Step 3b] Done.")
    return merged


if __name__ == "__main__":
    merge_speakers()