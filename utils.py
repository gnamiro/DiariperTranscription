# --- utils.py ---
import os
import json
import pandas as pd
from pathlib import Path


def _dataframe_to_list(df):
    """
    Convert WhisperX diarization DataFrame to a serializable list of dicts.
    DataFrame columns: segment (pyannote Segment), label, speaker
    """
    rows = []
    for _, row in df.iterrows():
        seg = row["segment"]
        rows.append({
            "start":   round(float(seg.start), 3),
            "end":     round(float(seg.end),   3),
            "speaker": row["speaker"]
        })
    return rows
 
 
# def _list_to_dataframe(segments_list):
#     """
#     Reconstruct a DataFrame from cached list so assign_word_speakers works.
#     """
#     from pyannote.core import Segment
#     records = []
#     for s in segments_list:
#         records.append({
#             "segment": Segment(s["start"], s["end"]),
#             "label":   s["speaker"],
#             "speaker": s["speaker"]
#         })
#     return pd.DataFrame(records)

def _list_to_dataframe(segments_list):
    records = []
    for s in segments_list:
        records.append({
            "start":   s["start"],
            "end":     s["end"],
            "label":   s["speaker"],
            "speaker": s["speaker"]
        })
    return pd.DataFrame(records)

def seconds_to_hms(seconds):
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved → {path}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def cache_exists(path):
    exists = os.path.exists(path)
    if exists:
        print(f"  Cache found → {path} (skipping step)")
    return exists

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
