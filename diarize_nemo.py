# --- diarize.py ---
"""
Step 3: Diarize audio and assign speakers to segments using NVIDIA NeMo.
NeMo's ClusteringDiarizer / MSDD pipeline replaces pyannote for higher accuracy.

Dependencies:
    pip install nemo_toolkit[asr] omegaconf torch torchaudio

Run independently:  python diarize.py
"""

import gc
import os
import json
import wget
import tempfile
import pandas as pd

import whisperx
from omegaconf import OmegaConf


from config import (
    AUDIO_FILE, DEVICE,
    CACHE_ALIGNED, CACHE_DIARIZED, CACHE_MERGED
)
from utils import (
    save_json, load_json, cache_exists, seconds_to_hms,
    _list_to_dataframe, _dataframe_to_list
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# def _rttm_to_dataframe(rttm_path: str) -> pd.DataFrame:
#     """
#     Parse a NeMo RTTM file into a DataFrame that matches the schema
#     WhisperX's assign_word_speakers expects:
#         columns: start  end  speaker
#     RTTM line format:
#         SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
#     """
#     rows = []
#     with open(rttm_path) as f:
#         for line in f:
#             line = line.strip()
#             if not line or not line.startswith("SPEAKER"):
#                 continue
#             parts = line.split()
#             start    = float(parts[3])
#             duration = float(parts[4])
#             speaker  = parts[7]
#             rows.append({
#                 "start":   start,
#                 "end":     start + duration,
#                 "speaker": speaker,
#             })

#     if not rows:
#         raise ValueError(f"No SPEAKER entries found in RTTM file: {rttm_path}")

#     df = pd.DataFrame(rows, columns=["start", "end", "speaker"])
#     return df.sort_values("start").reset_index(drop=True)

def _rttm_to_dataframe(rttm_path: str, time_threshold: float = 0.5) -> pd.DataFrame:
    """
    Parse a NeMo RTTM file into a DataFrame with columns [start, end, speaker].
    Merges consecutive same-speaker segments that are close together.

    RTTM format (space-separated):
        SPEAKER <file_id> <chn> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    columns = [
        "type", "file_id", "channel_id", "begin_time", "duration",
        "ortho", "speaker_type", "speaker_name", "confidence_score", "signal_lookahead"
    ]
    rttm_df = pd.read_csv(
        rttm_path,
        sep=r"\s+",
        names=columns,
        comment="#",
        engine="python"
    )
    print(rttm_df.head())

    # Merge consecutive same-speaker segments (same logic as original)
    audio_sections = []
    current_section = None
    for _, row in rttm_df.iterrows():
        begin_time = row["begin_time"]
        end_time   = begin_time + row["duration"]
        speaker    = row["speaker_name"]

        if row["duration"] > time_threshold:
            if (current_section is None
                    or current_section[2] != speaker
                    or begin_time - current_section[1] >= 1):
                if current_section:
                    audio_sections.append(current_section)
                current_section = [begin_time, end_time, speaker]
            else:
                current_section[1] = end_time  # extend current segment

    if current_section:
        audio_sections.append(current_section)

    if not audio_sections:
        raise ValueError(f"No valid SPEAKER entries found in RTTM: {rttm_path}")

    # Convert to DataFrame matching [start, end, speaker] schema
    df = pd.DataFrame(audio_sections, columns=["start", "end", "speaker"])
    return df.sort_values("start").reset_index(drop=True)


def _build_nemo_config(audio_path, model_config_url="https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"):
    # Prepare metadata for NeMo diarization
    meta = {'audio_filepath': audio_path, 'offset': 0, 'duration': None, 'label': 'infer', 'text': '-', 'num_speakers': None, 'rttm_filepath': None, 'uem_filepath': None}
    with open("input_manifest.json", 'w', encoding='utf-8') as fp:
        json.dump(meta, fp)
        fp.write('\n')

    # Load or download diarizer configuration
    model_config_path = "model_data/diar_infer_telephonic.yaml"
    if not os.path.exists(model_config_path):
        os.makedirs("model_data", exist_ok=True)
        model_config_path = wget.download(model_config_url, "model_data")
    
    # Set configuration parameters
    config = OmegaConf.load(model_config_path)
    config.diarizer.manifest_filepath = "input_manifest.json"
    config.diarizer.out_dir = "oracle_vad"
    config.device = "cpu" # CHANGE THIS TO cuda TO USE NVIDIA GPU
    config.num_workers = 0
    
    return config

# ---------------------------------------------------------------------------
# Main steps
# ---------------------------------------------------------------------------

def diarize(audio_file: str = AUDIO_FILE, force: bool = False) -> pd.DataFrame:
    """
    Run NeMo diarization on audio.
    Returns a DataFrame with columns [start, end, speaker].
    Caches as a list-of-dicts for serialisation.
    """
    if cache_exists(CACHE_DIARIZED) and not force:
        print("[Step 3a] Loading diarization from cache...")
        return _list_to_dataframe(load_json(CACHE_DIARIZED))

    print("[Step 3a] Running NeMo diarization...")

    # Import here so the module is still importable without NeMo installed
    # (cache-hit path never needs it)
    try:
        from nemo.collections.asr.models import ClusteringDiarizer
    except ImportError as exc:
        raise ImportError(
            "NeMo is required for diarization. "
            "Install it with:  pip install nemo_toolkit[asr]"
        ) from exc

    with tempfile.TemporaryDirectory() as work_dir:
        cfg = _build_nemo_config(audio_file)

        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        # NeMo writes RTTM to  <out_dir>/pred_rttms/<audio_stem>.rttm
        # audio_stem = os.path.splitext(os.path.basename(audio_file))[0]
        # rttm_path  = os.path.join(work_dir, "pred_rttms", f"{audio_stem}.rttm")
        rttm_path = 'oracle_vad/pred_rttms/temp_extracted_audio.rttm'
        if not os.path.exists(rttm_path):
            raise FileNotFoundError(
                f"NeMo did not produce an RTTM file at expected path: {rttm_path}\n"
                f"Contents of work_dir: {os.listdir(work_dir)}"
            )
        print("preparing to convert rttm to df")
        diarize_df = _rttm_to_dataframe(rttm_path)
    print("dataframe is ready!")
    print(diarize_df)
    # Cache as list-of-dicts
    # save_json(_dataframe_to_list(diarize_df), CACHE_DIARIZED)

    del diarizer
    gc.collect()

    unique_speakers = set(diarize_df["speaker"].unique())
    print(f"[Step 3a] Found {len(unique_speakers)} speaker(s): {unique_speakers}")
    print("[Step 3a] Done.")
    return diarize_df

def assign_speakers_to_words(aligned: dict, diarize_df) -> dict:
    """
    For each word in aligned segments, find the diarization speaker
    whose interval contains the word's midpoint. Assigns speaker to
    each word, then picks the majority speaker for the segment.
    """
    import pandas as pd
    from collections import Counter

    for seg in aligned["segments"]:
        word_speakers = []

        for word in seg.get("words", []):
            w_start = word.get("start")
            w_end   = word.get("end")

            if w_start is None or w_end is None:
                word["speaker"] = None
                continue

            midpoint = (w_start + w_end) / 2.0

            # Find rows where midpoint falls within [start, end]
            match = diarize_df[
                (diarize_df["start"] <= midpoint) &
                (diarize_df["end"]   >= midpoint)
            ]

            if not match.empty:
                speaker = match.iloc[0]["speaker"]
            else:
                # Fallback: nearest interval by distance to midpoint
                diarize_df["_dist"] = (
                    (diarize_df["start"] + diarize_df["end"]) / 2.0 - midpoint
                ).abs()
                speaker = diarize_df.loc[diarize_df["_dist"].idxmin(), "speaker"]
                diarize_df.drop(columns=["_dist"], inplace=True)

            word["speaker"] = speaker
            word_speakers.append(speaker)

        # Assign majority speaker to the segment
        if word_speakers:
            seg["speaker"] = Counter(word_speakers).most_common(1)[0][0]
        else:
            seg["speaker"] = None

    return aligned

def merge_speakers(audio_file: str = AUDIO_FILE, force: bool = False) -> list:
    if cache_exists(CACHE_MERGED) and not force:
        print("[Step 3b] Loading merged segments from cache...")
        return load_json(CACHE_MERGED)

    if not cache_exists(CACHE_ALIGNED):
        raise FileNotFoundError(
            f"Aligned transcription not found at '{CACHE_ALIGNED}'. "
            "Run step2_transcribe.py first."
        )
    aligned    = load_json(CACHE_ALIGNED)
    diarize_df = diarize(audio_file, force=force)

    print("[Step 3b] Assigning speakers to segments...")
    result = assign_speakers_to_words(aligned, diarize_df)   # <-- replaces whisperx call
    print(result)
    merged     = []
    last_known = "speaker_0"
    for seg in result["segments"]:
        speaker = seg.get("speaker") or "Unknown"
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
    # return merged


# def merge_speakers(audio_file: str = AUDIO_FILE, force: bool = False) -> list:
#     """
#     Assign speakers to WhisperX aligned segments using assign_word_speakers.
#     Returns merged segment list with hh:mm:ss timestamps.
#     Uses cache if available.
#     """
#     if cache_exists(CACHE_MERGED) and not force:
#         print("[Step 3b] Loading merged segments from cache...")
#         return load_json(CACHE_MERGED)

#     if not cache_exists(CACHE_ALIGNED):
#         raise FileNotFoundError(
#             f"Aligned transcription not found at '{CACHE_ALIGNED}'. "
#             "Run step2_transcribe.py first."
#         )
#     aligned = load_json(CACHE_ALIGNED)

#     diarize_df = diarize(audio_file, force=force)

#     print("[Step 3b] Assigning speakers to segments...")
#     result = whisperx.assign_word_speakers(diarize_df, aligned)

#     merged     = []
#     last_known = "SPEAKER_00"
#     for seg in result["segments"]:
#         speaker = seg.get("speaker", "Unknown")
#         if speaker != "Unknown":
#             last_known = speaker
#         else:
#             speaker = last_known

#         merged.append({
#             "speaker": speaker,
#             "start":   seconds_to_hms(seg["start"]),
#             "end":     seconds_to_hms(seg["end"]),
#             "text":    seg["text"].strip()
#         })

#     save_json(merged, CACHE_MERGED)
#     print(f"[Step 3b] Merged {len(merged)} segments.")
#     print("[Step 3b] Done.")
#     # return merged


if __name__ == "__main__":
    merge_speakers()