# --- main.py ---
"""
Main orchestrator — runs the full pipeline or individual steps.

Steps:
  1 = extract audio
  2 = transcribe + align
  3 = diarize + assign speakers
  4 = LLM post-processing
  5 = save outputs
"""

import argparse
import os
import shutil
from pathlib import Path

from extract_audio  import extract_audio
from transcribe     import align
from diarize_v2     import merge_speakers
# from diarize        import merge_speakers  # pyannote
# from diarize_nemo   import merge_speakers  # Nvidia nemo
from role_inference import process_transcript_json
from save_output    import save_transcriptions
from utils          import ensure_dirs
import config
from config         import CACHE_DIR, OUT_DIR


STEP_MAP = {
    1: ("Extract audio",      lambda force, file_path, **kw: extract_audio(input_video=file_path, force=force)),
    2: ("Transcribe + align", lambda force, **kw: align(force=force)),
    3: ("Diarize + assign",   lambda force, **kw: merge_speakers(force=force)),
    4: ("Role Inference",     lambda force, study_type, **kw: process_transcript_json(force=force, study_type=study_type)),
    5: ("Save outputs",       lambda file_path, **kw: save_transcriptions(input_file_path=file_path)),
}


def run_pipeline(data_dir, steps=None, force=False, study_type=None):
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise ValueError(f"Data folder not found: {data_dir}")

    # Collect all files in the folder (non-recursive; add rglob for nested)
    files = sorted([f for f in data_path.iterdir() if f.is_file()])
    if not files:
        print(f"No files found in {data_dir}")
        return

    if steps is None:
        steps = list(STEP_MAP.keys())

    print(f"\n{'='*60}")
    print(f"Files found   : {len(files)}")
    print(f"Steps         : {steps}")
    print(f"Force         : {force}")
    print(f"Study type    : {study_type}")
    print(f"{'='*60}\n")

    ensure_dirs(CACHE_DIR, OUT_DIR)

    for file_path in files:
        print(f"\n{'#'*60}")
        print(f"FILE: {file_path}")
        print(f"{'#'*60}")
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR)  # recreate empty for next file
            print(f"\n  Cache cleared: {CACHE_DIR}")

        for step_num in steps:
            if step_num not in STEP_MAP:
                print(f"  Unknown step {step_num}, skipping.")
                continue

            label, fn = STEP_MAP[step_num]
            print(f"\n  {'─'*56}")
            print(f"  STEP {step_num}: {label}")
            print(f"  {'─'*56}")

            try:
                fn(force=force, file_path=file_path, study_type=study_type)
            except Exception as e:
                print(f"  ✗ Step {step_num} failed for {file_path.name}: {e}")
                break  # Stop remaining steps for this file; continue to next

        
    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcription pipeline")
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to the folder containing files to process."
    )
    parser.add_argument(
        "--steps", nargs="+", type=int,
        help="Steps to run (e.g. --steps 3 4). Default: all steps."
    )
    parser.add_argument(
        "--force", action="store_true",
        # "--force", 
        help="Force re-run selected steps even if cache exists."
    )
    parser.add_argument(
        "-st", "--study-type", required=True,
        help="Define the type of study."
    )
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        steps=args.steps,
        force=args.force,
        study_type=args.study_type,
    )