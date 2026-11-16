"""
Runs in the diarizen_env conda environment.
Called as a subprocess from the main pipeline.
Usage: python diarize_worker.py --audio audio.wav --out cache/diarized.json --token hf_...
"""
import sys, os
sys.path.insert(0, "./DiariZen")

import argparse
import json
from diarizen.pipelines.inference import DiariZenPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out",   required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    print(f"[diarize_worker] Loading DiariZen pipeline...")
    pipeline = DiariZenPipeline.from_pretrained(
        "BUT-FIT/diarizen-wavlm-large-s80-md"
    )

    print(f"[diarize_worker] Running diarization on {args.audio}...")
    annotation = pipeline(args.audio)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start":   turn.start,
            "end":     turn.end,
            "speaker": speaker
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"[diarize_worker] Saved {len(segments)} segments to {args.out}")

if __name__ == "__main__":
    main()