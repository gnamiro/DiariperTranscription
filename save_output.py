"""
save_transcripts.py
"""

import json
import os
import re
import datetime

from pathlib import Path
from collections import Counter

from utils import (
    ensure_dir, load_json
)
from config import CACHE_ROLE, OUT_DIR
import config
import subprocess

def clean_text(text: str) -> str:
    """
    Remove correction tags:
    <fix original="...">corrected</fix>
    """

    return re.sub(
        r'<fix original="[^"]*">(.*?)</fix>',
        r'\1',
        text
    )


def parse_hms_to_seconds(hms: str) -> float:
    """
    Convert HH:MM:SS to seconds.
    """

    parts = hms.split(":")

    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])

    return h * 3600 + m * 60 + s


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02}:{m:02}:{s:02}"

def get_creation_time(file_path):
    '''
    Try ffprobe metadata first, fall back to file system modification time.
    '''
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-show_entries', 'format_tags=creation_time',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(file_path)
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        if output:
            output = output.replace("T", " ").split(".")[0]
            return output
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fall back to file system modification time
    try:
        ts = os.path.getmtime(file_path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return None

def build_metadata(
    segments,
    file_path: Path,
    file_name: str,
    study_code: str,
    session_id: str
):

    study_name = file_path.parent.parent.name

    creation_time = get_creation_time(file_path)

    starts = [
        parse_hms_to_seconds(seg["start"])
        for seg in segments
    ]

    ends = [
        parse_hms_to_seconds(seg["end"])
        for seg in segments
    ]

    duration_sec = max(ends) - min(starts)

    speakers = {
        seg.get("speaker", "UNKNOWN")
        for seg in segments
    }

    metadata = [
        f"# Study Type: {study_name}",
        f"# Title: {study_code}-{session_id}",
        f"# Original File: {file_name}",
        f"# Date Modified: {creation_time}",
        f"# Duration: {format_duration(duration_sec)}",
        f"# Number of Speakers: {len(speakers)}",
        f"# Number of Segments: {len(segments)}",
        f"# Format: Start | End | Speaker | Role | Text",
        f"# Time Format: hh:mm:ss",
        "# --------------------------------------------------",
        ""
    ]

    return metadata


def get_top_speakers(segments, top_n=2):
    """
    Get most common speakers based on segment count.
    """

    speaker_counts = Counter(
        seg.get("speaker", "UNKNOWN")
        for seg in segments
    )

    top_speakers = [
        speaker
        for speaker, _
        in speaker_counts.most_common(top_n)
    ]

    return top_speakers


def write_txt(
    segments,
    output_path: Path,
    file_path: Path,
    file_name: str,
    study_code: str,
    session_id: str
):

    if len(segments) == 0:
        return

    metadata_lines = build_metadata(
        segments=segments,
        file_path=file_path,
        file_name=file_name,
        study_code=study_code,
        session_id=session_id
    )

    with open(output_path, "w", encoding="utf-8") as f:

        # Write metadata
        for line in metadata_lines:
            f.write(line + "\n")

        # Write transcript
        for seg in segments:

            speaker = seg.get(
                "speaker",
                "UNKNOWN"
            )

            role = seg.get(
                "speaker_role",
                "???"
            )

            start = seg.get(
                "start",
                "00:00:00"
            )

            end = seg.get(
                "end",
                "00:00:00"
            )

            text = clean_text(
                seg.get("text", "")
            )

            line = (
                f"{start} | "
                f"{end} | "
                f"{speaker} | "
                f"{role} | "
                f"{text}"
            )

            f.write(line + "\n")

    print(f"Saved → {output_path}")


def save_transcriptions(
    input_file_path,
    output_directory=OUT_DIR
):
    """
    Parameters
    ----------
    json_path : str or Path
        Input transcript JSON.

    output_directory : str or Path
        Base output directory.
    """
    json_path = CACHE_ROLE
    input_file_path = Path(input_file_path)

    # =====================================================
    # Load JSON
    # =====================================================

    segments = load_json(json_path)

    if not segments:
        raise ValueError(
            "Transcript JSON is empty."
        )
    
    print(input_file_path)
    file_name = input_file_path.stem
    print(file_name)
    # Example:
    # ABC-001
    parts = file_name.split("-")

    study_code = parts[0]

    session_id = '-'.join(parts[1:]) if len(parts) > 1 else ""
    if session_id != "":
        session_id = session_id[:3]
    # Example:
    # study/videos/file.json
    # -> extract study folder

    try:
        study_name = input_file_path.parent.parent.name

    except Exception:
        study_name = "UnknownStudy"

    # =====================================================
    # Output Structure
    # =====================================================

    base_output = (
        Path(output_directory) /
        study_name
    )

    single_dir = base_output / "single"
    dyad_dir = base_output / "dyad"
    full_dir = base_output / "full"

    ensure_dir(single_dir)
    ensure_dir(dyad_dir)
    ensure_dir(full_dir)

    # =====================================================
    # Identify Top Speakers
    # =====================================================

    top_speakers = get_top_speakers(
        segments,
        top_n=2
    )

    print(f"Top speakers: {top_speakers}")

    # =====================================================
    # SINGLE FILES
    # =====================================================

    for idx, speaker in enumerate(
        top_speakers,
        start=1
    ):

        speaker_segments = [
            seg
            for seg in segments
            if seg.get("speaker") == speaker
        ]

        output_path = (
            single_dir
            / f"{file_name}_speaker_{idx}_single.txt"
        )

        write_txt(
            segments=speaker_segments,
            output_path=output_path,
            file_path=input_file_path,
            file_name=file_name,
            study_code=study_code,
            session_id=session_id
        )

    # =====================================================
    # DYAD FILE
    # =====================================================

    dyad_segments = [
        seg
        for seg in segments
        if seg.get("speaker") in top_speakers
    ]

    dyad_output = (
        dyad_dir
        / f"{file_name}_dyad.txt"
    )

    write_txt(
        segments=dyad_segments,
        output_path=dyad_output,
        file_path=input_file_path,
        file_name=file_name,
        study_code=study_code,
        session_id=session_id
    )


    # =====================================================
    # FULL TXT
    # =====================================================

    full_txt_output = (
        full_dir
        / f"{file_name}_full.txt"
    )

    write_txt(
        segments=segments,
        output_path=full_txt_output,
        file_path=input_file_path,
        file_name=file_name,
        study_code=study_code,
        session_id=session_id
    )

    print("\nAll outputs saved successfully.")


# Example usage
if __name__ == "__main__":

    save_transcriptions(
        input_file_path=config.FILE_PATH
    )