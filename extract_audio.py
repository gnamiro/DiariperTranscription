# --- extract_audio.py ---

import os
import subprocess
import soundfile as sf
import config

def extract_audio(
    input_video  = None,
    output_audio = None,
    start_time   = None,
    duration     = None,
    force        = False
):
    if input_video  is None: input_video  = config.FILE_PATH
    if output_audio is None: output_audio = config.AUDIO_FILE
    if start_time   is None: start_time   = config.EXTRACT_START
    if duration     is None: duration     = config.EXTRACT_DURATION

    if os.path.exists(output_audio) and not force:
        print(f"[Step 1] Audio already exists at '{output_audio}' — skipping extraction.")
        print("         Pass force=True to re-extract.")
    else:
        print(f"[Step 1] Extracting audio from '{input_video}'...")
        if os.path.exists(output_audio):
            os.remove(output_audio)

        cmd = ["ffmpeg", "-y"]
        if start_time:
            cmd += ["-ss", str(start_time)]
        if duration:
            cmd += ["-t", str(duration)]
        cmd += ["-i", input_video, "-ar", "16000", "-ac", "1", output_audio]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        print(f"[Step 1] Done.")

    info = sf.info(output_audio)
    print(f"[Step 1] Audio duration: {info.duration:.2f}s")
    return output_audio


if __name__ == "__main__":
    extract_audio()