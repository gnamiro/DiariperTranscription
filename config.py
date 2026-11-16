# --- config.py ---
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
FILE_PATH  = "../Data/temp/GS1-FtF30A.MP4"
AUDIO_FILE = "temp_extracted_audio.wav"
CACHE_DIR  = "./cache"                   # intermediate outputs saved here
OUT_DIR    = "./output"                  # final outputs saved here

# WhisperX
DEVICE       = "cpu"                     # "cuda" if GPU available
COMPUTE_TYPE = "int8"                    # "float16" for GPU
BATCH_SIZE   = 1
WHISPER_MODEL = "large-v2"

# Audio extraction
EXTRACT_START    = "00:00:00"
EXTRACT_DURATION = "00:15:00"

# HuggingFace
HF_TOKEN = os.getenv("HF_TK")

# Ollama
OLLAMA_MODEL = "llama3:8b-instruct-q8_0"
OLLAMA_URL   = "http://localhost:11434/api/generate"

FIX_PROMPT_FILE = "./fix_prompt.md"
SPEAKER_PROMPT_FILE = "./speaker_prompt.md"

# Cache file names
CACHE_TRANSCRIPTION = os.path.join(CACHE_DIR, "transcription.json")
CACHE_ALIGNED       = os.path.join(CACHE_DIR, "aligned.json")
CACHE_DIARIZED      = os.path.join(CACHE_DIR, "diarized.json")
CACHE_MERGED        = os.path.join(CACHE_DIR, "merged.json")
CACHE_ROLE      = os.path.join(CACHE_DIR, "role_inference.json")
CACHE_PASS_A = os.path.join(CACHE_DIR, "llm_pass_a.json")   # corrected segments
CACHE_PASS_B = os.path.join(CACHE_DIR, "llm_pass_b.json")   # speaker analysis
CACHE_LLM           = os.path.join(CACHE_DIR, "llm_result.json")



# ROLE 
PRIMARY_PAIR = ["speaker 1", "speaker 2"]

HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "mx"}

INTRO_PATTERNS = [
    r"\b(?:i am|i'm|im)\s+(?P<name>[a-z][a-z\-']+(?:\s+[a-z][a-z\-']+)*)\b",
    r"\bmy\s+name\s+is\s+(?P<name>[a-z][a-z\-']+(?:\s+[a-z][a-z\-']+)*)\b",
    r"\bthis\s+is\s+(?P<name>[a-z][a-z\-']+(?:\s+[a-z][a-z\-']+)*)\b",
    r"\b(?P<name>[a-z][a-z\-']+)\s+here\b",
    r"\b(?:hey|hi|hello)[,!\.\s]*(?:i'?m|i am)?\s*(?P<name>[a-z][a-z\-']+)\b",
]