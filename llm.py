# ── step4_llm.py ──────────────────────────────────────────────────────────────
"""
Step 4: Two-pass LLM post-processing via Ollama.
 
  Pass A — Fix transcription errors segment by segment (small, focused task)
  Pass B — Identify speakers from the fully corrected transcript (separate task)
 
Run independently:  python step4_llm.py
"""
 
import re
import json
import requests
from config import OLLAMA_MODEL, OLLAMA_URL, CACHE_MERGED, CACHE_LLM, CACHE_DIR
from utils import save_json, load_json, cache_exists
import os
 
# Intermediate cache paths
CACHE_PASS_A = os.path.join(CACHE_DIR, "llm_pass_a.json")   # corrected segments
CACHE_PASS_B = os.path.join(CACHE_DIR, "llm_pass_b.json")   # speaker analysis
 
 
# ── Ollama helper ─────────────────────────────────────────────────────────────
 
def query_ollama(system_prompt, user_message, model=OLLAMA_MODEL, use_json_format=True):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        }
    }

    if use_json_format:
        payload["format"] = "json"

    response = requests.post(OLLAMA_URL.replace("/generate", "/chat"), json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
 
 
def parse_json_response(raw):
    """Robustly extract and parse a JSON object from LLM output."""
    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.startswith("{"):
                raw = part
                break
 
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
 
    json_str = raw[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json.loads(json_str)
 
 
# ── Pass A: Semantic validation using context ─────────────────────────────────
 
PASS_A_SYSTEM = """You are an ASR (speech-to-text) error detector.

Your job is NOT to improve, rewrite, or paraphrase sentences.

You ONLY fix clear transcription errors where:
- A word was misheard (phonetic confusion)
- The word is incorrect given the context

Examples of VALID fixes:
- "where did you switch?" → "why did you switch?"
- "their going home" → "they're going home"

Examples of INVALID changes (DO NOT DO THESE):
- "That's very good for you" → "That's great"
- "I changed my mind" → "I switched my mind"
- Any stylistic rewrite
- Any grammar improvement
- Any shortening or rephrasing

STRICT RULES:
- You may change AT MOST 1 word (or 2 if it's a clear ASR phrase error)
- Do NOT rewrite the sentence
- Do NOT improve fluency
- Do NOT touch correct sentences
- Casual speech is ALWAYS valid ("yeah", "no way", "I mean", etc.)

If the sentence is acceptable spoken English → DO NOT CHANGE IT

Output ONLY:
{
  "changed": true or false,
  "original": "...",
  "correction": "..."
}
"""
 
 
def fix_segment(segment, context_segments, future_segments):
    """
    Run Pass A on a single segment using past + future context.
    Returns corrected segment dict.
    """

    # Combine: past + current + future
    all_lines = context_segments + [segment] + future_segments

    def clean_text(s):
        return re.sub(r'<fix original=\"[^\"]*\">(.*?)</fix>', r'\1', s['text'])

    # Build numbered lines
    numbered_lines = []
    for i, s in enumerate(all_lines):
        speaker = s.get("speaker", "UNKNOWN")
        numbered_lines.append(f"{i+1}: ({speaker}) {clean_text(s)}")

    numbered = "\n".join(numbered_lines)

    # Target index = current segment position
    target_index = len(context_segments) + 1

    user_msg = (
        f"{numbered}\n\n"
        f"IMPORTANT:\n"
        f"- Line {target_index} is the target\n"
        f"- Fix ONLY a wrong word if it is clearly misheard\n"
        f"- Do NOT rewrite or improve the sentence\n"
        f"- If the sentence is acceptable, return changed=false\n"
        f"Be careful:\n"
        f"- This is natural conversation.\n"
        f"- Do NOT change informal speech like: 'yeah', 'no way', 'I mean', 'you know', 'like'\n"
        f"- Do NOT fix grammar or style\n"
        f"- ONLY fix if there is a likely ASR error (wrong word that breaks meaning)\n\n"
        f"Focus especially on:\n"
        f"- Questions that don't match the answer\n"
        f"- Words that sound similar but change meaning (where/why, their/there, etc.)\n\n"
        f"Output JSON only."
    )

    raw = query_ollama(PASS_A_SYSTEM, user_msg)

    try:
        result = parse_json_response(raw)

        if result.get("changed") is True:
            original   = result.get("original")
            correction = result.get("correction")

            if not original or not correction:
                return segment
            
            # Reject if correction is too long (prevents rewrites)
            if len(correction.split()) > 2:
                print("[reject] correction too long → likely rewrite")
                return segment

            # Reject if original phrase is long (means model tried to replace sentence)
            if len(original.split()) > 3:
                print("[reject] original span too large → likely rewrite")
                return segment
            
            # Reject if sentence structure changes too much
            if len(segment["text"].split()) > 0:
                ratio = len(correction) / len(segment["text"])
                if ratio > 0.5:
                    print("[reject] too much change → likely paraphrase")
                    return segment

            original_text = segment["text"]

            # More tolerant matching
            pattern = re.compile(re.escape(original), re.IGNORECASE)

            if not pattern.search(original_text):
                print(f"[warn] '{original}' not found exactly, applying best-effort replacement")

            fix_tag = f'<fix original="{original}">{correction}</fix>'
            corrected_text = pattern.sub(fix_tag, original_text, count=1)

            print(f"[fix] '{original}' → '{correction}'")
            return {**segment, "text": corrected_text}

        return segment

    except json.JSONDecodeError:
        print("[Pass A] Parse failed, keeping original.")
        return segment
 
 
def run_pass_a(merged_segments, force=False, context_window=8):
    """
    Validate transcription using plain numbered conversation format.
    Each call sees the previous context_window segments + current as numbered lines.
    Returns list of validated segments.
    """
    if cache_exists(CACHE_PASS_A) and not force:
        print("[Step 4 / Pass A] Loading validated segments from cache...")
        return load_json(CACHE_PASS_A)
 
    print(f"[Step 4 / Pass A] Validating {len(merged_segments)} segments (context={context_window})...")
    corrected    = []
    change_count = 0
 
    for i, seg in enumerate(merged_segments):
        past = corrected[-context_window:] if corrected else []
        future = merged_segments[i+1:i+3]  # 1–2 future segments

        print(f"index {i}: with context:\n{past}\nand some future:\n{future}\n\n")
        fixed = fix_segment(seg, past, future)
        if fixed["text"] != seg["text"]:
            change_count += 1
        corrected.append(fixed)
 
    print(f"[Step 4 / Pass A] {change_count}/{len(merged_segments)} segments modified.")
    save_json(corrected, CACHE_PASS_A)
    print("[Step 4 / Pass A] Done.")
    return corrected
 
 
# ── Pass B: Identify speakers ─────────────────────────────────────────────────
 
PASS_B_SYSTEM = """You are a conversation analyst. You ONLY output valid JSON.
You will receive a transcript of a casual conversation between two people.
The speaker labels (SPEAKER_00, SPEAKER_01 etc.) may be wrong because the voices sound similar.
 
Your job is to re-identify the two speakers based ONLY on conversational patterns:
- Who initiates topics vs who responds
- Sentence length and complexity
- Vocabulary and tone differences
- Question-asking patterns
- Interruptions and reactions
 
Label the two speakers as PERSON_A and PERSON_B.
For each segment decide who is speaking and how confident you are.
 
Output ONLY this JSON, nothing else:
{
  "speaker_analysis": {
    "PERSON_A": "description of speech patterns that identify this person",
    "PERSON_B": "description of speech patterns that identify this person"
  },
  "segments": [
    {
      "index": 0,
      "speaker": "PERSON_A or PERSON_B",
      "speaker_confidence": "high or medium or low"
    }
  ]
}"""
 
 
def run_pass_b(corrected_segments, force=False):
    """
    Identify speakers from the corrected transcript.
    Returns speaker analysis and per-segment speaker assignments.
    """
    if cache_exists(CACHE_PASS_B) and not force:
        print("[Step 4 / Pass B] Loading speaker analysis from cache...")
        return load_json(CACHE_PASS_B)
 
    # Build a clean numbered transcript for the LLM (no timestamps — keep it simple)
    lines = []
    for i, seg in enumerate(corrected_segments):
        clean_text = re.sub(r'<fix original="[^"]*">(.*?)</fix>', r'\1', seg["text"])
        lines.append(f"[{i}] {seg['speaker']}: {clean_text}")
    transcript_text = "\n".join(lines)
 
    print("[Step 4 / Pass B] Identifying speakers...")
    raw = query_ollama(
        PASS_B_SYSTEM,
        f"Identify the two speakers in this transcript. Output only JSON.\n\n{transcript_text}"
    )
 
    try:
        result = parse_json_response(raw)
    except json.JSONDecodeError as e:
        print(f"[Step 4 / Pass B] JSON parse failed: {e}")
        print("Raw response:", raw)
        raise
 
    save_json(result, CACHE_PASS_B)
    print("[Step 4 / Pass B] Done.")
    return result
 
 
# ── Merge Pass A + Pass B ─────────────────────────────────────────────────────
 
def merge_passes(corrected_segments, speaker_result):
    """
    Combine corrected text from Pass A with speaker labels from Pass B.
    Returns final llm_result dict.
    """
    assignments = {
        item["index"]: item
        for item in speaker_result.get("segments", [])
    }
 
    final_segments = []
    for i, seg in enumerate(corrected_segments):
        assignment = assignments.get(i, {})
        final_segments.append({
            "speaker":            assignment.get("speaker", seg["speaker"]),
            "start":              seg["start"],
            "end":                seg["end"],
            "text":               seg["text"],
            "speaker_confidence": assignment.get("speaker_confidence", "low")
        })
 
    return {
        "speaker_analysis": speaker_result.get("speaker_analysis", {}),
        "segments":         final_segments
    }
 
 
# ── Main entry point ──────────────────────────────────────────────────────────
 
def run_llm(force=False):
    """
    Run two-pass LLM post-processing.
    Pass A: fix transcription errors (per segment)
    Pass B: identify speakers (full transcript)
    Returns merged llm_result dict.
    """
    if cache_exists(CACHE_LLM) and not force:
        print("[Step 4] Loading final LLM result from cache...")
        return load_json(CACHE_LLM)
 
    if not cache_exists(CACHE_MERGED):
        raise FileNotFoundError(
            f"Merged segments not found at '{CACHE_MERGED}'. "
            "Run step3_diarize.py first."
        )
 
    merged_segments = load_json(CACHE_MERGED)
 
    # Pass A — fix transcription
    corrected = run_pass_a(merged_segments, force=force)
 
    # Pass B — identify speakers
    speaker_result = run_pass_b(corrected, force=force)
 
    # Merge both passes into final result
    llm_result = merge_passes(corrected, speaker_result)
 
    save_json(llm_result, CACHE_LLM)
    print("[Step 4] Full LLM post-processing complete.")
    return llm_result
 
 
if __name__ == "__main__":
    run_llm()