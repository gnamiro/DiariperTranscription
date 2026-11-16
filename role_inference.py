import argparse

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Set

from config import (
    PRIMARY_PAIR, HONORIFICS,  
    INTRO_PATTERNS, CACHE_ROLE, CACHE_MERGED
)
from utils import (
    _list_to_dataframe,
    load_json, save_json, cache_exists
)


def get_study_config(study_type="default"):
    if study_type == "TRUST_V1":
        return {
            "target_names": ["Ali", "Rowan"],
            "name_aliases": {
                "ali": {"alli","ally","allie","aly","alley","alee","alie","allee",
                        "aalie","aali","ale","ely","elie","aley","ali"},
                "rowan": {"roan","rowen","rowenn","roann","rohan","rohanne",
                        "rowynn","rowin","rowe","roen","rowan"}
            }
        }

    if study_type == "GSGS":
        return {
            "target_names": ["Ali", "Bailey"],
            "name_aliases": {
                "ali": {"alli","ally","allie","aly","alley","alee","alie","allee",
                        "aalie","aali","ale","ely","elie","aley","ali"},
                "bailey": {"baily","baile","bailee","baylee","baileigh","beyley","baley",
                            "baly","beyli","beylie","baely","baele","bailli","bailey"}
            }
        }
    
    if study_type == "TRUST_V2":
        return {
            "target_names": ["Charlie", "Jayden"],
            "name_aliases": {
                "charlie": {
                    "charlie","charly","charley","charli","charlee"
                },
                "jayden": {
                    "jayden","jaydan","jaiden","jaden","jaydon","jaedon"
                }
            }
        }

    return {
        "target_names": ["SpeakerA", "SpeakerB"],
        "name_aliases": {}
    }

def normalize_token(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    if parts and parts[0] in HONORIFICS:
        parts = parts[1:]
    return parts[0] if parts else ""


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def detect_self_intro_name(text: str) -> str:
    t = text.lower()
    for pat in INTRO_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return normalize_token(m.group("name"))
    return None


def fuzzy_match_name(cand, target_names, aliases, threshold=0.8):
    c = normalize_token(cand)
    if not c:
        return None

    # exact
    for t in target_names:
        if c == t.lower():
            return t

    # alias
    for t in target_names:
        if c in aliases.get(t.lower(), set()):
            return t

    # fuzzy
    best, score = None, 0
    for t in target_names:
        s = sim(c, t.lower())
        if s > score:
            best, score = t, s

    return best if score >= threshold else None


def infer_roles(segments, target_names, aliases):
    speaker_to_role = {}
    role_taken = set()

    # detect intros
    for seg in segments:
        name = detect_self_intro_name(seg["text"])
        if not name:
            continue

        match = fuzzy_match_name(name, target_names, aliases)

        if match:
            spk = seg["speaker"]
            speaker_to_role[spk] = match
            role_taken.add(match)
    print(f"speaker 2 role: {speaker_to_role}")

    # counterpart rule
    if len(role_taken) == 1:
        detected = list(role_taken)[0]
        other = [t for t in target_names if t != detected][0]

        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_to_role:
                speaker_to_role[spk] = other
                break

    # assign RA
    if role_taken:
        for seg in segments:
            if seg["speaker"] not in speaker_to_role:
                speaker_to_role[seg["speaker"]] = "RA"

    return speaker_to_role


# ===================== Main JSON pipeline =====================

def process_transcript_json(force, study_type="default"):
    
    if cache_exists(CACHE_ROLE) and not force:
        print("[Step 3a] Loading diarization from cache...")
        segments_list = load_json(CACHE_ROLE)
        return _list_to_dataframe(segments_list)
    
    config = get_study_config(study_type)
    target_names = config["target_names"]
    aliases = config["name_aliases"]
    print(f"CONFIG: {config}")

    segments = load_json(CACHE_MERGED)

    # infer roles
    speaker_roles = infer_roles(segments, target_names, aliases)
    print(speaker_roles)

    # attach roles
    for seg in segments:
        seg["speaker_role"] = speaker_roles.get(seg["speaker"], None)

    # save
    save_json(segments, CACHE_ROLE)

    print(f"✅ Saved → {CACHE_ROLE}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transcription pipeline")
    parser.add_argument(
        "--fn", action="store_true",
        help="Force re-run selected steps even if cache exists."
    )
    parser.add_argument(
        "-st", "--study-type", required=True,
        help="Define the type of study"
    )
    args = parser.parse_args()
    process_transcript_json(args.fn, args.study_type)