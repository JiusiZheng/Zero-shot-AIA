"""Configuration constants for the Qwen2.5-Omni-7B ALM attacker against
anonymized speech."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Edit these constants directly; CLI flags are optional overrides.

PROJECT_DIR = str(Path(__file__).resolve().parent.parent / "hf_cache")
HF_CACHE_DIR = PROJECT_DIR
JSON_PATH = "/path/to/data/CV_test_meta/3_testsets_attribute_labels.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
MODEL_DISPLAY_NAME = "Qwen2.5-Omni-7B"

# Audio for every method is a flat dir of "common_voice_en_<id>.wav" files.
ANON_METHODS: dict[str, dict[str, str]] = {
    "sttts": dict(
        display="IMS-SttTts (ims_sttts_pc)",
        audio_dir="/path/to/data/Anon_CV/anon_speech_sttts/ims_sttts_pc/cv_balance",
    ),
    "asrbn": dict(
        display="ASR-BN (asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1)",
        audio_dir="/path/to/data/Anon_CV/cv_balance_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1/wav",
    ),
    "mcadams": dict(
        display="McAdams",
        audio_dir="/path/to/data/Anon_CV/cv_balance_mcadams/wav",
    ),
    "nac": dict(
        display="NAC",
        audio_dir="/path/to/data/Anon_CV/cv_balance_nac/wav",
    ),
}

# Identical to ALM_Infer.py.
N_VOTES = 3
GENERATION_TEMPERATURE = 0.1
MAX_NEW_TOKENS_UTTERANCE = 64
RANDOM_SEED = 42

ATTRIBUTES = ("gender", "age", "accent")

CONTINUE_ON_AUDIO_ERROR = True
SKIP_HF_PREFLIGHT = True  # Qwen2.5-Omni-7B weights are already cached locally
INSPECT_ONLY = False
ATTN_IMPLEMENTATION: str | None = None
