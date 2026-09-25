"""Configuration constants for the LLM (text) attacker against anonymized speech."""
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

LLM_DIR = Path(__file__).resolve().parent.parent.parent / "llm"
FEATURE_EXTRACTORS_DIR = LLM_DIR / "feature_extractors"
OPTIMIZED_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "llm" / "optimized"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DISPLAY_NAME = "Qwen2.5-7B-Instruct"

# Audio for every method is a flat dir of "common_voice_en_<id>.wav" files.
# Identical to alm_anon_eval.py's ANON_METHODS.
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

PROMPT_FILES = {
    "gender": OPTIMIZED_PROMPTS_DIR / "optimized_prompt_gender.txt",
    "age": OPTIMIZED_PROMPTS_DIR / "optimized_prompt_age.txt",
    "accent": OPTIMIZED_PROMPTS_DIR / "optimized_prompt_accent.txt",
}
FEATURE_PLACEHOLDER = "[SPEECH FEATURES WILL BE INSERTED HERE]"

# Only the features named in each optimized prompt's FEATURES section are
# kept; copied verbatim from run_cv_dev_optimized_prompt_eval.py.
FEATURE_SUBSET_KEYS = {
    "gender": [
        "f0_mean_hz", "f0_median_hz", "f0_min_hz", "f0_max_hz", "f0_sd_hz",
        "f1_hz", "f2_hz", "f3_hz",
        "formant_spacing_hz", "vtl_cm",
    ],
    "age": [
        "jitter_local_pct", "jitter_rap_pct",
        "shimmer_local_pct", "shimmer_apq3_pct",
        "hnr_db",
    ],
    "accent": [
        "f3_below_f2_rate",
        "vot_ms_mean", "vot_ms_median", "vot_short_fraction",
        "pct_vocalic", "delta_c_ms", "npvi",
    ],
}

# Majority-vote / sampling / generation, identical to
# run_cv_dev_optimized_prompt_eval.py.
N_REPEATS = 3
GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_NEW_TOKENS = 300
RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 6

PARSE_ERROR_LABEL = "PARSE_ERROR"  # sentinel: never equals a true label -> always scored wrong

ATTRIBUTES = ("gender", "age", "accent")

ATTRIBUTE_VALID_LABELS = {
    "gender": ["male_masculine", "female_feminine"],
    "age": ["less than 31", "31 to 50", "more than 50"],
    "accent": ["India and South Asia", "England", "Australian", "Canadian", "United States"],
}

INSPECT_ONLY = False
SKIP_HF_PREFLIGHT = True  # Qwen2.5-7B-Instruct weights are already cached locally
