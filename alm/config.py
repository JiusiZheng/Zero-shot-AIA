"""Module-level constants, ModelSpec dataclass and MODEL_SPECS list."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# config

# --- paths ---
PROJECT_DIR = str(Path(__file__).resolve().parent)
HF_CACHE_DIR = PROJECT_DIR
# EDIT ME: path to the dataset metadata JSON (must be supplied by the user).
DATASET_JSON_PATH = "/path/to/data/CV_test_meta/3_testsets_attribute_labels.json"  # e.g. "/path/to/data/dataset.json"; or pass --dataset-json

# --- which models to evaluate ---
MODELS_TO_RUN = ["all"]  # "all", or a list of backend names e.g. ["qwen25_omni", "voxtral"]

# --- majority-vote / sampling ---
N_VOTES = 3  # independent inferences per utterance per attribute
GENERATION_TEMPERATURE = 0.1  # >0 so the N_VOTES samples can actually differ
MAX_NEW_TOKENS_UTTERANCE = 64  # outputs are a single short category label
RANDOM_SEED = 42  # seeds the tie-break RNG when no strict majority; None = nondeterministic

# --- attributes & inference order per utterance ---
ATTRIBUTES = ("gender", "age", "accent")

# --- robustness / control flow ---
CONTINUE_ON_AUDIO_ERROR = True  # keep going after a single query_audio() failure
CONTINUE_ON_MODEL_ERROR = True  # keep going after an entire model fails to load/run
SKIP_HF_PREFLIGHT = False  # True skips the small-file HF access check before downloading models
INSPECT_ONLY = False  # True = only print label/prompt/subset info, never load models or touch audio

# None uses each model's default attention; set "flash_attention_2" only if installed
ATTN_IMPLEMENTATION: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    model_id: str
    backend: str
    extra_access_repos: tuple[str, ...] = ()


MODEL_SPECS = [
    ModelSpec(
        display_name="Qwen2.5-Omni-7B",
        model_id="Qwen/Qwen2.5-Omni-7B",
        backend="qwen25_omni",
    ),
    ModelSpec(
        display_name="Audio-Flamingo-3",
        model_id="nvidia/audio-flamingo-3-hf",
        backend="audio_flamingo3",
    ),
    ModelSpec(
        display_name="Gemma-3n-E4B-it",
        model_id="google/gemma-3n-E4B-it",
        backend="gemma3n",
    ),
    ModelSpec(
        display_name="Voxtral-Mini-3B-2507",
        model_id="mistralai/Voxtral-Mini-3B-2507",
        backend="voxtral",
    ),
]
