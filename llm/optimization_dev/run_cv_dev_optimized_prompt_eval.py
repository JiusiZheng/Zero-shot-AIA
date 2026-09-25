#!/usr/bin/env python3
"""Text-only LLM attribute-inference attack on the CommonVoice dev set,
using the trimmed ("optimized") prompts from prompts/llm/optimized/.
Sibling of ../run_cv_dev_attribute_inference.py: same methodology
(../_common/), but only FEATURE_SUBSET_KEYS' feature subset is filled in."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FEATURE_EXTRACTORS_DIR = PROJECT_DIR / "feature_extractors"
RESULTS_DIR = SCRIPT_DIR / "results"

# User-editable: path to the CV_dev.json metadata file you provide.
CV_DEV_JSON = "/path/to/data/CV_test_meta/CV_dev.json"
HF_HOME_DIR = str(PROJECT_DIR / "hf_cache")  # shared with ../run_cv_dev_attribute_inference.py

FEATURES_CACHE_PATH = RESULTS_DIR / "cv_dev_optimized_features_cache.json"
# Full-feature cache already built by ../run_cv_dev_attribute_inference.py; reused, never modified.
ORIGINAL_FEATURES_CACHE_PATH = PROJECT_DIR / "results" / "cv_dev_features_cache.json"

PROMPTS_DIR = PROJECT_DIR.parent / "prompts" / "llm" / "optimized"
PROMPT_FILES = {
    "gender": PROMPTS_DIR / "optimized_prompt_gender.txt",
    "age": PROMPTS_DIR / "optimized_prompt_age.txt",
    "accent": PROMPTS_DIR / "optimized_prompt_accent.txt",
}
FEATURE_PLACEHOLDER = "[SPEECH FEATURES WILL BE INSERTED HERE]"

# Only the features named in each optimized prompt's FEATURES section (confirmed with the user).
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

MODEL_LLMS = [
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
]

DEFAULT_BATCH_SIZE = 6
BATCH_SIZE_OVERRIDES = {
    "google/gemma-3-12b-it": 3,
}

N_REPEATS = 3
GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_NEW_TOKENS = 300
RANDOM_SEED = 42

sys.path.insert(0, str(PROJECT_DIR))  # so `_common` resolves as a top-level import
from _common.labels import ATTRIBUTE_VALID_LABELS, PARSE_ERROR_LABEL, TABLE_COLUMNS  # noqa: E402
from _common.model import generate_batch  # noqa: E402
from _common import model as _common_model  # noqa: E402
from _common.parsing import majority_vote, parse_prediction  # noqa: E402
from _common.scoring import compute_metrics  # noqa: E402

sys.path.insert(0, str(FEATURE_EXTRACTORS_DIR))
import extract_accent  # noqa: E402
import extract_age  # noqa: E402
import extract_gender  # noqa: E402

FEATURE_FUNCS = {
    "gender": extract_gender.extract_gender,
    "age": extract_age.extract_age,
    "accent": extract_accent.extract_accent,
}


def slugify(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def load_cv_dev() -> list[dict]:
    with open(CV_DEV_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template(attribute: str) -> str:
    with open(PROMPT_FILES[attribute], "r", encoding="utf-8") as f:
        text = f.read()
    n = text.count(FEATURE_PLACEHOLDER)
    if n != 1:
        raise ValueError(
            f"{PROMPT_FILES[attribute].name} must contain the placeholder "
            f"'{FEATURE_PLACEHOLDER}' exactly once, found {n} times."
        )
    return text


def subset_features(attribute: str, full_features: dict) -> dict:
    keys = FEATURE_SUBSET_KEYS[attribute]
    return {k: full_features.get(k) for k in keys}


def build_full_prompt(attribute: str, full_features: dict) -> str:
    template = load_prompt_template(attribute)
    subset = subset_features(attribute, full_features)
    features_json = json.dumps(subset, indent=2)
    return template.replace(FEATURE_PLACEHOLDER, features_json)


# Stage 1: feature extraction + cache (CPU only)

def load_original_feature_cache() -> dict:
    if not ORIGINAL_FEATURES_CACHE_PATH.exists():
        return {}
    with open(ORIGINAL_FEATURES_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_cache(limit: int | None = None) -> dict:
    """cache[audio_path][attribute] = full features dict (subsetting happens
    at prompt-build time). Reuses the sibling pipeline's cache when
    available; resumable."""
    records = load_cv_dev()
    if limit:
        records = records[:limit]

    cache: dict = {}
    if FEATURES_CACHE_PATH.exists():
        with open(FEATURES_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

    original_cache = load_original_feature_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(records)
    for i, rec in enumerate(records, start=1):
        path = rec["audio_path"]
        entry = cache.setdefault(path, {})
        for attribute, fn in FEATURE_FUNCS.items():
            if attribute in entry:
                continue
            if path in original_cache and attribute in original_cache[path]:
                entry[attribute] = original_cache[path][attribute]
            else:
                entry[attribute] = fn(path)
        if i % 10 == 0 or i == n:
            print(f"  [features] {i}/{n} files done", file=sys.stderr)
            with open(FEATURES_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f)
    with open(FEATURES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"[features] wrote {FEATURES_CACHE_PATH}", file=sys.stderr)
    return cache


def load_feature_cache() -> dict:
    if not FEATURES_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_CACHE_PATH} not found. Run with --stage features first."
        )
    with open(FEATURES_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Stage 2: LLM loading (delegates to _common/model.py)

def get_llm(model_name: str):
    return _common_model.get_llm(model_name, HF_HOME_DIR)


# Stage 2 driver: run one model over all 3 attributes

def run_inference_for_model(model_name: str, attributes: list[str], limit: int | None = None) -> dict:
    records = load_cv_dev()
    if limit:
        records = records[:limit]
    feature_cache = load_feature_cache()
    tokenizer, model = get_llm(model_name)
    batch_size = BATCH_SIZE_OVERRIDES.get(model_name, DEFAULT_BATCH_SIZE)

    all_results = {}
    for attribute in attributes:
        print(f"\n[infer] model={model_name} attribute={attribute} n={len(records)}", file=sys.stderr)
        rng = random.Random(RANDOM_SEED)

        prompts = []
        for rec in records:
            feats = feature_cache[rec["audio_path"]][attribute]
            prompts.append(build_full_prompt(attribute, feats))

        flat_prompts = [p for p in prompts for _ in range(N_REPEATS)]
        flat_responses = generate_batch(
            tokenizer, model, flat_prompts, GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE, batch_size,
        )

        sample_results = []
        for i, rec in enumerate(records):
            responses = flat_responses[i * N_REPEATS:(i + 1) * N_REPEATS]
            parsed = [parse_prediction(attribute, r) for r in responses]
            final_label, was_tie_break = majority_vote(parsed, rng)
            sample_results.append({
                "audio_path": rec["audio_path"],
                "true_label": rec[attribute],
                "raw_responses": responses,
                "parsed_labels": parsed,
                "final_label": final_label,
                "tie_break_random": was_tie_break,
            })
            if (i + 1) % 25 == 0 or (i + 1) == len(records):
                print(f"  [{i + 1}/{len(records)}] last true={rec[attribute]!r} pred={final_label!r}", file=sys.stderr)

        metrics = compute_metrics(sample_results, ATTRIBUTE_VALID_LABELS[attribute])
        all_results[attribute] = {"samples": sample_results, "metrics": metrics}
        print(
            f"  [{attribute}] balanced_accuracy={metrics['balanced_accuracy'] * 100:.2f}%  "
            f"full_parse_failures={metrics['n_full_parse_failures']}/{len(records)}",
            file=sys.stderr,
        )

    return {"model_name": model_name, "attributes": all_results}


# Stage 3: aggregate all models into a plain-text report

def results_path_for(model_name: str) -> Path:
    return RESULTS_DIR / f"cv_dev_optimized_results_{slugify(model_name)}.json"


def aggregate_and_print_table(model_names: list[str]) -> None:
    all_model_results = {}
    for model_name in model_names:
        path = results_path_for(model_name)
        if not path.exists():
            print(f"[aggregate] WARNING: missing {path}, skipping {model_name}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            all_model_results[model_name] = json.load(f)

    out_path = RESULTS_DIR / "cv_dev_optimized_attribute_inference_table.txt"
    lines = []
    lines.append("CommonVoice DEV set (CV_dev.json, n=150) -- LLM text-only attribute inference")
    lines.append("Using the TRIMMED/optimized prompts (optimization_dev/optimized_prompts/), each fed only")
    lines.append("the subset of extractor features actually named in that prompt's FEATURES section.")
    lines.append(f"3 independent samples per (model, attribute, utterance), majority vote, temperature={GENERATION_TEMPERATURE}")
    lines.append("=" * 100)

    for model_name, result in all_model_results.items():
        lines.append(f"\nModel: {model_name}")
        for attribute in ("gender", "age", "accent"):
            m = result["attributes"][attribute]["metrics"]
            lines.append(
                f"  {attribute:<8} balanced_accuracy={m['balanced_accuracy'] * 100:6.2f}%  "
                f"(full_parse_failures={m['n_full_parse_failures']}/{m['n_total']})"
            )
            for label, recall in m["per_class_recall"].items():
                r = f"{recall * 100:6.2f}%" if recall is not None else "  n/a "
                lines.append(f"      recall[{label!r:<24}] = {r}")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[aggregate] wrote {out_path}", file=sys.stderr)


# CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["features", "infer", "aggregate", "prompt_preview"])
    p.add_argument("--model_index", type=int, default=None, help="index into MODEL_LLMS (for --stage infer)")
    p.add_argument("--model_name", type=str, default=None, help="override model name directly (for --stage infer)")
    p.add_argument("--attributes", type=str, default="gender,age,accent")
    p.add_argument("--limit", type=int, default=None, help="use only the first N dev-set records (smoke test)")
    p.add_argument("--sample_index", type=int, default=0, help="for --stage prompt_preview")
    args = p.parse_args()

    attributes = args.attributes.split(",")

    if args.stage == "features":
        build_feature_cache(limit=args.limit)
        return

    if args.stage == "prompt_preview":
        feature_cache = build_feature_cache(limit=args.sample_index + 1)
        records = load_cv_dev()
        rec = records[args.sample_index]
        attribute = attributes[0]
        feats = feature_cache[rec["audio_path"]][attribute]
        print(build_full_prompt(attribute, feats))
        return

    if args.stage == "infer":
        if args.model_name:
            model_name = args.model_name
        elif args.model_index is not None:
            model_name = MODEL_LLMS[args.model_index]
        else:
            raise SystemExit("--stage infer requires --model_index or --model_name")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result = run_inference_for_model(model_name, attributes, limit=args.limit)
        out_path = results_path_for(model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[infer] wrote {out_path}", file=sys.stderr)
        return

    if args.stage == "aggregate":
        aggregate_and_print_table(MODEL_LLMS)
        return


if __name__ == "__main__":
    main()
