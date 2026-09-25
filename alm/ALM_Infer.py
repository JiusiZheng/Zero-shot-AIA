#!/usr/bin/env python3
"""Zero-shot ALM attribute-inference attack: majority-vote gender/age/accent
classification with balanced accuracy per test set. Ground truth is never
shown to the model."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    ATTRIBUTES,
    CONTINUE_ON_MODEL_ERROR,
    DATASET_JSON_PATH,
    INSPECT_ONLY,
    MODEL_SPECS,
    MODELS_TO_RUN,
    N_VOTES,
    RANDOM_SEED,
    SKIP_HF_PREFLIGHT,
    ModelSpec,
)
from data import discover_categories, group_by_subset, load_records
from prompts import build_user_prompt
from scoring import balanced_accuracy
from util import eprint


# CLI

def select_models(names: list[str]) -> list[ModelSpec]:
    if names == ["all"]:
        return MODEL_SPECS
    lookup = {spec.backend: spec for spec in MODEL_SPECS}
    unknown = sorted(set(names) - set(lookup))
    if unknown:
        raise ValueError(f"Unknown model backend(s): {unknown}")
    return [lookup[name] for name in dict.fromkeys(names)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json", default=DATASET_JSON_PATH,
        help="JSON list with audio_path, subset and attribute labels. "
             "Defaults to DATASET_JSON_PATH set at the top of this script.",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS_TO_RUN,
        choices=["all"] + [s.backend for s in MODEL_SPECS],
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Seed for the majority-vote tie-break RNG (optional).",
    )
    parser.add_argument(
        "--inspect", action="store_true", default=INSPECT_ONLY,
        help="Print label/subset/prompt info without loading models or touching audio.",
    )
    parser.add_argument("--skip-hf-preflight", action="store_true", default=SKIP_HF_PREFLIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_json:
        raise ValueError(
            "No dataset JSON given. Either set DATASET_JSON_PATH at the top of "
            "this script, or pass --dataset-json /path/to/dataset.json."
        )

    rows = load_records(args.dataset_json)
    json_dir = Path(args.dataset_json).resolve().parent
    categories = {attr: discover_categories(rows, attr) for attr in ATTRIBUTES}
    subsets = group_by_subset(rows)

    for attr in ATTRIBUTES:
        counts = Counter(row[attr] for row in rows)
        eprint(f"[labels] {attr}: {len(categories[attr])} categories: {dict(sorted(counts.items()))}")
    for attr in ATTRIBUTES:
        eprint(f"[prompt: {attr}]\n{build_user_prompt(attr, categories[attr])}")
    eprint(f"[subsets/test-sets] {list(subsets.keys())} (sizes: {[len(v) for v in subsets.values()]})")

    if args.inspect:
        print(f"Inspection complete: {len(rows)} audio records across {len(subsets)} test sets; attributes={list(ATTRIBUTES)}")
        return

    from backends import load_model, preflight_hf_access, release_model, require_transformers_version
    from inference import run_attribute_on_subset

    models = select_models(args.models)

    # Fail before downloading any model if audio paths are unavailable.
    for row in rows:
        path = Path(row["audio_path"])
        if not path.is_absolute():
            path = (json_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio does not exist: {path}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for these ALMs.")
    require_transformers_version()
    if not args.skip_hf_preflight:
        preflight_hf_access(models)

    failures = []
    for spec in models:
        model = None
        rng = random.Random(args.seed)  # fresh, reproducible tie-break RNG per model
        try:
            model = load_model(spec)
            # subset outer loop -> finish test set A (all 3 attributes) before B, then C.
            per_attr_balacc: dict[str, dict[str, float | None]] = {attr: {} for attr in ATTRIBUTES}
            per_attr_hallucinations: dict[str, dict[str, int]] = {attr: {} for attr in ATTRIBUTES}
            per_attr_errors: dict[str, dict[str, int]] = {attr: {} for attr in ATTRIBUTES}
            per_attr_invalid_predictions: dict[str, dict[str, int]] = {attr: {} for attr in ATTRIBUTES}
            for subset_name, subset_rows in subsets.items():
                for attribute in ATTRIBUTES:
                    print(json.dumps({
                        "type": "condition", "model": spec.display_name,
                        "subset": subset_name, "attribute": attribute,
                        "n_votes": N_VOTES,
                    }), flush=True)
                    results = run_attribute_on_subset(
                        model, subset_rows, attribute, categories[attribute],
                        subset_name, spec.display_name, json_dir, rng,
                    )
                    bal_acc = balanced_accuracy(results, categories[attribute])
                    n_hallucinations = sum(r["n_hallucinations"] for r in results)
                    n_errors = sum(r["n_errors"] for r in results)
                    n_invalid_predictions = sum(1 for r in results if r["prediction_invalid"])
                    per_attr_balacc[attribute][subset_name] = bal_acc
                    per_attr_hallucinations[attribute][subset_name] = n_hallucinations
                    per_attr_errors[attribute][subset_name] = n_errors
                    per_attr_invalid_predictions[attribute][subset_name] = n_invalid_predictions
                    print(json.dumps({
                        "type": "subset_metric", "model": spec.display_name,
                        "subset": subset_name, "attribute": attribute,
                        "balanced_accuracy": bal_acc, "n_utterances": len(results),
                        # already counted as wrong inside balanced_accuracy; these fields are just visibility
                        "n_hallucinated_votes": n_hallucinations,
                        "n_error_votes": n_errors,
                        "n_utterances_with_invalid_prediction": n_invalid_predictions,
                    }), flush=True)

            for attribute in ATTRIBUTES:
                values = [v for v in per_attr_balacc[attribute].values() if v is not None]
                mean = statistics.mean(values) if values else None
                sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
                print(json.dumps({
                    "type": "summary", "model": spec.display_name, "attribute": attribute,
                    "n_test_sets": len(values),
                    "per_test_set_balanced_accuracy": per_attr_balacc[attribute],
                    "mean_balanced_accuracy": mean,
                    "sample_std_balanced_accuracy": sample_std,
                    "total_hallucinated_votes": sum(per_attr_hallucinations[attribute].values()),
                    "total_error_votes": sum(per_attr_errors[attribute].values()),
                    "total_utterances_with_invalid_prediction": sum(per_attr_invalid_predictions[attribute].values()),
                }), flush=True)
        except Exception as exc:
            failures.append((spec.display_name, str(exc)))
            eprint(f"[model FAILED] {spec.display_name}: {exc}")
            if not CONTINUE_ON_MODEL_ERROR:
                raise
        finally:
            release_model(model)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
