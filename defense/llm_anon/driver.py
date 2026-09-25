"""Evaluation driver, HF preflight check, and summary printing for the LLM
(text) attacker."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    ANON_METHODS,
    ATTRIBUTE_VALID_LABELS,
    ATTRIBUTES,
    GENERATION_MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE,
    HF_CACHE_DIR,
    MODEL_DISPLAY_NAME,
    N_REPEATS,
    PARSE_ERROR_LABEL,
)
from data import anon_audio_path, build_full_prompt  # noqa: E402
from model import generate_batch  # noqa: E402
from scoring import aggregate_across_subsets, majority_vote, parse_prediction, subset_metrics  # noqa: E402
from util import eprint  # noqa: E402


def preflight_hf_access(model_id: str) -> None:
    from huggingface_hub import hf_hub_download, whoami
    try:
        account = whoami()
        eprint(f"[hf] Authenticated as: {account.get('name', '<unknown>')}")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication is unavailable. Run `hf auth login`."
        ) from exc
    try:
        hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=HF_CACHE_DIR)
        eprint(f"[hf-access OK] {model_id}")
    except Exception as exc:
        raise RuntimeError(f"Hugging Face access check failed for {model_id}: {exc}") from exc


def run_attribute_on_subset(
    tokenizer: Any, model: Any, rows: list[dict[str, Any]], attribute: str,
    subset_name: str, method_key: str, audio_dir: str,
    feature_cache: dict[str, dict[str, Any]], rng: random.Random, batch_size: int,
) -> list[dict[str, Any]]:
    prompts = []
    apaths = []
    for row in rows:
        apath = str(anon_audio_path(row["audio_path"], audio_dir))
        apaths.append(apath)
        prompts.append(build_full_prompt(attribute, feature_cache[apath][attribute]))

    flat_prompts = [p for p in prompts for _ in range(N_REPEATS)]
    flat_responses = generate_batch(
        tokenizer, model, flat_prompts, GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE, batch_size,
    )

    results = []
    for i, row in enumerate(rows):
        responses = flat_responses[i * N_REPEATS:(i + 1) * N_REPEATS]
        parsed = [parse_prediction(attribute, r) for r in responses]
        final_label, was_tie_break = majority_vote(parsed, rng)

        result = dict(
            anon_method=method_key, model=MODEL_DISPLAY_NAME, subset=subset_name,
            attribute=attribute, speaker_id=row.get("speaker_id"), anon_audio_path=apaths[i],
            raw_responses=responses, parsed_labels=parsed, tie_break_random=was_tie_break,
            prediction=final_label, true_label=row[attribute],
        )
        print(json.dumps({"type": "utterance_result", **result}, ensure_ascii=False), flush=True)
        results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            eprint(
                f"  [{method_key}][{subset_name}][{attribute}] {i + 1}/{len(rows)} "
                f"last true={row[attribute]!r} pred={final_label!r}"
            )
    return results


def evaluate_method(
    tokenizer: Any, model: Any, method_key: str, audio_dir: str,
    subsets: "OrderedDict[str, list[dict[str, Any]]]", feature_cache: dict[str, dict[str, Any]],
    seed: int, batch_size: int,
) -> dict[str, Any]:
    per_attribute: dict[str, Any] = {}
    for attribute in ATTRIBUTES:
        per_subset = []
        for subset_name, subset_rows in subsets.items():
            rng = random.Random(seed)
            print(json.dumps({
                "type": "condition", "anon_method": method_key, "model": MODEL_DISPLAY_NAME,
                "subset": subset_name, "attribute": attribute, "n_repeats": N_REPEATS,
            }), flush=True)
            results = run_attribute_on_subset(
                tokenizer, model, subset_rows, attribute, subset_name, method_key,
                audio_dir, feature_cache, rng, batch_size,
            )
            categories = tuple(ATTRIBUTE_VALID_LABELS[attribute])
            ba, recalls, support = subset_metrics(results, categories)
            n_parse_failures = sum(1 for r in results if r["prediction"] == PARSE_ERROR_LABEL)
            per_subset.append(dict(
                subset=subset_name, n=len(results), balanced_accuracy=ba,
                recalls=recalls, support=support, n_parse_failures=n_parse_failures,
            ))
            print(json.dumps({
                "type": "subset_metric", "anon_method": method_key, "model": MODEL_DISPLAY_NAME,
                "subset": subset_name, "attribute": attribute, "balanced_accuracy": ba,
                "n_utterances": len(results), "n_parse_failures": n_parse_failures,
            }), flush=True)
        categories = tuple(ATTRIBUTE_VALID_LABELS[attribute])
        per_attribute[attribute] = dict(
            per_subset=per_subset,
            **aggregate_across_subsets(per_subset, categories),
        )
        print(json.dumps({
            "type": "summary", "anon_method": method_key, "model": MODEL_DISPLAY_NAME,
            "attribute": attribute,
            "mean_balanced_accuracy": per_attribute[attribute]["balanced_accuracy_mean"],
            "sample_std_balanced_accuracy": per_attribute[attribute]["balanced_accuracy_sample_std"],
        }), flush=True)
    return dict(
        method=method_key, display=ANON_METHODS[method_key]["display"],
        audio_dir=audio_dir, model=MODEL_DISPLAY_NAME, per_attribute=per_attribute,
    )


def print_summary(method_key: str, result: dict[str, Any]) -> None:
    print(f"\n=== {MODEL_DISPLAY_NAME} (optimized prompts) attacker vs. anonymization method: "
          f"{method_key} ({result['display']}) ===")
    for attr in ATTRIBUTES:
        agg = result["per_attribute"][attr]
        print(f"\n-- {attr} --  Bal.acc = {100*agg['balanced_accuracy_mean']:.2f} "
              f"+/- {100*agg['balanced_accuracy_sample_std']:.2f} %")
        for label in ATTRIBUTE_VALID_LABELS[attr]:
            r = agg["per_class_recall"][label]
            print(f"   {label:24s} recall = {100*r['mean']:6.2f} +/- {100*r['sample_std']:5.2f} %")
