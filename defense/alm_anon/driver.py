"""Evaluation driver and summary printing for the ALM attacker."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from backend import Qwen25OmniAdapter, eprint  # noqa: E402
from config import (  # noqa: E402
    ANON_METHODS,
    ATTRIBUTES,
    CONTINUE_ON_AUDIO_ERROR,
    MAX_NEW_TOKENS_UTTERANCE,
    MODEL_DISPLAY_NAME,
    N_VOTES,
)
from data import anon_audio_path  # noqa: E402
from prompts import build_user_prompt  # noqa: E402
from scoring import aggregate_across_subsets, classify_vote, majority_vote, subset_metrics  # noqa: E402


def run_attribute_on_subset(
    model: Qwen25OmniAdapter,
    rows: list[dict[str, Any]],
    attribute: str,
    categories: tuple[str, ...],
    subset_name: str,
    method_key: str,
    audio_dir: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    prompt = build_user_prompt(attribute, categories)
    results = []
    for row in rows:
        apath = anon_audio_path(row["audio_path"], audio_dir)
        if not apath.is_file():
            raise FileNotFoundError(f"Anonymized audio not found: {apath}")

        # Ground truth is read here ONLY for later scoring; never shown to the model.
        true_label = row[attribute]

        raw_responses: list[str | None] = []
        errors: list[str | None] = []
        for _ in range(N_VOTES):
            try:
                response = model.query_audio(str(apath), prompt, MAX_NEW_TOKENS_UTTERANCE)
                raw_responses.append(response)
                errors.append(None)
            except Exception as exc:
                raw_responses.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
                if not CONTINUE_ON_AUDIO_ERROR:
                    raise

        parsed_votes: list[str | None] = []
        vote_statuses: list[str] = []
        for response in raw_responses:
            label, status = classify_vote(response, categories)
            parsed_votes.append(label)
            vote_statuses.append(status)
            if status == "hallucination":
                eprint(
                    f"[hallucination] method={method_key} model={MODEL_DISPLAY_NAME} "
                    f"subset={subset_name} attribute={attribute} "
                    f"speaker_id={row.get('speaker_id')} raw_response={response!r} "
                    f"allowed_categories={categories}"
                )

        prediction = majority_vote(parsed_votes, rng)
        prediction_invalid = prediction is None
        n_hallucinations = vote_statuses.count("hallucination")
        n_errors = vote_statuses.count("error")

        result = dict(
            anon_method=method_key, model=MODEL_DISPLAY_NAME, subset=subset_name,
            attribute=attribute, speaker_id=row.get("speaker_id"), anon_audio_path=str(apath),
            raw_responses=raw_responses, votes=parsed_votes, vote_statuses=vote_statuses,
            n_hallucinations=n_hallucinations, n_errors=n_errors, prediction=prediction,
            prediction_invalid=prediction_invalid, true_label=true_label, errors=errors,
        )
        print(json.dumps({"type": "utterance_result", **result}, ensure_ascii=False), flush=True)
        results.append(result)
    return results


def evaluate_method(
    model: Qwen25OmniAdapter,
    method_key: str,
    audio_dir: str,
    subsets: "OrderedDict[str, list[dict[str, Any]]]",
    categories: dict[str, tuple[str, ...]],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    per_attribute: dict[str, Any] = {}
    for attribute in ATTRIBUTES:
        per_subset = []
        for subset_name, subset_rows in subsets.items():
            print(json.dumps({
                "type": "condition", "anon_method": method_key, "model": MODEL_DISPLAY_NAME,
                "subset": subset_name, "attribute": attribute, "n_votes": N_VOTES,
            }), flush=True)
            results = run_attribute_on_subset(
                model, subset_rows, attribute, categories[attribute],
                subset_name, method_key, audio_dir, rng,
            )
            ba, recalls, support = subset_metrics(results, categories[attribute])
            n_hallucinations = sum(r["n_hallucinations"] for r in results)
            n_errors = sum(r["n_errors"] for r in results)
            n_invalid = sum(1 for r in results if r["prediction_invalid"])
            per_subset.append(dict(
                subset=subset_name, n=len(results), balanced_accuracy=ba,
                recalls=recalls, support=support, n_hallucinated_votes=n_hallucinations,
                n_error_votes=n_errors, n_utterances_with_invalid_prediction=n_invalid,
            ))
            print(json.dumps({
                "type": "subset_metric", "anon_method": method_key, "model": MODEL_DISPLAY_NAME,
                "subset": subset_name, "attribute": attribute, "balanced_accuracy": ba,
                "n_utterances": len(results), "n_hallucinated_votes": n_hallucinations,
                "n_error_votes": n_errors, "n_utterances_with_invalid_prediction": n_invalid,
            }), flush=True)
        per_attribute[attribute] = dict(
            per_subset=per_subset,
            **aggregate_across_subsets(per_subset, categories[attribute]),
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


def print_summary(method_key: str, result: dict[str, Any], labels: dict[str, tuple[str, ...]]) -> None:
    print(f"\n=== {MODEL_DISPLAY_NAME} ALM attacker vs. anonymization method: {method_key} "
          f"({result['display']}) ===")
    for attr in ATTRIBUTES:
        agg = result["per_attribute"][attr]
        print(f"\n-- {attr} --  Bal.acc = {100*agg['balanced_accuracy_mean']:.2f} "
              f"+/- {100*agg['balanced_accuracy_sample_std']:.2f} %")
        for label in labels[attr]:
            r = agg["per_class_recall"][label]
            print(f"   {label:24s} recall = {100*r['mean']:6.2f} +/- {100*r['sample_std']:5.2f} %")
