"""Response parsing, majority vote, and metrics aggregation for the LLM (text)
attacker."""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import ATTRIBUTE_VALID_LABELS, PARSE_ERROR_LABEL  # noqa: E402


# Parsing + majority vote, copied unchanged from run_cv_dev_optimized_prompt_eval.py.

def parse_prediction(attribute: str, response_text: str) -> str | None:
    labels = sorted(ATTRIBUTE_VALID_LABELS[attribute], key=len, reverse=True)
    idx = response_text.lower().find("prediction")
    window = response_text[idx:idx + 200] if idx != -1 else response_text[:200]
    for label in labels:
        if label.lower() in window.lower():
            return label
    for label in labels:
        if label.lower() in response_text.lower():
            return label
    return None


def majority_vote(parsed_labels: list[str | None], rng: random.Random) -> tuple[str, bool]:
    """Returns (final_label, was_tie_break_random)."""
    from collections import Counter
    valid = [l for l in parsed_labels if l is not None]
    if not valid:
        return PARSE_ERROR_LABEL, False
    counts = Counter(valid)
    max_count = max(counts.values())
    winners = sorted(l for l, c in counts.items() if c == max_count)
    if len(winners) == 1:
        return winners[0], False
    return rng.choice(winners), True


# Same statistic as alm_anon_eval.py / classifier_anon_eval.py: balanced
# accuracy mean +/- sample std across the 3 test subsets.

def subset_metrics(results: list[dict[str, Any]], categories: tuple[str, ...]):
    recalls, support = {}, {}
    for label in categories:
        idx = [i for i, r in enumerate(results) if r["true_label"] == label]
        if not idx:
            raise ValueError(f"Missing ground-truth class in this subset: {label}")
        support[label] = len(idx)
        recalls[label] = sum(results[i]["prediction"] == label for i in idx) / len(idx)
    ba = statistics.mean(recalls.values())
    return ba, recalls, support


def aggregate_across_subsets(per_subset: list[dict[str, Any]], labels: tuple[str, ...]):
    ba_values = [s["balanced_accuracy"] for s in per_subset]
    ba_mean = statistics.mean(ba_values)
    ba_std = statistics.stdev(ba_values) if len(ba_values) > 1 else 0.0
    recall_summary = {}
    for label in labels:
        values = [s["recalls"][label] for s in per_subset]
        recall_summary[label] = dict(
            mean=statistics.mean(values),
            sample_std=statistics.stdev(values) if len(values) > 1 else 0.0,
        )
    return dict(
        balanced_accuracy_mean=ba_mean,
        balanced_accuracy_sample_std=ba_std,
        per_class_recall=recall_summary,
    )
