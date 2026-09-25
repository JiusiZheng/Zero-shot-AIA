"""Vote classification, majority vote, and metrics aggregation for the ALM
attacker."""
from __future__ import annotations

import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def classify_vote(response: str | None, categories: tuple[str, ...]) -> tuple[str | None, str]:
    if response is None:
        return None, "error"
    label = response.strip()
    if label in categories:
        return label, "valid"
    return None, "hallucination"


def majority_vote(parsed_votes: list[str | None], rng: random.Random) -> str | None:
    counts = Counter(parsed_votes)
    max_count = max(counts.values())
    winners = [label for label, c in counts.items() if c == max_count]
    if len(winners) == 1 and max_count > 1:
        return winners[0]
    return rng.choice(parsed_votes)


def subset_metrics(results: list[dict[str, Any]], categories: tuple[str, ...]):
    """Balanced accuracy (macro recall) + per-class recall/support for one subset.
    A None prediction (hallucination/error) can never equal a true_label string,
    so it is automatically scored as incorrect."""
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
