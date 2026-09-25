"""Per-vote classification, majority vote and balanced-accuracy scoring."""

from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Any


def parse_classification(response: str | None, categories: tuple[str, ...]) -> str | None:
    if response is None:
        return None
    label = response.strip()
    return label if label in categories else None


def classify_vote(response: str | None, categories: tuple[str, ...]) -> tuple[str | None, str]:
    """Returns (parsed_label_or_None, status), status in {"valid", "hallucination", "error"}."""
    if response is None:
        return None, "error"
    label = response.strip()
    if label in categories:
        return label, "valid"
    return None, "hallucination"


def majority_vote(parsed_votes: list[str | None], rng: random.Random) -> str | None:
    """Falls back to a uniformly random pick among the votes when there is no strict majority."""
    counts = Counter(parsed_votes)
    max_count = max(counts.values())
    winners = [label for label, c in counts.items() if c == max_count]
    if len(winners) == 1 and max_count > 1:
        return winners[0]
    # No strict majority (e.g. 3 distinct votes) -> random tie-break among the votes.
    return rng.choice(parsed_votes)


def balanced_accuracy(results: list[dict[str, Any]], categories: tuple[str, ...]) -> float | None:
    """Macro-averaged per-class recall, skipping classes with zero support."""
    per_class_recall = []
    for label in categories:
        support = sum(1 for r in results if r["true_label"] == label)
        if support == 0:
            continue
        correct = sum(1 for r in results if r["true_label"] == label and r["prediction"] == label)
        per_class_recall.append(correct / support)
    return statistics.mean(per_class_recall) if per_class_recall else None
