"""Response parsing and majority-vote tie-breaking, shared by every
LLM-supported attack script."""
from __future__ import annotations

import random
from collections import Counter

from .labels import ATTRIBUTE_VALID_LABELS, PARSE_ERROR_LABEL


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
    valid = [l for l in parsed_labels if l is not None]
    if not valid:
        return PARSE_ERROR_LABEL, False
    counts = Counter(valid)
    max_count = max(counts.values())
    winners = sorted(l for l, c in counts.items() if c == max_count)
    if len(winners) == 1:
        return winners[0], False
    return rng.choice(winners), True
