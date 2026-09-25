"""Per-utterance metrics (balanced accuracy + per-class recall), shared by
every LLM-supported attack script."""
from __future__ import annotations

from .labels import PARSE_ERROR_LABEL


def compute_metrics(sample_results: list[dict], valid_labels: list[str]) -> dict:
    per_class_recall = {}
    for label in valid_labels:
        support = [s for s in sample_results if s["true_label"] == label]
        correct = [s for s in support if s["final_label"] == label]
        per_class_recall[label] = (len(correct) / len(support)) if support else None
    scored_recalls = [r for r in per_class_recall.values() if r is not None]
    balanced_accuracy = sum(scored_recalls) / len(scored_recalls) if scored_recalls else None
    n_full_parse_failures = sum(1 for s in sample_results if s["final_label"] == PARSE_ERROR_LABEL)
    return {
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": per_class_recall,
        "n_full_parse_failures": n_full_parse_failures,
        "n_total": len(sample_results),
    }
