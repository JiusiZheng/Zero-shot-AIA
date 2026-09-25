"""Dataset JSON loading, category discovery and subset grouping."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any


def load_records(json_path: str) -> list[dict[str, Any]]:
    with open(json_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON list of audio records.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} must be an object.")
        if not isinstance(row.get("audio_path"), str) or not row["audio_path"].strip():
            raise ValueError(f"Record {index}: missing audio_path.")
        if not isinstance(row.get("subset"), str) or not row["subset"].strip():
            raise ValueError(f"Record {index}: missing subset (test-set) label.")
    return rows


def discover_categories(rows: list[dict[str, Any]], attribute: str) -> tuple[str, ...]:
    values = set()
    for index, row in enumerate(rows):
        value = row.get(attribute)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Record {index}: missing/empty label for {attribute}.")
        if value != value.strip():
            raise ValueError(f"Record {index}: label has boundary whitespace: {value!r}")
        values.add(value)
    return tuple(sorted(values))


def group_by_subset(rows: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    """Preserve first-appearance order so subsets map onto test sets A/B/C in order."""
    grouped: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(row["subset"], []).append(row)
    return grouped
