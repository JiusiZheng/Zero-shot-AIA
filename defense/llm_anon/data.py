"""Dataset loading, anonymized-audio paths, and the feature-extraction/prompt
pipeline for the LLM (text) attacker."""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    ATTRIBUTE_VALID_LABELS,
    ATTRIBUTES,
    FEATURE_EXTRACTORS_DIR,
    FEATURE_PLACEHOLDER,
    FEATURE_SUBSET_KEYS,
    PROMPT_FILES,
)
from util import eprint  # noqa: E402

sys.path.insert(0, str(FEATURE_EXTRACTORS_DIR))
import extract_accent  # noqa: E402
import extract_age  # noqa: E402
import extract_gender  # noqa: E402

FEATURE_FUNCS = {
    "gender": extract_gender.extract_gender,
    "age": extract_age.extract_age,
    "accent": extract_accent.extract_accent,
}


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
        for attribute in ATTRIBUTES:
            value = row.get(attribute)
            if value not in ATTRIBUTE_VALID_LABELS[attribute]:
                raise ValueError(f"Record {index}: invalid/missing {attribute} label: {value!r}")
    return rows


def group_by_subset(rows: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(row["subset"], []).append(row)
    return grouped


def anon_audio_path(original_audio_path: str, audio_dir: str) -> Path:
    stem = Path(original_audio_path).stem  # e.g. "common_voice_en_20321970"
    return Path(audio_dir) / f"{stem}.wav"


def feature_cache_path(out_dir: Path, method_key: str) -> Path:
    return out_dir / f"llm_features_cache_{method_key}.json"


def build_feature_cache_for_method(
    method_key: str, audio_dir: str, rows: list[dict[str, Any]], out_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Resumable: entries already present in an existing cache file are not
    recomputed. Subsetting happens later, at prompt-build time."""
    cache_path = feature_cache_path(out_dir, method_key)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    for i, row in enumerate(rows, start=1):
        apath = str(anon_audio_path(row["audio_path"], audio_dir))
        entry = cache.setdefault(apath, {})
        for attribute, fn in FEATURE_FUNCS.items():
            if attribute in entry:
                continue
            entry[attribute] = fn(apath)
        if i % 10 == 0 or i == n:
            eprint(f"  [features:{method_key}] {i}/{n} files done")
            cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    eprint(f"[features:{method_key}] wrote {cache_path}")
    return cache


def load_prompt_template(attribute: str) -> str:
    text = PROMPT_FILES[attribute].read_text(encoding="utf-8")
    n = text.count(FEATURE_PLACEHOLDER)
    if n != 1:
        raise ValueError(
            f"{PROMPT_FILES[attribute].name} must contain the placeholder "
            f"'{FEATURE_PLACEHOLDER}' exactly once, found {n} times."
        )
    return text


def subset_features(attribute: str, full_features: dict[str, Any]) -> dict[str, Any]:
    keys = FEATURE_SUBSET_KEYS[attribute]
    return {k: full_features.get(k) for k in keys}


def build_full_prompt(attribute: str, full_features: dict[str, Any]) -> str:
    template = load_prompt_template(attribute)
    subset = subset_features(attribute, full_features)
    features_json = json.dumps(subset, indent=2)
    return template.replace(FEATURE_PLACEHOLDER, features_json)
