#!/usr/bin/env python3
"""Majority-vote Qwen2.5-Omni-7B ALM gender/age/accent attacker, evaluated
against ANONYMIZED speech from 4 anonymization systems. This is the
anonymized-speech counterpart of CommonVoice_Experiments/ALM/ALM_Infer.py,
restricted to Qwen2.5-Omni-7B. Ground truth is never shown to the model."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "alm_anon"))

import config
import data
import prompts
import backend
import scoring
import driver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-json", default=config.JSON_PATH)
    p.add_argument("--out-dir", default=config.OUT_DIR)
    p.add_argument("--methods", nargs="+", default=list(config.ANON_METHODS.keys()),
                    choices=list(config.ANON_METHODS.keys()))
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--inspect", action="store_true", default=config.INSPECT_ONLY)
    p.add_argument("--skip-hf-preflight", action="store_true", default=config.SKIP_HF_PREFLIGHT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = data.load_records(args.dataset_json)
    categories = {attr: data.discover_categories(rows, attr) for attr in config.ATTRIBUTES}
    subsets = data.group_by_subset(rows)

    for attr in config.ATTRIBUTES:
        counts = Counter(row[attr] for row in rows)
        backend.eprint(f"[labels] {attr}: {len(categories[attr])} categories: {dict(sorted(counts.items()))}")
    backend.eprint(f"[subsets/test-sets] {list(subsets.keys())} (sizes: {[len(v) for v in subsets.values()]})")
    backend.eprint(f"[anon methods] {args.methods}")

    if args.inspect:
        print(f"Inspection complete: {len(rows)} audio records across {len(subsets)} test sets; "
              f"attributes={list(config.ATTRIBUTES)}; methods={args.methods}")
        return

    for method_key in args.methods:
        audio_dir = config.ANON_METHODS[method_key]["audio_dir"]
        for row in rows:
            apath = data.anon_audio_path(row["audio_path"], audio_dir)
            if not apath.is_file():
                raise FileNotFoundError(f"Anonymized audio does not exist: {apath}")

    import torch as torch_module
    backend.torch = torch_module
    if not torch_module.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Qwen2.5-Omni-7B.")
    backend.require_transformers_version()
    if not args.skip_hf_preflight:
        backend.preflight_hf_access(config.MODEL_ID)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend.eprint(f"[model] Loading {config.MODEL_ID} ...")
    model = backend.Qwen25OmniAdapter(config.MODEL_ID)
    try:
        for method_key in args.methods:
            audio_dir = config.ANON_METHODS[method_key]["audio_dir"]
            backend.eprint(f"\n[method] {method_key} -> {audio_dir}")
            result = driver.evaluate_method(model, method_key, audio_dir, subsets, categories, args.seed)
            out_path = out_dir / f"alm_{method_key}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            backend.eprint(f"[saved] {out_path}")
            driver.print_summary(method_key, result, categories)
    finally:
        backend.release_model(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
