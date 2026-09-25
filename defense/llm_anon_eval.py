#!/usr/bin/env python3
"""Majority-vote Qwen2.5-7B-Instruct text-only LLM attacker (optimized
prompts/feature-subset), evaluated against ANONYMIZED speech from 4
anonymization systems. This is the anonymized-speech counterpart of
CommonVoice_Experiments/LLM/optimization_dev/run_cv_dev_optimized_prompt_eval.py,
restricted to Qwen2.5-7B-Instruct. Ground truth is never shown to the model."""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "llm_anon"))

import config
import data
import model
import scoring
import driver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-json", default=config.JSON_PATH)
    p.add_argument("--out-dir", default=config.OUT_DIR)
    p.add_argument("--methods", nargs="+", default=list(config.ANON_METHODS.keys()),
                    choices=list(config.ANON_METHODS.keys()))
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None,
                    help="use only the first N records per subset (smoke test)")
    p.add_argument("--inspect", action="store_true", default=config.INSPECT_ONLY)
    p.add_argument("--skip-hf-preflight", action="store_true", default=config.SKIP_HF_PREFLIGHT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = data.load_records(args.dataset_json)
    subsets = data.group_by_subset(rows)
    if args.limit:
        subsets = OrderedDict((name, subset_rows[:args.limit]) for name, subset_rows in subsets.items())
        rows = [row for subset_rows in subsets.values() for row in subset_rows]

    for attr in config.ATTRIBUTES:
        driver.eprint(f"[labels] {attr}: {config.ATTRIBUTE_VALID_LABELS[attr]}")
    driver.eprint(f"[subsets/test-sets] {list(subsets.keys())} (sizes: {[len(v) for v in subsets.values()]})")
    driver.eprint(f"[anon methods] {args.methods}")

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
    model.torch = torch_module
    if not torch_module.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Qwen2.5-7B-Instruct.")
    if not args.skip_hf_preflight:
        driver.preflight_hf_access(config.MODEL_ID)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, llm_model = model.load_llm(config.MODEL_ID)
    for method_key in args.methods:
        audio_dir = config.ANON_METHODS[method_key]["audio_dir"]
        driver.eprint(f"\n[method] {method_key} -> {audio_dir}")
        feature_cache = data.build_feature_cache_for_method(method_key, audio_dir, rows, out_dir)
        result = driver.evaluate_method(
            tokenizer, llm_model, method_key, audio_dir, subsets, feature_cache, args.seed, args.batch_size,
        )
        out_path = out_dir / f"llm_{method_key}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        driver.eprint(f"[saved] {out_path}")
        driver.print_summary(method_key, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
