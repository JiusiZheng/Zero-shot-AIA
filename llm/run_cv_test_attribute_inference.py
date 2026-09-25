#!/usr/bin/env python3
"""Text-only LLM attribute-inference attack on the 3 CommonVoice test sets
(full-feature prompts). Reuses run_cv_dev_attribute_inference.py's prompt
assembly, model loading, and parsing unchanged (`import ... as base`); adds
per-subset splitting and mean +/- sample-std aggregation across CV_test_1/2/3,
matching the LR-classifier / ALM methodology in ../Defense."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_cv_dev_attribute_inference as base  # noqa: E402

RESULTS_DIR = SCRIPT_DIR / "results"

# User-editable: path to the 3_testsets_attribute_labels.json metadata file you provide.
CV_TEST_JSON = "/path/to/data/CV_test_meta/3_testsets_attribute_labels.json"
SUBSETS = ["CV_test_1", "CV_test_2", "CV_test_3"]  # fixed, known order

MODEL_LLMS = base.MODEL_LLMS
ATTRIBUTE_VALID_LABELS = base.ATTRIBUTE_VALID_LABELS
TABLE_COLUMNS = base.TABLE_COLUMNS
N_REPEATS = base.N_REPEATS
GENERATION_TEMPERATURE = base.GENERATION_TEMPERATURE
GENERATION_MAX_NEW_TOKENS = base.GENERATION_MAX_NEW_TOKENS
RANDOM_SEED = base.RANDOM_SEED
PARSE_ERROR_LABEL = base.PARSE_ERROR_LABEL
DEFAULT_BATCH_SIZE = base.DEFAULT_BATCH_SIZE
BATCH_SIZE_OVERRIDES = base.BATCH_SIZE_OVERRIDES


def feature_cache_path(subset: str) -> Path:
    return RESULTS_DIR / f"cv_test_features_cache_{subset}.json"


def results_path_for(model_name: str) -> Path:
    return RESULTS_DIR / f"cv_test_results_{base.slugify(model_name)}.json"


# Data loading, split by subset

def load_cv_test() -> list[dict]:
    with open(CV_TEST_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)
    found = sorted({r["subset"] for r in rows})
    if found != SUBSETS:
        raise ValueError(f"Expected subsets {SUBSETS}, found {found}")
    return rows


def load_subset_records(subset: str) -> list[dict]:
    return [r for r in load_cv_test() if r["subset"] == subset]


# Stage 1: feature extraction + cache (CPU only), one cache file per subset

def build_feature_cache(subset: str, limit: int | None = None) -> dict:
    """cache[audio_path][attribute] = features dict. Resumable."""
    records = load_subset_records(subset)
    if limit:
        records = records[:limit]

    cache_path = feature_cache_path(subset)
    cache: dict = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(records)
    for i, rec in enumerate(records, start=1):
        path = rec["audio_path"]
        entry = cache.setdefault(path, {})
        for attribute, fn in base.FEATURE_FUNCS.items():
            if attribute in entry:
                continue
            entry[attribute] = fn(path)
        if i % 10 == 0 or i == n:
            print(f"  [features:{subset}] {i}/{n} files done", file=sys.stderr)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"[features:{subset}] wrote {cache_path}", file=sys.stderr)
    return cache


def load_feature_cache(subset: str) -> dict:
    path = feature_cache_path(subset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run with --stage features --subset {subset} first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Stage 2: run one model over all 3 subsets x 3 attributes

def run_inference_for_model(model_name: str, attributes: list[str], limit: int | None = None) -> dict:
    tokenizer, model = base.get_llm(model_name)
    batch_size = BATCH_SIZE_OVERRIDES.get(model_name, DEFAULT_BATCH_SIZE)

    per_subset: dict[str, dict] = {}
    for subset in SUBSETS:
        records = load_subset_records(subset)
        if limit:
            records = records[:limit]
        feature_cache = load_feature_cache(subset)

        subset_attr_results = {}
        for attribute in attributes:
            print(f"\n[infer] model={model_name} subset={subset} attribute={attribute} n={len(records)}",
                  file=sys.stderr)
            rng = random.Random(RANDOM_SEED)

            prompts = []
            for rec in records:
                feats = feature_cache[rec["audio_path"]][attribute]
                prompts.append(base.build_full_prompt(attribute, feats))

            flat_prompts = [p for p in prompts for _ in range(N_REPEATS)]
            flat_responses = base.generate_batch(
                tokenizer, model, flat_prompts, GENERATION_MAX_NEW_TOKENS,
                GENERATION_TEMPERATURE, batch_size,
            )

            sample_results = []
            for i, rec in enumerate(records):
                responses = flat_responses[i * N_REPEATS:(i + 1) * N_REPEATS]
                parsed = [base.parse_prediction(attribute, r) for r in responses]
                final_label, was_tie_break = base.majority_vote(parsed, rng)
                sample_results.append({
                    "audio_path": rec["audio_path"],
                    "true_label": rec[attribute],
                    "raw_responses": responses,
                    "parsed_labels": parsed,
                    "final_label": final_label,
                    "tie_break_random": was_tie_break,
                })
                if (i + 1) % 25 == 0 or (i + 1) == len(records):
                    print(f"  [{subset}] [{i + 1}/{len(records)}] last true={rec[attribute]!r} pred={final_label!r}",
                          file=sys.stderr)

            metrics = base.compute_metrics(sample_results, ATTRIBUTE_VALID_LABELS[attribute])
            subset_attr_results[attribute] = {"samples": sample_results, "metrics": metrics}
            print(
                f"  [{subset}][{attribute}] balanced_accuracy={metrics['balanced_accuracy'] * 100:.2f}%  "
                f"full_parse_failures={metrics['n_full_parse_failures']}/{len(records)}",
                file=sys.stderr,
            )
        per_subset[subset] = subset_attr_results

    aggregated = aggregate_across_subsets(per_subset, attributes)
    return {"model_name": model_name, "per_subset": per_subset, "aggregated": aggregated}


def aggregate_across_subsets(per_subset: dict[str, dict], attributes: list[str]) -> dict:
    """Mean +/- sample std (ddof=1) across the 3 subsets, matching the
    LR-classifier / ALM tables in ../Defense."""
    out = {}
    for attribute in attributes:
        ba_values = [per_subset[s][attribute]["metrics"]["balanced_accuracy"] for s in SUBSETS]
        ba_values = [v for v in ba_values if v is not None]
        ba_mean = statistics.mean(ba_values) if ba_values else None
        ba_std = statistics.stdev(ba_values) if len(ba_values) > 1 else 0.0

        recall_summary = {}
        for label in ATTRIBUTE_VALID_LABELS[attribute]:
            values = [per_subset[s][attribute]["metrics"]["per_class_recall"].get(label) for s in SUBSETS]
            values = [v for v in values if v is not None]
            recall_summary[label] = dict(
                mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else 0.0,
            )
        out[attribute] = dict(
            balanced_accuracy_mean=ba_mean,
            balanced_accuracy_sample_std=ba_std,
            per_class_recall=recall_summary,
        )
    return out


# Stage 3: aggregate all 4 models into a plain-text + LaTeX-ready report

def render_plain(model_name: str, aggregated: dict) -> list[str]:
    lines = [f"\nModel: {model_name}"]
    for attribute in ("gender", "age", "accent"):
        agg = aggregated[attribute]
        ba_mean = agg["balanced_accuracy_mean"]
        ba_str = f"{ba_mean * 100:6.2f}" if ba_mean is not None else "   n/a"
        lines.append(f"  {attribute:<8} balanced_accuracy = {ba_str} +/- {agg['balanced_accuracy_sample_std'] * 100:5.2f} %")
        for label, r in agg["per_class_recall"].items():
            if r["mean"] is None:
                lines.append(f"      recall[{label!r:<24}] =    n/a")
            else:
                lines.append(f"      recall[{label!r:<24}] = {r['mean'] * 100:6.2f} +/- {r['sample_std'] * 100:5.2f} %")
    return lines


def render_latex_row(model_name: str, aggregated: dict) -> list[str]:
    cells = []
    for attribute in ("gender", "age", "accent"):
        agg = aggregated[attribute]
        ba_mean = agg["balanced_accuracy_mean"] * 100 if agg["balanced_accuracy_mean"] is not None else float("nan")
        ba_std = agg["balanced_accuracy_sample_std"] * 100
        cells.append(f"\\overallcell{{{ba_mean:.2f}}}{{{ba_std:.2f}}}")
        for true_label, _display in TABLE_COLUMNS[attribute]:
            r = agg["per_class_recall"].get(true_label, {"mean": None, "sample_std": 0.0})
            val = r["mean"] * 100 if r["mean"] is not None else float("nan")
            cells.append(f"\\recallcell{{{val:.2f}}}{{{r['sample_std'] * 100:.2f}}}")
    display_name = model_name.split("/")[-1]
    return [f"        {display_name}", "        & " + "\n        & ".join(cells) + " \\\\\n"]


def aggregate_and_print_table(model_names: list[str]) -> None:
    all_model_results = {}
    for model_name in model_names:
        path = results_path_for(model_name)
        if not path.exists():
            print(f"[aggregate] WARNING: missing {path}, skipping {model_name}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            all_model_results[model_name] = json.load(f)

    out_path = RESULTS_DIR / "cv_test_attribute_inference_table.txt"
    lines = []
    lines.append("3 CommonVoice test sets (3_testsets_attribute_labels.json, "
                  "CV_test_1/2/3, n=150 each) -- LLM text-only attribute inference")
    lines.append(f"3 independent samples per (model, subset, attribute, utterance), majority vote, "
                 f"temperature={GENERATION_TEMPERATURE}")
    lines.append("Balanced accuracy / per-class recall reported as mean +/- sample std (ddof=1) across the 3 test sets.")
    lines.append("=" * 100)

    for model_name, result in all_model_results.items():
        lines.extend(render_plain(model_name, result["aggregated"]))

    lines.append("\n" + "=" * 100)
    lines.append("LaTeX rows, ready to paste into the tabular block (columns: Gender bal.acc/Male/Female | "
                  "Age bal.acc/>50/<31/31-50 | Accent bal.acc/Ind.S.Asia/England/Aus./Can./US):\n")

    for model_name, result in all_model_results.items():
        lines.extend(render_latex_row(model_name, result["aggregated"]))

    lines.append("Random guess")
    lines.append(
        "& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
            50.00, 50.00, 50.00,
            33.33, 33.33, 33.33, 33.33,
            20.00, 20.00, 20.00, 20.00, 20.00, 20.00,
        )
    )

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[aggregate] wrote {out_path}", file=sys.stderr)


# CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["features", "infer", "aggregate", "prompt_preview"])
    p.add_argument("--subset", type=str, default="all", choices=SUBSETS + ["all"],
                    help="for --stage features / prompt_preview")
    p.add_argument("--model_index", type=int, default=None, help="index into MODEL_LLMS (for --stage infer)")
    p.add_argument("--model_name", type=str, default=None, help="override model name directly (for --stage infer)")
    p.add_argument("--attributes", type=str, default="gender,age,accent")
    p.add_argument("--limit", type=int, default=None, help="use only the first N records per subset (smoke test)")
    p.add_argument("--sample_index", type=int, default=0, help="for --stage prompt_preview")
    args = p.parse_args()

    attributes = args.attributes.split(",")

    if args.stage == "features":
        subsets = SUBSETS if args.subset == "all" else [args.subset]
        for subset in subsets:
            build_feature_cache(subset, limit=args.limit)
        return

    if args.stage == "prompt_preview":
        subset = SUBSETS[0] if args.subset == "all" else args.subset
        feature_cache = build_feature_cache(subset, limit=args.sample_index + 1)
        records = load_subset_records(subset)
        rec = records[args.sample_index]
        attribute = attributes[0]
        feats = feature_cache[rec["audio_path"]][attribute]
        print(base.build_full_prompt(attribute, feats))
        return

    if args.stage == "infer":
        if args.model_name:
            model_name = args.model_name
        elif args.model_index is not None:
            model_name = MODEL_LLMS[args.model_index]
        else:
            raise SystemExit("--stage infer requires --model_index or --model_name")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result = run_inference_for_model(model_name, attributes, limit=args.limit)
        out_path = results_path_for(model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[infer] wrote {out_path}", file=sys.stderr)
        return

    if args.stage == "aggregate":
        aggregate_and_print_table(MODEL_LLMS)
        return


if __name__ == "__main__":
    main()
