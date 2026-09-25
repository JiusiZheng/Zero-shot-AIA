#!/usr/bin/env python3
"""Evaluate the pretrained LR-classifier attacker (frozen VoxCeleb-pretrained
ECAPA-TDNN speaker embeddings + per-attribute logistic-regression heads
trained by CommonVoice_Experiments/Classifier/CV_Extra_Classifier_Train_Eval.py)
against ANONYMIZED speech. The attacker itself is not retrained here."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Edit these constants directly; CLI flags are optional overrides.

JSON_PATH = "/path/to/data/CV_test_meta/3_testsets_attribute_labels.json"

# Trained once by CV_Extra_Classifier_Train_Eval.py (CLASSIFIER_TYPE="logreg").
TRAINED_MODEL_DIR = Path(__file__).resolve().parent.parent / "classifier" / "cv_extra_trained_models" / "logreg"

CACHE_DIR = str(Path(__file__).resolve().parent / "model_cache")
ECAPA_SAVEDIR = CACHE_DIR + "/spkrec-ecapa-voxceleb"
HF_HOME = CACHE_DIR + "/huggingface"

OUT_DIR = Path(__file__).resolve().parent / "results"

# Audio for every method is a flat dir of "common_voice_en_<id>.wav" files.
ANON_METHODS: dict[str, dict[str, str]] = {
    "sttts": dict(
        display="IMS-SttTts (ims_sttts_pc)",
        audio_dir="/path/to/data/Anon_CV/anon_speech_sttts/ims_sttts_pc/cv_balance",
    ),
    "asrbn": dict(
        display="ASR-BN (asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1)",
        audio_dir="/path/to/data/Anon_CV/cv_balance_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1/wav",
    ),
    "mcadams": dict(
        display="McAdams",
        audio_dir="/path/to/data/Anon_CV/cv_balance_mcadams/wav",
    ),
    "nac": dict(
        display="NAC",
        audio_dir="/path/to/data/Anon_CV/cv_balance_nac/wav",
    ),
}

ATTRIBUTES = ["gender", "age", "accent"]
LABELS = {
    "age": ["less than 31", "31 to 50", "more than 50"],
    "gender": ["female_feminine", "male_masculine"],
    "accent": ["England", "India and South Asia", "Australian", "Canadian", "United States"],
}

DEVICE = "auto"  # "auto" / "cuda" / "cpu"
MAX_SECONDS = 15  # same truncation used to train/evaluate the original attacker


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_rows(json_path: str) -> list[dict[str, Any]]:
    rows = json.loads(Path(json_path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON list")
    for i, r in enumerate(rows):
        for k in ["subset", "audio_path", "speaker_id"] + ATTRIBUTES:
            if k not in r:
                raise ValueError(f"Row {i} missing {k!r}")
        for attr in ATTRIBUTES:
            if r[attr] not in LABELS[attr]:
                raise ValueError(f"Row {i}: unknown {attr} label {r[attr]!r}")
    subsets = sorted({r["subset"] for r in rows})
    if len(subsets) != 3:
        raise ValueError(f"Expected 3 test subsets, got {subsets}")
    return rows


def anon_audio_path(original_audio_path: str, audio_dir: str) -> Path:
    stem = Path(original_audio_path).stem  # e.g. "common_voice_en_20321970"
    return Path(audio_dir) / f"{stem}.wav"


def resolve_device(device: str) -> str:
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_audio(path: Path, max_seconds: float):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.mean(x, axis=1).copy())
    except Exception:
        x, sr = torchaudio.load(str(path))
        wav = x.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if not wav.numel() or not torch.isfinite(wav).all():
        raise ValueError(f"Empty/nonfinite audio: {path}")
    wav = wav[: int(max_seconds * 16000)]
    if wav.numel() < 400:
        raise ValueError(f"Audio too short for convolutional encoder: {path}")
    return wav.unsqueeze(0)


def load_encoder(device: str):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ECAPA_SAVEDIR,
        run_opts={"device": device},
    )


def embed(encoder, wav, device: str):
    import torch
    with torch.inference_mode():
        vec = encoder.encode_batch(wav.to(device))
    return vec.reshape(-1).cpu().numpy()


def load_trained_classifiers(model_dir: str) -> dict[str, Any]:
    import joblib
    models = {}
    for attr in ATTRIBUTES:
        path = Path(model_dir) / f"{attr}_classifier.joblib"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing trained classifier: {path}. Run "
                "CV_Extra_Classifier_Train_Eval.py (CLASSIFIER_TYPE='logreg') first."
            )
        models[attr] = joblib.load(path)
    return models


def subset_metrics(truth: list[str], prediction: list[str | None], labels: list[str]):
    """Balanced accuracy (macro recall) + per-class recall/support for one subset."""
    recalls, support = {}, {}
    for label in labels:
        idx = [i for i, t in enumerate(truth) if t == label]
        if not idx:
            raise ValueError(f"Missing ground-truth class in this subset: {label}")
        support[label] = len(idx)
        recalls[label] = sum(prediction[i] == label for i in idx) / len(idx)
    ba = statistics.mean(recalls.values())
    return ba, recalls, support


def aggregate_across_subsets(per_subset: list[dict[str, Any]], labels: list[str]):
    """mean +/- sample std (ddof=1) of BA and of each per-class recall, across subsets."""
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


def evaluate_method(
    method_key: str,
    audio_dir: str,
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    encoder,
    device: str,
) -> dict[str, Any]:
    subsets = sorted({r["subset"] for r in rows})
    per_utterance = []
    for i, row in enumerate(rows):
        apath = anon_audio_path(row["audio_path"], audio_dir)
        if not apath.is_file():
            raise FileNotFoundError(f"Anonymized audio not found: {apath}")
        wav = load_audio(apath, MAX_SECONDS)
        feat = embed(encoder, wav, device).reshape(1, -1)
        prediction = {attr: str(models[attr].predict(feat)[0]) for attr in ATTRIBUTES}
        result = dict(
            index=i, subset=row["subset"], speaker_id=row["speaker_id"],
            anon_audio_path=str(apath), truth={a: row[a] for a in ATTRIBUTES},
            prediction=prediction,
        )
        per_utterance.append(result)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            eprint(f"[{method_key}] [{i + 1}/{len(rows)}] {row['subset']}: "
                   f"truth={result['truth']} prediction={prediction}")

    per_attribute = {}
    for attr in ATTRIBUTES:
        per_subset = []
        for subset in subsets:
            subset_rows = [r for r in per_utterance if r["subset"] == subset]
            truth = [r["truth"][attr] for r in subset_rows]
            prediction = [r["prediction"][attr] for r in subset_rows]
            ba, recalls, support = subset_metrics(truth, prediction, LABELS[attr])
            per_subset.append(dict(subset=subset, n=len(subset_rows),
                                    balanced_accuracy=ba, recalls=recalls, support=support))
        per_attribute[attr] = dict(
            per_subset=per_subset,
            **aggregate_across_subsets(per_subset, LABELS[attr]),
        )

    return dict(
        method=method_key, display=ANON_METHODS[method_key]["display"],
        audio_dir=audio_dir, n_utterances=len(per_utterance),
        per_attribute=per_attribute, per_utterance=per_utterance,
    )


def print_summary(method_key: str, result: dict[str, Any]) -> None:
    print(f"\n=== LR classifier attacker vs. anonymization method: {method_key} "
          f"({result['display']}) ===")
    for attr in ATTRIBUTES:
        agg = result["per_attribute"][attr]
        print(f"\n-- {attr} --  Bal.acc = {100*agg['balanced_accuracy_mean']:.2f} "
              f"+/- {100*agg['balanced_accuracy_sample_std']:.2f} %")
        for label in LABELS[attr]:
            r = agg["per_class_recall"][label]
            print(f"   {label:24s} recall = {100*r['mean']:6.2f} +/- {100*r['sample_std']:5.2f} %")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=JSON_PATH)
    p.add_argument("--model-dir", default=TRAINED_MODEL_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--methods", nargs="+", default=list(ANON_METHODS.keys()),
                    choices=list(ANON_METHODS.keys()))
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    return p.parse_args()


def main() -> None:
    global MAX_SECONDS
    args = parse_args()
    MAX_SECONDS = args.max_seconds

    import os
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = str(Path(HF_HOME) / "hub")

    rows = load_rows(args.json)
    eprint(f"[data] {len(rows)} utterances across "
           f"{sorted({r['subset'] for r in rows})}")

    device = resolve_device(args.device)
    eprint(f"[device] {device}")

    encoder = load_encoder(device)
    models = load_trained_classifiers(args.model_dir)
    eprint(f"[attacker] loaded frozen LR classifiers from {args.model_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for method_key in args.methods:
        audio_dir = ANON_METHODS[method_key]["audio_dir"]
        eprint(f"\n[method] {method_key} -> {audio_dir}")
        result = evaluate_method(method_key, audio_dir, rows, models, encoder, device)
        # Drop the large per-utterance field from the saved summary JSON.
        summary_only = {k: v for k, v in result.items() if k != "per_utterance"}
        out_path = out_dir / f"classifier_{method_key}.json"
        out_path.write_text(json.dumps(summary_only, indent=2, ensure_ascii=False))
        eprint(f"[saved] {out_path}")
        print_summary(method_key, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
