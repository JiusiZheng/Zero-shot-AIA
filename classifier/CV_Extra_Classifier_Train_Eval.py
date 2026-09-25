#!/usr/bin/env python3
import hashlib
import json
import statistics
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Classifier_Attack as CA  # reuse LABELS / load_audio / balanced_accuracy / summarize / print_results

# user configuration
MANIFEST_PATH = Path("/path/to/data/CommonVoice_Extra/train_dev_manifest_v3.json")  # EDIT: point this at your own manifest

CACHE_DIR = Path(__file__).resolve().parent / "model_cache"
ECAPA_SAVEDIR = CACHE_DIR / "spkrec-ecapa-voxceleb"
HF_HOME = CACHE_DIR / "huggingface"

# Cached per split so switching CLASSIFIER_TYPE or re-evaluating skips re-running ECAPA-TDNN.
EMBEDDING_CACHE_DIR = Path("/path/to/data/CommonVoice_Extra/embedding_cache")  # EDIT: point this at your own embedding cache directory

MODEL_OUT_DIR = Path(__file__).resolve().parent / "cv_extra_trained_models"
RESULTS_OUT_DIR = Path(__file__).resolve().parent / "cv_extra_results"
DEVICE = "auto"          # "auto" / "cuda" / "cpu"
MAX_SECONDS = 15
ATTRIBUTES = ["gender", "age", "accent"]
CV_TEST_JSON = "/path/to/data/CV_test_meta/3_testsets_attribute_labels.json"  # EDIT: point this at your own test-set attribute labels JSON
CLASSIFIER_TYPE = "logreg"  # "logreg" (linear) / "mlp" (2-hidden-layer nonlinear head)


def resolve_device():
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def load_encoder(device):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ECAPA_SAVEDIR),
        run_opts={"device": device},
    )


def embed(encoder, wav, device):
    with torch.inference_mode():
        vec = encoder.encode_batch(wav.to(device))
    return vec.reshape(-1).cpu().numpy()


def cache_key(records):
    """Hashes (audio_path, label) pairs so the cache auto-invalidates when the manifest changes."""
    h = hashlib.sha256()
    for r in records:
        h.update(r["audio_path"].encode())
        for a in ATTRIBUTES:
            h.update(str(r.get(a)).encode())
    h.update(str(MAX_SECONDS).encode())
    return h.hexdigest()[:16]


def extract_features(records, get_encoder, device, cache_name):
    """get_encoder is only called on a cache miss, so a fully cached run skips loading the model."""
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(records)
    cache_path = EMBEDDING_CACHE_DIR / f"{cache_name}_{key}.npz"
    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        labels = {a: data[a].tolist() for a in ATTRIBUTES}
        return data["X"], labels

    encoder = get_encoder()
    feats, labels = [], {a: [] for a in ATTRIBUTES}
    for r in records:
        try:
            wav, _ = CA.load_audio(r["audio_path"], MAX_SECONDS)
        except Exception as e:
            print(f"skip {r['audio_path']}: {e!r}", flush=True)
            continue
        feats.append(embed(encoder, wav, device))
        for a in ATTRIBUTES:
            labels[a].append(r[a])
    X = np.stack(feats)
    np.savez(cache_path, X=X, **{a: np.array(labels[a], dtype=object) for a in ATTRIBUTES})
    print(f"Cached embeddings: {cache_path}", flush=True)
    return X, labels


def make_head():
    if CLASSIFIER_TYPE == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced")
    if CLASSIFIER_TYPE == "mlp":
        # class_weight isn't supported by MLPClassifier; relies on the balanced training data instead.
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                              alpha=1e-3, early_stopping=True, n_iter_no_change=15,
                              max_iter=500, random_state=42)
    raise ValueError(f"Unknown CLASSIFIER_TYPE: {CLASSIFIER_TYPE!r}")


def train_classifiers(X, labels):
    models = {}
    for attr in ATTRIBUTES:
        y = labels[attr]
        classes = sorted(set(y))
        print(f"Training {attr} classifier ({CLASSIFIER_TYPE}): {len(y)} examples, "
              f"classes present: {classes}", flush=True)
        clf = Pipeline([("scale", StandardScaler()), ("head", make_head())])
        clf.fit(X, y)
        models[attr] = clf
    return models


def evaluate_dev(models, X_dev, labels_dev):
    print("\n=== Dev set report ===", flush=True)
    for attr in ATTRIBUTES:
        y_true = labels_dev[attr]
        y_pred = models[attr].predict(X_dev)
        print(f"\n-- {attr} --")
        print(classification_report(y_true, y_pred, labels=CA.LABELS[attr], zero_division=0))


def precision_per_class(truth, prediction, labels):
    """Complements CA.balanced_accuracy()'s recall-only view: high recall with low
    precision means the model over-predicts this class rather than reliably detecting it."""
    precisions = {}
    for label in labels:
        pred_idx = [i for i, p in enumerate(prediction) if p == label]
        precisions[label] = (sum(truth[i] == label for i in pred_idx) / len(pred_idx)
                              if pred_idx else float("nan"))
    return precisions


def print_per_class_report(records, per_set, attributes):
    """Surfaces per-class recall+precision (mean ± std across subsets) that
    CA.print_results() only reports at the attribute level."""
    subsets = sorted({r["subset"] for r in records})
    print("\nPer-class recall & precision, mean ± sample std across 3 test sets (%):", flush=True)
    print(f'{"Attribute":10s} {"Class":24s} {"Support/set":>12s} {"Recall (%)":>14s} {"Precision (%)":>16s}')
    class_summary = []
    for attr in attributes:
        rows = [r for r in per_set if r["attribute"] == attr]
        precisions_by_subset = []
        for subset in subsets:
            subset_records = [r for r in records if r["subset"] == subset]
            precisions_by_subset.append(precision_per_class(
                [r["truth"][attr] for r in subset_records],
                [r["prediction"][attr] for r in subset_records], CA.LABELS[attr]))
        for cls in CA.LABELS[attr]:
            recalls = [100 * r["recalls"][cls] for r in rows]
            supports = [r["support"][cls] for r in rows]
            precisions = [100 * p[cls] for p in precisions_by_subset]
            r_mean, r_std = statistics.mean(recalls), statistics.stdev(recalls) if len(recalls) > 1 else 0.0
            p_mean = statistics.mean(precisions)
            p_std = statistics.stdev(precisions) if len(precisions) > 1 else 0.0
            print(f'{attr:10s} {cls:24s} {str(supports):>12s} '
                  f'{r_mean:6.2f} ± {r_std:5.2f} {p_mean:9.2f} ± {p_std:5.2f}')
            class_summary.append(dict(attribute=attr, cls=cls, support_per_set=supports,
                                       recall_mean_percent=r_mean, recall_sample_std_percent=r_std,
                                       precision_mean_percent=p_mean, precision_sample_std_percent=p_std))
    return class_summary


def evaluate_on_cv_testsets(models, get_encoder, device):
    rows = json.loads(Path(CV_TEST_JSON).read_text())
    test_records = [dict(audio_path=row["audio_path"], **{a: row[a] for a in ATTRIBUTES}) for row in rows]
    X_test, _ = extract_features(test_records, get_encoder, device, cache_name="test")

    records = []
    for i, (row, feat) in enumerate(zip(rows, X_test)):
        result = dict(index=i, subset=row["subset"], speaker_id=row["speaker_id"],
                      audio_path=row["audio_path"], truth={a: row[a] for a in ATTRIBUTES},
                      prediction={}, errors={})
        feat = feat.reshape(1, -1)
        for attr in ATTRIBUTES:
            result["prediction"][attr] = str(models[attr].predict(feat)[0])
        records.append(result)
        print(f"[{i+1}/{len(rows)}] {row['subset']}: truth={result['truth']} "
              f"prediction={result['prediction']}", flush=True)

    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = RESULTS_OUT_DIR / f"utterance_predictions_{CLASSIFIER_TYPE}.json"
    predictions_path.write_text(json.dumps(records, indent=2))
    print(f"\nSaved utterance-level predictions to {predictions_path}", flush=True)

    per_set, summary = CA.summarize(records, ATTRIBUTES)
    CA.print_results(per_set, summary)
    class_summary = print_per_class_report(records, per_set, ATTRIBUTES)

    report_path = RESULTS_OUT_DIR / f"test_report_{CLASSIFIER_TYPE}.json"
    report_path.write_text(json.dumps(dict(per_set=per_set, summary=summary,
                                            per_class=class_summary), indent=2))
    print(f"Saved per-set/per-class report to {report_path}", flush=True)
    return per_set, summary


def main():
    import os
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")

    device = resolve_device()
    print(f"Device: {device}, classifier head: {CLASSIFIER_TYPE}", flush=True)

    manifest = json.loads(MANIFEST_PATH.read_text())

    # Loaded lazily; skipped entirely on a fully cached re-run.
    _encoder_holder = {}
    def get_encoder():
        if "enc" not in _encoder_holder:
            _encoder_holder["enc"] = load_encoder(device)
        return _encoder_holder["enc"]

    print(f"\nTrain embeddings ({len(manifest['train'])} clips)...", flush=True)
    X_train, y_train = extract_features(manifest["train"], get_encoder, device, cache_name="train")
    print(f"\nDev embeddings ({len(manifest['dev'])} clips)...", flush=True)
    X_dev, y_dev = extract_features(manifest["dev"], get_encoder, device, cache_name="dev")

    models = train_classifiers(X_train, y_train)
    evaluate_dev(models, X_dev, y_dev)

    out_dir = MODEL_OUT_DIR / CLASSIFIER_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    for attr, clf in models.items():
        joblib.dump(clf, out_dir / f"{attr}_classifier.joblib")
    print(f"\nSaved models to {out_dir}", flush=True)

    print("\n=== Evaluating on the 3 CommonVoice test subsets ===", flush=True)
    evaluate_on_cv_testsets(models, get_encoder, device)


if __name__ == "__main__":
    main()
