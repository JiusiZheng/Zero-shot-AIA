#!/usr/bin/env python3
"""Shared label/scoring helpers for the conventional LR-classifier attack.
Used by CV_Extra_Classifier_Train_Eval.py; not meant to be run directly."""
import statistics
import sys

LABELS = {
    'age': ['less than 31', '31 to 50', 'more than 50'],
    'gender': ['female_feminine', 'male_masculine'],
    'accent': ['England', 'India and South Asia', 'Australian', 'Canadian', 'United States'],
}


def balanced_accuracy(truth, predictions, labels):
    if len(truth) != len(predictions) or not truth:
        raise ValueError('Empty or unequal prediction/target lengths')
    recalls, support = {}, {}
    for label in labels:
        indexes = [i for i, t in enumerate(truth) if t == label]
        if not indexes:
            raise ValueError(f'Missing ground-truth class: {label}')
        support[label] = len(indexes)
        recalls[label] = sum(predictions[i] == label for i in indexes) / len(indexes)
    return statistics.mean(recalls.values()), recalls, support


def summarize(records, attributes):
    per_set, summary = [], []
    subsets = sorted({r['subset'] for r in records})
    if len(subsets) != 3:
        raise ValueError(f'Expected 3 subsets, got {subsets}')
    for attr in attributes:
        scores = []
        for subset in subsets:
            rows = [r for r in records if r['subset'] == subset]
            ba, recalls, support = balanced_accuracy(
                [r['truth'][attr] for r in rows],
                [r['prediction'].get(attr) for r in rows], LABELS[attr])
            scores.append(ba)
            per_set.append(dict(subset=subset, attribute=attr, n=len(rows),
                                balanced_accuracy=ba, balanced_accuracy_percent=100*ba,
                                failed=sum(attr in r['errors'] for r in rows),
                                recalls=recalls, support=support))
        mean, std = statistics.mean(scores), statistics.stdev(scores)
        summary.append(dict(attribute=attr, n_test_sets=3, mean=mean,
                            sample_std=std, mean_percent=100*mean,
                            sample_std_percent=100*std, ddof=1))
    return per_set, summary


def load_audio(path, max_seconds):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    try:
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        wav = torch.from_numpy(np.mean(x, axis=1).copy())
    except Exception:
        x, sr = torchaudio.load(path)
        wav = x.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if not wav.numel() or not torch.isfinite(wav).all():
        raise ValueError(f'Empty/nonfinite audio: {path}')
    duration = wav.numel()/16000
    wav = wav[:int(max_seconds*16000)]
    if wav.numel() < 400:
        raise ValueError(f'Audio too short for convolutional encoder: {path}')
    return wav.unsqueeze(0), duration


def print_results(per_set, summary):
    print("\nBalanced accuracy per test set (%):", flush=True)
    print(f'{"Test set":16s} {"Attribute":10s} {"N":>5s} {"BA (%)":>10s} {"Failed":>8s}')
    for row in sorted(per_set, key=lambda r: (r['subset'], r['attribute'])):
        print(f'{row["subset"]:16s} {row["attribute"]:10s} {row["n"]:5d} '
              f'{row["balanced_accuracy_percent"]:10.2f} {row["failed"]:8d}')
    print('\nAcross 3 test sets: mean ± sample std (ddof=1, %):')
    for row in summary:
        print(f'{row["attribute"]:8s}: {row["mean_percent"]:.2f} ± {row["sample_std_percent"]:.2f}')
    sys.stdout.flush()
