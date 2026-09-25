#!/usr/bin/env python3
"""Extract the feature set for the AGE prediction prompt."""
import argparse
import json
import sys

import numpy as np
import speech_features as sf


# Age-specific helpers (not in speech_features)

def _true_runs(mask):
    """Return [(start, end)] inclusive index pairs for contiguous True runs."""
    mask = np.asarray(mask, dtype=bool)
    runs, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def centralization_ratio(tracks):
    """Vowel-centralization proxy: fraction of voiced frames in the schwa region."""
    f1, f2 = tracks.get("F1"), tracks.get("F2")
    if f1 is None or f2 is None:
        return None
    m = np.isfinite(f1) & np.isfinite(f2)
    f1, f2 = f1[m], f2[m]
    if f1.size == 0:
        return None
    center = (f1 >= 400) & (f1 <= 700) & (f2 >= 1200) & (f2 <= 1800)
    return float(np.mean(center))


def pause_metrics(sound, min_pitch=75.0, time_step=0.01, min_pause_s=0.05):
    """Pause count/duration/rate from silent gaps inside the speech region."""
    out = {"pause_count": None, "pause_mean_s": None,
           "pause_total_s": None, "pause_rate_per_s": None}
    try:
        intens = sound.to_intensity(minimum_pitch=min_pitch, time_step=time_step)
        db = np.asarray(intens.values).ravel()
        db = db[np.isfinite(db)]
        if db.size == 0:
            return out
        speech = db > (np.max(db) - 25.0)
        sp_idx = np.where(speech)[0]
        if sp_idx.size == 0:
            return out
        lo, hi = sp_idx[0], sp_idx[-1]
        durs = []
        for a, b in _true_runs(~speech):
            if lo < a and b < hi:
                dur = (b - a + 1) * time_step
                if dur >= min_pause_s:
                    durs.append(dur)
        if durs:
            durs = np.array(durs)
            out["pause_count"] = int(len(durs))
            out["pause_mean_s"] = float(np.mean(durs))
            out["pause_total_s"] = float(np.sum(durs))
            speech_dur = float((hi - lo + 1) * time_step)
            out["pause_rate_per_s"] = (float(len(durs) / speech_dur)
                                       if speech_dur > 0 else None)
    except Exception:
        pass
    return out


def extract_age(path):
    sound, y, sr = sf.load(path)
    f0, times = sf.f0_track(sound)
    voiced = f0 > 0.0
    tracks = sf.formant_values(sound, times, voiced)

    out = {}
    # f0_stats computed first -- needed by h1_h2_db below.
    out.update(sf.f0_stats(f0, times))

    # Voice quality: jitter, shimmer, HNR, breathiness.
    out.update(sf.jitter_shimmer(sound))
    out["hnr_db"] = sf.hnr_db(sound)
    out["h1_h2_db"] = sf.h1_h2_db(y, sr, out.get("f0_mean_hz"))

    # Formants and vowel space.
    out.update(sf.formant_stats(tracks))
    out.update(sf.vowel_space(tracks))
    out["centralization_ratio"] = centralization_ratio(tracks)

    # Timing / prosody.
    out.update(sf.speaking_rate(sound))
    out.update(pause_metrics(sound))
    out.update(sf.vot_ms(y, sr, f0))

    # Spectral / level.
    out["spectral_tilt_db_per_octave"] = sf.spectral_tilt_db_oct(y, sr)
    out["spectral_centroid_hz"] = sf.spectral_centroid_hz(y, sr)
    out.update(sf.intensity_stats(sound))
    out.update(sf.mfcc_features(y, sr))

    out["duration_s"] = float(sound.duration)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="path to an audio file (wav/mp3/...)")
    p.add_argument("--out", default=None, help="optional output JSON path")
    args = p.parse_args(argv)

    res = extract_age(args.audio)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
