#!/usr/bin/env python3
"""Extract the feature set for the ACCENT prediction prompt."""
import argparse
import json
import sys

import numpy as np
import scipy.signal
import speech_features as sf


# Accent-specific helpers (not in speech_features)

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


def f3_below_f2_rate(tracks):
    """Rhoticity proxy: fraction of voiced frames with F3 < F2 (low-F3 /r/ cue)."""
    f2, f3 = tracks.get("F2"), tracks.get("F3")
    if f2 is None or f3 is None:
        return None
    m = np.isfinite(f2) & np.isfinite(f3)
    f2, f3 = f2[m], f3[m]
    return float(np.mean(f3 < f2)) if f2.size else None


def centralization_ratio(tracks):
    """Vowel-reduction proxy: fraction of voiced frames in the schwa region."""
    f1, f2 = tracks.get("F1"), tracks.get("F2")
    if f1 is None or f2 is None:
        return None
    m = np.isfinite(f1) & np.isfinite(f2)
    f1, f2 = f1[m], f2[m]
    if f1.size == 0:
        return None
    center = (f1 >= 400) & (f1 <= 700) & (f2 >= 1200) & (f2 <= 1800)
    return float(np.mean(center))


def final_f0_slope(f0, times):
    """F0 slope (Hz/s) over the last voiced run; positive = rising (uptalk)."""
    runs = _true_runs(f0 > 0.0)
    if not runs:
        return None
    a, b = runs[-1]
    tt, ff = times[a:b + 1], f0[a:b + 1]
    if tt.size < 2 or (tt[-1] - tt[0]) < 0.15:
        return None
    return float(np.polyfit(tt, ff, 1)[0])


def flap_like_closures(sound, f0, min_pitch=75.0, time_step=0.01):
    """Intervocalic /t/ proxy: unvoiced gaps of 20-45 ms inside speech."""
    out = {"flap_like_closures": 0, "flap_like_rate_per_s": None}
    try:
        intens = sound.to_intensity(minimum_pitch=min_pitch, time_step=time_step)
        db = np.asarray(intens.values).ravel()
        db = db[np.isfinite(db)]
        n = min(len(f0), len(db))
        f0, db = f0[:n], db[:n]
        if db.size == 0:
            return out
        speech = db > (np.max(db) - 25.0)
        voiced = f0 > 0.0
        flaps = 0
        for a, b in _true_runs((~voiced) & speech):
            dur = (b - a + 1) * time_step
            if 0.020 <= dur <= 0.045:
                flaps += 1
        speech_dur = float(speech.sum() * time_step)
        out["flap_like_closures"] = flaps
        out["flap_like_rate_per_s"] = (float(flaps / speech_dur)
                                       if speech_dur > 0 else None)
    except Exception:
        pass
    return out


def vot_summary(y, sr, f0, time_step=0.01):
    """VOT mean/median/n and short(<40 ms) fraction via high-freq burst onset."""
    out = {"vot_ms_mean": None, "vot_ms_median": None,
           "vot_n_bursts": 0, "vot_short_fraction": None}
    try:
        sos = scipy.signal.butter(4, [1500.0, 6000.0], btype="bandpass",
                                  fs=sr, output="sos")
        hf = scipy.signal.sosfilt(sos, y)
        frame = max(int(0.010 * sr), 1)
        hop = max(int(0.005 * sr), 1)
        if len(hf) < frame:
            return out
        frames = np.lib.stride_tricks.sliding_window_view(hf, frame)[::hop]
        env = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        env_times = (np.arange(len(env)) * hop + frame / 2) / sr
        rise = np.diff(env)
        rise = np.where(rise > 0, rise, 0.0)
        if rise.size == 0:
            return out
        thr = np.percentile(rise, 95.0)
        peaks, _ = scipy.signal.find_peaks(rise, height=thr,
                                           distance=int(0.02 / (hop / sr)))
        burst_times = env_times[peaks + 1]
        f0_times = np.arange(len(f0)) * time_step
        vots = []
        for b in burst_times:
            j = int(np.searchsorted(f0_times, b))
            while j < len(f0) and f0[j] <= 0.0:
                j += 1
            if j < len(f0):
                v = (f0_times[j] - b) * 1000.0
                if 5.0 <= v <= 150.0:
                    vots.append(v)
        if vots:
            vots = np.array(vots)
            out["vot_ms_mean"] = float(np.mean(vots))
            out["vot_ms_median"] = float(np.median(vots))
            out["vot_n_bursts"] = int(len(vots))
            out["vot_short_fraction"] = float(np.mean(vots < 40.0))
    except Exception:
        pass
    return out


def extract_accent(path):
    sound, y, sr = sf.load(path)
    f0, times = sf.f0_track(sound)
    voiced = f0 > 0.0
    tracks = sf.formant_values(sound, times, voiced)

    out = {}
    # Vowels, rhoticity, diphthong movement, reduction.
    out.update(sf.formant_stats(tracks))
    out.update(sf.vowel_space(tracks))
    out.update(sf.formant_movement(tracks))
    out["f3_below_f2_rate"] = f3_below_f2_rate(tracks)
    out["centralization_ratio"] = centralization_ratio(tracks)

    # Prosody, rhythm, timing.
    out.update(sf.f0_stats(f0, times))
    out.update(sf.rhythm_metrics(sound, f0))
    out.update(sf.speaking_rate(sound))
    out.update(vot_summary(y, sr, f0))
    out.update(flap_like_closures(sound, f0))
    out["final_f0_slope_hz_per_s"] = final_f0_slope(f0, times)

    # Spectral.
    out.update(sf.mfcc_features(y, sr))
    out["spectral_centroid_hz"] = sf.spectral_centroid_hz(y, sr)

    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="path to an audio file (wav/mp3/...)")
    p.add_argument("--out", default=None, help="optional output JSON path")
    args = p.parse_args(argv)

    res = extract_accent(args.audio)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
