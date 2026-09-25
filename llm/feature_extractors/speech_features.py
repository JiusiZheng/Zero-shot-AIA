"""Shared acoustic-feature helpers for the gender/age/accent extractors
(parselmouth/Praat, librosa, scipy/numpy). Vowel-space area, speaking rate,
rhythm metrics, VOT, and rhoticity are computed as proxies since no forced
phonetic aligner is used. All functions return None instead of raising on
missing/unvoiced data."""
from __future__ import annotations

import numpy as np
import scipy.signal
import parselmouth
from parselmouth.praat import call
import librosa

SPEED_OF_SOUND_CM_S = 34300.0  # cm/s, for vocal-tract-length estimation


def load(path: str):
    """Load audio as both a Praat Sound and a numpy array."""
    sound = parselmouth.Sound(path)
    y, sr = librosa.load(path, sr=None, mono=True)
    return sound, y, sr


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


# Fundamental frequency (F0)
def f0_track(sound, time_step: float = 0.01, pitch_floor: float = 60.0,
             pitch_ceiling: float = 500.0):
    """Return (f0 array, time array). Unvoiced frames are 0.0 Hz."""
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor,
                           pitch_ceiling=pitch_ceiling)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)
    times = np.asarray(pitch.xs(), dtype=float)
    return f0, times


def f0_stats(f0, times):
    """Summarise F0 over voiced frames."""
    voiced = f0[f0 > 0.0]
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.01
    out = {
        "voiced_fraction": float(np.mean(f0 > 0.0)) if len(f0) else None,
        "phonation_time_s": float(np.sum(f0 > 0.0) * dt) if len(f0) else None,
    }
    if voiced.size == 0:
        for k in ("f0_mean_hz", "f0_median_hz", "f0_min_hz", "f0_max_hz",
                  "f0_sd_hz", "f0_range_hz"):
            out[k] = None
    else:
        out.update({
            "f0_mean_hz": float(np.mean(voiced)),
            "f0_median_hz": float(np.median(voiced)),
            "f0_min_hz": float(np.min(voiced)),
            "f0_max_hz": float(np.max(voiced)),
            "f0_sd_hz": float(np.std(voiced)),
            "f0_range_hz": float(np.max(voiced) - np.min(voiced)),
        })
    return out


# Formants + vocal-tract length
def formant_values(sound, times, voiced, n_formants: int = 5,
                   max_formant: float = 5000.0):
    """Return F1..F{n_formants} arrays sampled at `times`, masked to voiced frames."""
    formant = sound.to_formant_burg(time_step=None,
                                    max_number_of_formants=float(n_formants),
                                    maximum_formant=max_formant,
                                    window_length=0.025,
                                    pre_emphasis_from=50.0)
    tracks = {}
    for i in range(1, n_formants + 1):
        vals = np.array([formant.get_value_at_time(i, float(t)) for t in times])
        keep = voiced & (vals > 0.0) & (vals < max_formant)
        tracks[f"F{i}"] = np.where(keep, vals, np.nan)
    return tracks


def formant_stats(tracks):
    """Mean F1..F4 over voiced, valid frames."""
    out = {}
    for key in ("F1", "F2", "F3", "F4"):
        if key in tracks:
            vals = _finite(tracks[key])
            out[f"{key.lower()}_hz"] = float(np.mean(vals)) if vals.size else None
    return out


def formant_spacing_vtl(tracks):
    """Mean adjacent-formant spacing and a vocal-tract-length estimate."""
    f1, f2, f3, f4 = (tracks.get("F1"), tracks.get("F2"),
                      tracks.get("F3"), tracks.get("F4"))
    if any(x is None for x in (f1, f2, f3, f4)):
        return {"formant_spacing_hz": None, "vtl_cm": None}
    ok = (np.isfinite(f1) & np.isfinite(f2) & np.isfinite(f3) & np.isfinite(f4))
    if not ok.any():
        return {"formant_spacing_hz": None, "vtl_cm": None}
    spacing = (np.abs(f2 - f1) + np.abs(f3 - f2) + np.abs(f4 - f3)) / 3.0
    spacing = spacing[ok]
    mean_spacing = float(np.mean(spacing))
    vtl = SPEED_OF_SOUND_CM_S / (2.0 * mean_spacing) if mean_spacing > 0 else None
    return {"formant_spacing_hz": mean_spacing, "vtl_cm": vtl}


# Voice quality: HNR, jitter, shimmer, intensity
def hnr_db(sound, time_step: float = 0.01, min_pitch: float = 75.0):
    try:
        h = call(sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)
        val = call(h, "Get mean", 0, 0)
        return float(val) if np.isfinite(val) else None
    except Exception:
        return None


def jitter_shimmer(sound, min_pitch: float = 75.0, max_pitch: float = 600.0):
    out = {"jitter_local_pct": None, "jitter_rap_pct": None,
           "shimmer_local_pct": None, "shimmer_apq3_pct": None}
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", min_pitch, max_pitch)
        out["jitter_local_pct"] = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)) * 100.0
        out["jitter_rap_pct"] = float(call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)) * 100.0
        out["shimmer_local_pct"] = float(call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)) * 100.0
        out["shimmer_apq3_pct"] = float(call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)) * 100.0
    except Exception:
        pass
    return out


def intensity_stats(sound, min_pitch: float = 75.0, time_step: float = 0.01):
    none = {"intensity_mean_db": None, "intensity_sd_db": None,
            "intensity_min_db": None, "intensity_max_db": None}
    try:
        intens = sound.to_intensity(minimum_pitch=min_pitch, time_step=time_step)
        db = _finite(np.asarray(intens.values).ravel())
        if db.size == 0:
            return none
        return {"intensity_mean_db": float(np.mean(db)),
                "intensity_sd_db": float(np.std(db)),
                "intensity_min_db": float(np.min(db)),
                "intensity_max_db": float(np.max(db))}
    except Exception:
        return none


# Spectral features (librosa / scipy)
def spectral_centroid_hz(y, sr):
    try:
        return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    except Exception:
        return None


def mfcc_features(y, sr, n_mfcc: int = 13):
    try:
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return {"mfcc_mean": m.mean(axis=1).tolist(),
                "mfcc_std": m.std(axis=1).tolist()}
    except Exception:
        return {"mfcc_mean": None, "mfcc_std": None}


def spectral_tilt_db_oct(y, sr, fmin: float = 200.0, fmax: float = 4000.0):
    """Slope of the long-term spectrum in dB per octave (a source-tilt proxy)."""
    try:
        if len(y) < 512:
            return None
        freqs, psd = scipy.signal.welch(y, fs=sr, nperseg=min(2048, len(y)))
        band = (freqs >= fmin) & (freqs <= fmax)
        freqs, psd = freqs[band], psd[band]
        if freqs.size < 8:
            return None
        db = 10.0 * np.log10(psd + 1e-12)
        octaves = np.log2(freqs / freqs[0])
        slope = np.polyfit(octaves, db, 1)[0]  # dB per octave
        return float(slope)
    except Exception:
        return None


def h1_h2_db(y, sr, f0_mean):
    """H1-H2 (dB) over a voiced window; larger values indicate a breathier voice."""
    try:
        if not f0_mean or f0_mean <= 0:
            return None
        win = int(0.06 * sr)  # 60 ms
        if len(y) < win:
            return None
        start = max(0, (len(y) - win) // 2)
        seg = y[start:start + win] * np.hanning(win)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        db = 20.0 * np.log10(spec + 1e-12)

        def peak(center, lo, hi):
            m = (freqs >= center * lo) & (freqs <= center * hi)
            return np.max(db[m]) if m.any() else None

        h1 = peak(f0_mean, 0.7, 1.3)
        h2 = peak(2 * f0_mean, 0.85, 1.15)
        if h1 is None or h2 is None:
            return None
        return float(h1 - h2)
    except Exception:
        return None


# Speaking rate, rhythm, VOT (proxy measures)
def _run_lengths(mask):
    """Lengths (in frames) of consecutive True runs."""
    mask = np.asarray(mask, dtype=bool)
    runs, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return runs


def speaking_rate(sound, min_pitch: float = 75.0, time_step: float = 0.01):
    """Syllable count/rate via intensity-envelope peaks (de Jong & Wempe proxy)."""
    out = {"syllable_count": None, "speaking_rate_syll_per_s": None}
    try:
        intens = sound.to_intensity(minimum_pitch=min_pitch, time_step=time_step)
        db = _finite(np.asarray(intens.values).ravel())
        if db.size == 0 or sound.duration <= 0:
            return out
        floor = np.max(db) - 30.0
        speech = db > floor
        if not speech.any():
            return out
        thr = np.percentile(db[speech], 50.0)
        peaks, _ = scipy.signal.find_peaks(db, height=thr,
                                           distance=int(0.05 / time_step))
        peaks = [p for p in peaks if speech[p]]
        out["syllable_count"] = int(len(peaks))
        out["speaking_rate_syll_per_s"] = float(len(peaks) / sound.duration)
    except Exception:
        pass
    return out


def rhythm_metrics(sound, f0, min_pitch: float = 75.0, time_step: float = 0.01):
    """%V, delta-C and nPVI from voiced/unvoiced runs within the speech region."""
    out = {"pct_vocalic": None, "delta_c_ms": None, "npvi": None}
    try:
        intens = sound.to_intensity(minimum_pitch=min_pitch, time_step=time_step)
        db = _finite(np.asarray(intens.values).ravel())
        n = min(len(f0), len(db))
        f0 = f0[:n]
        db = db[:n]
        speech = db > (np.max(db) - 25.0) if db.size else np.zeros(n, bool)
        voiced = f0 > 0.0
        vocalic = voiced & speech
        consonantal = (~voiced) & speech
        v_runs = _run_lengths(vocalic)
        c_runs = _run_lengths(consonantal)
        speech_frames = int(speech.sum())
        if speech_frames == 0:
            return out
        out["pct_vocalic"] = float(100.0 * vocalic.sum() / speech_frames)
        if c_runs:
            c_s = np.array(c_runs) * time_step
            out["delta_c_ms"] = float(np.std(c_s) * 1000.0)
        if len(v_runs) >= 2:
            v_s = np.array(v_runs, dtype=float) * time_step
            diffs = np.abs(np.diff(v_s))
            means = (v_s[:-1] + v_s[1:]) / 2.0
            out["npvi"] = float(100.0 * np.mean(diffs / means))
    except Exception:
        pass
    return out


def vot_ms(y, sr, f0, time_step: float = 0.01):
    """Approximate positive VOT: high-freq burst onset -> next voicing onset (no forced alignment)."""
    out = {"vot_ms_mean": None, "vot_ms_median": None, "vot_n_bursts": 0}
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
                v = (f0_times[j] - b) * 1000.0  # ms
                if 5.0 <= v <= 150.0:
                    vots.append(v)
        if vots:
            out["vot_ms_mean"] = float(np.mean(vots))
            out["vot_ms_median"] = float(np.median(vots))
            out["vot_n_bursts"] = int(len(vots))
    except Exception:
        pass
    return out


# Vowel space and formant dynamics
def vowel_space(tracks):
    """Approx. vowel-space area = convex-hull area of (F2, F1), plus F1/F2 ranges."""
    out = {"vowel_space_area_hz2": None, "f1_range_hz": None, "f2_range_hz": None}
    try:
        f1 = tracks.get("F1")
        f2 = tracks.get("F2")
        if f1 is None or f2 is None:
            return out
        m = (np.isfinite(f1) & np.isfinite(f2) & (f1 > 50) & (f1 < 1500)
             & (f2 > 300) & (f2 < 3500))
        f1, f2 = f1[m], f2[m]
        if f1.size < 3:
            return out
        lo1, hi1 = np.percentile(f1, [2.5, 97.5])
        lo2, hi2 = np.percentile(f2, [2.5, 97.5])
        keep = (f1 >= lo1) & (f1 <= hi1) & (f2 >= lo2) & (f2 <= hi2)
        f1, f2 = f1[keep], f2[keep]
        if f1.size < 3:
            return out
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.column_stack([f2, f1]))
        out["vowel_space_area_hz2"] = float(hull.volume)
        out["f1_range_hz"] = float(np.max(f1) - np.min(f1))
        out["f2_range_hz"] = float(np.max(f2) - np.min(f2))
    except Exception:
        pass
    return out


def formant_movement(tracks, time_step: float = 0.01):
    """Mean |dF1| and |dF2| per second over voiced frames (diphthongisation proxy)."""
    out = {"formant_movement_f1_hz_per_s": None,
           "formant_movement_f2_hz_per_s": None}
    try:
        f1 = tracks.get("F1")
        f2 = tracks.get("F2")
        if f1 is None or f2 is None:
            return out
        d1 = np.abs(np.diff(f1))
        d1 = d1[np.isfinite(d1)]
        d2 = np.abs(np.diff(f2))
        d2 = d2[np.isfinite(d2)]
        if d1.size:
            out["formant_movement_f1_hz_per_s"] = float(np.mean(d1) / time_step)
        if d2.size:
            out["formant_movement_f2_hz_per_s"] = float(np.mean(d2) / time_step)
    except Exception:
        pass
    return out
