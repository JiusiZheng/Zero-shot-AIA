#!/usr/bin/env python3
"""Extract the feature set for the GENDER prediction prompt."""
import argparse
import json
import sys

import speech_features as sf


def extract_gender(path):
    sound, y, sr = sf.load(path)
    f0, times = sf.f0_track(sound)
    voiced = f0 > 0.0
    tracks = sf.formant_values(sound, times, voiced)

    out = {}
    out.update(sf.f0_stats(f0, times))
    out.update(sf.formant_stats(tracks))
    out.update(sf.formant_spacing_vtl(tracks))
    out["hnr_db"] = sf.hnr_db(sound)
    out.update(sf.jitter_shimmer(sound))
    out.update(sf.intensity_stats(sound))
    out["spectral_tilt_db_per_octave"] = sf.spectral_tilt_db_oct(y, sr)
    out["spectral_centroid_hz"] = sf.spectral_centroid_hz(y, sr)
    out["h1_h2_db"] = sf.h1_h2_db(y, sr, out.get("f0_mean_hz"))
    mf = sf.mfcc_features(y, sr)
    out["mfcc_mean"] = mf["mfcc_mean"]
    out["mfcc_std"] = mf["mfcc_std"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="path to an audio file (wav/mp3/...)")
    p.add_argument("--out", default=None, help="optional output JSON path")
    args = p.parse_args(argv)

    res = extract_gender(args.audio)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
