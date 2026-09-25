#!/usr/bin/env python3
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

EXTRA_ROOT = Path("/path/to/data/CommonVoice_Extra")  # EDIT: point this at your own CommonVoice-Extra data root
CANDIDATES_TSV = EXTRA_ROOT / "filtered_candidates_train.tsv"
TRAIN_AUDIO_DIR = EXTRA_ROOT / "audio/en/train"
AUDIO_OUT = EXTRA_ROOT / "audio_extracted_v3"
MANIFEST_OUT = EXTRA_ROOT / "train_dev_manifest_v3.json"

SEED = 42
MAX_CLIPS_PER_SPEAKER = 6
DEV_CLIPS_PER_SPEAKER = 5
DEV_MIN_CLIPS = 5

GENDERS = ["male_masculine", "female_feminine"]
AGES = ["less than 31", "31 to 50", "more than 50"]
ACCENTS = ["Australian", "Canadian", "India and South Asia", "England", "United States"]

# Zero speakers anywhere in the corpus; filled from FALLBACK_CELL instead.
EMPTY_CELL = ("female_feminine", "more than 50", "India and South Asia")
FALLBACK_CELL = ("female_feminine", "31 to 50", "India and South Asia")

TRAIN_SPEAKER_TARGET = {
    "Australian": "all",
    "Canadian": "all",
    "India and South Asia": 900,
    "England": 900,
    "United States": 900,
}


def list_shard_members():
    basename_to_tar = {}
    shard_paths = sorted(TRAIN_AUDIO_DIR.glob("en_train_*.tar"))
    print(f"Found {len(shard_paths)} downloaded train shards: {[p.name for p in shard_paths]}")
    for shard in shard_paths:
        result = subprocess.run(["tar", "-tf", str(shard)], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.endswith(".mp3"):
                basename_to_tar[line.rsplit("/", 1)[-1]] = (shard, line)
    print(f"Total mp3 files available across downloaded shards: {len(basename_to_tar)}")
    return basename_to_tar


def pick_dev_speakers(rng, by_speaker_meta, eligible_ids):
    """One speaker per (gender, age, accent) cell, +1 extra to cover EMPTY_CELL."""
    cells = defaultdict(list)
    for sid in eligible_ids:
        meta = by_speaker_meta[sid]
        cells[(meta["gender"], meta["age_label"], meta["accent_label"])].append(sid)

    plan = [(g, a, acc) for g in GENDERS for a in AGES for acc in ACCENTS]
    assert EMPTY_CELL in plan
    plan.remove(EMPTY_CELL)
    plan.append(FALLBACK_CELL)  # picked twice: once as itself, once as EMPTY_CELL's stand-in
    assert len(plan) == 30

    chosen = []
    used = set()
    for cell in plan:
        candidates = sorted(c for c in cells.get(cell, []) if c not in used)
        if not candidates:
            raise RuntimeError(f"No eligible speaker left for cell {cell} (used={len(used)})")
        rng.shuffle(candidates)
        pick = candidates[0]
        used.add(pick)
        chosen.append(pick)

    assert len(chosen) == 30 and len(set(chosen)) == 30
    return chosen


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(CANDIDATES_TSV, sep="\t", quoting=3, low_memory=False)
    df["fname"] = df["path"].apply(lambda p: p if p.endswith(".mp3") else p + ".mp3")

    basename_to_tar = list_shard_members()
    df = df[df["fname"].isin(basename_to_tar)]
    print(f"Candidate rows with audio actually available in downloaded shards: {len(df)}")

    by_speaker = df.groupby("client_id")
    clip_counts = by_speaker.size()
    eligible_ids = set(clip_counts[clip_counts >= DEV_MIN_CLIPS].index)

    meta = df.drop_duplicates("client_id").set_index("client_id")[["gender", "age_label", "accent_label"]]
    by_speaker_meta = meta.to_dict("index")

    dev_ids = pick_dev_speakers(rng, by_speaker_meta, eligible_ids)
    dev_ids_set = set(dev_ids)
    print(f"\nDev speakers chosen: {len(dev_ids_set)}")

    accent_of = meta["accent_label"].to_dict()
    speakers_by_accent = defaultdict(list)
    for sid, accent in accent_of.items():
        if sid in dev_ids_set:
            continue
        speakers_by_accent[accent].append(sid)

    train_ids = set()
    for accent, sids in speakers_by_accent.items():
        sids = sorted(sids)
        rng.shuffle(sids)
        target = TRAIN_SPEAKER_TARGET.get(accent, len(sids))
        n_train = len(sids) if target == "all" else min(target, len(sids))
        train_ids.update(sids[:n_train])

    def build_records(ids, clips_per_speaker):
        records = []
        for sid in sorted(set(ids)):
            rows = by_speaker.get_group(sid)
            fnames = sorted(rows["fname"].unique().tolist())
            rng.shuffle(fnames)
            chosen = fnames[:clips_per_speaker]
            for fname in chosen:
                row = rows[rows["fname"] == fname].iloc[0]
                shard_path, member = basename_to_tar[fname]
                records.append(dict(
                    speaker_id=sid, filename=fname, shard=shard_path.name, tar_member=member,
                    gender=row["gender"], age=row["age_label"], accent=row["accent_label"],
                ))
        return records

    manifest = dict(
        seed=SEED,
        train=build_records(train_ids, MAX_CLIPS_PER_SPEAKER),
        dev=build_records(dev_ids, DEV_CLIPS_PER_SPEAKER),
    )

    for split_name in ("train", "dev"):
        recs = manifest[split_name]
        print(f"\n{split_name}: {len(recs)} clips, {len({r['speaker_id'] for r in recs})} speakers")
        for attr in ("gender", "age", "accent"):
            counts = {}
            for r in recs:
                counts[r[attr]] = counts.get(r[attr], 0) + 1
            print(f"  {attr}: {counts}")
        if split_name == "dev":
            speaker_attrs = {r["speaker_id"]: (r["gender"], r["age"], r["accent"]) for r in recs}
            for attr_idx, attr in enumerate(("gender", "age", "accent")):
                counts = {}
                for v in speaker_attrs.values():
                    counts[v[attr_idx]] = counts.get(v[attr_idx], 0) + 1
                print(f"  speaker-level {attr}: {counts}")

    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    members_by_shard = defaultdict(set)
    for split_name in ("train", "dev"):
        for r in manifest[split_name]:
            members_by_shard[r["shard"]].add(r["tar_member"])

    for shard_name, members in members_by_shard.items():
        shard_path = TRAIN_AUDIO_DIR / shard_name
        list_file = EXTRA_ROOT / f"extract_list_v3_{shard_name}.txt"
        list_file.write_text("\n".join(sorted(members)))
        print(f"\nExtracting {len(members)} files from {shard_name}...")
        subprocess.run(["tar", "-xf", str(shard_path), "-C", str(AUDIO_OUT), "-T", str(list_file)], check=True)

    for split_name in ("train", "dev"):
        for r in manifest[split_name]:
            r["audio_path"] = str(AUDIO_OUT / r["tar_member"])

    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
