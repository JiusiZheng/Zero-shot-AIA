#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

# EDIT: point this at your own CommonVoice_Extra data root (same path you
# use for EXTRA_ROOT in cv_extra_split_v3.py).
EXTRA_ROOT = Path("/path/to/data/CommonVoice_Extra")
TRAIN_TSV = EXTRA_ROOT / "transcript/en/train.tsv"
# EDIT: same test-set label JSON used throughout this project.
CV_TEST_JSON = "/path/to/data/CV_test_meta/3_testsets_attribute_labels.json"
OUT_TSV = EXTRA_ROOT / "filtered_candidates_train.tsv"

# Must match Classifier_Attack.py's LABELS exactly.
ACCENT_MAP = {
    "United States English": "United States",
    "England English": "England",
    "Canadian English": "Canadian",
    "Australian English": "Australian",
    "India and South Asia (India, Pakistan, Sri Lanka)": "India and South Asia",
}
AGE_MAP = {
    "teens": "less than 31", "twenties": "less than 31",
    "thirties": "31 to 50", "fourties": "31 to 50",
    "fifties": "more than 50", "sixties": "more than 50",
    "seventies": "more than 50", "eighties": "more than 50", "nineties": "more than 50",
}
GENDERS = {"male_masculine", "female_feminine"}


def main():
    df = pd.read_csv(TRAIN_TSV, sep="\t", quoting=3, low_memory=False)
    print(f"total train.tsv rows: {len(df)}")

    our_test_speakers = {r["speaker_id"] for r in json.loads(Path(CV_TEST_JSON).read_text())}
    df = df[~df["client_id"].isin(our_test_speakers)]

    df = df[df["accents"].isin(ACCENT_MAP) & df["age"].isin(AGE_MAP) & df["gender"].isin(GENDERS)]
    df["accent_label"] = df["accents"].map(ACCENT_MAP)
    df["age_label"] = df["age"].map(AGE_MAP)

    print(f"usable rows after filtering + excluding our {len(our_test_speakers)} test speakers: {len(df)}")
    print(f"unique speakers: {df['client_id'].nunique()}")

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
