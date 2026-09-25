"""Per-attribute-per-subset inference driver (N_VOTES loop + majority vote)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from backends import AudioLanguageModel
from config import CONTINUE_ON_AUDIO_ERROR, MAX_NEW_TOKENS_UTTERANCE, N_VOTES
from prompts import build_user_prompt
from scoring import classify_vote, majority_vote
from util import eprint


def run_attribute_on_subset(
    model: AudioLanguageModel,
    rows: list[dict[str, Any]],
    attribute: str,
    categories: tuple[str, ...],
    subset_name: str,
    model_name: str,
    json_dir: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    prompt = build_user_prompt(attribute, categories)
    results = []
    for row in rows:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (json_dir / audio_path).resolve()

        # read only for later scoring -- never inserted into `prompt` or passed to query_audio()
        true_label = row[attribute]

        raw_responses: list[str | None] = []
        errors: list[str | None] = []
        for _ in range(N_VOTES):
            try:
                response = model.query_audio(str(audio_path), prompt, MAX_NEW_TOKENS_UTTERANCE)
                raw_responses.append(response)
                errors.append(None)
            except Exception as exc:
                raw_responses.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
                if not CONTINUE_ON_AUDIO_ERROR:
                    raise

        parsed_votes: list[str | None] = []
        vote_statuses: list[str] = []
        for response in raw_responses:
            label, status = classify_vote(response, categories)
            parsed_votes.append(label)
            vote_statuses.append(status)
            if status == "hallucination":
                eprint(
                    f"[hallucination] model={model_name} subset={subset_name} "
                    f"attribute={attribute} speaker_id={row.get('speaker_id')} "
                    f"raw_response={response!r} allowed_categories={categories}"
                )

        prediction = majority_vote(parsed_votes, rng)
        prediction_invalid = prediction is None

        n_hallucinations = vote_statuses.count("hallucination")
        n_errors = vote_statuses.count("error")

        result = {
            "model": model_name,
            "subset": subset_name,
            "attribute": attribute,
            "speaker_id": row.get("speaker_id"),
            "audio_path": str(audio_path),
            "raw_responses": raw_responses,
            "votes": parsed_votes,
            "vote_statuses": vote_statuses,
            "n_hallucinations": n_hallucinations,
            "n_errors": n_errors,
            "prediction": prediction,
            "prediction_invalid": prediction_invalid,
            "true_label": true_label,
            "errors": errors,
        }
        print(json.dumps({"type": "utterance_result", **result}, ensure_ascii=False), flush=True)
        results.append(result)
    return results
