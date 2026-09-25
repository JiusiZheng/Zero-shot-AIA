# Speaker Attribute-Inference Attacks on Common Voice

Code and prompts for four attackers — LR classifier, LLM, ALM, autonomous
agent — inferring a speaker's gender, age, and accent from Common Voice
speech, plus a defense evaluation against 4 anonymization methods
(`defense/`).

## Overview

- **[Supplementary Table 1](#supplementary-table-1--llm-medium-effort-optimized-prompt-results)**
  — LLM attacker, medium-effort (optimized-prompt) results across all 3 test
  sets. (Low-effort numbers and every other attacker are in the paper's main
  results table.)
- **[Supplementary Table 2](#supplementary-table-2--agents-chosen-pretrained-models)**
  — pretrained models/methods the autonomous agent chose, per test set.
- **All prompts**, exactly as sent to each model → [`prompts/`](prompts/)
  (`llm/`, `alm/`, `agent/`).
- **All code**:

  ```
  classifier/   LR classifier attack
  llm/          LLM-supported attack (feature_extractors/, optimization_dev/, optimization_test/)
  alm/          Zero-shot ALM attack
  defense/      Anonymized-speech re-evaluation against the 4 anonymization methods
  ```

Setup, data download, and how to run each attack are further down, after
the two supplementary tables.

## Supplementary Table 1 — LLM medium-effort (optimized prompt) results

Balanced accuracy (Bal. acc.) and per-class recall (%), mean ± sample SD
across the 3 test sets. Low-effort numbers and every other attacker are in
the paper's main results table.

| Model                 | Gender Bal. acc. | Male        | Female      | Age Bal. acc. | >50          | <31          | 31–50      | Accent Bal. acc. | Ind./S. Asia | England    | Aus.       | Can.       | US           |
| --------------------- | ---------------- | ----------- | ----------- | ------------- | ------------ | ------------ | ----------- | ---------------- | ------------ | ---------- | ---------- | ---------- | ------------ |
| Gemma-3-12B-it        | 84.00±1.76      | 74.67±3.53 | 93.33±0.00 | 33.11±2.04   | 56.67±5.03  | 0.00±0.00   | 42.67±4.16 | 19.11±1.02      | 77.78±1.92  | 0.00±0.00 | 2.22±1.92 | 5.56±1.92 | 10.00±3.33  |
| Llama-3.1-8B-Instruct | 79.78±1.39      | 63.56±2.78 | 96.00±0.00 | 34.89±7.13   | 26.00±6.00  | 40.00±10.39 | 38.67±7.02 | 20.22±0.38      | 100.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 1.11±1.92   |
| Qwen2.5-7B-Instruct   | 84.00±0.67      | 76.00±1.33 | 92.00±1.33 | 32.67±1.15   | 89.33±2.31  | 0.00±0.00   | 8.67±3.06  | 20.00±0.00      | 0.00±0.00   | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 100.00±0.00 |
| Olmo-3-7B-Instruct    | 64.00±0.67      | 30.67±1.33 | 97.33±0.00 | 33.33±0.00   | 100.00±0.00 | 0.00±0.00   | 0.00±0.00  | 20.67±1.76      | 5.56±6.94   | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 97.78±1.92  |
| Random guess          | 50.00            | 50.00       | 50.00       | 33.33         | 33.33        | 33.33        | 33.33       | 20.00            | 20.00        | 20.00      | 20.00      | 20.00      | 20.00        |

## Supplementary Table 2 — Agent's chosen pretrained models

Pretrained models/methods chosen autonomously by the agent (see
`prompts/agent/`); age predictions mapped to the project's 3 age groups.

| Attribute | Test set | Model or method                                                                                                            |
| --------- | -------- | -------------------------------------------------------------------------------------------------------------------------- |
| Gender    | 1        | pYIN pitch threshold (165 Hz)                                                                                              |
| Gender    | 2        | [prithivMLmods/Common-Voice-Gender-Detection](https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection)           |
| Gender    | 3        | YIN pitch threshold (165 Hz)                                                                                               |
| Age       | 1–3     | [audeering/wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) |
| Accent    | 1        | [MilesPurvis/english-accent-classifier](https://huggingface.co/MilesPurvis/english-accent-classifier)                       |
| Accent    | 2, 3     | [dima806/english_accents_classification](https://huggingface.co/dima806/english_accents_classification)                     |

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is required for anything that loads a model;
feature extraction and result aggregation are CPU-only.

## Data

Nothing is bundled here — grep `/path/to/data` to find every path to edit.

- **Train / dev / test** — download the English Common Voice dataset:
  [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets).
- **Anonymized speech** (for `defense/`) — generated with the 4 baseline
  anonymization systems from the
  [Voice Privacy Challenge 2026](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026).

## Running the attacks

### 1. LR classifier (`classifier/`)

```bash
python classifier/cv_extra_filter_candidates.py
python classifier/cv_extra_split_v3.py
python classifier/CV_Extra_Classifier_Train_Eval.py
```

### 2. LLM (`llm/`)

Low-effort = full feature set + `prompts/llm/original/`. Medium-effort =
top-3 ranked feature *categories* only (e.g. gender keeps F0 + formants +
formant spacing/VTL → 10 numbers) + `prompts/llm/optimized/`. Both evaluate
Gemma-3-12B-it, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Olmo-3-7B-Instruct.

```bash
# Dev set, low-effort
python llm/run_cv_dev_attribute_inference.py --stage features
python llm/run_cv_dev_attribute_inference.py --stage infer --model_index 0   # 0..3
python llm/run_cv_dev_attribute_inference.py --stage aggregate

# Test set, low-effort
python llm/run_cv_test_attribute_inference.py --stage features --subset all
python llm/run_cv_test_attribute_inference.py --stage infer --model_index 0  # 0..3
python llm/run_cv_test_attribute_inference.py --stage aggregate

# Medium-effort: same stages, from optimization_dev/ and optimization_test/
python llm/optimization_dev/run_cv_dev_optimized_prompt_eval.py --stage features
python llm/optimization_test/run_cv_test_optimized_prompt_eval.py --stage features --subset all
```

`--stage prompt_preview` prints one assembled prompt without touching a GPU.

### 3. ALM (`alm/`)

```bash
python alm/ALM_Infer.py --inspect   # sanity-check, no model/audio
python alm/ALM_Infer.py             # full run
```

### 4. Agent

No script — prompt per attribute in `prompts/agent/`; run manually with an
agent of your choice.

### 5. Defense (`defense/`)

Re-evaluates 3 frozen attackers — the LR classifier, the ALM attacker
(Qwen2.5-Omni-7B), and the LLM text attacker (Qwen2.5-7B-Instruct) — against
speech processed by the 4 Voice Privacy Challenge 2026 anonymization
baselines, without any retraining. For every utterance, the original test
audio is swapped for its anonymized counterpart (same `common_voice_en_<id>`
basename) before being shown to the attacker.

```bash
python defense/classifier_anon_eval.py
python defense/llm_anon_eval.py
python defense/alm_anon_eval.py
```

Each script writes one JSON file per anonymization method to
`defense/results/` (`classifier_<method>.json`, `llm_<method>.json`,
`alm_<method>.json`), holding balanced accuracy and per-class recall.
`classifier_anon_eval.py` needs a working speechbrain/ECAPA stack;
`llm_anon_eval.py` and `alm_anon_eval.py` need the Transformers/Qwen stack
(including `qwen-omni-utils` for the ALM attacker) — the original project
ran these in two separate conda environments for dependency reasons, not a
hard requirement of the code itself.

## Prompts (`prompts/`)

Exactly as sent to the model. `llm/original/` and `llm/optimized/` still
have a feature-JSON placeholder filled in at run time; `alm/` and `agent/`
are fully instantiated, one file per attribute. `llm/metaprompt_<attribute>.txt`
is a separate, one-time offline step: the prompt given to DeepSeek-V4-Pro to
generate `llm/optimized/`'s prompts in the first place (not read by any
script here).
