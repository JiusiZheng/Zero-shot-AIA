"""Text-LLM loading and batched generation, shared by every LLM-supported
attack script."""
from __future__ import annotations

import sys

_llm_cache: dict = {}


def get_llm(model_name: str, cache_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _llm_cache.get("model_name") != model_name:
        print(f"[llm] loading {model_name} ...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _llm_cache["model_name"] = model_name
        _llm_cache["model"] = (tokenizer, model)
    return _llm_cache["model"]


def _apply_chat_template(tokenizer, user_content: str):
    messages = [{"role": "user", "content": user_content}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
    # transformers version gotcha: return_tensors="pt" may give a tensor or a BatchEncoding.
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return encoded[0]


def _generate_one_batch(tokenizer, model, user_contents: list[str], max_new_tokens: int, temperature: float) -> list[str]:
    import torch

    encoded = [_apply_chat_template(tokenizer, c) for c in user_contents]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id")

    max_len = max(ids.shape[0] for ids in encoded)
    input_ids = torch.full((len(encoded), max_len), pad_token_id, dtype=encoded[0].dtype)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(encoded):
        input_ids[row, -ids.shape[0]:] = ids
        attention_mask[row, -ids.shape[0]:] = 1

    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
    generated = output_ids[:, input_ids.shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def generate_batch(tokenizer, model, user_contents: list[str], max_new_tokens: int, temperature: float, batch_size: int) -> list[str]:
    import torch

    results: list[str | None] = [None] * len(user_contents)
    i = 0
    bs = batch_size
    while i < len(user_contents):
        chunk = user_contents[i:i + bs]
        try:
            out = _generate_one_batch(tokenizer, model, chunk, max_new_tokens, temperature)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f"  [warn] CUDA OOM, retrying with batch_size={bs}", file=sys.stderr)
            continue
        for j, text in enumerate(out):
            results[i + j] = text
        i += len(chunk)
    return results  # type: ignore[return-value]
