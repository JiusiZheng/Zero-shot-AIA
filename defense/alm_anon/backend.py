"""Qwen2.5-Omni model backend: generation helpers, the model adapter, HF
preflight check, and process teardown."""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import ATTN_IMPLEMENTATION, GENERATION_TEMPERATURE, HF_CACHE_DIR  # noqa: E402

torch = None  # imported only for real inference; --inspect needs no ML deps


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def generation_kwargs(max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": GENERATION_TEMPERATURE > 0,
    }
    if GENERATION_TEMPERATURE > 0:
        kwargs["temperature"] = GENERATION_TEMPERATURE
    return kwargs


def preferred_dtype() -> "torch.dtype":
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def first_model_device(model: Any) -> "torch.device":
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def move_inputs(inputs: Any, device, floating_dtype) -> Any:
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=floating_dtype)
            else:
                inputs[key] = value.to(device=device)
    return inputs


def require_transformers_version() -> None:
    import transformers
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.54.0"):
        raise RuntimeError(
            f"transformers {transformers.__version__} is installed, but "
            "Qwen2.5-Omni requires transformers>=4.54.0 (use the psst_3 env)."
        )
    eprint(f"[environment] transformers={transformers.__version__}")


def preflight_hf_access(model_id: str) -> None:
    from huggingface_hub import hf_hub_download, whoami
    try:
        account = whoami()
        eprint(f"[hf] Authenticated as: {account.get('name', '<unknown>')}")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication is unavailable. Run `hf auth login`."
        ) from exc
    try:
        hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=HF_CACHE_DIR)
        eprint(f"[hf-access OK] {model_id}")
    except Exception as exc:
        raise RuntimeError(f"Hugging Face access check failed for {model_id}: {exc}") from exc


# Copied from ALM_Infer.py, Qwen-only.
class Qwen25OmniAdapter:
    def __init__(self, model_id: str):
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

        self.dtype = preferred_dtype()
        load_kwargs: dict[str, Any] = {
            "cache_dir": HF_CACHE_DIR,
            "dtype": self.dtype,
            "device_map": "auto",
        }
        if ATTN_IMPLEMENTATION:
            load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

        self.process_mm_info = process_mm_info
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

        # Widen the stop set to include "<|im_end|>" (see ALM_Infer.py).
        tokenizer = self.processor.tokenizer
        eos_ids: list[int] = []
        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            im_end_id is not None
            and im_end_id != tokenizer.unk_token_id
            and im_end_id not in eos_ids
        ):
            eos_ids.append(im_end_id)
        self.eos_token_ids = eos_ids or None

    def _query(self, conversation: list[dict[str, Any]], max_new_tokens: int) -> str:
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = self.process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        device = first_model_device(self.model)
        inputs = move_inputs(inputs, device, self.dtype)
        input_length = inputs["input_ids"].shape[1]

        extra_kwargs: dict[str, Any] = {}
        if self.eos_token_ids is not None:
            extra_kwargs["eos_token_id"] = self.eos_token_ids

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                **extra_kwargs,
                **generation_kwargs(max_new_tokens),
            )
        generated_ids = output_ids[:, input_length:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def query_audio(self, audio_path: str, prompt: str, max_new_tokens: int) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._query(conversation, max_new_tokens)

    def close(self) -> None:
        del self.processor
        del self.model


def release_model(model: Any | None) -> None:
    if model is not None:
        try:
            model.close()
        finally:
            del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
