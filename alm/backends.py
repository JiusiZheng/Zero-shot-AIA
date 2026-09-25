"""torch/transformers-backed utilities and audio-language-model adapters.

Imported lazily (after the --inspect early-return in ALM_Infer.py), so the
top-level `import torch` here never runs in --inspect mode."""

from __future__ import annotations

import gc
from typing import Any, Iterable

import torch

from config import ATTN_IMPLEMENTATION, GENERATION_TEMPERATURE, HF_CACHE_DIR, ModelSpec
from util import eprint


def generation_kwargs(max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": GENERATION_TEMPERATURE > 0,
    }
    if GENERATION_TEMPERATURE > 0:
        kwargs["temperature"] = GENERATION_TEMPERATURE
    return kwargs


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def first_model_device(model: Any) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def move_inputs(inputs: Any, device: torch.device | str, floating_dtype: torch.dtype) -> Any:
    """Move a BatchFeature/dict without converting token IDs to floating point."""
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=floating_dtype)
            else:
                inputs[key] = value.to(device=device)
    return inputs


def require_transformers_version() -> None:
    try:
        import transformers
        from packaging.version import Version
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers>=4.54, packaging and the other dependencies "
            "before running this script."
        ) from exc

    if Version(transformers.__version__) < Version("4.54.0"):
        raise RuntimeError(
            f"transformers {transformers.__version__} is installed, but Voxtral "
            "requires transformers>=4.54.0."
        )
    eprint(f"[environment] transformers={transformers.__version__}")


def preflight_hf_access(model_specs: Iterable[ModelSpec]) -> None:
    """Check small config files before attempting multi-GB model downloads."""
    try:
        from huggingface_hub import hf_hub_download, whoami
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required.") from exc

    try:
        account = whoami()
        eprint(f"[hf] Authenticated as: {account.get('name', '<unknown>')}")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication is unavailable. Run `hf auth login` "
            "with the same account that has model access."
        ) from exc

    repos: list[str] = []
    for spec in model_specs:
        repos.append(spec.model_id)
        repos.extend(spec.extra_access_repos)

    failures: list[tuple[str, str]] = []
    for repo_id in dict.fromkeys(repos):
        try:
            hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=HF_CACHE_DIR)
            eprint(f"[hf-access OK] {repo_id}")
        except Exception as exc:
            failures.append((repo_id, f"{type(exc).__name__}: {exc}"))
            eprint(f"[hf-access FAILED] {repo_id}: {exc}")

    if failures:
        lines = ["Hugging Face access check failed:"]
        lines.extend(f"  - {repo}: {error}" for repo, error in failures)
        lines.append(
            "Accept the relevant model license in the browser, then run "
            "`hf auth login` on this server with the same account."
        )
        raise RuntimeError("\n".join(lines))


# audio-language model adapters

class AudioLanguageModel:
    def query_audio(self, audio_path: str, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Qwen25OmniAdapter(AudioLanguageModel):
    def __init__(self, model_id: str):
        try:
            from qwen_omni_utils import process_mm_info
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-Omni requires qwen-omni-utils and a Transformers "
                "version containing Qwen2_5OmniThinkerForConditionalGeneration."
            ) from exc

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

        # Widen the EOS set with <|im_end|>: the default generation_config often lacks it,
        # so generate() would otherwise run on and hallucinate extra turns past the answer.
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


class AudioFlamingo3Adapter(AudioLanguageModel):
    def __init__(self, model_id: str):
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Audio Flamingo 3 requires a Transformers build containing "
                "AudioFlamingo3ForConditionalGeneration. Run: "
                "python -m pip install -U transformers accelerate librosa soundfile"
            ) from exc

        self.dtype = preferred_dtype()
        load_kwargs: dict[str, Any] = {
            "cache_dir": HF_CACHE_DIR,
            "dtype": self.dtype,
            "device_map": "auto",
        }
        if ATTN_IMPLEMENTATION:
            load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def _query(self, conversation: list[dict[str, Any]], max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = first_model_device(self.model)
        inputs = move_inputs(inputs, device, self.dtype)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs(max_new_tokens))
        return self.processor.batch_decode(
            output_ids[:, input_length:], skip_special_tokens=True
        )[0].strip()

    def query_audio(self, audio_path: str, prompt: str, max_new_tokens: int) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._query(conversation, max_new_tokens)

    def close(self) -> None:
        del self.processor
        del self.model


class VoxtralAdapter(AudioLanguageModel):
    def __init__(self, model_id: str):
        try:
            from transformers import AutoProcessor, VoxtralForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Voxtral requires transformers>=4.54 and mistral-common[audio]>=1.8.1."
            ) from exc

        self.dtype = preferred_dtype()
        load_kwargs: dict[str, Any] = {
            "cache_dir": HF_CACHE_DIR,
            "dtype": self.dtype,
            "device_map": "auto",
        }
        if ATTN_IMPLEMENTATION:
            load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
        self.model = VoxtralForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def _query(self, conversation: list[dict[str, Any]], max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(conversation)
        device = first_model_device(self.model)
        inputs = move_inputs(inputs, device, self.dtype)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs(max_new_tokens))
        return self.processor.batch_decode(
            output_ids[:, input_length:], skip_special_tokens=True
        )[0].strip()

    def query_audio(self, audio_path: str, prompt: str, max_new_tokens: int) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._query(conversation, max_new_tokens)

    def close(self) -> None:
        del self.processor
        del self.model


class Gemma3nAdapter(AudioLanguageModel):
    def __init__(self, model_id: str):
        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 3n requires transformers>=4.53.1 with Gemma3nForConditionalGeneration."
            ) from exc

        self.dtype = preferred_dtype()
        load_kwargs: dict[str, Any] = {
            "cache_dir": HF_CACHE_DIR,
            "dtype": self.dtype,
            "device_map": "auto",
        }
        if ATTN_IMPLEMENTATION:
            load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def _query(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = first_model_device(self.model)
        inputs = move_inputs(inputs, device, self.dtype)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs(max_new_tokens))
        return self.processor.decode(
            output_ids[0, input_length:], skip_special_tokens=True
        ).strip()

    def query_audio(self, audio_path: str, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._query(messages, max_new_tokens)

    def close(self) -> None:
        del self.processor
        del self.model


def load_model(spec: ModelSpec) -> AudioLanguageModel:
    eprint(f"[model] Loading {spec.model_id} ...")
    if spec.backend == "qwen25_omni":
        return Qwen25OmniAdapter(spec.model_id)
    if spec.backend == "audio_flamingo3":
        return AudioFlamingo3Adapter(spec.model_id)
    if spec.backend == "voxtral":
        return VoxtralAdapter(spec.model_id)
    if spec.backend == "gemma3n":
        return Gemma3nAdapter(spec.model_id)
    raise ValueError(f"Unknown model backend: {spec.backend}")


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
