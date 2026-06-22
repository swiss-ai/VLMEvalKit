"""Apertus 1.5 VLM wrapper for VLMEvalKit.

It renders the chat template outside vLLM and passes prompt_token_ids directly.
That keeps tokenization batch size at 1 and makes add_special_tokens=False
explicit, avoiding a double-BOS path in vLLM chat tokenization.
"""
import hashlib
import logging
import os
import re

from PIL import Image

from vlmeval.smp.distributed_env import without_torchrun_env
from vlmeval.vlm.base import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-8B-sft-capfilter-lr6e-5-constant-innovator-fix-it23409"
DEFAULT_TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
DEFAULT_CHAT_TEMPLATE = os.path.join(DEFAULT_TOKENIZER_PATH, "chat_template.jinja")

_THINKING_SUFFIXES = ("</think>", "<|inner_suffix|>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>|</?think>")


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class Apertus1p5(BaseModel):
    """Apertus 1.5 8B evaluated via vLLM with correct tokenization."""

    INSTALL_REQ = False
    INTERLEAVE = True
    allowed_types = ["text", "image"]

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        chat_template=DEFAULT_CHAT_TEMPLATE,
        max_new_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        tp_size=1,
        gpu_memory_utilization=0.6,
        max_model_len=131072,
        enable_thinking=False,
        skip_mm_profiling=None,
        **kwargs,
    ):
        super().__init__()
        # Lazy-imported here (not at module top) so importing this wrapper for
        # introspection — config.py registration, tests — does not require
        # transformers + vllm to be installed.
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.enable_thinking = enable_thinking
        if skip_mm_profiling is None:
            skip_mm_profiling = _env_bool("VLLM_APERTUS_SKIP_MM_PROFILING", True)
        self.skip_mm_profiling = skip_mm_profiling
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=False
        )

        # Load chat template from file if provided
        self.chat_template_str = None
        self.chat_template_hash = None
        if chat_template and os.path.isfile(chat_template):
            with open(chat_template, encoding="utf-8") as f:
                self.chat_template_str = f.read()
            self.chat_template_hash = hashlib.sha256(
                self.chat_template_str.encode("utf-8")
            ).hexdigest()

        logger.info(
            f"Loading Apertus 1.5 via vLLM: {model_path} "
            f"(TP={tp_size}, thinking={enable_thinking})"
        )
        with without_torchrun_env():
            self.llm = LLM(
                model=model_path,
                tokenizer=tokenizer_path,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=False,
                max_model_len=max_model_len,
                hf_overrides={"max_position_embeddings": max_model_len},
                limit_mm_per_prompt={"image": 32},
                skip_mm_profiling=self.skip_mm_profiling,
            )

        # All fields are fixed at construction; build SamplingParams once
        # rather than per-row in generate_inner.
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            seed=0,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=not enable_thinking,
        )
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_model_len = max_model_len
        # Greedy decoding (temperature == 0) is deterministic and therefore
        # cacheable; sampling (e.g. the thinking recipe) is not. response_cache.py
        # reads response_cache_deterministic directly and uses generate_kwargs in
        # the cache fingerprint, alongside chat_template_hash above.
        self.generate_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "enable_thinking": enable_thinking,
        }
        self.response_cache_deterministic = temperature <= 0.0
        self.vllm_prompt_contract = "prompt_token_ids"
        self.tokenizer_batch_size = 1
        self.tokenizer_add_special_tokens = False

    def preproc_content(self, inputs):
        from vlmeval.smp.file import parse_file

        if self.check_content(inputs) != "listdict":
            return super().preproc_content(inputs)

        for item in inputs:
            assert "type" in item and "value" in item
            mime, path = parse_file(item["value"])
            if mime is None:
                assert item["type"] == "text"
            elif mime == "unknown" and item["type"] == "image" and os.path.isfile(path):
                item["value"] = path
            else:
                assert mime.split("/")[0] == item["type"]
                item["value"] = path
        return inputs

    def _build_messages(self, message):
        """Convert VLMEvalKit message format to the Apertus chat-template shape."""
        content_parts = []
        images = []
        for item in message:
            if item["type"] == "text":
                content_parts.append({"type": "text", "text": item["value"]})
            elif item["type"] == "image":
                img = Image.open(item["value"]).convert("RGB")
                images.append(img)
                content_parts.append({"type": "image"})
        return [{"role": "user", "content": {"parts": content_parts}}], images

    def _tokenize_messages(self, msgs):
        prompt = self.tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
            chat_template=self.chat_template_str,
            enable_thinking=self.enable_thinking,
        )
        tokenized = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return tokenized["input_ids"]

    @staticmethod
    def _strip_thinking(text):
        for suffix in _THINKING_SUFFIXES:
            if suffix in text:
                text = text.rsplit(suffix, 1)[1]
                break
        return _SPECIAL_TOKEN_RE.sub("", text).strip()

    def generate_inner(self, message, dataset=None):
        msgs, images = self._build_messages(message)
        prompt_data = {"prompt_token_ids": self._tokenize_messages(msgs)}
        if images:
            prompt_data["multi_modal_data"] = {"image": images}

        outputs = self.llm.generate(
            prompts=[prompt_data],
            sampling_params=self.sampling_params,
        )
        generated_text = outputs[0].outputs[0].text

        if self.enable_thinking:
            generated_text = self._strip_thinking(generated_text)

        return generated_text.strip()
