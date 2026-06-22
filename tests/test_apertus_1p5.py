import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def load_apertus_module():
    vlmeval = types.ModuleType("vlmeval")
    vlmeval.__path__ = ["vlmeval"]

    vlm = types.ModuleType("vlmeval.vlm")
    vlm.__path__ = ["vlmeval/vlm"]

    base = types.ModuleType("vlmeval.vlm.base")
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(device_count=lambda: 1)

    class BaseModel:
        def __init__(self):
            pass

    base.BaseModel = BaseModel
    modules = {
        "vlmeval": vlmeval,
        "vlmeval.vlm": vlm,
        "vlmeval.vlm.base": base,
        "torch": torch,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "vlmeval.vlm.apertus_1p5",
            Path("vlmeval/vlm/apertus_1p5.py"),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["vlmeval.vlm.apertus_1p5"] = module
        spec.loader.exec_module(module)
        sys.modules.pop("vlmeval.vlm.apertus_1p5", None)
        return module


class FakeTokenizer:

    def __init__(self):
        self.template_calls = []
        self.tokenize_calls = []

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        chat_template,
        enable_thinking=None,
        **kwargs,
    ):
        self.template_calls.append({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
            "chat_template": chat_template,
            "enable_thinking": enable_thinking,
            "kwargs": kwargs,
        })
        return "<bos> rendered prompt"

    def __call__(self, text, add_special_tokens, return_attention_mask):
        self.tokenize_calls.append({
            "text": text,
            "add_special_tokens": add_special_tokens,
            "return_attention_mask": return_attention_mask,
        })
        return {"input_ids": [11, 22, 33]}


class TestApertus1p5Tokenization(unittest.TestCase):

    def test_llm_construction_hides_torchrun_env_from_vllm_child(self):
        module = load_apertus_module()
        seen_env = {}
        seen_kwargs = {}

        class FakeAutoTokenizer:

            @staticmethod
            def from_pretrained(tokenizer_path, trust_remote_code):
                return FakeTokenizer()

        class FakeLLM:

            def __init__(self, **kwargs):
                seen_kwargs.update(kwargs)
                for key in (
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "TORCHELASTIC_RUN_ID",
                ):
                    seen_env[key] = os.environ.get(key)
                seen_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
                seen_env["VLLM_HOST_IP"] = os.environ.get("VLLM_HOST_IP")

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = lambda **kwargs: kwargs

        torchrun_env = {
            "RANK": "2",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "2",
            "LOCAL_WORLD_SIZE": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29541",
            "TORCHELASTIC_RUN_ID": "none",
            "CUDA_VISIBLE_DEVICES": "2",
            "VLLM_HOST_IP": "127.0.0.1",
        }
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
            with mock.patch.dict(os.environ, torchrun_env, clear=False):
                module.Apertus1p5(model_path="model", tokenizer_path="tokenizer", chat_template=None)

                for key in (
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "TORCHELASTIC_RUN_ID",
                ):
                    self.assertIsNone(seen_env[key])
                    self.assertEqual(os.environ[key], torchrun_env[key])

        self.assertEqual(seen_env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(seen_env["VLLM_HOST_IP"], "127.0.0.1")
        self.assertEqual(seen_kwargs["gpu_memory_utilization"], 0.6)
        self.assertTrue(seen_kwargs["skip_mm_profiling"])

    def test_llm_construction_honors_skip_mm_profiling_env_override(self):
        module = load_apertus_module()
        seen_kwargs = {}

        class FakeAutoTokenizer:

            @staticmethod
            def from_pretrained(tokenizer_path, trust_remote_code):
                return FakeTokenizer()

        class FakeLLM:

            def __init__(self, **kwargs):
                seen_kwargs.update(kwargs)

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = lambda **kwargs: kwargs

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
            with mock.patch.dict(os.environ, {"VLLM_APERTUS_SKIP_MM_PROFILING": "false"}, clear=False):
                module.Apertus1p5(model_path="model", tokenizer_path="tokenizer", chat_template=None)

        self.assertFalse(seen_kwargs["skip_mm_profiling"])

    def test_tokenize_messages_uses_single_prompt_and_no_added_special_tokens(self):
        module = load_apertus_module()
        model = object.__new__(module.Apertus1p5)
        model.tokenizer = FakeTokenizer()
        model.chat_template_str = "template"
        model.enable_thinking = False

        token_ids = model._tokenize_messages([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])

        self.assertEqual(token_ids, [11, 22, 33])
        self.assertEqual(model.tokenizer.template_calls[0]["tokenize"], False)
        self.assertEqual(model.tokenizer.tokenize_calls[0]["add_special_tokens"], False)
        self.assertIsInstance(model.tokenizer.tokenize_calls[0]["text"], str)

    def test_build_messages_uses_apertus_parts_mapping_for_chat_template(self):
        module = load_apertus_module()
        model = object.__new__(module.Apertus1p5)
        image = mock.Mock()
        image.convert.return_value = "rgb-image"

        with mock.patch.object(module.Image, "open", return_value=image):
            messages, images = model._build_messages([
                {"type": "text", "value": "question"},
                {"type": "image", "value": "/tmp/image.png"},
            ])

        self.assertEqual(messages, [{
            "role": "user",
            "content": {
                "parts": [
                    {"type": "text", "text": "question"},
                    {"type": "image"},
                ]
            },
        }])
        self.assertEqual(images, ["rgb-image"])
        image.convert.assert_called_once_with("RGB")

    def test_generate_inner_sends_one_prompt_token_id_request_to_vllm(self):
        module = load_apertus_module()
        model = object.__new__(module.Apertus1p5)
        model.sampling_params = SimpleNamespace()
        model.enable_thinking = False
        prompts_seen = []
        sampling_seen = []

        def fake_build_messages(message):
            return [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], []

        def fake_tokenize_messages(messages):
            return [11, 22, 33]

        class FakeLLM:

            def generate(self, prompts, sampling_params):
                prompts_seen.extend(prompts)
                sampling_seen.append(sampling_params)
                return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")])]

        model._build_messages = fake_build_messages
        model._tokenize_messages = fake_tokenize_messages
        model.llm = FakeLLM()

        self.assertEqual(model.generate_inner([{"type": "text", "value": "hello"}]), "answer")
        self.assertEqual(prompts_seen, [{"prompt_token_ids": [11, 22, 33]}])
        self.assertIs(sampling_seen[0], model.sampling_params)


if __name__ == "__main__":
    unittest.main()
