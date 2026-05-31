import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_response_cache_module():
    spec = importlib.util.spec_from_file_location(
        "vlmeval.smp.response_cache",
        Path("vlmeval/smp/response_cache.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_response_cache_class():
    return load_response_cache_module().ResponseCache


class FakeModel:

    def __init__(self, model_path="model-a"):
        self.calls = 0
        self.model_path = model_path
        self.tokenizer_path = "tokenizer-a"
        self.chat_template_hash = "template-a"
        self.generate_kwargs = {"temperature": 0.0, "max_new_tokens": 8}
        self.vllm_prompt_contract = "prompt_token_ids"
        self.tokenizer_batch_size = 1

    def generate(self, message, dataset=None):
        self.calls += 1
        return f"response-{self.calls}-{dataset}"


class TestResponseCache(unittest.TestCase):

    def test_disabled_cache_is_falsy_and_noop(self):
        ResponseCache = load_response_cache_class()
        cache = ResponseCache.create(cache_root="")

        self.assertFalse(cache)
        self.assertIsNone(cache.lookup("missing"))
        self.assertFalse(cache.store_response("key", "value", {}))
        cache.close()
        cache.finalize(use_distributed_barrier=False)

    def test_get_or_generate_uses_cached_response_on_second_call(self):
        ResponseCache = load_response_cache_class()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-a",
                rank=0,
                world_size=1,
                model_name="FakeModel",
            )
            model = FakeModel()
            message = [{"type": "text", "value": "hello"}]

            first = cache.get_or_generate(
                model=model,
                model_name="FakeModel",
                dataset_name="DatasetA",
                sample_index=7,
                message=message,
                generate_fn=lambda: model.generate(message=message, dataset="DatasetA"),
            )
            second = cache.get_or_generate(
                model=model,
                model_name="FakeModel",
                dataset_name="DatasetA",
                sample_index=7,
                message=message,
                generate_fn=lambda: model.generate(message=message, dataset="DatasetA"),
            )

            self.assertEqual(first, "response-1-DatasetA")
            self.assertEqual(second, "response-1-DatasetA")
            self.assertEqual(model.calls, 1)
            cache.close()

    def test_finalize_merges_per_rank_databases_into_root_cache(self):
        ResponseCache = load_response_cache_class()
        with tempfile.TemporaryDirectory() as tmpdir:
            rank1 = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-a",
                rank=1,
                world_size=2,
                model_name="FakeModel",
            )
            rank1.store_response("rank1-key", "rank1-value", {"rank": 1})
            rank1.close()

            rank0 = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-a",
                rank=0,
                world_size=2,
                model_name="FakeModel",
            )
            rank0.store_response("rank0-key", "rank0-value", {"rank": 0})
            rank0.finalize(use_distributed_barrier=False)

            reader = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-b",
                rank=0,
                world_size=1,
                model_name="FakeModel",
            )
            self.assertEqual(reader.lookup("rank0-key"), "rank0-value")
            self.assertEqual(reader.lookup("rank1-key"), "rank1-value")
            reader.close()

    def test_root_cache_is_created_only_when_rank_zero_merges(self):
        ResponseCache = load_response_cache_class()
        with tempfile.TemporaryDirectory() as tmpdir:
            root_db = Path(tmpdir) / "cache.db"

            rank1 = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-a",
                rank=1,
                world_size=2,
                model_name="FakeModel",
            )
            self.assertFalse(root_db.exists())
            self.assertIsNone(rank1.root_conn)
            rank1.store_response("rank1-key", "rank1-value", {"rank": 1})
            rank1.close()

            rank0 = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-a",
                rank=0,
                world_size=2,
                model_name="FakeModel",
            )
            self.assertFalse(root_db.exists())
            self.assertIsNone(rank0.root_conn)
            rank0.store_response("rank0-key", "rank0-value", {"rank": 0})
            rank0.finalize(use_distributed_barrier=False)
            self.assertTrue(root_db.exists())

            reader = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-b",
                rank=1,
                world_size=2,
                model_name="FakeModel",
            )
            self.assertEqual(reader.root_conn.execute("PRAGMA query_only").fetchone()[0], 1)
            self.assertEqual(reader.lookup("rank0-key"), "rank0-value")
            self.assertEqual(reader.lookup("rank1-key"), "rank1-value")
            self.assertTrue(reader.store_response("rank1-new-key", "rank1-new-value", {"rank": 1}))
            reader.close()

    def test_cache_key_includes_model_fingerprint(self):
        ResponseCache = load_response_cache_class()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-a",
                rank=0,
                world_size=1,
                model_name="FakeModel",
            )
            message = [{"type": "text", "value": "hello"}]

            key_a = cache.make_key(
                model=FakeModel(model_path="model-a"),
                model_name="FakeModel",
                dataset_name="DatasetA",
                sample_index=7,
                message=message,
            )
            key_b = cache.make_key(
                model=FakeModel(model_path="model-b"),
                model_name="FakeModel",
                dataset_name="DatasetA",
                sample_index=7,
                message=message,
            )

            self.assertNotEqual(key_a, key_b)
            cache.close()

    def test_non_deterministic_model_is_not_stored(self):
        ResponseCache = load_response_cache_class()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache.create(
                cache_root=tmpdir,
                run_id="run-a",
                rank=0,
                world_size=1,
                model_name="FakeModel",
            )
            model = FakeModel()
            model.generate_kwargs["temperature"] = 0.7
            message = [{"type": "text", "value": "hello"}]

            first = cache.get_or_generate(
                model=model,
                model_name="FakeModel",
                dataset_name="DatasetA",
                sample_index=7,
                message=message,
                generate_fn=lambda: model.generate(message=message, dataset="DatasetA"),
            )
            second = cache.get_or_generate(
                model=model,
                model_name="FakeModel",
                dataset_name="DatasetA",
                sample_index=7,
                message=message,
                generate_fn=lambda: model.generate(message=message, dataset="DatasetA"),
            )

            self.assertEqual(first, "response-1-DatasetA")
            self.assertEqual(second, "response-2-DatasetA")
            self.assertEqual(model.calls, 2)
            cache.close()


if __name__ == "__main__":
    unittest.main()
