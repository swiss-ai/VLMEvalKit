import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "vlmeval" / "smp" / "distributed_env.py"
spec = importlib.util.spec_from_file_location("distributed_env", MODULE_PATH)
distributed_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distributed_env)
split_cuda_visible_devices = distributed_env.split_cuda_visible_devices


def test_split_cuda_visible_devices_assigns_one_gpu_per_rank():
    assert split_cuda_visible_devices("0,1,2,3", local_world_size=4, local_rank=2) == "2"


def test_split_cuda_visible_devices_leaves_already_isolated_spawn_child_unchanged():
    assert split_cuda_visible_devices("2", local_world_size=4, local_rank=2) is None


def test_split_cuda_visible_devices_ignores_single_process():
    assert split_cuda_visible_devices("0,1,2,3", local_world_size=1, local_rank=0) is None
