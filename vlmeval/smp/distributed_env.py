"""Distributed-mode env helpers shared by Apertus wrappers and submitters."""
import os
from contextlib import contextmanager

# Variables torch.distributed.run / torchelastic inject into worker processes.
# When a wrapper spins up vLLM (which forks its own distributed group),
# leaving these in the environment makes vLLM rejoin the existing torchelastic
# rendezvous and crash. without_torchrun_env() drops them for the duration
# of the inner block and restores them after.
_TORCHRUN_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
)


@contextmanager
def without_torchrun_env():
    saved = {key: os.environ.get(key) for key in _TORCHRUN_ENV_VARS}
    for key in _TORCHRUN_ENV_VARS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_cuda_visible_devices(cuda_visible_devices):
    if not cuda_visible_devices:
        return []
    return [device.strip() for device in cuda_visible_devices.split(",") if device.strip()]


def split_cuda_visible_devices(cuda_visible_devices, local_world_size, local_rank):
    """Return per-rank CUDA_VISIBLE_DEVICES, or None when no remap is needed.

    torchrun workers first see all visible GPUs and should be split across them.
    Child processes spawned after that split may only see one GPU while still
    inheriting LOCAL_WORLD_SIZE from torchrun; those are already isolated.
    """
    devices = _parse_cuda_visible_devices(cuda_visible_devices)
    if local_world_size <= 1 or not devices:
        return None
    if len(devices) < local_world_size:
        return None

    gpu_per_proc = len(devices) // local_world_size
    if gpu_per_proc < 1:
        return None

    start = gpu_per_proc * local_rank
    selected = devices[start: start + gpu_per_proc]
    if not selected:
        return None
    return ",".join(selected)
