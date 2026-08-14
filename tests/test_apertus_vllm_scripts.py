import os
import subprocess


def run_script(args, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=merged_env,
    )


def test_eval_dry_run_defaults_to_vlmeval_owned_image_token_cache(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    env = {
        "BENCHMARK_ROOT": str(benchmark_root),
    }
    expected_cache_base = benchmark_root / "VLMEval_Cache" / "image_token_cache"

    result = run_script(
        [
            "bash",
            "scripts/apertus-vllm/eval.sh",
            "--data",
            "3DSRBench",
            "--dry-run",
        ],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"--image-token-cache-base {expected_cache_base}" in result.stdout
    assert "--image-token-cache-mode fill" in result.stdout
    assert "--image-token-cache-readonly 0" in result.stdout
    assert "--image-token-cache-write-misses 1" in result.stdout
    assert "--image-token-cache-preload 0" in result.stdout


def test_eval_dry_run_honors_image_token_cache_base_override(tmp_path):
    image_token_cache_base = tmp_path / "image_token_cache"
    env = {
        "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
        "IMAGE_TOKEN_CACHE_BASE": str(image_token_cache_base),
    }
    result = run_script(
        [
            "bash",
            "scripts/apertus-vllm/eval.sh",
            "--data",
            "3DSRBench",
            "--dry-run",
        ],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"--image-token-cache-base {image_token_cache_base}" in result.stdout
    assert "--image-token-cache-mode fill" in result.stdout
    assert "--enable-image-token-cache true" in result.stdout
    assert "--image-token-cache-local-copy 0" in result.stdout
    assert "--image-token-cache-readonly 0" in result.stdout
    assert "--image-token-cache-write-misses 1" in result.stdout
    assert "--image-token-cache-preload 0" in result.stdout


def test_eval_dry_run_can_use_readonly_image_token_cache(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    env = {
        "BENCHMARK_ROOT": str(benchmark_root),
        "CACHE_BASE": str(tmp_path / "image_token_cache"),
    }
    expected_cache_base = benchmark_root / "VLMEval_Cache" / "image_token_cache"
    result = run_script(
        [
            "bash",
            "scripts/apertus-vllm/eval.sh",
            "--data",
            "3DSRBench",
            "--image-token-cache-mode",
            "readonly",
            "--dry-run",
        ],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"--image-token-cache-base {expected_cache_base}" in result.stdout
    assert "--image-token-cache-mode readonly" in result.stdout
    assert "--image-token-cache-readonly 1" in result.stdout
    assert "--image-token-cache-write-misses 0" in result.stdout
    assert "--image-token-cache-preload 1" in result.stdout


def test_eval_job_help_documents_image_token_cache_flags():
    result = run_script(["bash", "scripts/apertus-vllm/eval_job.slurm", "--help"])

    assert result.returncode == 0, result.stderr
    assert "--enable-image-token-cache <true|false>" in result.stdout
    assert "--image-token-cache-mode <fill|readonly>" in result.stdout
    assert "--image-token-cache-base <path>" in result.stdout
    assert "--image-token-cache-readonly <0|1|true|false>" in result.stdout


def test_container_build_uses_lmms_eval_vllm_branch_with_image_token_cache():
    dockerfile = open(
        "scripts/apertus-vllm/Dockerfile.vlmevalkit-prod-cu130",
        encoding="utf-8",
    ).read()

    assert "apertus-lmms-eval" in dockerfile
    assert "apertus_integration" not in dockerfile
