#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-vlmevalkit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/VLMEval_Cache/logs/container-build/%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/VLMEval_Cache/logs/container-build/%j.err

set -euo pipefail

REPO_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/VLMEvalKit"
DOCKERFILE="scripts/apertus-vllm/Dockerfile.vlmevalkit-prod-cu130"
IMAGE_NAME="apertus-vlmevalkit-vllm-0.19-cu130-aarch64"
OUTPUT_SQSH="/capstor/store/cscs/swissai/infra01/container-images/${IMAGE_NAME}.sqsh"
EMU35_SOURCE="/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/Tokenizer/submodules/Emu3.5"
AUDIO_TOKENIZER_SOURCE="/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer"
WAVTOKENIZER_SOURCE="/iopsstor/scratch/cscs/xyixuan/dev/WavTokenizer"
VLLM_VERSION="0.19.0"

usage() {
  cat <<'EOF'
Usage:
  sbatch build_container.slurm -- [options]

Options:
  --repo-dir <path>        VLMEvalKit checkout used as build context.
  --dockerfile <path>      Dockerfile path, relative to repo-dir unless absolute.
  --image-name <name>      Local Podman image tag prefix.
  --output-sqsh <path>     Final SquashFS image path.
  --emu35-source <path>    Patched Emu3.5 working tree to copy into the build context.
  --audio-tokenizer-source <path>
                           benchmark-audio-tokenizer checkout used for the WavTokenizer wrapper.
  --wavtokenizer-source <path>
                           Patched WavTokenizer working tree copied under the wrapper layout.
  --vllm-version <version> vLLM release used for precompiled wheel. Default: 0.19.0.
  --help                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --dockerfile) DOCKERFILE="$2"; shift 2 ;;
    --image-name) IMAGE_NAME="$2"; shift 2 ;;
    --output-sqsh) OUTPUT_SQSH="$2"; shift 2 ;;
    --emu35-source) EMU35_SOURCE="$2"; shift 2 ;;
    --audio-tokenizer-source) AUDIO_TOKENIZER_SOURCE="$2"; shift 2 ;;
    --wavtokenizer-source) WAVTOKENIZER_SOURCE="$2"; shift 2 ;;
    --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Repo not found: ${REPO_DIR}" >&2
  exit 1
fi

if [[ "${DOCKERFILE}" != /* ]]; then
  DOCKERFILE="${REPO_DIR}/${DOCKERFILE}"
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

if [[ ! -f "${EMU35_SOURCE}/src/vision_tokenizer/__init__.py" ]]; then
  echo "Patched Emu3.5 source not found or incomplete: ${EMU35_SOURCE}" >&2
  exit 1
fi

if [[ ! -f "${AUDIO_TOKENIZER_SOURCE}/src/audio_tokenizers/implementations/wavtokenizer.py" ]]; then
  echo "benchmark-audio-tokenizer wrapper source not found or incomplete: ${AUDIO_TOKENIZER_SOURCE}" >&2
  exit 1
fi

if [[ ! -f "${WAVTOKENIZER_SOURCE}/encoder/utils.py" || ! -f "${WAVTOKENIZER_SOURCE}/decoder/pretrained.py" ]]; then
  echo "WavTokenizer source not found or incomplete: ${WAVTOKENIZER_SOURCE}" >&2
  exit 1
fi

command -v podman >/dev/null || { echo "podman not found" >&2; exit 1; }
command -v enroot >/dev/null || { echo "enroot not found" >&2; exit 1; }
command -v tar >/dev/null || { echo "tar not found" >&2; exit 1; }

# The submit shell often runs inside a Pyxis container. Do not let its preload,
# library, or certificate paths affect host Podman/Enroot on the build node.
export LD_LIBRARY_PATH="${BUILD_JOB_LD_LIBRARY_PATH:-}"
unset LD_PRELOAD REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_DIR
for ca_bundle in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do
  if [[ -r "${ca_bundle}" ]]; then
    export SSL_CERT_FILE="${ca_bundle}"
    break
  fi
done

# Batch nodes have no D-Bus session and /run/user/<uid> may not exist.
export DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/podman-runtime-$$"
mkdir -p "${XDG_RUNTIME_DIR}"

CONFIG_ROOT="${XDG_CONFIG_HOME:-${HOME}/.config}"
mkdir -p "${CONFIG_ROOT}/containers"
if [[ ! -s "${CONFIG_ROOT}/containers/storage.conf" ]]; then
  cat > "${CONFIG_ROOT}/containers/storage.conf" <<EOF
[storage]
driver = "overlay"
runroot = "/dev/shm/${USER}/runroot"
graphroot = "/dev/shm/${USER}/root"
EOF
fi

SCRATCH_DIR="${SCRATCH:-${TMPDIR:-/tmp}}"
SCRATCH_SQSH="${SCRATCH_DIR}/${IMAGE_NAME}.sqsh"
BUILD_CONTEXT="${SCRATCH_DIR}/${IMAGE_NAME}-context-${SLURM_JOB_ID:-manual}"
IMAGE_TAG="${IMAGE_NAME}:${SLURM_JOB_ID:-manual}"

cleanup() {
  podman rmi "${IMAGE_TAG}" 2>/dev/null || true
  rm -f "${SCRATCH_SQSH}" 2>/dev/null || true
  rm -rf "${BUILD_CONTEXT}" 2>/dev/null || true
  rm -rf "${XDG_RUNTIME_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Building ${IMAGE_TAG} on $(hostname) at $(date) ==="
echo "REPO_DIR=${REPO_DIR}"
echo "DOCKERFILE=${DOCKERFILE}"
echo "OUTPUT_SQSH=${OUTPUT_SQSH}"
echo "EMU35_SOURCE=${EMU35_SOURCE}"
echo "AUDIO_TOKENIZER_SOURCE=${AUDIO_TOKENIZER_SOURCE}"
echo "WAVTOKENIZER_SOURCE=${WAVTOKENIZER_SOURCE}"
echo "VLLM_VERSION=${VLLM_VERSION}"
podman info | sed -n '/store:/,/registries:/p'

echo "=== Staging build context ==="
rm -rf "${BUILD_CONTEXT}"
mkdir -p \
  "${BUILD_CONTEXT}/external/Emu3.5" \
  "${BUILD_CONTEXT}/external/benchmark-audio-tokenizer/src/audio_tokenizers" \
  "${BUILD_CONTEXT}/external/benchmark-audio-tokenizer/src/repos/wavtokenizer"
tar \
  --exclude='.git' \
  --exclude='.agents' \
  --exclude='.codex' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.uv_cache' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.egg-info' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='core_*' \
  --exclude='logs' \
  --exclude='outputs' \
  --exclude='results' \
  --exclude='runtime' \
  --exclude='VLMEval_Cache' \
  --exclude='VLMEval_Data' \
  --exclude='VLMEval_Outputs' \
  -C "${REPO_DIR}" -cf - . | tar -C "${BUILD_CONTEXT}" -xf -
tar \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  -C "${EMU35_SOURCE}" -cf - . | tar -C "${BUILD_CONTEXT}/external/Emu3.5" -xf -
tar \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  -C "${AUDIO_TOKENIZER_SOURCE}/src/audio_tokenizers" -cf - . | tar -C "${BUILD_CONTEXT}/external/benchmark-audio-tokenizer/src/audio_tokenizers" -xf -
tar \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='tests' \
  -C "${WAVTOKENIZER_SOURCE}" -cf - . | tar -C "${BUILD_CONTEXT}/external/benchmark-audio-tokenizer/src/repos/wavtokenizer" -xf -

podman build \
  --format docker \
  --build-arg VLLM_VERSION="${VLLM_VERSION}" \
  --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
  -t "${IMAGE_TAG}" \
  -f "${BUILD_CONTEXT}/${DOCKERFILE#"${REPO_DIR}/"}" \
  "${BUILD_CONTEXT}"

echo "=== Converting ${IMAGE_TAG} to SquashFS ==="
rm -f "${SCRATCH_SQSH}"
enroot import -x mount -o "${SCRATCH_SQSH}" "podman://${IMAGE_TAG}" || true
if [[ ! -s "${SCRATCH_SQSH}" ]]; then
  echo "ERROR: enroot import produced no output at ${SCRATCH_SQSH}" >&2
  exit 1
fi

echo "=== Saving to ${OUTPUT_SQSH} ==="
mkdir -p "$(dirname "${OUTPUT_SQSH}")"
cp "${SCRATCH_SQSH}" "${OUTPUT_SQSH}.tmp"
mv "${OUTPUT_SQSH}.tmp" "${OUTPUT_SQSH}"
chmod o+rx "${OUTPUT_SQSH}"

echo "=== Done: ${OUTPUT_SQSH} at $(date) ==="
