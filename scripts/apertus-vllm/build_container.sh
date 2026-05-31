#!/usr/bin/env bash
# Submit a CSCS Podman -> Enroot build for the Apertus VLMEvalKit image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${SCRIPT_DIR}/build_container.slurm}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark}"
LOG_DIR="${LOG_DIR:-${BENCHMARK_ROOT}/VLMEval_Cache/logs/container-build}"
IMAGE_NAME="${IMAGE_NAME:-apertus-vlmevalkit-vllm-0.19-cu130-aarch64}"
OUTPUT_SQSH="${OUTPUT_SQSH:-/capstor/store/cscs/swissai/infra01/container-images/${IMAGE_NAME}.sqsh}"
DOCKERFILE="${DOCKERFILE:-scripts/apertus-vllm/Dockerfile.vlmevalkit-prod-cu130}"
EMU35_SOURCE="${EMU35_SOURCE:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/Tokenizer/submodules/Emu3.5}"
AUDIO_TOKENIZER_SOURCE="${AUDIO_TOKENIZER_SOURCE:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}"
WAVTOKENIZER_SOURCE="${WAVTOKENIZER_SOURCE:-/iopsstor/scratch/cscs/xyixuan/dev/WavTokenizer}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-infra01}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_RESERVATION="${SBATCH_RESERVATION:-SD-69241-apertus-1-5-0}"
DRY_RUN=0

usage() {
  sed -n '2,3p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --image-name <name>          Local Podman image name and output basename.
  --output-sqsh <path>         Final SquashFS path.
  --dockerfile <path>          Dockerfile path relative to repo root.
  --emu35-source <path>        Patched Emu3.5 working tree. Default: benchmark-image-tokenzier submodule.
  --audio-tokenizer-source <path>
                               benchmark-audio-tokenizer checkout used for the WavTokenizer wrapper.
  --wavtokenizer-source <path> Patched WavTokenizer working tree copied under the wrapper layout.
  --vllm-version <version>     vLLM wheel release. Default: 0.19.0.
  --time <hh:mm:ss>            Slurm time limit. Default: 04:00:00.
  --account <name>             Slurm account. Default: infra01.
  --partition <name>           Optional Slurm partition.
  --reservation <name>         Optional Slurm reservation.
  --log-dir <path>             Slurm stdout/stderr directory.
  --dry-run                    Print sbatch command without submitting.
  -h, --help                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-name) IMAGE_NAME="$2"; shift 2 ;;
    --output-sqsh) OUTPUT_SQSH="$2"; shift 2 ;;
    --dockerfile) DOCKERFILE="$2"; shift 2 ;;
    --emu35-source) EMU35_SOURCE="$2"; shift 2 ;;
    --audio-tokenizer-source) AUDIO_TOKENIZER_SOURCE="$2"; shift 2 ;;
    --wavtokenizer-source) WAVTOKENIZER_SOURCE="$2"; shift 2 ;;
    --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --account) SBATCH_ACCOUNT="$2"; shift 2 ;;
    --partition) SBATCH_PARTITION="$2"; shift 2 ;;
    --reservation) SBATCH_RESERVATION="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

[[ -f "${SLURM_TEMPLATE}" ]] || { echo "slurm template not found: ${SLURM_TEMPLATE}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"

# sbatch from inside an existing Pyxis container can inherit SPANK variables that
# conflict with new jobs. Match the eval submission wrapper.
while IFS='=' read -r key _; do
  [[ "${key}" == SLURM_SPANK* ]] && unset "${key}"
done < <(env)
export LD_LIBRARY_PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

CMD=(
  sbatch
  --account "${SBATCH_ACCOUNT}"
  --job-name "build-${IMAGE_NAME}"
  --output "${LOG_DIR}/${IMAGE_NAME}_%j.out"
  --error "${LOG_DIR}/${IMAGE_NAME}_%j.err"
  --time "${SBATCH_TIME}"
  --export "ALL,LD_LIBRARY_PATH=,LD_PRELOAD=,SSL_CERT_FILE=,SSL_CERT_DIR=,REQUESTS_CA_BUNDLE=,CURL_CA_BUNDLE="
)

if [[ -n "${SBATCH_PARTITION}" ]]; then
  CMD+=(--partition "${SBATCH_PARTITION}")
fi

if [[ -n "${SBATCH_RESERVATION}" ]]; then
  CMD+=(--reservation "${SBATCH_RESERVATION}")
fi

CMD+=(
  "${SLURM_TEMPLATE}"
  --repo-dir "${REPO_DIR}"
  --dockerfile "${DOCKERFILE}"
  --image-name "${IMAGE_NAME}"
  --output-sqsh "${OUTPUT_SQSH}"
  --emu35-source "${EMU35_SOURCE}"
  --audio-tokenizer-source "${AUDIO_TOKENIZER_SOURCE}"
  --wavtokenizer-source "${WAVTOKENIZER_SOURCE}"
  --vllm-version "${VLLM_VERSION}"
)

echo "========================================"
echo "Apertus VLMEvalKit container build"
echo "  repo:        ${REPO_DIR}"
echo "  dockerfile:  ${DOCKERFILE}"
echo "  emu3.5:      ${EMU35_SOURCE}"
echo "  audio wrap:  ${AUDIO_TOKENIZER_SOURCE}"
echo "  wavtoken:    ${WAVTOKENIZER_SOURCE}"
echo "  vLLM:        ${VLLM_VERSION}"
echo "  output:      ${OUTPUT_SQSH}"
echo "  logs:        ${LOG_DIR}"
echo "========================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf ' %q' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}"
fi
