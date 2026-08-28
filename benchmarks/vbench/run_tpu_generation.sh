#!/usr/bin/env bash
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
# VBench TPU Video Generation Script
#
# This script:
#   1. Clones/syncs MaxDiffusion repository on the TPU VM.
#   2. Installs Python dependencies via setup.sh for TPU devices.
#   3. Executes WAN 2.2 27B inference on 110 VBench benchmark prompts.
#   4. Stores the generated video files directly in the specified GCS bucket.
#
# Usage (Directly on TPU VM):
#   bash benchmarks/vbench/run_tpu_generation.sh GCS_BUCKET=<my-bucket-name> [KEY=VALUE ...]
#
# Usage (Orchestrated remotely via SSH from local machine):
#   bash benchmarks/vbench/run_tpu_generation.sh --ssh GCS_BUCKET=<my-bucket-name> TPU_NAME=<tpu-vm-name> [KEY=VALUE ...]
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Default Configuration & Parameters
# ------------------------------------------------------------------------------
GCS_BUCKET="${GCS_BUCKET:-}"
RUN_NAME="${RUN_NAME:-wan-inference}"
SEED="${SEED:-12345}"
PROMPT_FILE="${PROMPT_FILE:-./benchmarks/vbench/prompts_3.txt}"
CONFIG_FILE="${CONFIG_FILE:-src/maxdiffusion/configs/base_wan_27b.yml}"
ATTENTION="${ATTENTION:-ulysses_custom}"
NUM_STEPS="${NUM_STEPS:-40}"
NUM_FRAMES="${NUM_FRAMES:-81}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-0.125}"
VAE_SPATIAL="${VAE_SPATIAL:-4}"
VAE_DECODE_CHUNK="${VAE_DECODE_CHUNK:-4}"
VAE_WEIGHTS_DTYPE="${VAE_WEIGHTS_DTYPE:-bfloat16}"
VAE_DTYPE="${VAE_DTYPE:-bfloat16}"
TEXT_ENCODER_DTYPE="${TEXT_ENCODER_DTYPE:-bfloat16}"
COMPILE_TEXT_ENCODER="${COMPILE_TEXT_ENCODER:-true}"
ICI_DATA_PARALLELISM="${ICI_DATA_PARALLELISM:-2}"
ICI_CONTEXT_PARALLELISM="${ICI_CONTEXT_PARALLELISM:-4}"
FPS="${FPS:-16}"
USE_KV_CACHE="${USE_KV_CACHE:-true}"
USE_BASE2_EXP="${USE_BASE2_EXP:-true}"
USE_EXPERIMENTAL_SCHEDULER="${USE_EXPERIMENTAL_SCHEDULER:-true}"
USE_BATCHED_TEXT_ENCODER="${USE_BATCHED_TEXT_ENCODER:-true}"
FLASH_BLOCK_SIZES='{"block_q" : 3328, "block_kv_compute" : 256, "block_kv" : 2816, "block_kv_compute_in" : 256, "block_q_dkv": 3328, "block_kv_dkv" : 2816, "block_kv_dkv_compute" : 256, "block_q_dq" : 3328, "block_kv_dq" : 2816, "heads_per_tile" : 1}'

# Remote TPU VM details (used if --ssh mode is requested)
TPU_NAME="${TPU_NAME:-jalwaniya-v6e-8}"
TPU_ZONE="${TPU_ZONE:-southamerica-west1-a}"
TPU_PROJECT="${TPU_PROJECT:-tpu-prod-env-one-vm}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_DIR="${REMOTE_DIR:-~/maxdiffusion}"
GIT_REPO="${GIT_REPO:-https://github.com/jitendra-jalwaniya/maxdiffusion.git}"
GIT_BRANCH="${GIT_BRANCH:-two_machines}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"

SSH_MODE=false

# ------------------------------------------------------------------------------
# Parse Command Line Arguments
# ------------------------------------------------------------------------------
for arg in "$@"; do
  case "${arg}" in
    --ssh)
      SSH_MODE=true
      ;;
    --help|-h)
      echo "Usage: $0 [--ssh] [GCS_BUCKET=bucket_name] [TPU_NAME=tpu_name] [RUN_NAME=run_name] ..."
      echo ""
      echo "Options:"
      echo "  --ssh              Execute commands remotely on TPU VM over SSH"
      echo "  GCS_BUCKET         (Required) GCS bucket for saving generated videos"
      echo "  RUN_NAME           Name of this benchmarking run (default: wan-inference)"
      echo "  TPU_NAME           TPU VM name (for SSH mode)"
      echo "  TPU_ZONE           TPU VM zone (for SSH mode)"
      echo "  TPU_PROJECT        GCP project of the TPU VM"
      echo "  PROMPT_FILE        Path to prompt file (default: ./benchmarks/vbench/prompts_110.txt)"
      exit 0
      ;;
    *=*)
      KEY="${arg%%=*}"
      VALUE="${arg#*=}"
      export "${KEY}"="${VALUE}"
      ;;
    *)
      # If single argument provided without key=, treat as GCS_BUCKET if not set
      if [[ -z "${GCS_BUCKET}" ]]; then
        GCS_BUCKET="${arg}"
      fi
      ;;
  esac
done

if [[ -z "${GCS_BUCKET}" ]]; then
  echo "=========================================================================="
  echo "ERROR: GCS_BUCKET is required!"
  echo "Example: bash benchmarks/vbench/run_tpu_generation.sh GCS_BUCKET=my-bucket"
  echo "=========================================================================="
  exit 1
fi

# Clean bucket name (strip leading gs:// and trailing /)
GCS_BUCKET="${GCS_BUCKET#gs://}"
GCS_BUCKET="${GCS_BUCKET%/}"

# ------------------------------------------------------------------------------
# Remote SSH Mode Execution
# ------------------------------------------------------------------------------
if [[ "${SSH_MODE}" == "true" ]]; then
  echo "=========================================================================="
  echo "Executing VBench generation on TPU VM remotely via SSH"
  echo "  TPU Name:    ${TPU_NAME}"
  echo "  TPU Zone:    ${TPU_ZONE}"
  echo "  TPU Project: ${TPU_PROJECT}"
  echo "  GCS Bucket:  gs://${GCS_BUCKET}"
  echo "=========================================================================="

  REMOTE_SCRIPT=$(cat <<EOF
set -euo pipefail

echo "==> [TPU VM] Ensuring MaxDiffusion repository is set up at ${REMOTE_DIR}..."
if [ ! -d "${REMOTE_DIR}" ]; then
  mkdir -p "\$(dirname "${REMOTE_DIR}")"
  git clone "${GIT_REPO}" "${REMOTE_DIR}"
fi

cd "${REMOTE_DIR}"
git checkout "${GIT_BRANCH}" || true
git pull origin "${GIT_BRANCH}" || true

echo "==> [TPU VM] Running dependency setup..."
if ! python3 -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null; then
  echo "Setting up Python 3.12 virtualenv..."
  python3 -m pip install --upgrade uv || true
  if [ ! -d "\$HOME/maxdiffusion_venv" ]; then
    uv venv "\$HOME/maxdiffusion_venv" --python 3.12 --seed
  fi
  source "\$HOME/maxdiffusion_venv/bin/activate"
fi

if [ -f "\$HOME/maxdiffusion_venv/bin/activate" ]; then
  source "\$HOME/maxdiffusion_venv/bin/activate"
fi

bash setup.sh MODE=stable DEVICE=tpu
python3 -m uv pip install -e .

echo "==> [TPU VM] Launching WAN 2.2 27B video generation..."
python3 src/maxdiffusion/generate_wan.py "${CONFIG_FILE}" \\
  run_name="${RUN_NAME}" \\
  seed="${SEED}" \\
  attention="${ATTENTION}" \\
  num_inference_steps="${NUM_STEPS}" \\
  num_frames="${NUM_FRAMES}" \\
  width="${WIDTH}" \\
  height="${HEIGHT}" \\
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \\
  vae_spatial="${VAE_SPATIAL}" \\
  vae_decode_chunk="${VAE_DECODE_CHUNK}" \\
  vae_weights_dtype="${VAE_WEIGHTS_DTYPE}" \\
  vae_dtype="${VAE_DTYPE}" \\
  text_encoder_dtype="${TEXT_ENCODER_DTYPE}" \\
  compile_text_encoder="${COMPILE_TEXT_ENCODER}" \\
  ici_data_parallelism="${ICI_DATA_PARALLELISM}" \\
  ici_context_parallelism="${ICI_CONTEXT_PARALLELISM}" \\
  fps="${FPS}" \\
  use_kv_cache="${USE_KV_CACHE}" \\
  use_base2_exp="${USE_BASE2_EXP}" \\
  use_experimental_scheduler="${USE_EXPERIMENTAL_SCHEDULER}" \\
  use_batched_text_encoder="${USE_BATCHED_TEXT_ENCODER}" \\
  flash_block_sizes='${FLASH_BLOCK_SIZES}' \\
  prompt_file="${PROMPT_FILE}" \\
  base_output_directory="gs://${GCS_BUCKET}"

echo "==> [TPU VM] Syncing any remaining generated files to GCS bucket..."
if ls wan_output_*.mp4 1> /dev/null 2>&1; then
  gcloud storage cp wan_output_*.mp4 "gs://${GCS_BUCKET}/${RUN_NAME}/videos/" || gsutil -m cp wan_output_*.mp4 "gs://${GCS_BUCKET}/${RUN_NAME}/videos/" || true
fi

# Upload benchmark metadata for GPU eval step
if [ -f "${PROMPT_FILE}" ]; then
  gcloud storage cp "${PROMPT_FILE}" "gs://${GCS_BUCKET}/${RUN_NAME}/" || true
fi
if [ -f "./benchmarks/vbench/VBench_full_info_sub3.json" ]; then
  gcloud storage cp "./benchmarks/vbench/VBench_full_info_sub3.json" "gs://${GCS_BUCKET}/${RUN_NAME}/" || true
fi

echo "==> [TPU VM] Video generation completed successfully!"
EOF
)

  # Execute via gcloud compute tpus tpu-vm ssh
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${TPU_ZONE}" \
    --project="${TPU_PROJECT}" \
    --command="${REMOTE_SCRIPT}"

  echo ""
  echo "=========================================================================="
  echo "TPU video generation complete!"
  echo "Generated videos saved to: gs://${GCS_BUCKET}/${RUN_NAME}/videos/"
  echo ""
  echo "Next Step: SSH into your GPU VM and run benchmarks/vbench/run_gpu_eval.sh:"
  echo "  bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=${GCS_BUCKET} RUN_NAME=${RUN_NAME}"
  echo "=========================================================================="
  exit 0
fi

# ------------------------------------------------------------------------------
# Local TPU VM Execution
# ------------------------------------------------------------------------------
echo "=========================================================================="
echo "Starting VBench video generation on TPU VM (Local Execution)"
echo "  Run Name:    ${RUN_NAME}"
echo "  GCS Bucket:  gs://${GCS_BUCKET}"
echo "  Prompt File: ${PROMPT_FILE}"
echo "  Config File: ${CONFIG_FILE}"
echo "=========================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 1. Environment Setup
echo "==> Step 1/3: Verifying Python environment and dependencies..."
if ! python3 -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null; then
  echo "Creating Python 3.12 virtualenv..."
  python3 -m pip install --upgrade uv || true
  if [ ! -d "$HOME/maxdiffusion_venv" ]; then
    uv venv "$HOME/maxdiffusion_venv" --python 3.12 --seed
  fi
  source "$HOME/maxdiffusion_venv/bin/activate"
fi

if [ -f "$HOME/maxdiffusion_venv/bin/activate" ]; then
  source "$HOME/maxdiffusion_venv/bin/activate"
fi

bash setup.sh MODE=stable DEVICE=tpu
python3 -m uv pip install -e .

# 2. Run Video Generation Command
echo "==> Step 2/3: Running WAN 2.2 27B inference on VBench prompts..."
python3 src/maxdiffusion/generate_wan.py "${CONFIG_FILE}" \
  run_name="${RUN_NAME}" \
  seed="${SEED}" \
  attention="${ATTENTION}" \
  num_inference_steps="${NUM_STEPS}" \
  num_frames="${NUM_FRAMES}" \
  width="${WIDTH}" \
  height="${HEIGHT}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  vae_spatial="${VAE_SPATIAL}" \
  vae_decode_chunk="${VAE_DECODE_CHUNK}" \
  vae_weights_dtype="${VAE_WEIGHTS_DTYPE}" \
  vae_dtype="${VAE_DTYPE}" \
  text_encoder_dtype="${TEXT_ENCODER_DTYPE}" \
  compile_text_encoder="${COMPILE_TEXT_ENCODER}" \
  ici_data_parallelism="${ICI_DATA_PARALLELISM}" \
  ici_context_parallelism="${ICI_CONTEXT_PARALLELISM}" \
  fps="${FPS}" \
  use_kv_cache="${USE_KV_CACHE}" \
  use_base2_exp="${USE_BASE2_EXP}" \
  use_experimental_scheduler="${USE_EXPERIMENTAL_SCHEDULER}" \
  use_batched_text_encoder="${USE_BATCHED_TEXT_ENCODER}" \
  flash_block_sizes="${FLASH_BLOCK_SIZES}" \
  prompt_file="${PROMPT_FILE}" \
  base_output_directory="gs://${GCS_BUCKET}"

# 3. Post-run verification & backup upload
echo "==> Step 3/3: Verifying and syncing video outputs to GCS..."
if ls wan_output_*.mp4 1> /dev/null 2>&1; then
  echo "Syncing any remaining local videos to GCS: gs://${GCS_BUCKET}/${RUN_NAME}/videos/"
  gcloud storage cp wan_output_*.mp4 "gs://${GCS_BUCKET}/${RUN_NAME}/videos/" || gsutil -m cp wan_output_*.mp4 "gs://${GCS_BUCKET}/${RUN_NAME}/videos/" || true
fi

# Upload benchmark metadata for GPU eval step
if [ -f "${PROMPT_FILE}" ]; then
  gcloud storage cp "${PROMPT_FILE}" "gs://${GCS_BUCKET}/${RUN_NAME}/" || true
fi
if [ -f "./benchmarks/vbench/VBench_full_info_sub3.json" ]; then
  gcloud storage cp "./benchmarks/vbench/VBench_full_info_sub3.json" "gs://${GCS_BUCKET}/${RUN_NAME}/" || true
fi

echo ""
echo "=========================================================================="
echo "TPU video generation complete!"
echo "Generated videos saved to: gs://${GCS_BUCKET}/${RUN_NAME}/videos/"
echo ""
echo "Next Step: SSH into your GPU VM and run benchmarks/vbench/run_gpu_eval.sh:"
echo "  bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=${GCS_BUCKET} RUN_NAME=${RUN_NAME}"
echo "=========================================================================="
