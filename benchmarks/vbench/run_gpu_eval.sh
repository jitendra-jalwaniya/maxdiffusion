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
# VBench GPU Evaluation Script
#
# This script:
#   1. Clones and sets up the VBench repository and dependencies on a GPU VM.
#   2. Downloads generated WAN 2.2 videos from the specified GCS bucket.
#   3. Aligns video files to VBench prompt specifications using VBench_full_info_sub110.json.
#   4. Evaluates the videos across VBench dimensions using evaluate.py.
#   5. Publishes evaluation metrics and JSON results back to the GCS bucket.
#
# Usage (Directly on GPU VM):
#   bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=<my-bucket-name> [KEY=VALUE ...]
#
# Usage (Orchestrated remotely via SSH from local machine):
#   bash benchmarks/vbench/run_gpu_eval.sh --ssh GCS_BUCKET=<my-bucket-name> GPU_NAME=<gpu-vm-name> [KEY=VALUE ...]
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Default Configuration & Parameters
# ------------------------------------------------------------------------------
GCS_BUCKET="${GCS_BUCKET:-}"
RUN_NAME="${RUN_NAME:-wan-inference}"
GCS_VIDEO_DIR="${GCS_VIDEO_DIR:-${RUN_NAME}/videos}"
GCS_RESULTS_DIR="${GCS_RESULTS_DIR:-${RUN_NAME}/vbench_results}"
WORK_DIR_OVERRIDE=""
VBENCH_REPO="${VBENCH_REPO:-https://github.com/Vchitect/VBench.git}"
VBENCH_BRANCH="${VBENCH_BRANCH:-master}"
MAXDIFFUSION_REPO="${MAXDIFFUSION_REPO:-https://github.com/google/maxdiffusion.git}"
BENCHMARK_JSON_PATH="${BENCHMARK_JSON_PATH:-}"

# Dimensions to evaluate (defaults to all 16 standard VBench dimensions)
DIMENSIONS=(
  "temporal_flickering"
  "motion_smoothness"
  "subject_consistency"
  "background_consistency"
  "overall_consistency"
  "aesthetic_quality"
  "imaging_quality"
  "color"
  "object_class"
  "multiple_objects"
  "human_action"
  "spatial_relationship"
  "scene"
  "temporal_style"
  "appearance_style"
  "dynamic_degree"
)

# Remote GPU VM details (used if --ssh mode is requested)
GPU_NAME="${GPU_NAME:-}"
GPU_ZONE="${GPU_ZONE:-us-central1-a}"
GPU_PROJECT="${GPU_PROJECT:-}"

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
      echo "Usage: $0 [--ssh] [GCS_BUCKET=bucket_name] [RUN_NAME=run_name] [DIMENSIONS=\"dim1 dim2\"] ..."
      echo ""
      echo "Options:"
      echo "  --ssh              Execute commands remotely on GPU VM over SSH"
      echo "  GCS_BUCKET         (Required) GCS bucket where videos are stored"
      echo "  RUN_NAME           Run name used during generation (default: wan-inference)"
      echo "  GCS_VIDEO_DIR      GCS prefix path to videos (default: \${RUN_NAME}/videos)"
      echo "  GCS_RESULTS_DIR    GCS prefix path to upload results (default: \${RUN_NAME}/vbench_results)"
      echo "  WORK_DIR           Working directory on GPU VM (default: \$HOME/vbench_evaluation)"
      echo "  GPU_NAME           GPU VM name (for SSH mode)"
      echo "  GPU_ZONE           GPU VM zone (for SSH mode)"
      echo "  GPU_PROJECT        GCP project of the GPU VM"
      echo "  DIMENSIONS         Space-separated list of dimensions to evaluate (default: all 16)"
      exit 0
      ;;
    WORK_DIR=*)
      WORK_DIR_OVERRIDE="${arg#*=}"
      ;;
    *=*)
      KEY="${arg%%=*}"
      VALUE="${arg#*=}"
      if [[ "${KEY}" == "DIMENSIONS" ]]; then
        # shellcheck disable=SC2206
        DIMENSIONS=(${VALUE})
      else
        export "${KEY}"="${VALUE}"
      fi
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
  echo "Example: bash benchmarks/vbench/run_gpu_eval.sh GCS_BUCKET=my-bucket"
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
  if [[ -z "${GPU_NAME}" ]]; then
    echo "ERROR: GPU_NAME must be specified when using --ssh mode!"
    exit 1
  fi

  echo "=========================================================================="
  echo "Executing VBench evaluation on GPU VM remotely via SSH"
  echo "  GPU Name:    ${GPU_NAME}"
  echo "  GPU Zone:    ${GPU_ZONE}"
  echo "  GCS Bucket:  gs://${GCS_BUCKET}"
  echo "  Video Dir:   gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
  echo "  Results Dir: gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}"
  echo "=========================================================================="

  DIMS_STR="${DIMENSIONS[*]}"
  REMOTE_SCRIPT=$(cat <<EOF
set -euo pipefail

WORK_DIR="${WORK_DIR_OVERRIDE:-\$HOME/vbench_evaluation}"
mkdir -p "\${WORK_DIR}"
cd "\${WORK_DIR}"

echo "==> [GPU VM] Ensuring build and video codec system packages..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
fi

echo "==> [GPU VM] Setting up Python virtual environment (Python 3.10-3.12)..."
export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || python3 -m pip install --user uv 2>/dev/null || true
  export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
fi

if [ -d "venv" ]; then
  if ! venv/bin/python3 -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13)' 2>/dev/null; then
    echo "Recreating venv with compatible Python version (3.10-3.12)..."
    rm -rf venv
  fi
fi

if [ ! -d "venv" ]; then
  if python3 -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13)' 2>/dev/null; then
    python3 -m venv venv
  elif command -v uv >/dev/null 2>&1; then
    uv venv venv --python 3.11 --seed || uv venv venv --python 3.12 --seed || uv venv venv --python 3.10 --seed || python3 -m venv venv
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv venv
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv venv
  elif command -v python3.10 >/dev/null 2>&1; then
    python3.10 -m venv venv
  else
    python3 -m venv venv
  fi
fi
source venv/bin/activate

pip install --upgrade pip setuptools wheel

echo "==> [GPU VM] Cloning VBench repository..."
if [ ! -d "VBench" ]; then
  git clone -b "${VBENCH_BRANCH}" "${VBENCH_REPO}" VBench
else
  (cd VBench && git pull origin "${VBENCH_BRANCH}" || true)
fi

echo "==> [GPU VM] Installing VBench & GPU dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision
pip install "numpy<2"
(cd VBench && pip install --no-build-isolation -e .)
pip install pandas tabulate google-cloud-storage tqdm opencv-python decord

echo "==> [GPU VM] Fetching VBench_full_info_sub110.json..."
if [ ! -f "VBench_full_info_sub110.json" ]; then
  # Try downloading from GCS first
  if ! gcloud storage cp "gs://${GCS_BUCKET}/${RUN_NAME}/VBench_full_info_sub110.json" ./ 2>/dev/null; then
    # Fallback to cloning maxdiffusion
    if [ ! -d "maxdiffusion" ]; then
      git clone --depth 1 "${MAXDIFFUSION_REPO}" maxdiffusion
    fi
    cp maxdiffusion/benchmarks/vbench/VBench_full_info_sub110.json ./
  fi
fi

# Download Videos
mkdir -p downloaded_videos
echo "==> [GPU VM] Downloading generated videos from GCS: gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/..."
gcloud storage cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" downloaded_videos/ || gsutil -m cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" downloaded_videos/

# Prepare Videos
mkdir -p vbench_videos
python3 - <<'PYEOF'
import json, os, glob, shutil

with open('VBench_full_info_sub110.json', 'r', encoding='utf-8') as f:
    bench_data = json.load(f)

downloaded = sorted(glob.glob('downloaded_videos/*.mp4'))
print(f"Downloaded {len(downloaded)} videos from GCS.")

matched = 0
for idx, item in enumerate(bench_data):
    prompt = item['prompt_en']
    target_name = f"{prompt}-0.mp4"
    target_path = os.path.join('vbench_videos', target_name)
    
    # Try finding matching video by index or filename
    src_video = None
    candidate_idx_name = f"wan_output_{idx}.mp4"
    candidate_glob = glob.glob(f"downloaded_videos/*_{idx}.mp4")
    
    if candidate_glob:
        src_video = candidate_glob[0]
    elif idx < len(downloaded):
        src_video = downloaded[idx]
    
    if src_video and os.path.exists(src_video):
        if os.path.exists(target_path):
            os.remove(target_path)
        try:
            os.symlink(os.path.abspath(src_video), target_path)
        except OSError:
            shutil.copy2(src_video, target_path)
        matched += 1

print(f"Successfully prepared and linked {matched}/{len(bench_data)} videos for VBench evaluation.")
PYEOF

echo "==> [GPU VM] Running VBench evaluation..."
cd VBench
mkdir -p "\${WORK_DIR}/evaluation_results"
python3 evaluate.py \
  --videos_path "\${WORK_DIR}/vbench_videos" \
  --full_json_dir "\${WORK_DIR}/VBench_full_info_sub110.json" \
  --output_path "\${WORK_DIR}/evaluation_results" \
  --dimension ${DIMS_STR} \
  --mode vbench_standard

echo "==> [GPU VM] Publishing evaluation results to GCS..."
gcloud storage cp -r "\${WORK_DIR}/evaluation_results/*" "gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/" || \
gsutil -m cp -r "\${WORK_DIR}/evaluation_results/*" "gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/"

echo "==> [GPU VM] VBench evaluation complete! Results published to gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/"
EOF
)

  SSH_CMD=("gcloud" "compute" "ssh" "${GPU_NAME}" "--zone=${GPU_ZONE}")
  if [[ -n "${GPU_PROJECT}" ]]; then
    SSH_CMD+=("--project=${GPU_PROJECT}")
  fi
  SSH_CMD+=("--command=${REMOTE_SCRIPT}")

  "${SSH_CMD[@]}"
  exit 0
fi

# ------------------------------------------------------------------------------
# Local GPU VM Execution
# ------------------------------------------------------------------------------
WORK_DIR="${WORK_DIR_OVERRIDE:-${WORK_DIR:-$HOME/vbench_evaluation}}"
echo "=========================================================================="
echo "Starting VBench Evaluation on GPU VM (Local Execution)"
echo "  GCS Bucket:      gs://${GCS_BUCKET}"
echo "  Video Source:    gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}"
echo "  Results Target:  gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}"
echo "  Working Dir:     ${WORK_DIR}"
echo "  Dimensions:      ${DIMENSIONS[*]}"
echo "=========================================================================="

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# 1. Environment & VBench Setup
echo "==> Step 1/5: Setting up Python environment and VBench..."
echo "Ensuring build and video codec system packages..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
fi

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || python3 -m pip install --user uv 2>/dev/null || true
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

if [ -d "venv" ]; then
  if ! venv/bin/python3 -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13)' 2>/dev/null; then
    echo "Recreating venv with compatible Python version (3.10-3.12)..."
    rm -rf venv
  fi
fi

if [ ! -d "venv" ]; then
  if python3 -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13)' 2>/dev/null; then
    python3 -m venv venv
  elif command -v uv >/dev/null 2>&1; then
    uv venv venv --python 3.11 --seed || uv venv venv --python 3.12 --seed || uv venv venv --python 3.10 --seed || python3 -m venv venv
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv venv
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv venv
  elif command -v python3.10 >/dev/null 2>&1; then
    python3.10 -m venv venv
  else
    python3 -m venv venv
  fi
fi
source venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel

if [ ! -d "VBench" ]; then
  echo "Cloning VBench repository from ${VBENCH_REPO}..."
  git clone -b "${VBENCH_BRANCH}" "${VBENCH_REPO}" VBench
else
  echo "Updating VBench repository..."
  (cd VBench && git pull origin "${VBENCH_BRANCH}" || true)
fi

echo "Installing VBench dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision
pip install "numpy<2"
(cd VBench && pip install --no-build-isolation -e .)
pip install pandas tabulate google-cloud-storage tqdm opencv-python decord

# 2. Benchmark JSON Metadata Setup
echo "==> Step 2/5: Locating VBench_full_info_sub110.json..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JSON_DEST="${WORK_DIR}/VBench_full_info_sub110.json"
if [[ -n "${BENCHMARK_JSON_PATH}" && -f "${BENCHMARK_JSON_PATH}" ]]; then
  cp "${BENCHMARK_JSON_PATH}" "${JSON_DEST}"
elif [ -f "${REPO_ROOT}/benchmarks/vbench/VBench_full_info_sub110.json" ]; then
  cp "${REPO_ROOT}/benchmarks/vbench/VBench_full_info_sub110.json" "${JSON_DEST}"
elif [ -f "${WORK_DIR}/VBench_full_info_sub110.json" ]; then
  echo "Using existing ${JSON_DEST}"
else
  echo "Attempting to download VBench_full_info_sub110.json from GCS..."
  if ! gcloud storage cp "gs://${GCS_BUCKET}/${RUN_NAME}/VBench_full_info_sub110.json" "${JSON_DEST}" 2>/dev/null; then
    echo "Cloning MaxDiffusion to fetch VBench_full_info_sub110.json..."
    if [ ! -d "maxdiffusion" ]; then
      git clone --depth 1 "${MAXDIFFUSION_REPO}" maxdiffusion
    fi
    cp maxdiffusion/benchmarks/vbench/VBench_full_info_sub110.json "${JSON_DEST}"
  fi
fi

# 3. Download Generated Videos from GCS
echo "==> Step 3/5: Downloading generated videos from GCS: gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/..."
mkdir -p "${WORK_DIR}/downloaded_videos"
gcloud storage cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" "${WORK_DIR}/downloaded_videos/" || \
gsutil -m cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" "${WORK_DIR}/downloaded_videos/"

# 4. Align Videos to VBench Expected Naming Format
echo "==> Step 4/5: Aligning video filenames for VBench evaluation..."
mkdir -p "${WORK_DIR}/vbench_videos"
python3 - <<PYEOF
import json, os, glob, shutil

json_file = '${JSON_DEST}'
download_dir = '${WORK_DIR}/downloaded_videos'
vbench_dir = '${WORK_DIR}/vbench_videos'

with open(json_file, 'r', encoding='utf-8') as f:
    bench_data = json.load(f)

downloaded = sorted(glob.glob(os.path.join(download_dir, '*.mp4')))
print(f"Downloaded {len(downloaded)} videos from GCS.")

matched = 0
for idx, item in enumerate(bench_data):
    prompt = item['prompt_en']
    target_name = f"{prompt}-0.mp4"
    target_path = os.path.join(vbench_dir, target_name)
    
    # Locate candidate video by index suffix or position
    candidate_glob = glob.glob(os.path.join(download_dir, f"*_{idx}.mp4"))
    src_video = None
    if candidate_glob:
        src_video = candidate_glob[0]
    elif idx < len(downloaded):
        src_video = downloaded[idx]
        
    if src_video and os.path.exists(src_video):
        if os.path.exists(target_path):
            os.remove(target_path)
        try:
            os.symlink(os.path.abspath(src_video), target_path)
        except OSError:
            shutil.copy2(src_video, target_path)
        matched += 1

print(f"Successfully matched and mapped {matched}/{len(bench_data)} videos to standard VBench prompt names.")
PYEOF

# 5. Run VBench Evaluation & Publish Results
echo "==> Step 5/5: Executing VBench evaluate.py..."
mkdir -p "${WORK_DIR}/evaluation_results"
cd "${WORK_DIR}/VBench"

python3 evaluate.py \
  --videos_path "${WORK_DIR}/vbench_videos" \
  --full_json_dir "${JSON_DEST}" \
  --output_path "${WORK_DIR}/evaluation_results" \
  --dimension "${DIMENSIONS[@]}" \
  --mode vbench_standard

echo ""
echo "==> Publishing results to GCS bucket: gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/..."
gcloud storage cp -r "${WORK_DIR}/evaluation_results/*" "gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/" || \
gsutil -m cp -r "${WORK_DIR}/evaluation_results/*" "gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/"

echo ""
echo "=========================================================================="
echo "VBench Evaluation Finished Successfully!"
echo "Results published to: gs://${GCS_BUCKET}/${GCS_RESULTS_DIR}/"
echo "Local results stored in: ${WORK_DIR}/evaluation_results/"
echo "=========================================================================="
