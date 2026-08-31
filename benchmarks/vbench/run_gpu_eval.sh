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
RUN_NAME="${RUN_NAME:-wan-inference-aug-31-1}"
GCS_VIDEO_DIR="${GCS_VIDEO_DIR:-${RUN_NAME}/videos}"
GCS_RESULTS_DIR="${GCS_RESULTS_DIR:-${RUN_NAME}/vbench_results}"
WORK_DIR_OVERRIDE=""
VBENCH_REPO="${VBENCH_REPO:-https://github.com/Vchitect/VBench.git}"
VBENCH_BRANCH="${VBENCH_BRANCH:-master}"
MAXDIFFUSION_REPO="${MAXDIFFUSION_REPO:-https://github.com/jitendra-jalwaniya/maxdiffusion.git}"
MAXDIFFUSION_BRANCH="${MAXDIFFUSION_BRANCH:-two_machines}"
BENCHMARK_JSON="${BENCHMARK_JSON:-VBench_full_info_sub3.json}"
BENCHMARK_JSON_URL="${BENCHMARK_JSON_URL:-}"
BENCHMARK_JSON_PATH="${BENCHMARK_JSON_PATH:-}"

# Dimensions to evaluate (defaults to dimensions specified in the benchmark JSON if not explicitly provided)
DIMENSIONS=()

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
      echo "Usage: $0 [--ssh] [GCS_BUCKET=bucket_name] [RUN_NAME=run_name] [BENCHMARK_JSON=VBench_full_info_sub3.json] [DIMENSIONS=\"dim1 dim2\"] ..."
      echo ""
      echo "Options:"
      echo "  --ssh              Execute commands remotely on GPU VM over SSH"
      echo "  GCS_BUCKET         (Required) GCS bucket where videos are stored"
      echo "  RUN_NAME           Run name used during generation (default: wan-inference)"
      echo "  BENCHMARK_JSON     Benchmark JSON file name (default: VBench_full_info_sub3.json)"
      echo "  BENCHMARK_JSON_URL URL to download benchmark JSON from (default: raw GitHub URL)"
      echo "  BENCHMARK_JSON_PATH Local path to benchmark JSON file (optional override)"
      echo "  GCS_VIDEO_DIR      GCS prefix path to videos (default: \${RUN_NAME}/videos)"
      echo "  GCS_RESULTS_DIR    GCS prefix path to upload results (default: \${RUN_NAME}/vbench_results)"
      echo "  WORK_DIR           Working directory on GPU VM (default: \$HOME/vbench_evaluation)"
      echo "  GPU_NAME           GPU VM name (for SSH mode)"
      echo "  GPU_ZONE           GPU VM zone (for SSH mode)"
      echo "  GPU_PROJECT        GCP project of the GPU VM"
      echo "  DIMENSIONS         Space-separated list of dimensions to evaluate (default: auto-extracted from benchmark JSON)"
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

# Default download URL for benchmark JSON from GitHub repository if not explicitly set
BENCHMARK_JSON_URL="${BENCHMARK_JSON_URL:-https://raw.githubusercontent.com/jitendra-jalwaniya/maxdiffusion/two_machines/benchmarks/vbench/${BENCHMARK_JSON}}"

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

  DIMS_STR="${DIMENSIONS[*]:-}"
  REMOTE_SCRIPT=$(cat <<EOF
set -euo pipefail

WORK_DIR="${WORK_DIR_OVERRIDE:-\$HOME/vbench_evaluation}"
mkdir -p "\${WORK_DIR}"
cd "\${WORK_DIR}"

echo "==> [GPU VM] Ensuring build, git, and video codec system packages..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y git curl wget unzip zip python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y git curl wget unzip zip python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
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

pip install --upgrade "pip<25" "setuptools<71" wheel packaging

echo "==> [GPU VM] Cloning VBench repository..."
if [ ! -d "VBench" ]; then
  git clone -b "${VBENCH_BRANCH}" "${VBENCH_REPO}" VBench
else
  (cd VBench && git pull origin "${VBENCH_BRANCH}" || true)
fi

echo "==> [GPU VM] Checking NVIDIA GPU and driver status..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. If this VM has an NVIDIA GPU, ensure NVIDIA drivers are installed."
fi

echo "==> [GPU VM] Installing PyTorch with CUDA support..."
if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "Installing CUDA-enabled PyTorch..."
  pip uninstall -y torch torchvision 2>/dev/null || true
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple || \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple || \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
fi
pip install "numpy<2"

echo "==> [GPU VM] Installing VBench & GPU dependencies..."
if [ -d "VBench" ]; then
  (cd VBench && git checkout setup.py vbench/distributed.py evaluate.py 2>/dev/null || true)
  python3 -c "
import os

# 1. Patch setup.py to bypass restrictive torch/cuda check during install
setup_path = 'VBench/setup.py'
if os.path.exists(setup_path):
    with open(setup_path, 'r') as f:
        s = f.read()
    s = s.replace('def check_torch_version():', 'def check_torch_version():\n    return\n')
    with open(setup_path, 'w') as f:
        f.write(s)

# 2. Patch vbench/distributed.py to support gloo fallback when CUDA is not present
dist_path = 'VBench/vbench/distributed.py'
if os.path.exists(dist_path):
    with open(dist_path, 'r') as f:
        s = f.read()
    s = s.replace(
        \"backend = 'gloo' if os.name == 'nt' else 'nccl'\",
        \"backend = 'gloo' if (os.name == 'nt' or not torch.cuda.is_available()) else 'nccl'\"
    )
    s = s.replace(
        \"torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))\",
        \"if torch.cuda.is_available():\\n        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))\"
    )
    with open(dist_path, 'w') as f:
        f.write(s)

# 3. Patch evaluate.py for safe device initialization
eval_path = 'VBench/evaluate.py'
if os.path.exists(eval_path):
    with open(eval_path, 'r') as f:
        s = f.read()
    s = s.replace(
        'device = torch.device(\"cuda\")',
        'device = torch.device(f\"cuda:{int(os.environ.get(\\'LOCAL_RANK\\', \\'0\\'))}\") if torch.cuda.is_available() else torch.device(\"cpu\")'
    )
    with open(eval_path, 'w') as f:
        f.write(s)
" 2>/dev/null || true
fi
(cd VBench && pip install --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/cu121)
pip install --no-build-isolation git+https://github.com/openai/CLIP.git --extra-index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' --extra-index-url https://download.pytorch.org/whl/cu121 || true
pip install pandas tabulate google-cloud-storage tqdm opencv-python decord

# Verify PyTorch CUDA state after dependency resolution
if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Reinstalling CUDA-enabled PyTorch to ensure GPU support..."
    pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
  fi
fi

python3 -c "import torch; print(f'==> [GPU VM] PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo "==> [GPU VM] Fetching ${BENCHMARK_JSON}..."
if [ ! -f "${BENCHMARK_JSON}" ]; then
  echo "Downloading ${BENCHMARK_JSON} from ${BENCHMARK_JSON_URL}..."
  if ! curl -fLsS "${BENCHMARK_JSON_URL}" -o "${BENCHMARK_JSON}" 2>/dev/null; then
    python3 -c "import urllib.request; urllib.request.urlretrieve('${BENCHMARK_JSON_URL}', '${BENCHMARK_JSON}')" 2>/dev/null || {
      echo "Attempting fallback to GCS or git clone..."
      if ! gcloud storage cp "gs://${GCS_BUCKET}/${RUN_NAME}/${BENCHMARK_JSON}" ./ 2>/dev/null; then
        if [ ! -d "maxdiffusion" ]; then
          git clone --depth 1 -b "${MAXDIFFUSION_BRANCH}" "${MAXDIFFUSION_REPO}" maxdiffusion
        fi
        cp "maxdiffusion/benchmarks/vbench/${BENCHMARK_JSON}" ./
      fi
    }
  fi
fi

# Download Videos
mkdir -p downloaded_videos
echo "==> [GPU VM] Downloading generated videos from GCS: gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/..."
gcloud storage cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" downloaded_videos/ || gsutil -m cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" downloaded_videos/

# Prepare Videos
mkdir -p vbench_videos
python3 - <<'PYEOF'
import json, os, glob, re, shutil

json_file = '${BENCHMARK_JSON}'
download_dir = 'downloaded_videos'
vbench_dir = 'vbench_videos'

with open(json_file, 'r', encoding='utf-8') as f:
    bench_data = json.load(f)

downloaded = sorted(glob.glob(os.path.join(download_dir, '*.mp4')))
print(f"Downloaded {len(downloaded)} videos from GCS.")

# Group downloaded videos by prompt index (e.g. wan_output_<seed>_<prompt_idx>.mp4)
prompt_video_map = {}
for vid_path in downloaded:
    fname = os.path.basename(vid_path)
    m = re.search(r'_(\d+)\.mp4$', fname)
    if m:
        p_idx = int(m.group(1))
        prompt_video_map.setdefault(p_idx, []).append(vid_path)

total_prompts = len(bench_data)
NUM_SLOTS = 5  # Standard VBench benchmark evaluates 5 samples per prompt (0..4)

matched_prompts = 0
total_linked = 0

for idx, item in enumerate(bench_data):
    prompt = item['prompt_en']
    
    # Locate candidate video(s) by extracted prompt index or positional order
    candidates = prompt_video_map.get(idx, [])
    if not candidates and idx < len(downloaded):
        candidates = [downloaded[idx]]
        
    if not candidates:
        print(f"Warning: No matching video found for prompt {idx} ('{prompt[:40]}...')")
        continue

    matched_prompts += 1
    
    # Populate all 5 VBench sample slots (<prompt>-0.mp4 .. <prompt>-4.mp4)
    for slot in range(NUM_SLOTS):
        target_name = f"{prompt}-{slot}.mp4"
        target_path = os.path.join(vbench_dir, target_name)
        src_video = candidates[slot % len(candidates)]
        
        if os.path.exists(target_path) or os.path.islink(target_path):
            os.remove(target_path)
        try:
            os.symlink(os.path.abspath(src_video), target_path)
        except OSError:
            shutil.copy2(src_video, target_path)
        total_linked += 1

print(f"Successfully prepared {total_linked} video links for {matched_prompts}/{total_prompts} prompts for VBench evaluation.")
PYEOF

echo "==> [GPU VM] Running VBench evaluation..."
cd VBench
mkdir -p "\${WORK_DIR}/evaluation_results"
if [ -n "${DIMS_STR}" ]; then
  EVAL_DIMS="${DIMS_STR}"
else
  echo "==> [GPU VM] Extracting dimensions from \${WORK_DIR}/${BENCHMARK_JSON}..."
  EVAL_DIMS=\$(python3 -c "import json; print(' '.join(dict.fromkeys(d for item in json.load(open('\${WORK_DIR}/${BENCHMARK_JSON}')) for d in item.get('dimension', []))))")
fi
echo "==> [GPU VM] Evaluating dimensions: \${EVAL_DIMS}"

python3 evaluate.py \
  --videos_path "\${WORK_DIR}/vbench_videos" \
  --full_json_dir "\${WORK_DIR}/${BENCHMARK_JSON}" \
  --output_path "\${WORK_DIR}/evaluation_results" \
  --dimension \${EVAL_DIMS} \
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
echo "  Benchmark JSON:  ${BENCHMARK_JSON}"
echo "  Dimensions:      ${DIMENSIONS[*]:-(auto-extracted from ${BENCHMARK_JSON})}"
echo "=========================================================================="

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# 1. Environment & VBench Setup
echo "==> Step 1/5: Setting up Python environment and VBench..."
echo "Ensuring build, git, and video codec system packages..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y git curl wget unzip zip python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y git curl wget unzip zip python3-dev python3-venv build-essential pkg-config ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 || true
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

python3 -m pip install --upgrade "pip<25" "setuptools<71" wheel packaging

if [ ! -d "VBench" ]; then
  echo "Cloning VBench repository from ${VBENCH_REPO}..."
  git clone -b "${VBENCH_BRANCH}" "${VBENCH_REPO}" VBench
else
  echo "Updating VBench repository..."
  (cd VBench && git pull origin "${VBENCH_BRANCH}" || true)
fi

echo "Checking NVIDIA GPU and driver status..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. If this VM has an NVIDIA GPU, ensure NVIDIA drivers are installed."
fi

echo "Installing PyTorch with CUDA support..."
if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "Installing CUDA-enabled PyTorch..."
  pip uninstall -y torch torchvision 2>/dev/null || true
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple || \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple || \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
fi
pip install "numpy<2"

echo "Installing VBench & GPU dependencies..."
if [ -d "VBench" ]; then
  (cd VBench && git checkout setup.py vbench/distributed.py evaluate.py 2>/dev/null || true)
  python3 -c "
import os

# 1. Patch setup.py to bypass restrictive torch/cuda check during install
setup_path = 'VBench/setup.py'
if os.path.exists(setup_path):
    with open(setup_path, 'r') as f:
        s = f.read()
    s = s.replace('def check_torch_version():', 'def check_torch_version():\n    return\n')
    with open(setup_path, 'w') as f:
        f.write(s)

# 2. Patch vbench/distributed.py to support gloo fallback when CUDA is not present
dist_path = 'VBench/vbench/distributed.py'
if os.path.exists(dist_path):
    with open(dist_path, 'r') as f:
        s = f.read()
    s = s.replace(
        \"backend = 'gloo' if os.name == 'nt' else 'nccl'\",
        \"backend = 'gloo' if (os.name == 'nt' or not torch.cuda.is_available()) else 'nccl'\"
    )
    s = s.replace(
        \"torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))\",
        \"if torch.cuda.is_available():\\n        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))\"
    )
    with open(dist_path, 'w') as f:
        f.write(s)

# 3. Patch evaluate.py for safe device initialization
eval_path = 'VBench/evaluate.py'
if os.path.exists(eval_path):
    with open(eval_path, 'r') as f:
        s = f.read()
    s = s.replace(
        'device = torch.device(\"cuda\")',
        'device = torch.device(f\"cuda:{int(os.environ.get(\\'LOCAL_RANK\\', \\'0\\'))}\") if torch.cuda.is_available() else torch.device(\"cpu\")'
    )
    with open(eval_path, 'w') as f:
        f.write(s)
" 2>/dev/null || true
fi
(cd VBench && pip install --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/cu121)
pip install --no-build-isolation git+https://github.com/openai/CLIP.git --extra-index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' --extra-index-url https://download.pytorch.org/whl/cu121 || true
pip install pandas tabulate google-cloud-storage tqdm opencv-python decord

# Verify PyTorch CUDA state after dependency resolution
if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Reinstalling CUDA-enabled PyTorch to ensure GPU support..."
    pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
  fi
fi

python3 -c "import torch; print(f'==> PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# 2. Benchmark JSON Metadata Setup
echo "==> Step 2/5: Locating ${BENCHMARK_JSON}..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JSON_DEST="${WORK_DIR}/${BENCHMARK_JSON}"
if [[ -n "${BENCHMARK_JSON_PATH}" && -f "${BENCHMARK_JSON_PATH}" ]]; then
  cp "${BENCHMARK_JSON_PATH}" "${JSON_DEST}"
elif [ -f "${REPO_ROOT}/benchmarks/vbench/${BENCHMARK_JSON}" ]; then
  cp "${REPO_ROOT}/benchmarks/vbench/${BENCHMARK_JSON}" "${JSON_DEST}"
elif [ -f "${WORK_DIR}/${BENCHMARK_JSON}" ]; then
  echo "Using existing ${JSON_DEST}"
else
  echo "Downloading ${BENCHMARK_JSON} from ${BENCHMARK_JSON_URL}..."
  if ! curl -fLsS "${BENCHMARK_JSON_URL}" -o "${JSON_DEST}" 2>/dev/null; then
    python3 -c "import urllib.request; urllib.request.urlretrieve('${BENCHMARK_JSON_URL}', '${JSON_DEST}')" 2>/dev/null || {
      echo "Attempting fallback to GCS or git clone..."
      if ! gcloud storage cp "gs://${GCS_BUCKET}/${RUN_NAME}/${BENCHMARK_JSON}" "${JSON_DEST}" 2>/dev/null; then
        echo "Cloning MaxDiffusion to fetch ${BENCHMARK_JSON}..."
        if [ ! -d "maxdiffusion" ]; then
          git clone --depth 1 -b "${MAXDIFFUSION_BRANCH}" "${MAXDIFFUSION_REPO}" maxdiffusion
        fi
        cp "maxdiffusion/benchmarks/vbench/${BENCHMARK_JSON}" "${JSON_DEST}"
      fi
    }
  fi
fi

# Extract dimensions from benchmark JSON if not explicitly passed
if [[ ${#DIMENSIONS[@]} -eq 0 ]]; then
  echo "Extracting dimensions from ${JSON_DEST}..."
  read -r -a DIMENSIONS <<< "$(python3 -c "import json; print(' '.join(dict.fromkeys(d for item in json.load(open('${JSON_DEST}')) for d in item.get('dimension', []))))")"
fi
echo "Dimensions to evaluate: ${DIMENSIONS[*]}"

# 3. Download Generated Videos from GCS
echo "==> Step 3/5: Downloading generated videos from GCS: gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/..."
mkdir -p "${WORK_DIR}/downloaded_videos"
gcloud storage cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" "${WORK_DIR}/downloaded_videos/" || \
gsutil -m cp "gs://${GCS_BUCKET}/${GCS_VIDEO_DIR}/*.mp4" "${WORK_DIR}/downloaded_videos/"

# 4. Align Videos to VBench Expected Naming Format
echo "==> Step 4/5: Aligning video filenames for VBench evaluation..."
mkdir -p "${WORK_DIR}/vbench_videos"
python3 - <<PYEOF
import json, os, glob, re, shutil

json_file = '${JSON_DEST}'
download_dir = '${WORK_DIR}/downloaded_videos'
vbench_dir = '${WORK_DIR}/vbench_videos'

with open(json_file, 'r', encoding='utf-8') as f:
    bench_data = json.load(f)

downloaded = sorted(glob.glob(os.path.join(download_dir, '*.mp4')))
print(f"Downloaded {len(downloaded)} videos from GCS.")

# Group downloaded videos by prompt index (e.g. wan_output_<seed>_<prompt_idx>.mp4)
prompt_video_map = {}
for vid_path in downloaded:
    fname = os.path.basename(vid_path)
    m = re.search(r'_(\d+)\.mp4$', fname)
    if m:
        p_idx = int(m.group(1))
        prompt_video_map.setdefault(p_idx, []).append(vid_path)

total_prompts = len(bench_data)
NUM_SLOTS = 5  # Standard VBench benchmark evaluates 5 samples per prompt (0..4)

matched_prompts = 0
total_linked = 0

for idx, item in enumerate(bench_data):
    prompt = item['prompt_en']
    
    # Locate candidate video(s) by extracted prompt index or positional order
    candidates = prompt_video_map.get(idx, [])
    if not candidates and idx < len(downloaded):
        candidates = [downloaded[idx]]
        
    if not candidates:
        print(f"Warning: No matching video found for prompt {idx} ('{prompt[:40]}...')")
        continue

    matched_prompts += 1
    
    # Populate all 5 VBench sample slots (<prompt>-0.mp4 .. <prompt>-4.mp4)
    for slot in range(NUM_SLOTS):
        target_name = f"{prompt}-{slot}.mp4"
        target_path = os.path.join(vbench_dir, target_name)
        src_video = candidates[slot % len(candidates)]
        
        if os.path.exists(target_path) or os.path.islink(target_path):
            os.remove(target_path)
        try:
            os.symlink(os.path.abspath(src_video), target_path)
        except OSError:
            shutil.copy2(src_video, target_path)
        total_linked += 1

print(f"Successfully prepared {total_linked} video links for {matched_prompts}/{total_prompts} prompts for VBench evaluation.")
PYEOF

# 5. Run VBench Evaluation & Publish Results
echo "==> Step 5/5: Executing VBench evaluate.py for dimensions: ${DIMENSIONS[*]}..."
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
