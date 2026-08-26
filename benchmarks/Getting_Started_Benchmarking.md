# Getting Started with Benchmark Running in MaxDiffusion

This directory organizes quality benchmarks and performance profiling tools for **MaxDiffusion**, following the benchmarking patterns established in [MaxText](https://github.com/AI-Hypercomputer/maxtext).

---

## Directory Structure

```
benchmarks/
├── Getting_Started_Benchmarking.md  # Main benchmarking guide
├── benchmark_utils.py               # Shared benchmark helpers & utilities
├── __init__.py
└── vbench/                          # VBench video quality benchmark suite
    ├── README.md                    # VBench user guide
    ├── VBench_full_info_sub110.json # Bundled 110-prompt benchmark dataset
    ├── __init__.py
    ├── vbench_categories.py         # Dimension categories and prompt filtering
    └── vbench_eval.py               # Primary evaluation runner
```

---

## Approaches to Benchmarking

MaxDiffusion provides multiple benchmark tools depending on the evaluation goal:

1. **Quality & Alignment Evaluation (VBench)**:
   Run standardized prompt datasets through diffusion pipelines (e.g. WAN 2.1, WAN 2.2) to generate videos formatted for automatic scoring across 16 quality and text-alignment dimensions.

2. **Block-Level & Attention Microbenchmarks**:
   Isolate and benchmark individual attention blocks and tile configurations (e.g. `src/maxdiffusion/utils/wan_block_benchmark.py` and `ltx2_block_benchmark.py`).

---

## Running VBench Benchmarks

### 1. Zero-Config Run (Default 110-Prompt Subset)
```bash
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  output_dir=/tmp/vbench_run
```

### 2. High-Performance Fast Inference (Tuned 2D Ring Attention)
```bash
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_27b.yml \
  output_dir=gs://<your-bucket>/benchmarks/wan22 \
  attention=ulysses_ring_custom_fixed_m \
  ulysses_shards=2 \
  ici_data_parallelism=2 ici_fsdp_parallelism=1 \
  ici_context_parallelism=4 ici_tensor_parallelism=1 \
  per_device_batch_size=0.125 \
  num_inference_steps=40 num_frames=81 width=1280 height=720 \
  guidance_scale_low=3.0 guidance_scale_high=4.0
```

### 3. Filter by Dimension or Slicing
```bash
# Evaluate only motion smoothness prompts
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  --dimension=motion_smoothness --num_prompts=10
```

For more details on scoring and evaluation metrics, see [benchmarks/vbench/README.md](vbench/README.md).
