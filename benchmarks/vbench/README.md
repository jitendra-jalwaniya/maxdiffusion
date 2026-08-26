# VBench Video Generation Benchmarking

This directory contains the VBench benchmarking suite for video diffusion models in **MaxDiffusion**, modeled after the benchmarking structure in [MaxText](https://github.com/AI-Hypercomputer/maxtext).

## Overview

[VBench](https://github.com/Vchitect/VBench) is a comprehensive benchmark suite for video generative models, evaluating both **video quality** and **video-text alignment** across 16 core dimensions.

### Supported Dimensions

| Category | Dimension | Description |
| :--- | :--- | :--- |
| **Video Quality** | `subject_consistency` | Consistency of the main subject across frames |
| | `background_consistency` | Consistency of the background scene across frames |
| | `temporal_flickering` | Absence of high-frequency flickering artifacts |
| | `motion_smoothness` | Smoothness and naturalness of video motion |
| | `dynamic_degree` | Degree of dynamic motion and movement |
| | `aesthetic_quality` | Photographic and artistic visual appeal |
| | `imaging_quality` | Clarity, sharpness, and absence of compression artifacts |
| **Video-Text Alignment** | `object_class` | Accurate representation of requested object classes |
| | `multiple_objects` | Accurate rendering of multiple specified objects |
| | `human_action` | Accurate depiction of specified human actions |
| | `color` | Faithful color rendering matching prompt descriptions |
| | `spatial_relationship` | Correct relative spatial positioning of objects |
| | `scene` | Accurate representation of the requested scene |
| | `appearance_style` | Faithfulness to artistic or photographic styles |
| | `temporal_style` | Temporal pacing and stylistic motion consistency |
| | `overall_consistency` | Holistic alignment and perceptual coherence |

---

## Quick Start

### 1. Generate Benchmark Videos

#### WAN 2.1 14B on TPU:
```bash
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  output_dir=/tmp/vbench_wan21 \
  num_inference_steps=40 width=1280 height=720 num_frames=81
```

#### WAN 2.2 27B on TPU:
```bash
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_27b.yml \
  output_dir=/tmp/vbench_wan22 \
  guidance_scale_low=3.0 guidance_scale_high=4.0 \
  num_inference_steps=40 width=1280 height=720 num_frames=81
```

#### Upload Directly to GCS:
```bash
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  output_dir=gs://<your-gcs-bucket>/vbench_eval/ \
  run_name=wan21_14b_vbench
```

---

## Filtering and Slicing

You can filter prompts by dimension or slice subsets:

### Filter by Dimension:
```bash
# Evaluate only motion smoothness prompts
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  --dimension=motion_smoothness
```

### Filter by Category:
```bash
# Evaluate all video quality dimensions
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  --dimension=video_quality
```

### Slice Prompts with Index Offsets:
```bash
# Run 10 prompts starting at index 20
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  --start_idx=20 --num_prompts=10
```

---

## Prompt Datasets

- `VBench_full_info_sub110.json`: Bundled 110-prompt curated subset covering all 16 evaluation dimensions (default).
- Full VBench dataset (`VBench_full_info.json`): Pass `--vbench_json=VBench_full_info.json`. If not found locally, it is downloaded automatically to `~/.cache/vbench/`.

---

## Evaluating Generated Videos

Once videos are generated in the output folder, evaluate them using the official VBench package:

```bash
# Run VBench evaluation on generated videos
vbench evaluate \
  --videos_path /tmp/vbench_wan21/videos \
  --dimension motion_smoothness temporal_flickering subject_consistency \
  --output_path /tmp/vbench_eval_results
```

---

## Command-Line Arguments

| Flag | Pyconfig Override | Default | Description |
| :--- | :--- | :--- | :--- |
| `--vbench_json` | `vbench_json=<path>` | `VBench_full_info_sub110.json` | Path to VBench JSON prompts file |
| `--num_prompts` | `num_prompts=<int>` / `vbench_num_prompts=<int>` | `None` (all) | Number of prompts to evaluate |
| `--start_idx` | `start_idx=<int>` / `vbench_start_idx=<int>` | `0` | Starting index offset |
| `--dimension` | `dimension=<name>` | `None` (all) | Specific dimension or category to filter |
