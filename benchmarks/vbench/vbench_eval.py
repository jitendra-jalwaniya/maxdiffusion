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

"""VBench benchmark evaluation runner for MaxDiffusion video models.

Usage:
# WAN 2.1 14B on a single TPU v6e-8 VM
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  output_dir=gs://<bucket>/vbench_out \
  num_inference_steps=40 width=1280 height=720 num_frames=81

# WAN 2.2 27B on TPU v6e-8 (with custom dimension filter and count):
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_27b.yml \
  output_dir=gs://<bucket>/vbench_out \
  --dimension=motion_smoothness --num_prompts=10

# Slicing prompt indices:
python3 -m benchmarks.vbench.vbench_eval src/maxdiffusion/configs/base_wan_14b.yml \
  --start_idx=0 --num_prompts=20
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Sequence

from benchmarks.benchmark_utils import (
    format_benchmark_summary,
    resolve_benchmark_asset_path,
    str2bool,
)
from benchmarks.vbench.vbench_categories import (
    filter_prompts_by_dimension,
    validate_dimension,
)

# Optional heavy ML imports for execution on accelerator environments
try:
  import flax
  import jax

  jax.config.update("jax_use_shardy_partitioner", True)
except ImportError:
  flax = None
  jax = None

try:
  from absl import app
  from maxdiffusion import aot_cache, max_logging, max_utils, pyconfig
  from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
  from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
  from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
  from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
  from maxdiffusion.common_types import WAN2_1, WAN2_2
  from maxdiffusion.generate_wan import call_pipeline, maybe_tune_block_sizes
  from maxdiffusion.loaders.wan_lora_nnx_loader import Wan2_1NNXLoraLoader, Wan2_2NNXLoraLoader
  from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1
  from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2
  from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p1 import WanPipelineI2V_2_1
  from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2
  from maxdiffusion.train_utils import transformer_engine_context
  from maxdiffusion.utils import export_to_video
except ImportError:
  app = None
  aot_cache = None
  max_logging = None
  max_utils = None
  pyconfig = None
  transformer_engine_context = None

DEFAULT_VBENCH_JSON = "VBench_full_info_sub110.json"


def _log(msg: str) -> None:
  """Helper to log via max_logging if available, otherwise standard logging."""
  if max_logging is not None:
    max_logging.log(msg)
  else:
    logging.info(msg)


def load_vbench_prompts(
    json_path: str = DEFAULT_VBENCH_JSON,
    start_idx: int = 0,
    count: int | None = None,
    dimension: str | None = None,
) -> list[str]:
  """Loads prompts from a VBench JSON, searching candidate paths or downloading if needed."""
  filename = os.path.basename(json_path)
  vbench_dir = os.path.dirname(os.path.abspath(__file__))
  download_url = f"https://raw.githubusercontent.com/Vchitect/VBench/master/vbench/{filename}"

  resolved_path = resolve_benchmark_asset_path(
      filename_or_path=json_path,
      search_dirs=[
          vbench_dir,
          os.path.join(vbench_dir, "assets"),
      ],
      download_url=download_url,
  )

  _log(f"Loading VBench prompts from: {resolved_path}")
  with open(resolved_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  if dimension:
    validate_dimension(dimension)
    data = filter_prompts_by_dimension(data, dimension)
    _log(f"Filtered {len(data)} prompts for dimension/category: '{dimension}'")

  if count is not None:
    sliced_data = data[start_idx : start_idx + count]
  else:
    sliced_data = data[start_idx:]

  prompts = [item["prompt_en"] for item in sliced_data]
  return prompts


def load_pipeline(config):
  """Initializes and loads the WAN model pipeline and optional LoRAs."""
  if pyconfig is None or jax is None:
    raise RuntimeError("JAX and MaxDiffusion libraries must be installed to load pipeline.")

  model_key = config.model_name
  model_type = config.model_type
  load_start = time.perf_counter()

  if model_key == WAN2_1:
    pipeline_cls = WanPipelineI2V_2_1 if model_type == "I2V" else WanPipeline2_1
    pretrained_state_sources = (("wan_state", "transformer"),)
    pretrained_config_transformer_attr = "transformer"
    if model_type == "I2V":
      checkpoint_loader = WanCheckpointerI2V_2_1(config=config)
    else:
      checkpoint_loader = WanCheckpointer2_1(config=config)
  elif model_key == WAN2_2:
    pipeline_cls = WanPipelineI2V_2_2 if model_type == "I2V" else WanPipeline2_2
    pretrained_state_sources = (
        ("low_noise_transformer_state", "low_noise_transformer"),
        ("high_noise_transformer_state", "high_noise_transformer"),
    )
    pretrained_config_transformer_attr = "low_noise_transformer"
    if model_type == "I2V":
      checkpoint_loader = WanCheckpointerI2V_2_2(config=config)
    else:
      checkpoint_loader = WanCheckpointer2_2(config=config)
  else:
    raise ValueError(f"Unsupported model_name for checkpointer: {model_key}")

  checkpoint_step = checkpoint_loader.checkpoint_manager.latest_step()
  if checkpoint_step is not None:
    pipeline, _, _ = checkpoint_loader.load_checkpoint(checkpoint_step)
  else:
    pipeline = checkpoint_loader.load_pretrained_pipeline_or_diffusers(
        config,
        pipeline_cls,
        pretrained_state_sources,
        pretrained_config_transformer_attr,
    )
  load_time = time.perf_counter() - load_start
  _log(f"Pipeline loaded in {load_time:.1f}s")

  # Optional LoRA injection
  if (
      config.enable_lora
      and hasattr(config, "lora_config")
      and config.lora_config
      and config.lora_config["lora_model_name_or_path"]
  ):
    if model_key == WAN2_1:
      lora_loader = Wan2_1NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            transformer_weight_name=lora_config["weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )
    elif model_key == WAN2_2:
      lora_loader = Wan2_2NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            high_noise_weight_name=lora_config["high_noise_weight_name"][i],
            low_noise_weight_name=lora_config["low_noise_weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )

  # Per-shape AOT executable cache
  aot_cache.install(
      getattr(config, "aot_cache_dir", ""),
      meta={
          "model": config.pretrained_model_name_or_path,
          "attention": config.attention,
          "flash_block_sizes": str(config.flash_block_sizes),
          "mesh_shape": str(pipeline.mesh.shape),
          "weights_dtype": str(config.weights_dtype),
          "activations_dtype": str(config.activations_dtype),
          "scan_layers": str(config.scan_layers),
          "jax": jax.__version__,
      },
      mesh=pipeline.mesh,
  )
  aot_cache.wait_for_loads()
  return pipeline, load_time


def run_vbench_batch(
    config: Any,
    vbench_json: str = DEFAULT_VBENCH_JSON,
    start_idx: int = 0,
    num_prompts: int | None = None,
    dimension: str | None = None,
    commit_hash: str | None = None,
) -> dict[str, float | int]:
  """Executes batch generation over VBench prompts."""
  prompts = load_vbench_prompts(
      json_path=vbench_json,
      start_idx=start_idx,
      count=num_prompts,
      dimension=dimension,
  )
  total_prompts = len(prompts)
  _log(f"Successfully loaded {total_prompts} prompts (offset {start_idx})")

  maybe_tune_block_sizes(config)

  try:
    writer = max_utils.initialize_summary_writer(config)
    if jax.process_index() == 0 and writer and commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
  except Exception as e:
    _log(f"Note: Summary writer initialization skipped or failed: {e}")

  pipeline, load_time = load_pipeline(config)

  # Disable profiler during warmup compilation
  if "enable_profiler" in config.get_keys():
    config.get_keys()["enable_profiler"] = False

  # Warmup compilation with 2 denoising steps
  warmup_prompt = [prompts[0]] * config.global_batch_size_to_train_on
  warmup_neg = [config.negative_prompt] * config.global_batch_size_to_train_on
  warmup_steps = min(2, config.num_inference_steps)
  _log(f"Compiling warmup graph ({warmup_steps} denoising steps)...")
  s0_warmup = time.perf_counter()
  with aot_cache.warmup_mode():
    _ = call_pipeline(
        config,
        pipeline,
        warmup_prompt,
        warmup_neg,
        num_inference_steps=warmup_steps,
    )
  aot_cache.save_pending()
  compile_time = time.perf_counter() - s0_warmup
  _log(f"Compilation/warmup completed in {compile_time:.1f}s")

  dest_dir = (
      os.path.join(config.output_dir, config.run_name, "videos")
      if config.output_dir.startswith("gs://")
      else config.output_dir
  )
  _log("===================== VBench Generation Start =====================")
  _log(f"Model: {config.model_name} ({config.pretrained_model_name_or_path})")
  _log(
      f"Resolution: {config.width}x{config.height}, Frames: {config.num_frames}, Steps: {config.num_inference_steps}"
  )
  _log(f"Destination: {dest_dir}")
  _log(f"Total Videos to Generate: {total_prompts}")
  _log("====================================================================")

  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on
  generation_timings = []

  for idx, prompt_text in enumerate(prompts, start=1):
    _log(f"\n[{idx}/{total_prompts}] Generating prompt: \"{prompt_text}\"")
    prompt = [prompt_text] * config.global_batch_size_to_train_on

    t0 = time.perf_counter()
    outputs = call_pipeline(config, pipeline, prompt, negative_prompt)
    if isinstance(outputs, tuple):
      videos, _ = outputs
    else:
      videos = outputs
    gen_time = time.perf_counter() - t0
    generation_timings.append(gen_time)

    # VBench standard video naming: "<prompt_en>-0.mp4"
    video_filename = f"{prompt_text}-0.mp4"
    for i in range(len(videos)):
      cur_video_name = video_filename if i == 0 else f"{prompt_text}-{i}.mp4"
      export_to_video(videos[i], cur_video_name, fps=config.fps)
      _log(f"Saved: {cur_video_name} (time: {gen_time:.1f}s)")

      if config.output_dir.startswith("gs://"):
        gcs_destination = os.path.join(config.output_dir, config.run_name)
        try:
          max_utils.upload_file_to_gcs(gcs_destination, cur_video_name, subdir="videos")
          max_utils.delete_file(f"./{cur_video_name}")
        except Exception as e:
          _log(f"Warning: Failed to upload {cur_video_name} to GCS: {e}")
          _log(f"Video is preserved locally at: ./{cur_video_name}")

    avg_time = sum(generation_timings) / len(generation_timings)
    remaining_prompts = total_prompts - idx
    est_remaining_sec = remaining_prompts * avg_time
    _log(
        f"Progress: {idx}/{total_prompts} (Avg: {avg_time:.1f}s/video, "
        f"Est. remaining: {est_remaining_sec / 60:.1f} min)"
    )

  total_gen_time = sum(generation_timings)
  avg_time = total_gen_time / max(1, total_prompts)

  summary_metrics = {
      "load_time": load_time,
      "compile_time": compile_time,
      "total_gen_time": total_gen_time,
      "average_per_video": avg_time,
      "total_videos": total_prompts,
  }
  _log(format_benchmark_summary(summary_metrics, title="VBENCH BATCH GENERATION SUMMARY"))

  return summary_metrics


def parse_vbench_cli_args(argv: Sequence[str]):
  """Separates VBench benchmark CLI options from pyconfig model arguments."""
  parser = argparse.ArgumentParser(
      description="Run VBench video generation benchmark in MaxDiffusion."
  )
  parser.add_argument(
      "--vbench_json",
      type=str,
      default=DEFAULT_VBENCH_JSON,
      help="Path or filename of VBench JSON prompt dataset (default: VBench_full_info_sub110.json).",
  )
  parser.add_argument(
      "--num_prompts",
      "--vbench_num_prompts",
      dest="num_prompts",
      type=int,
      default=None,
      help="Number of prompts to generate.",
  )
  parser.add_argument(
      "--start_idx",
      "--vbench_start_idx",
      dest="start_idx",
      type=int,
      default=0,
      help="Starting prompt index (offset).",
  )
  parser.add_argument(
      "--dimension",
      type=str,
      default=None,
      help="Optional VBench dimension to filter (e.g. motion_smoothness, temporal_flickering).",
  )

  if len(argv) > 1 and any(arg in ("-h", "--help") for arg in argv[1:]):
    parser.print_help()
    sys.exit(0)

  raw_args = list(argv[1:]) if len(argv) > 1 else []
  extracted_flags = []
  model_args = [argv[0]] if argv else ["vbench_eval"]

  vbench_json = DEFAULT_VBENCH_JSON
  num_prompts = None
  start_idx = 0
  dimension = None

  for arg in raw_args:
    if arg.startswith("vbench_json="):
      vbench_json = arg.split("=", 1)[1]
    elif arg.startswith("vbench_num_prompts=") or arg.startswith("num_prompts="):
      num_prompts = int(arg.split("=", 1)[1])
    elif arg.startswith("vbench_start_idx=") or arg.startswith("start_idx="):
      start_idx = int(arg.split("=", 1)[1])
    elif arg.startswith("dimension="):
      dimension = arg.split("=", 1)[1]
    elif arg.startswith("-"):
      extracted_flags.append(arg)
    else:
      model_args.append(arg)

  if extracted_flags:
    parsed_flags, extra_unknown = parser.parse_known_args(extracted_flags)
    if parsed_flags.vbench_json != DEFAULT_VBENCH_JSON:
      vbench_json = parsed_flags.vbench_json
    if parsed_flags.num_prompts is not None:
      num_prompts = parsed_flags.num_prompts
    if parsed_flags.start_idx != 0:
      start_idx = parsed_flags.start_idx
    if parsed_flags.dimension is not None:
      dimension = parsed_flags.dimension
    for u in extra_unknown:
      model_args.append(u)

  return model_args, vbench_json, start_idx, num_prompts, dimension


def main(argv: Sequence[str]) -> None:
  model_args, vbench_json, start_idx, num_prompts, dimension = parse_vbench_cli_args(argv)

  if pyconfig is None or max_utils is None:
    raise RuntimeError(
        "MaxDiffusion and JAX must be installed to run VBench evaluation. "
        "Please ensure your virtual environment has the required dependencies."
    )

  commit_hash = max_utils.get_git_commit_hash()

  pyconfig.initialize(model_args)
  if flax is not None:
    try:
      flax.config.update("flax_always_shard_variable", False)
    except LookupError:
      pass

  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run_vbench_batch(
      config=pyconfig.config,
      vbench_json=vbench_json,
      start_idx=start_idx,
      num_prompts=num_prompts,
      dimension=dimension,
      commit_hash=commit_hash,
  )


if __name__ == "__main__":
  if app is not None and transformer_engine_context is not None:
    with transformer_engine_context():
      app.run(main)
  else:
    main(sys.argv)
