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

"""Batch generation of VBench benchmark videos using WAN 2.1 / WAN 2.2."""

import json
import os
import time
import urllib.request
from typing import Sequence
from absl import app
import flax
import jax
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

jax.config.update("jax_use_shardy_partitioner", True)


def load_vbench_prompts(json_path: str, start_idx: int = 0, count: int | None = None) -> list[str]:
  """Loads prompts from a VBench json, searching multiple candidate paths or downloading if needed."""
  filename = os.path.basename(json_path)
  candidates = [
      os.path.expanduser(json_path),
      os.path.abspath(json_path),
      os.path.join(os.path.expanduser("~"), json_path),
      os.path.join(os.path.expanduser("~"), "Documents", json_path),
      os.path.join(os.path.expanduser("~"), "Documents", "maxdiffusion", json_path),
      os.path.join(os.path.expanduser("~"), "Documents", "VBench", "vbench", filename),
      os.path.join(os.path.expanduser("~"), "VBench", "vbench", filename),
      os.path.join(os.getcwd(), json_path),
      os.path.join(os.getcwd(), filename),
      os.path.join(os.getcwd(), "..", "VBench", "vbench", filename),
      os.path.join(os.getcwd(), "VBench", "vbench", filename),
      os.path.expanduser(f"~/.cache/vbench/{filename}"),
  ]

  resolved_path = None
  for p in candidates:
    if os.path.exists(p):
      resolved_path = p
      break

  if resolved_path is None:
    cache_dir = os.path.expanduser("~/.cache/vbench")
    os.makedirs(cache_dir, exist_ok=True)
    resolved_path = os.path.join(cache_dir, filename)
    max_logging.log(f"VBench json not found locally. Downloading to {resolved_path}...")
    url = f"https://raw.githubusercontent.com/Vchitect/VBench/master/vbench/{filename}"
    try:
      urllib.request.urlretrieve(url, resolved_path)
      max_logging.log(f"Downloaded VBench json successfully to {resolved_path}")
    except Exception as e:
      raise FileNotFoundError(
          f"Could not find or download VBench json file from '{url}'. Error: {e}"
      ) from e

  max_logging.log(f"Loading VBench prompts from: {resolved_path}")
  with open(resolved_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  if count is not None:
    sliced_data = data[start_idx : start_idx + count]
  else:
    sliced_data = data[start_idx:]
  prompts = [item["prompt_en"] for item in sliced_data]
  return prompts


def load_pipeline(config):
  """Initializes and loads the WAN model pipeline and optional LoRAs."""
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
  max_logging.log(f"Pipeline loaded in {load_time:.1f}s")

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
    config,
    vbench_json: str,
    start_idx: int = 0,
    num_prompts: int | None = None,
    commit_hash: str | None = None,
):
  """Executes batch generation over VBench prompts."""
  # Load prompts first to validate early
  prompts = load_vbench_prompts(vbench_json, start_idx=start_idx, count=num_prompts)
  total_prompts = len(prompts)
  max_logging.log(f"Successfully loaded {total_prompts} prompts (offset {start_idx})")

  maybe_tune_block_sizes(config)

  writer = None
  try:
    writer = max_utils.initialize_summary_writer(config)
    if jax.process_index() == 0 and writer and commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
  except Exception as e:
    max_logging.log(f"Note: Summary writer initialization skipped or failed: {e}")

  pipeline, load_time = load_pipeline(config)

  # Disable profiler for warmup
  if "enable_profiler" in config.get_keys():
    config.get_keys()["enable_profiler"] = False

  # Warmup compilation with 2 denoising steps
  warmup_prompt = [prompts[0]] * config.global_batch_size_to_train_on
  warmup_neg = [config.negative_prompt] * config.global_batch_size_to_train_on
  warmup_steps = min(2, config.num_inference_steps)
  max_logging.log(f"Compiling warmup graph ({warmup_steps} denoising steps)...")
  s0_warmup = time.perf_counter()
  with aot_cache.warmup_mode():
    _ = call_pipeline(config, pipeline, warmup_prompt, warmup_neg, num_inference_steps=warmup_steps)
  aot_cache.save_pending()
  compile_time = time.perf_counter() - s0_warmup
  max_logging.log(f"Compilation/warmup completed in {compile_time:.1f}s")

  max_logging.log("===================== VBench Generation Start =====================")
  max_logging.log(f"Model: {config.model_name} ({config.pretrained_model_name_or_path})")
  max_logging.log(f"Resolution: {config.width}x{config.height}, Frames: {config.num_frames}, Steps: {config.num_inference_steps}")
  max_logging.log(f"Destination: {os.path.join(config.output_dir, config.run_name, 'videos') if config.output_dir.startswith('gs://') else config.output_dir}")
  max_logging.log(f"Total Videos to Generate: {total_prompts}")
  max_logging.log("====================================================================")

  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on
  generation_timings = []

  for idx, prompt_text in enumerate(prompts, start=1):
    max_logging.log(f"\n[{idx}/{total_prompts}] Generating prompt: \"{prompt_text}\"")
    prompt = [prompt_text] * config.global_batch_size_to_train_on

    t0 = time.perf_counter()
    outputs = call_pipeline(config, pipeline, prompt, negative_prompt)
    if isinstance(outputs, tuple):
      videos, _ = outputs
    else:
      videos = outputs
    gen_time = time.perf_counter() - t0
    generation_timings.append(gen_time)

    # VBench standard naming: "<prompt_en>-0.mp4"
    video_filename = f"{prompt_text}-0.mp4"
    for i in range(len(videos)):
      cur_video_name = video_filename if i == 0 else f"{prompt_text}-{i}.mp4"
      export_to_video(videos[i], cur_video_name, fps=config.fps)
      max_logging.log(f"Saved: {cur_video_name} (time: {gen_time:.1f}s)")

      if config.output_dir.startswith("gs://"):
        gcs_destination = os.path.join(config.output_dir, config.run_name)
        try:
          max_utils.upload_file_to_gcs(gcs_destination, cur_video_name, subdir="videos")
          max_utils.delete_file(f"./{cur_video_name}")
        except Exception as e:
          max_logging.log(f"Warning: Failed to upload {cur_video_name} to GCS: {e}")
          max_logging.log(f"Video is preserved locally at: ./{cur_video_name}")

    avg_time = sum(generation_timings) / len(generation_timings)
    remaining_prompts = total_prompts - idx
    est_remaining_sec = remaining_prompts * avg_time
    max_logging.log(
        f"Progress: {idx}/{total_prompts} (Avg: {avg_time:.1f}s/video, Est. remaining: {est_remaining_sec / 60:.1f} min)"
    )

  total_gen_time = sum(generation_timings)
  max_logging.log(f"\n{'=' * 50}")
  max_logging.log("  VBENCH BATCH GENERATION SUMMARY")
  max_logging.log(f"{'=' * 50}")
  max_logging.log(f"  Load Time:          {load_time:>7.1f}s")
  max_logging.log(f"  Compile Time:       {compile_time:>7.1f}s")
  max_logging.log(f"  Total Gen Time:     {total_gen_time:>7.1f}s")
  max_logging.log(f"  Average Per Video:  {total_gen_time / max(1, total_prompts):>7.1f}s")
  max_logging.log(f"  Total Videos:       {total_prompts:>7d}")
  max_logging.log(f"{'=' * 50}\n")


def main(argv: Sequence[str]) -> None:
  commit_hash = max_utils.get_git_commit_hash()

  # Extract script-specific arguments before passing argv to pyconfig
  vbench_json = "VBench_full_info_sub110.json"
  vbench_num_prompts = None
  vbench_start_idx = 0

  filtered_argv = [argv[0], argv[1]] if len(argv) >= 2 else list(argv)
  for arg in argv[2:]:
    if arg.startswith("vbench_json="):
      vbench_json = arg.split("=", 1)[1]
    elif arg.startswith("vbench_num_prompts="):
      vbench_num_prompts = int(arg.split("=", 1)[1])
    elif arg.startswith("vbench_start_idx="):
      vbench_start_idx = int(arg.split("=", 1)[1])
    else:
      filtered_argv.append(arg)

  pyconfig.initialize(filtered_argv)
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass

  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run_vbench_batch(
      config=pyconfig.config,
      vbench_json=vbench_json,
      start_idx=vbench_start_idx,
      num_prompts=vbench_num_prompts,
      commit_hash=commit_hash,
  )


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
