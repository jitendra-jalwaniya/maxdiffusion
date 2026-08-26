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

"""This file contains common utility functions for MaxDiffusion benchmarking."""

import os
import urllib.request
from typing import Any, Sequence


def str2bool(v: str | bool) -> bool:
  """Convert a string of truth to True or False.

  Args:
    v: input value.

  Returns:
    bool: True or False.
  """
  if isinstance(v, bool):
    return v
  v_lower = v.lower()
  true_values = ["y", "yes", "t", "true", "1"]
  false_values = ["n", "no", "f", "false", "0"]
  if v_lower in true_values:
    return True
  if v_lower in false_values:
    return False
  raise ValueError(f"Invalid boolean value '{v}'!")


def resolve_benchmark_asset_path(
    filename_or_path: str,
    search_dirs: Sequence[str] | None = None,
    download_url: str | None = None,
) -> str:
  """Resolves path to a benchmark asset file, checking candidates or downloading.

  Args:
    filename_or_path: relative or absolute path or filename.
    search_dirs: optional list of directories to look in.
    download_url: optional URL to download file if not found.

  Returns:
    Resolved absolute or accessible path to file.

  Raises:
    FileNotFoundError if file cannot be found or downloaded.
  """
  if os.path.isabs(filename_or_path) and os.path.exists(filename_or_path):
    return filename_or_path

  filename = os.path.basename(filename_or_path)
  candidates = [
      os.path.expanduser(filename_or_path),
      os.path.abspath(filename_or_path),
  ]

  if search_dirs:
    for d in search_dirs:
      candidates.append(os.path.join(d, filename_or_path))
      candidates.append(os.path.join(d, filename))

  # Default fallbacks based on repo conventions
  module_dir = os.path.dirname(os.path.abspath(__file__))
  candidates.extend([
      os.path.join(module_dir, "vbench", filename),
      os.path.join(module_dir, "vbench", "assets", filename),
      os.path.join(os.getcwd(), "benchmarks", "vbench", filename),
      os.path.join(os.getcwd(), filename),
      os.path.join(os.path.expanduser("~"), "Documents", "maxdiffusion", "benchmarks", "vbench", filename),
      os.path.join(os.path.expanduser("~"), "Documents", "VBench", "vbench", filename),
      os.path.join(os.path.expanduser("~"), "VBench", "vbench", filename),
      os.path.expanduser(f"~/.cache/vbench/{filename}"),
  ])

  for p in candidates:
    if os.path.exists(p):
      return os.path.abspath(p)

  if download_url:
    cache_dir = os.path.expanduser("~/.cache/vbench")
    os.makedirs(cache_dir, exist_ok=True)
    resolved_path = os.path.join(cache_dir, filename)
    try:
      urllib.request.urlretrieve(download_url, resolved_path)
      return resolved_path
    except Exception as e:
      raise FileNotFoundError(
          f"Could not find or download benchmark asset from '{download_url}'. Error: {e}"
      ) from e

  raise FileNotFoundError(
      f"Benchmark asset '{filename_or_path}' not found in candidate locations: {candidates}"
  )


def format_benchmark_summary(metrics: dict[str, Any], title: str = "BENCHMARK SUMMARY") -> str:
  """Formats a dictionary of benchmark metrics into a readable report banner."""
  lines = [
      f"\n{'=' * 50}",
      f"  {title}",
      f"{'=' * 50}",
  ]
  for k, v in metrics.items():
    label = k.replace("_", " ").title()
    if isinstance(v, float):
      lines.append(f"  {label:<22}: {v:>8.1f}s")
    elif isinstance(v, int):
      lines.append(f"  {label:<22}: {v:>8d}")
    else:
      lines.append(f"  {label:<22}: {str(v):>8}")
  lines.append(f"{'=' * 50}\n")
  return "\n".join(lines)
