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

"""VBench evaluation dimensions, categories, and prompt helper utilities."""

from typing import Any

# All 16 standard evaluation dimensions defined in VBench
VBENCH_DIMENSIONS: tuple[str, ...] = (
    "aesthetic_quality",
    "appearance_style",
    "background_consistency",
    "color",
    "dynamic_degree",
    "human_action",
    "imaging_quality",
    "motion_smoothness",
    "multiple_objects",
    "object_class",
    "overall_consistency",
    "scene",
    "spatial_relationship",
    "subject_consistency",
    "temporal_flickering",
    "temporal_style",
)

# Higher-level category groupings (Video Quality vs. Video-Text Alignment)
DIMENSION_CATEGORIES: dict[str, str] = {
    "aesthetic_quality": "video_quality",
    "appearance_style": "video_text_alignment",
    "background_consistency": "video_quality",
    "color": "video_text_alignment",
    "dynamic_degree": "video_quality",
    "human_action": "video_text_alignment",
    "imaging_quality": "video_quality",
    "motion_smoothness": "video_quality",
    "multiple_objects": "video_text_alignment",
    "object_class": "video_text_alignment",
    "overall_consistency": "video_text_alignment",
    "scene": "video_text_alignment",
    "spatial_relationship": "video_text_alignment",
    "subject_consistency": "video_quality",
    "temporal_flickering": "video_quality",
    "temporal_style": "video_text_alignment",
}

CATEGORY_TO_DIMENSIONS: dict[str, list[str]] = {
    "video_quality": [
        "aesthetic_quality",
        "background_consistency",
        "dynamic_degree",
        "imaging_quality",
        "motion_smoothness",
        "subject_consistency",
        "temporal_flickering",
    ],
    "video_text_alignment": [
        "appearance_style",
        "color",
        "human_action",
        "multiple_objects",
        "object_class",
        "overall_consistency",
        "scene",
        "spatial_relationship",
        "temporal_style",
    ],
}

DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "aesthetic_quality": "Overall photographic and artistic visual appeal",
    "appearance_style": "Faithfulness to artistic or photographic appearance styles",
    "background_consistency": "Consistency of the background scene across frames",
    "color": "Faithful color rendering matching prompt descriptions",
    "dynamic_degree": "Degree of dynamic motion and movement in the video",
    "human_action": "Accurate depiction of specified human actions",
    "imaging_quality": "Clarity, sharpness, and absence of compression artifacts",
    "motion_smoothness": "Smoothness and naturalness of video motion",
    "multiple_objects": "Accurate rendering of multiple specified objects",
    "object_class": "Accurate representation of requested object classes",
    "overall_consistency": "Holistic alignment and perceptual video-text coherence",
    "scene": "Accurate representation of the requested scene/environment",
    "spatial_relationship": "Correct relative spatial positioning of objects",
    "subject_consistency": "Consistency of the main subject across frames",
    "temporal_flickering": "Absence of sudden high-frequency flickering artifacts",
    "temporal_style": "Temporal pacing and stylistic motion consistency",
}


def validate_dimension(dimension: str) -> None:
  """Validates if a dimension or category name is recognized by VBench."""
  dim_clean = dimension.strip().lower()
  if dim_clean in VBENCH_DIMENSIONS:
    return
  if dim_clean in CATEGORY_TO_DIMENSIONS:
    return
  raise ValueError(
      f"Unknown VBench dimension/category '{dimension}'. "
      f"Supported dimensions: {sorted(VBENCH_DIMENSIONS)}. "
      f"Supported categories: {sorted(CATEGORY_TO_DIMENSIONS.keys())}."
  )


def filter_prompts_by_dimension(
    items: list[dict[str, Any]],
    dimension: str | None,
) -> list[dict[str, Any]]:
  """Filters prompt items by VBench dimension or category name.

  Args:
    items: list of VBench prompt dictionaries with 'prompt_en' and 'dimension'.
    dimension: target dimension (e.g. 'motion_smoothness') or category
      (e.g. 'video_quality'). If None or 'all', returns all items unchanged.

  Returns:
    Filtered list of prompt items.
  """
  if not dimension or dimension.strip().lower() == "all":
    return items

  dim_clean = dimension.strip().lower()
  validate_dimension(dim_clean)

  if dim_clean in CATEGORY_TO_DIMENSIONS:
    allowed_dims = set(CATEGORY_TO_DIMENSIONS[dim_clean])
    return [
        item for item in items
        if any(d in allowed_dims for d in item.get("dimension", []))
    ]

  return [
      item for item in items
      if dim_clean in item.get("dimension", [])
  ]
