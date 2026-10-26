"""Unit tests for models/classifier.py"""
from __future__ import annotations

import pytest
from ez_comfy.models.classifier import classify_checkpoint


@pytest.mark.parametrize("filename,expected_family,expected_variant", [
    # SDXL Lightning variants
    ("realvisxlV50_v50Lightning.safetensors",     "sdxl", "lightning"),
    ("dreamshaperXL_v21TurboDPMSDE.safetensors",  "sdxl", "turbo"),
    # Pony — must be detected BEFORE SDXL
    ("ponyDiffusionV6XL_v6StartWithThisOne.safetensors", "pony", None),
    # SD1.5
    ("v1-5-pruned-emaonly.safetensors",            "sd15", None),
    ("dreamshaper_8.safetensors",                  "sd15", None),
    # Flux
    ("flux1-dev.safetensors",                      "flux", None),
    ("flux1-schnell.safetensors",                  "flux", "schnell"),
    # SVD
    ("svd_xt.safetensors",                         "svd",  None),
    # Stable Audio
    ("stable_audio_open_1.0.safetensors",          "stable_audio", None),
    # Unknown
    ("some_random_model.ckpt",                     "unknown", None),
])
def test_classify_checkpoint(filename, expected_family, expected_variant):
    family, variant = classify_checkpoint(filename)
    assert family == expected_family, f"{filename}: expected family={expected_family!r}, got {family!r}"
    assert variant == expected_variant, f"{filename}: expected variant={expected_variant!r}, got {variant!r}"


def test_classify_sdxl_base():
    family, variant = classify_checkpoint("sd_xl_base_1.0.safetensors")
    assert family == "sdxl"
    assert variant is None


# ---------------------------------------------------------------------------
# WI-3: modern families. File names come from the workflow templates shipped with
# ComfyUI Desktop 1.1.8 (comfyui_workflow_templates_json 0.1.122), not from any user's inventory.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_family", [
    ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan_video"),
    ("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "wan_video"),
    ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "wan_video"),
    ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "wan_video"),
    ("wan2.2_ti2v_5B_fp16.safetensors", "wan_video"),
    ("wan2.1_t2v_1.3B_fp16.safetensors", "wan_video"),
    ("wan2.1_i2v_480p_14B_fp16.safetensors", "wan_video"),
    ("ltx-2.3-22b-dev-fp8.safetensors", "ltx_video"),
    ("ltx-video-2b-v0.9.safetensors", "ltx_video"),
    ("qwen_image_fp8_e4m3fn.safetensors", "qwen_image"),
    ("qwen_image_2.1_int8_convrot.safetensors", "qwen_image"),
    ("hidream_i1_dev_fp8.safetensors", "hidream"),
    ("flux1-dev.safetensors", "flux"),
    ("flux1-krea-dev_fp8_scaled.safetensors", "flux"),
    ("flux2_dev_fp8mixed.safetensors", "flux2"),
    ("flux3-image-klein-base-int8-convrot.safetensors", "flux3"),
    ("sd_xl_base_1.0.safetensors", "sdxl"),                 # unchanged
    ("v1-5-pruned-emaonly-fp16.safetensors", "sd15"),       # unchanged
    ("stable_audio_open_1.0.safetensors", "stable_audio"),  # unchanged
])
def test_modern_family_filenames(filename, expected_family):
    assert classify_checkpoint(filename)[0] == expected_family


def test_folder_prefix_does_not_hide_the_family():
    assert classify_checkpoint("Wan\\wan2.2_ti2v_5B_fp16.safetensors")[0] == "wan_video"


@pytest.mark.parametrize("filename", [
    "swan2.safetensors",             # letter before "wan"
    "wanderer_v3.safetensors",       # "wan" inside a word
    "wanted_2.safetensors",
    "dewan_2023.ckpt",
    "wan.safetensors",               # no version-like context
    "oswan-1.safetensors",
])
def test_wan_false_positives(filename):
    assert classify_checkpoint(filename)[0] != "wan_video"


@pytest.mark.parametrize("filename", [
    "multx_v1.safetensors",
    "ltxd_model.safetensors",
    "ltx.safetensors",
    "saltx2.safetensors",
])
def test_ltx_false_positives(filename):
    assert classify_checkpoint(filename)[0] != "ltx_video"


def test_qwen_text_encoder_is_not_an_image_model():
    assert classify_checkpoint("qwen_2.5_vl_7b_fp8_scaled.safetensors")[0] == "unknown"


def test_unrecognised_merge_without_xl_in_the_name_stays_unknown():
    """Decision D1(b): no allow-list for name-only SDXL merges; model_overrides covers them."""
    assert classify_checkpoint("exampleMergeV9.safetensors")[0] == "unknown"


def test_video_families_are_checked_before_image_families():
    # "xl" would otherwise make this SDXL; the video token wins.
    assert classify_checkpoint("wan2.2_xl_variant.safetensors")[0] == "wan_video"


def test_override_wins_over_every_rule():
    name = "exampleMergeV9.safetensors"
    assert classify_checkpoint(name, overrides={name: "sdxl"})[0] == "sdxl"
    assert classify_checkpoint("flux1-dev.safetensors", overrides={"flux1-dev.safetensors": "sd15"})[0] == "sd15"


def test_override_for_another_file_is_ignored():
    assert classify_checkpoint("flux1-dev.safetensors", overrides={"other.safetensors": "sd15"})[0] == "flux"
