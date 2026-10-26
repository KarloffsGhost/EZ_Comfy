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
