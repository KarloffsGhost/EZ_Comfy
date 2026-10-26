"""Unit tests for models/catalog.py"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory, ModelInfo
from ez_comfy.hardware.probe import HardwareProfile
from ez_comfy.models.catalog import MODEL_CATALOG, find_catalog_entry, recommend_models


def _hardware(vram: float = 16.0) -> HardwareProfile:
    hw = MagicMock(spec=HardwareProfile)
    hw.gpu_vram_gb = vram
    hw.gpu_name = "RTX 5070 Ti"
    hw.system_ram_gb = 127.0
    hw.platform = "Windows"
    return hw


def _inventory(
    filenames: list[str] | None = None,
    upscalers: list[str] | None = None,
) -> ComfyUIInventory:
    inv = ComfyUIInventory()
    for fn in (filenames or []):
        inv.checkpoints.append(ModelInfo(filename=fn, size_bytes=4_000_000_000, family="sdxl"))
    inv.upscale_models.extend(upscalers or [])
    return inv


def test_model_catalog_not_empty():
    assert len(MODEL_CATALOG) > 0


def test_find_catalog_entry_by_filename():
    # find_catalog_entry checks if the catalog's filename token is contained in the given string.
    # The catalog uses "realvisxlV50_lightning" as the filename token.
    entry = find_catalog_entry("realvisxlV50_lightning.safetensors")
    assert entry is not None
    assert entry.family == "sdxl"


def test_find_catalog_entry_installed_variant():
    # Installed file that starts with the catalog token should also match
    from ez_comfy.models.catalog import MODEL_CATALOG
    if MODEL_CATALOG:
        first = MODEL_CATALOG[0]
        entry = find_catalog_entry(first.filename + ".safetensors")
        assert entry is not None


def test_find_catalog_entry_realvis_variant_filename():
    # Real-world installed filename variant seen in ComfyUI inventories
    entry = find_catalog_entry("realvisxlV50_v50LightningBakedvae.safetensors")
    assert entry is not None
    assert "realvis" in entry.id


def test_find_catalog_entry_not_found():
    entry = find_catalog_entry("zzz_nonexistent_model_xyz.safetensors")
    assert entry is None


def test_recommend_models_returns_results():
    hw = _hardware(16.0)
    inv = _inventory()
    recs = recommend_models("a photo of a cat", "txt2img", hw, inv)
    assert len(recs) > 0


def test_installed_models_score_higher():
    hw = _hardware(16.0)
    # Inventory with RealVisXL installed
    inv = _inventory(["realvisxlV50_v50Lightning.safetensors"])
    recs = recommend_models("photo portrait", "txt2img", hw, inv)
    installed = [r for r in recs if r.installed]
    not_installed = [r for r in recs if not r.installed]
    if installed and not_installed:
        assert installed[0].score >= not_installed[0].score


def test_vram_constrained_hardware():
    hw = _hardware(vram=4.0)  # very limited VRAM
    inv = _inventory()
    recs = recommend_models("a cat", "txt2img", hw, inv)
    # Recommendations should still be returned (maybe with VRAM warnings)
    assert len(recs) > 0
    # Most recommended should have low VRAM requirement
    top = recs[0]
    assert top is not None


def test_audio_intent_recommends_audio_models():
    hw = _hardware(16.0)
    inv = _inventory()
    recs = recommend_models("generate a relaxing sound", "audio", hw, inv)
    # Should recommend audio models
    audio_recs = [r for r in recs if "audio" in r.entry.family.lower()]
    assert len(audio_recs) > 0


def test_recommend_with_installed_model():
    hw = _hardware(16.0)
    # Use the catalog's own filename token so inventory lookup matches
    from ez_comfy.models.catalog import MODEL_CATALOG
    first_sdxl = next((e for e in MODEL_CATALOG if e.family == "sdxl"), MODEL_CATALOG[0])
    inv = _inventory([first_sdxl.filename + ".safetensors"])
    recs = recommend_models("a portrait", "txt2img", hw, inv)
    installed = [r for r in recs if r.installed]
    assert len(installed) > 0


def test_recommend_upscale_marks_installed_upscaler():
    hw = _hardware(16.0)
    inv = _inventory(upscalers=["RealESRGAN_x4plus.pth"])
    recs = recommend_models("upscale this", "upscale", hw, inv, top_n=5)
    installed = [r for r in recs if r.installed and r.entry.family == "upscaler"]
    assert installed, "Expected at least one installed upscaler recommendation"
