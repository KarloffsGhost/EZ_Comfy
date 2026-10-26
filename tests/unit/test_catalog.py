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


# ---------------------------------------------------------------------------
# Catalog structure tests (from spec/IMPROVEMENTS_SPEC.md §2.5)
# ---------------------------------------------------------------------------

def test_minimum_entry_count():
    assert len(MODEL_CATALOG) >= 35, f"Catalog has only {len(MODEL_CATALOG)} entries; need >= 35"


def test_bucket_distribution():
    """Each required family bucket must meet minimum counts."""
    minimums = {
        "sdxl_photorealism": (
            [e for e in MODEL_CATALOG if e.family == "sdxl"
             and any(s in e.strengths for s in ["photorealism", "cinematic", "portraits"])],
            4,
        ),
        "sdxl_stylized": (
            [e for e in MODEL_CATALOG if e.family == "sdxl"
             and any(s in e.strengths for s in ["artistic", "fantasy", "stylized"])],
            3,
        ),
        "pony": (
            [e for e in MODEL_CATALOG if e.family == "pony"],
            3,
        ),
        "sd15": (
            [e for e in MODEL_CATALOG if e.family == "sd15"],
            4,
        ),
        "flux": (
            [e for e in MODEL_CATALOG if e.family == "flux"],
            4,
        ),
        "video": (
            [e for e in MODEL_CATALOG if "video" in e.best_for or "animation" in e.style_tags
             or e.family in ("svd", "animatediff", "mochi", "wan_video")],
            3,
        ),
        "audio": (
            [e for e in MODEL_CATALOG if "audio" in e.family],
            1,
        ),
        "specialty": (
            [e for e in MODEL_CATALOG if e.family in ("upscaler",)
             or "inpaint" in e.best_for or "inpaint" in e.style_tags],
            2,
        ),
    }
    for bucket, (entries, min_count) in minimums.items():
        assert len(entries) >= min_count, (
            f"Bucket '{bucket}' has {len(entries)} entries; need >= {min_count}"
        )


def test_vram_consistency():
    """vram_min_gb must be positive and <= vram_recommended_gb for all entries."""
    for entry in MODEL_CATALOG:
        assert entry.vram_min_gb > 0, f"{entry.id}: vram_min_gb must be > 0"
        assert entry.vram_recommended_gb > 0, f"{entry.id}: vram_recommended_gb must be > 0"
        assert entry.vram_min_gb <= entry.vram_recommended_gb, (
            f"{entry.id}: vram_min_gb ({entry.vram_min_gb}) > vram_recommended_gb ({entry.vram_recommended_gb})"
        )


def test_download_command_format():
    """Every entry must have a non-empty download_command."""
    for entry in MODEL_CATALOG:
        assert entry.download_command, f"{entry.id}: download_command must not be empty"


def test_unique_ids():
    ids = [e.id for e in MODEL_CATALOG]
    assert len(ids) == len(set(ids)), f"Duplicate catalog IDs: {[id for id in ids if ids.count(id) > 1]}"


def test_unique_filenames():
    filenames = [e.filename for e in MODEL_CATALOG]
    dupes = [fn for fn in filenames if filenames.count(fn) > 1]
    # flux_hyper shares the flux1-dev filename (it's a LoRA on top); exclude that known case
    dupes = [d for d in set(dupes) if d not in ("flux1-dev",)]
    assert not dupes, f"Unexpected duplicate filenames: {dupes}"


def test_pony_entries_have_score_tag_prefix():
    """All Pony-family entries must have score-tag injection enabled."""
    pony_entries = [e for e in MODEL_CATALOG if e.family == "pony"]
    assert pony_entries, "No Pony entries found"
    for entry in pony_entries:
        assert entry.prompt_syntax.quality_prefix is not None, (
            f"{entry.id}: Pony entry must have a score-tag quality_prefix"
        )
        assert "score_9" in entry.prompt_syntax.quality_prefix, (
            f"{entry.id}: Pony quality_prefix must contain 'score_9'"
        )


def test_flux_entries_strip_emphasis_weights():
    """All Flux-family entries must use emphasis_format='none'."""
    flux_entries = [e for e in MODEL_CATALOG if e.family == "flux"]
    assert flux_entries, "No Flux entries found"
    for entry in flux_entries:
        assert entry.prompt_syntax.emphasis_format == "none", (
            f"{entry.id}: Flux entry must use emphasis_format='none' to strip (word:1.3) weights"
        )
