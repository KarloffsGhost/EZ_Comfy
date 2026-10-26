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
    dupes = sorted({fn for fn in filenames if filenames.count(fn) > 1})
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


# ---------------------------------------------------------------------------
# WI-7a: one installed file maps to at most one catalog entry
# ---------------------------------------------------------------------------
import logging

from ez_comfy.models.catalog import (
    ModelCatalogEntry,
    _model_name_matches,
    match_installed_file,
    resolve_installed_filename,
)


def _entry(entry_id: str, filename: str, family: str = "sdxl") -> ModelCatalogEntry:
    base = next(e for e in MODEL_CATALOG if e.family == family)
    return ModelCatalogEntry(
        id=entry_id, name=entry_id, family=family, variant=None, source="test",
        filename=filename, size_gb=1.0, vram_min_gb=1.0, vram_recommended_gb=2.0,
        strengths=[], weaknesses=[], best_for=[], style_tags=[],
        settings=base.settings, prompt_syntax=base.prompt_syntax,
    )


def test_self_resolution_all_entries():
    for entry in MODEL_CATALOG:
        resolved = match_installed_file(entry.filename + ".safetensors", MODEL_CATALOG)
        assert resolved is entry, f"{entry.id}: own filename resolved to {getattr(resolved, 'id', None)}"


def test_cross_matching_siblings_resolve_to_owner_for_variant_filenames():
    """Files that only *contain* an entry's token (not exact) must still go to the right entry.

    The old per-entry rule let a sibling claim these; self-resolution alone cannot catch that
    because an entry's own filename is always an exact match.
    """
    confused = {
        b.id for a in MODEL_CATALOG for b in MODEL_CATALOG
        if a is not b and _model_name_matches(a.filename, b.filename + ".safetensors")
    }
    assert confused, "expected the old rule to still report cross-matches (guards this test)"
    by_id = {e.id: e for e in MODEL_CATALOG}
    for owner_id in confused:
        owner = by_id[owner_id]
        variant = owner.filename + "_v2.safetensors"
        assert match_installed_file(variant, MODEL_CATALOG) is owner, (owner_id, variant)


@pytest.mark.parametrize("installed, expected_id", [
    ("flux1-dev.safetensors", "flux_dev"),
    ("flux1-schnell.safetensors", "flux_schnell"),
    ("flux1-dev-fp8.safetensors", "flux_dev_fp8"),
    ("flux1-schnell-fp8.safetensors", "flux_schnell_fp8"),
])
def test_flux_dev_vs_schnell_not_confused(installed, expected_id):
    assert match_installed_file(installed, MODEL_CATALOG).id == expected_id


def test_flux_dev_installed_only_marks_dev_entries():
    inv = _inventory(["flux1-dev.safetensors"])
    by_id = {e.id: e for e in MODEL_CATALOG}
    assert resolve_installed_filename(by_id["flux_dev"], inv) == "flux1-dev.safetensors"
    for other in ("flux_schnell", "flux_dev_fp8", "flux_schnell_fp8", "flux_hyper_8step", "flux_merged_gguf_q8"):
        assert resolve_installed_filename(by_id[other], inv) is None, other


def test_pony_vs_pony_turbo():
    assert match_installed_file("ponyDiffusionV6XL.safetensors", MODEL_CATALOG).id == "pony_diffusion_v6_xl"
    assert match_installed_file("ponyDiffusionV6XL_turbo.safetensors", MODEL_CATALOG).id == "pony_diffusion_v6_xl_turbo"


def test_realvis_fuzzy_variant_still_matches():
    """Non-colliding names keep the old fuzzy behaviour."""
    entry = match_installed_file("realvisxlV50_v50LightningBakedvae.safetensors", MODEL_CATALOG)
    assert entry is not None and entry.id == "realvis_xl_v50_lightning"


def test_tie_returns_none_and_logs(caplog):
    twins = [_entry("twin_a", "sameModelName"), _entry("twin_b", "sameModelName")]
    with caplog.at_level(logging.WARNING, logger="ez_comfy.models.catalog"):
        assert match_installed_file("sameModelName.safetensors", twins) is None
    assert "twin_a" in caplog.text and "twin_b" in caplog.text


def test_unrelated_file_matches_nothing():
    assert match_installed_file("zzz_nonexistent_model_xyz.safetensors", MODEL_CATALOG) is None


def test_new_field_defaults():
    entry = _entry("defaults", "defaultsModel")
    assert entry.loader == "checkpoint"
    assert entry.requires_text_encoders == []
    assert entry.requires_vae is None
    assert entry.execution_supported is True
    assert entry.verified_on is None


def test_entries_without_workflow_builder_are_flagged():
    unsupported = {e.id for e in MODEL_CATALOG if not e.execution_supported}
    assert unsupported == {"animatediff_lightning", "mochi_preview", "wan_video_1_3b"}


def test_unsupported_entries_not_selected_for_generation(monkeypatch):
    import ez_comfy.models.catalog as cat
    # Make the video-family entries reachable so the filter itself is what is under test.
    monkeypatch.setattr(cat, "_intent_compatible_families",
                        lambda intent: {"svd", "animatediff", "mochi", "wan_video"})
    hw = _hardware(16.0)
    inv = _inventory()
    planner_view = recommend_models("a video of waves", "video", hw, inv, top_n=20, runnable_only=True)
    listing_view = recommend_models("a video of waves", "video", hw, inv, top_n=20)
    assert planner_view and all(r.entry.execution_supported for r in planner_view)
    unrunnable = [r for r in listing_view if not r.entry.execution_supported]
    assert unrunnable, "listing view should still show unsupported entries"
    assert all("not yet runnable" in " ".join(r.warnings) for r in unrunnable)


@pytest.mark.parametrize("token, expected_id", [
    ("flux", "flux_dev"),
    ("pony", "pony_diffusion_v6_xl"),
    ("sdxl", "sdxl_base_10"),
    ("flux1-dev", "flux_dev"),
    ("realvisxlV50_lightning", "realvis_xl_v50_lightning"),
])
def test_override_alias_tokens_resolve_like_before(token, expected_id):
    """checkpoint_override aliases (CLI --checkpoint) must keep reaching the same entry."""
    assert find_catalog_entry(token).id == expected_id


@pytest.mark.parametrize("installed, expected_id", [
    ("flux1-dev-fp8-e4m3fn.safetensors", "flux_dev_fp8"),
    (r"SDXL\sd_xl_base_1.0.safetensors", "sdxl_base_10"),
    ("DreamShaper_8_pruned.safetensors", "dreamshaper_8"),
    ("stable_audio_open_1.0.safetensors", "stable_audio_open_10"),
])
def test_real_world_variant_names(installed, expected_id):
    assert match_installed_file(installed, MODEL_CATALOG).id == expected_id


@pytest.mark.parametrize("installed", [
    "ltx-2.3-22b-dev-fp8.safetensors",   # shares only generic words ("dev", "fp8") with flux_dev_fp8
    "exampleMergeV9.safetensors",
    "v1-5-pruned-emaonly-fp16.safetensors",
])
def test_unrelated_real_files_match_nothing(installed):
    assert match_installed_file(installed, MODEL_CATALOG) is None


def test_generic_words_alone_do_not_mark_flux_installed():
    inv = _inventory(["ltx-2.3-22b-dev-fp8.safetensors"])
    by_id = {e.id: e for e in MODEL_CATALOG}
    assert resolve_installed_filename(by_id["flux_dev_fp8"], inv) is None
    recs = recommend_models("a poster with the text HELLO", "txt2img", _hardware(16.0), inv, top_n=40)
    assert not any(r.installed and r.entry.family == "flux" for r in recs)


def test_upscaler_resolution_uses_upscale_models():
    inv = _inventory(upscalers=["RealESRGAN_x4plus.safetensors", "4x-UltraSharp.pth"])
    by_id = {e.id: e for e in MODEL_CATALOG}
    assert resolve_installed_filename(by_id["realesrgan_x4plus"], inv) == "RealESRGAN_x4plus.safetensors"
    assert resolve_installed_filename(by_id["4x_ultrasharp"], inv) == "4x-UltraSharp.pth"
    # checkpoint entries never claim upscaler files
    assert resolve_installed_filename(by_id["realvis_xl_v50_lightning"], inv) is None


def test_recommend_work_scales_with_files_times_entries(monkeypatch):
    """Guard against re-ranking every file once per entry (was ~45x slower at 300 files)."""
    import ez_comfy.models.catalog as cat
    cat._OWNER_CACHE.clear()
    calls = {"n": 0}
    real = cat._match_rank

    def counting(expected, installed):
        calls["n"] += 1
        return real(expected, installed)

    monkeypatch.setattr(cat, "_match_rank", counting)
    files = [f"unrelated_model_{i}.safetensors" for i in range(300)]
    recommend_models("a cat", "txt2img", _hardware(16.0), _inventory(files), top_n=5)
    ranked_entries = len([e for e in MODEL_CATALOG if e.family != "upscaler"])
    assert calls["n"] <= 2 * len(files) * ranked_entries, calls["n"]


def test_tie_warning_logged_once_per_recommend(caplog):
    import ez_comfy.models.catalog as cat
    cat._OWNER_CACHE.clear()
    twins = [_entry("twin_a", "sameModelName"), _entry("twin_b", "sameModelName")]
    inv = _inventory(["sameModelName.safetensors"])
    with caplog.at_level(logging.WARNING, logger="ez_comfy.models.catalog"):
        for entry in twins * 5:
            resolve_installed_filename(entry, inv, catalog=twins)
    assert caplog.text.count("equally well") == 1
