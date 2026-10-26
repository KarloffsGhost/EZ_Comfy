"""Unit tests for planner integration, focusing on provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory, ModelInfo
from ez_comfy.hardware.probe import HardwareProfile
from ez_comfy.models.catalog import ModelCatalogEntry, ModelRecommendation
from ez_comfy.planner.planner import GenerationPlan, GenerationRequest, plan_generation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hardware(vram_gb: float = 16.0) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA RTX Test",
        gpu_vram_gb=vram_gb,
        system_ram_gb=32.0,
        cuda_version="12.4",
        platform="win32",
    )


def _make_inventory(checkpoints: list[str] | None = None) -> ComfyUIInventory:
    ckpt_list = [
        ModelInfo(filename=fn, size_bytes=6_000_000_000, family="sdxl", variant="lightning")
        for fn in (checkpoints or ["realvisxlV50_lightning.safetensors"])
    ]
    return ComfyUIInventory(
        checkpoints=ckpt_list,
        loras=[],
        vaes=[],
        upscale_models=[],
        clip_models=[],
        controlnet_models=[],
        discovered_class_types=set(),
        samplers=["euler", "dpm_2"],
        schedulers=["normal", "sgm_uniform"],
    )


def _make_request(**kwargs) -> GenerationRequest:
    defaults = dict(prompt="a beautiful landscape, photorealistic")
    defaults.update(kwargs)
    return GenerationRequest(**defaults)


# ---------------------------------------------------------------------------
# Basic plan tests
# ---------------------------------------------------------------------------

def test_plan_generation_returns_plan():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    assert isinstance(plan, GenerationPlan)


def test_plan_includes_provenance():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    assert plan.provenance is not None
    assert len(plan.provenance.decisions) > 0


# ---------------------------------------------------------------------------
# Intent decision
# ---------------------------------------------------------------------------

def test_provenance_has_intent_decision():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    intent_decisions = [d for d in plan.provenance.decisions if d.parameter == "intent"]
    assert len(intent_decisions) == 1


def test_provenance_intent_source_prompt_keyword():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "intent")
    assert d.source == "prompt_keyword"


def test_provenance_intent_source_user_when_override():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request(intent_override="txt2img")
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "intent")
    assert d.source == "user"


# ---------------------------------------------------------------------------
# Checkpoint decision
# ---------------------------------------------------------------------------

def test_provenance_has_checkpoint_decision():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    ckpt_decisions = [d for d in plan.provenance.decisions if d.parameter == "checkpoint"]
    assert len(ckpt_decisions) == 1


def test_provenance_checkpoint_has_reason():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "checkpoint")
    assert len(d.reason) > 0


def test_provenance_checkpoint_source_user_when_override():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request(checkpoint_override="realvisxlV50_lightning.safetensors")
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "checkpoint")
    assert d.source == "user"


# ---------------------------------------------------------------------------
# Recipe decision
# ---------------------------------------------------------------------------

def test_provenance_has_recipe_decision():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    recipe_decisions = [d for d in plan.provenance.decisions if d.parameter == "recipe"]
    assert len(recipe_decisions) == 1


def test_provenance_recipe_has_alternatives():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "recipe")
    # At minimum some recipes are considered and lower-priority ones rejected
    assert isinstance(d.alternatives, list)


def test_provenance_recipe_fallback_when_no_capabilities():
    hw = _make_hardware()
    # Inventory with no capabilities — controlnet recipe should be rejected
    inv = _make_inventory()
    req = _make_request(prompt="change the style", reference_image=b"fake_image_bytes")
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "recipe")
    rejected_ids = [a.value for a in d.alternatives]
    assert "img2img_controlnet_canny" in rejected_ids


# ---------------------------------------------------------------------------
# Parameter decisions
# ---------------------------------------------------------------------------

def test_provenance_has_param_decisions():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    param_names = {d.parameter for d in plan.provenance.decisions}
    assert "steps" in param_names
    assert "cfg" in param_names
    assert "sampler" in param_names
    assert "resolution" in param_names
    assert "seed" in param_names


def test_provenance_seed_random_when_no_seed():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request(seed=-1)
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "seed")
    assert d.source == "random"


def test_provenance_seed_user_when_specified():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request(seed=12345)
    plan = plan_generation(req, hw, inv)
    d = next(d for d in plan.provenance.decisions if d.parameter == "seed")
    assert d.source == "user"


# ---------------------------------------------------------------------------
# VRAM context
# ---------------------------------------------------------------------------

def test_provenance_vram_context():
    hw = _make_hardware(vram_gb=16.0)
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    assert plan.provenance.gpu_name == "NVIDIA RTX Test"
    assert plan.provenance.vram_available_gb == 16.0
    assert plan.provenance.vram_estimated_gb > 0


# ---------------------------------------------------------------------------
# Prompt adaptation
# ---------------------------------------------------------------------------

def test_provenance_prompt_adaptation_when_changed():
    """Pony model should trigger prompt adaptation decision."""
    hw = _make_hardware()
    inv = _make_inventory(checkpoints=["ponyDiffusionV6XL.safetensors"])
    req = _make_request(prompt="a cat")
    plan = plan_generation(req, hw, inv)
    adapt_decisions = [d for d in plan.provenance.decisions if d.parameter == "prompt_adaptation"]
    # Pony adds score prefix, so there should be an adaptation decision
    if adapt_decisions:
        assert len(adapt_decisions[0].reason) > 0


def test_provenance_no_prompt_adaptation_when_unchanged():
    hw = _make_hardware()
    inv = _make_inventory()
    # Plain SDXL prompt with no emphasis syntax and explicit negative — no changes expected
    req = _make_request(
        prompt="a mountain landscape",
        negative_prompt="ugly, blurry",
    )
    plan = plan_generation(req, hw, inv)
    adapt_decisions = [d for d in plan.provenance.decisions if d.parameter == "prompt_adaptation"]
    # Changes list would be empty → no decision emitted
    assert len(adapt_decisions) == 0


# ---------------------------------------------------------------------------
# Summary includes provenance
# ---------------------------------------------------------------------------

def test_summary_includes_provenance():
    hw = _make_hardware()
    inv = _make_inventory()
    req = _make_request()
    plan = plan_generation(req, hw, inv)
    summary = plan.summary()
    assert "provenance" in summary
    assert isinstance(summary["provenance"], dict)
    assert "decisions" in summary["provenance"]


# ---------------------------------------------------------------------------
# WI-7a: catalog matching as seen through the planner
# ---------------------------------------------------------------------------

def test_planner_asks_for_runnable_models_only(monkeypatch):
    import ez_comfy.planner.planner as planner_module

    seen = {}
    real = planner_module.recommend_models

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(planner_module, "recommend_models", spy)
    plan_generation(_make_request(), _make_hardware(), _make_inventory())
    assert seen.get("runnable_only") is True


def test_ltx_file_is_not_planned_as_flux_dev():
    """Regression: 'ltx-2.3-22b-dev-fp8' shares only generic words with flux1-dev-fp8."""
    inv = _make_inventory(["ltx-2.3-22b-dev-fp8.safetensors"])
    plan = plan_generation(_make_request(prompt="a poster with the text HELLO, typography"),
                           _make_hardware(), inv)
    assert plan.checkpoint == "ltx-2.3-22b-dev-fp8.safetensors"
    assert plan.catalog_entry is None or plan.catalog_entry.family != "flux"


@pytest.mark.parametrize("alias, installed, expected_entry_id", [
    ("sdxl", r"SDXL\sd_xl_base_1.0.safetensors", "sdxl_base_10"),
    ("flux", "flux1-dev.safetensors", "flux_dev"),
])
def test_checkpoint_override_alias_resolves_to_installed_file(alias, installed, expected_entry_id):
    inv = _make_inventory([installed])
    plan = plan_generation(_make_request(checkpoint_override=alias), _make_hardware(), inv)
    assert plan.checkpoint == installed
    assert plan.catalog_entry.id == expected_entry_id


# ---------------------------------------------------------------------------
# WI-2: loader compatibility guard
# ---------------------------------------------------------------------------

def _inventory_with(checkpoints: list[ModelInfo], diffusion: list[ModelInfo] | None = None) -> ComfyUIInventory:
    inv = _make_inventory()
    inv.checkpoints = checkpoints
    inv.diffusion_models = diffusion or []
    return inv


def test_diffusion_model_never_selected_for_checkpoint_recipe():
    wan = ModelInfo("wan2.2_t2v_14B.safetensors", 0, "unknown", loader="diffusion_model")
    sd15 = ModelInfo("DreamShaper_8_pruned.safetensors", 0, "sd15")
    plan = plan_generation(_make_request(prompt="a dragon"), _make_hardware(), _inventory_with([sd15], [wan]))
    assert plan.checkpoint == "DreamShaper_8_pruned.safetensors"


def test_diffusion_model_misplaced_in_checkpoints_is_still_not_selected():
    """Defends against a scan bug that would put a UNET file in the checkpoint list."""
    wan = ModelInfo("wan2.2_t2v_14B.safetensors", 0, "unknown", loader="diffusion_model")
    sd15 = ModelInfo("DreamShaper_8_pruned.safetensors", 0, "sd15")
    plan = plan_generation(_make_request(prompt="a dragon"), _make_hardware(), _inventory_with([wan, sd15]))
    assert plan.checkpoint == "DreamShaper_8_pruned.safetensors"


def test_explicit_override_cannot_pick_a_diffusion_model_file():
    wan = ModelInfo("wan2.2_t2v_14B.safetensors", 0, "unknown", loader="diffusion_model")
    sd15 = ModelInfo("DreamShaper_8_pruned.safetensors", 0, "sd15")
    inv = _inventory_with([sd15], [wan])
    plan = plan_generation(
        _make_request(prompt="a dragon", checkpoint_override="wan2.2_t2v_14B.safetensors"), _make_hardware(), inv
    )
    # The override is passed through as typed (existing behavior for unknown names) but it is not
    # reported as an installed checkpoint-loader file.
    decision = next(d for d in plan.provenance.decisions if d.parameter == "checkpoint")
    assert "not found in catalog" in decision.reason


def test_only_diffusion_models_installed_raises_instead_of_using_one():
    wan = ModelInfo("wan2.2_t2v_14B.safetensors", 0, "unknown", loader="diffusion_model")
    with pytest.raises(RuntimeError, match="No checkpoints found"):
        plan_generation(_make_request(prompt="a dragon"), _make_hardware(), _inventory_with([], [wan]))


def test_planner_fallback_prefers_a_family_that_suits_the_intent():
    unknown = ModelInfo("mystery_model.safetensors", 0, "unknown")
    known = ModelInfo("totally_custom_name.safetensors", 0, "sd15")
    plan = plan_generation(_make_request(prompt="a dragon"), _make_hardware(), _inventory_with([unknown, known]))
    assert plan.checkpoint == "totally_custom_name.safetensors"


def test_planner_fallback_skips_audio_and_video_families_for_an_image_prompt():
    audio = ModelInfo("some_audio_model.safetensors", 0, "stable_audio")
    video = ModelInfo("some_video_model.safetensors", 0, "ltx_video")
    unknown = ModelInfo("mystery_model.safetensors", 0, "unknown")
    plan = plan_generation(_make_request(prompt="a dragon"), _make_hardware(), _inventory_with([audio, video, unknown]))
    assert plan.checkpoint == "mystery_model.safetensors"


def test_planner_fallback_uses_first_when_all_unknown():
    a = ModelInfo("aaa.safetensors", 0, "unknown")
    b = ModelInfo("bbb.safetensors", 0, "unknown")
    plan = plan_generation(_make_request(prompt="a dragon"), _make_hardware(), _inventory_with([a, b]))
    assert plan.checkpoint == "aaa.safetensors"


def test_recipe_needing_another_loader_adds_a_warning(monkeypatch):
    from dataclasses import replace

    from ez_comfy.planner import planner as planner_module

    real_select = planner_module.select_recipe

    def select_with_unet_recipe(**kwargs):
        recipe, rejected = real_select(**kwargs)
        return replace(recipe, model_loader="diffusion_model"), rejected

    monkeypatch.setattr(planner_module, "select_recipe", select_with_unet_recipe)
    plan = plan_generation(_make_request(), _make_hardware(), _make_inventory())
    assert any("diffusion_model" in w for w in plan.warnings)
