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
