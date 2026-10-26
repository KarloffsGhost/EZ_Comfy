"""Tests for photo_realism_v1 recipe and workflow builder."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ez_comfy.planner.intent import PipelineIntent
from ez_comfy.planner.param_resolver import ResolvedParams
from ez_comfy.workflows.recipes import select_recipe


# ---------------------------------------------------------------------------
# Recipe selection
# ---------------------------------------------------------------------------

def _class_types(*extra: str) -> set[str]:
    return {"CheckpointLoaderSimple", "KSampler", *extra}


def test_photo_realism_selected_for_portrait_prompt():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="close-up portrait of a woman, natural lighting",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=_class_types(),
    )
    assert recipe.id == "photo_realism_v1"


def test_photo_realism_selected_for_photorealistic_prompt():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="photorealistic landscape, 35mm photography",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=_class_types(),
    )
    assert recipe.id == "photo_realism_v1"


def test_photo_realism_selected_for_cinematic_prompt():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="cinematic shot of a forest at dusk",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=_class_types(),
    )
    assert recipe.id == "photo_realism_v1"


def test_hires_fix_selected_for_detail_prompt_without_photo_keywords():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="detailed fantasy dragon, ultra sharp",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=_class_types(),
    )
    assert recipe.id == "txt2img_hires_fix"


def test_non_photo_prompt_does_not_select_photo_realism():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="a red apple on a table",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=_class_types(),
    )
    assert recipe.id != "photo_realism_v1"


def test_photo_realism_selectable_by_override():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="a cat",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=_class_types(),
        recipe_override="photo_realism_v1",
    )
    assert recipe.id == "photo_realism_v1"


# ---------------------------------------------------------------------------
# Builder node structure
# ---------------------------------------------------------------------------

def _make_plan(width: int = 1024, height: int = 1024, steps: int = 20) -> MagicMock:
    plan = MagicMock()
    plan.checkpoint = "realvisxlV50_v50LightningBakedvae.safetensors"
    plan.prompt = "a portrait of a woman, photorealistic"
    plan.negative_prompt = "ugly, deformed"
    plan.loras = []
    plan.vae_override = None
    plan.params = ResolvedParams(
        steps=steps, cfg_scale=7.0, sampler="euler", scheduler="normal",
        width=width, height=height, clip_skip=1, denoise_strength=0.75,
        seed=42, batch_size=1, sources={},
    )
    return plan


def test_photo_realism_v1_builds_without_error():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan()
    nodes = build_photo_realism_v1(plan)
    assert isinstance(nodes, dict)
    assert len(nodes) > 0


def test_photo_realism_v1_has_two_ksampler_passes():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan()
    nodes = build_photo_realism_v1(plan)
    ksamplers = [n for n in nodes.values() if n["class_type"] == "KSampler"]
    assert len(ksamplers) == 2, f"Expected 2 KSampler nodes, got {len(ksamplers)}"


def test_photo_realism_v1_pass1_is_reduced_resolution():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan(width=1024, height=1024)
    nodes = build_photo_realism_v1(plan)
    latent_node = nodes["4"]
    assert latent_node["class_type"] == "EmptyLatentImage"
    p1_w = latent_node["inputs"]["width"]
    p1_h = latent_node["inputs"]["height"]
    assert p1_w < 1024, f"Pass 1 width {p1_w} should be < 1024"
    assert p1_h < 1024, f"Pass 1 height {p1_h} should be < 1024"
    assert p1_w >= 512
    assert p1_h >= 512


def test_photo_realism_v1_uses_pixel_space_upscale():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan()
    nodes = build_photo_realism_v1(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "ImageScale" in class_types, "Must use ImageScale for pixel-space upscale"
    assert "LatentUpscale" not in class_types, "Should not use LatentUpscale"


def test_photo_realism_v1_pixel_upscale_targets_full_resolution():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan(width=1024, height=1024)
    nodes = build_photo_realism_v1(plan)
    scale_node = nodes["10"]
    assert scale_node["class_type"] == "ImageScale"
    assert scale_node["inputs"]["width"] == 1024
    assert scale_node["inputs"]["height"] == 1024
    assert scale_node["inputs"]["upscale_method"] == "lanczos"


def test_photo_realism_v1_pass2_has_low_denoise():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan()
    nodes = build_photo_realism_v1(plan)
    pass2 = nodes["12"]
    assert pass2["class_type"] == "KSampler"
    assert pass2["inputs"]["denoise"] < 0.6, "Pass 2 denoise should be low (preserve composition)"
    assert pass2["inputs"]["denoise"] > 0.2, "Pass 2 denoise should not be too low (needs to add detail)"


def test_photo_realism_v1_saves_with_photo_prefix():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan()
    nodes = build_photo_realism_v1(plan)
    save_nodes = [n for n in nodes.values() if n["class_type"] == "SaveImage"]
    assert len(save_nodes) == 1
    assert "photo" in save_nodes[0]["inputs"]["filename_prefix"]


def test_photo_realism_v1_with_vae_override():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan()
    plan.vae_override = "vae-ft-mse-840000-ema-pruned.safetensors"
    nodes = build_photo_realism_v1(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "VAELoader" in class_types


def test_photo_realism_v1_with_loras():
    from ez_comfy.workflows.txt2img import build_photo_realism_v1
    plan = _make_plan()
    plan.loras = [("detail_tweaker_xl.safetensors", 0.5)]
    nodes = build_photo_realism_v1(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "LoraLoader" in class_types


def test_photo_realism_v1_dispatches_via_composer():
    from unittest.mock import MagicMock
    from ez_comfy.workflows.composer import compose_workflow
    from ez_comfy.workflows.recipes import get_recipe

    plan = _make_plan()
    plan.recipe = get_recipe("photo_realism_v1")
    nodes = compose_workflow(plan)
    assert isinstance(nodes, dict)
    assert len(nodes) > 0
