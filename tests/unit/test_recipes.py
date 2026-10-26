"""Unit tests for workflows/recipes.py"""
from __future__ import annotations

import pytest
from ez_comfy.planner.intent import PipelineIntent
from ez_comfy.workflows.recipes import RECIPES, get_recipe, list_recipes, select_recipe


def test_all_recipes_have_builders():
    """Every recipe must have a builder name that matches a real function."""
    from ez_comfy.workflows.composer import _BUILDERS
    for r in RECIPES:
        assert r.builder in _BUILDERS, f"Recipe {r.id!r} builder {r.builder!r} not in composer"


def test_select_txt2img_basic():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="a cat",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=set(),
    )
    assert recipe.intent == PipelineIntent.TXT2IMG
    assert not recipe.requires_reference_image


def test_select_hires_fix_on_detail_prompt():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="ultra detailed 4k fantasy dragon, crisp, sharp",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=set(),
    )
    assert recipe.id == "txt2img_hires_fix"


def test_select_img2img_basic_without_controlnet():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.IMG2IMG,
        prompt="change the style",
        has_reference_image=True,
        has_mask=False,
        discovered_class_types=set(),  # no ControlNet
    )
    assert recipe.id == "img2img_basic"


def test_select_controlnet_when_available():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.IMG2IMG,
        prompt="keep the structure",
        has_reference_image=True,
        has_mask=False,
        discovered_class_types={"ControlNetLoader", "ControlNetApply"},
    )
    assert recipe.id == "img2img_controlnet_canny"


def test_select_inpaint():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.INPAINT,
        prompt="remove the background",
        has_reference_image=True,
        has_mask=True,
        discovered_class_types=set(),
    )
    assert recipe.id == "inpaint_basic"


def test_select_upscale_raises_without_capabilities():
    # UPSCALE has no capability-free fallback recipes — should raise, not silently return txt2img
    with pytest.raises(RuntimeError, match="No recipe available for intent='upscale'"):
        select_recipe(
            intent=PipelineIntent.UPSCALE,
            prompt="upscale this",
            has_reference_image=True,
            has_mask=False,
            discovered_class_types=set(),  # no upscale_model capability
        )


def test_select_upscale_succeeds_with_capability():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.UPSCALE,
        prompt="upscale this",
        has_reference_image=True,
        has_mask=False,
        discovered_class_types={"UpscaleModelLoader", "ImageUpscaleWithModel"},
    )
    assert recipe.intent == PipelineIntent.UPSCALE


def test_recipe_override():
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="anything",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=set(),
        recipe_override="txt2img_hires_fix",
    )
    assert recipe.id == "txt2img_hires_fix"


def test_unknown_recipe_override_raises():
    with pytest.raises(ValueError, match="Unknown recipe"):
        select_recipe(
            intent=PipelineIntent.TXT2IMG,
            prompt="test",
            has_reference_image=False,
            has_mask=False,
            discovered_class_types=set(),
            recipe_override="nonexistent_recipe_xyz",
        )


def test_get_recipe():
    r = get_recipe("txt2img_basic")
    assert r.id == "txt2img_basic"


def test_get_recipe_not_found():
    with pytest.raises(ValueError):
        get_recipe("not_a_real_recipe")


def test_list_recipes_returns_all():
    recipes = list_recipes()
    assert len(recipes) == len(RECIPES)


# ---------------------------------------------------------------------------
# Rejected list
# ---------------------------------------------------------------------------

def test_rejected_includes_capability_filtered():
    recipe, rejected = select_recipe(
        intent=PipelineIntent.IMG2IMG,
        prompt="change the style",
        has_reference_image=True,
        has_mask=False,
        discovered_class_types=set(),  # no ControlNet
    )
    assert recipe.id == "img2img_basic"
    rejected_ids = [r.id for r, _ in rejected]
    assert "img2img_controlnet_canny" in rejected_ids


def test_rejected_reason_mentions_capability():
    recipe, rejected = select_recipe(
        intent=PipelineIntent.IMG2IMG,
        prompt="keep structure",
        has_reference_image=True,
        has_mask=False,
        discovered_class_types=set(),
    )
    controlnet_rejections = [(r, reason) for r, reason in rejected if r.id == "img2img_controlnet_canny"]
    assert controlnet_rejections
    assert "controlnet" in controlnet_rejections[0][1].lower()


def test_rejected_empty_for_user_override():
    recipe, rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="anything",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=set(),
        recipe_override="txt2img_hires_fix",
    )
    assert recipe.id == "txt2img_hires_fix"
    assert rejected == []


def test_rejected_includes_lower_priority():
    recipe, rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="a simple cat",  # no photorealism/detail keywords
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=set(),
    )
    assert len(rejected) > 0
