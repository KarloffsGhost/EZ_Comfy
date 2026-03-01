"""Unit tests for planner/param_resolver.py"""
from __future__ import annotations

import pytest
from ez_comfy.models.profiles import get_profile
from ez_comfy.planner.param_resolver import resolve_params


def _sdxl_profile():
    return get_profile("sdxl")


def test_resolve_defaults():
    profile = _sdxl_profile()
    params = resolve_params(profile)
    assert params.width == 1024
    assert params.height == 1024
    assert params.steps > 0
    assert params.cfg_scale > 0
    assert params.sampler != ""
    assert params.seed >= 0  # randomized
    assert params.sources["seed"] == "random"


def test_user_overrides_take_priority():
    profile = _sdxl_profile()
    params = resolve_params(
        profile,
        user_overrides={"steps": 42, "cfg": 8.5, "seed": 12345},
    )
    assert params.steps == 42
    assert params.cfg_scale == 8.5
    assert params.seed == 12345
    assert params.sources["steps"] == "user"
    assert params.sources["cfg"] == "user"
    assert params.sources["seed"] == "user"


def test_recipe_overrides_above_profile():
    profile = _sdxl_profile()
    params = resolve_params(
        profile,
        recipe_overrides={"steps": 12, "denoise_default": 0.3},
    )
    assert params.steps == 12
    assert params.denoise_strength == 0.3
    assert params.sources["steps"] == "recipe"


def test_user_beats_recipe():
    profile = _sdxl_profile()
    params = resolve_params(
        profile,
        recipe_overrides={"steps": 12},
        user_overrides={"steps": 25},
    )
    assert params.steps == 25
    assert params.sources["steps"] == "user"


def test_aspect_ratio_landscape():
    profile = _sdxl_profile()
    params = resolve_params(profile, aspect_ratio="16:9")
    assert params.width > params.height
    assert (params.width, params.height) in profile.resolution_buckets


def test_aspect_ratio_portrait():
    profile = _sdxl_profile()
    params = resolve_params(profile, aspect_ratio="9:16")
    assert params.height > params.width
    assert (params.width, params.height) in profile.resolution_buckets


def test_aspect_ratio_square():
    profile = _sdxl_profile()
    params = resolve_params(profile, aspect_ratio="1:1")
    assert params.width == params.height


def test_user_dimensions_snapped():
    profile = _sdxl_profile()
    params = resolve_params(
        profile,
        user_overrides={"width": 1000, "height": 1000},
    )
    assert (params.width, params.height) in profile.resolution_buckets
    assert params.sources["width"] == "resolution_bucket"


def test_random_seed_generated():
    profile = _sdxl_profile()
    p1 = resolve_params(profile, user_overrides={"seed": -1})
    p2 = resolve_params(profile, user_overrides={"seed": -1})
    # Random seeds should be valid (very unlikely to be equal)
    assert 0 <= p1.seed < 2**32
    assert 0 <= p2.seed < 2**32
