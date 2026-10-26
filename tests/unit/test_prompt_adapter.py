"""Unit tests for planner/prompt_adapter.py"""
from __future__ import annotations

import pytest
from ez_comfy.models.profiles import PromptSyntax
from ez_comfy.planner.prompt_adapter import adapt_prompt, get_style_preset


def _sdxl_syntax() -> PromptSyntax:
    return PromptSyntax(
        emphasis_format="weighted",
        quality_prefix=None,
        quality_suffix=None,
        negative_required=True,
        default_negative="ugly, deformed",
    )


def _flux_syntax() -> PromptSyntax:
    return PromptSyntax(
        emphasis_format="none",
        quality_prefix=None,
        quality_suffix=None,
        negative_required=False,
        default_negative="",
    )


def _pony_syntax() -> PromptSyntax:
    return PromptSyntax(
        emphasis_format="weighted",
        quality_prefix="score_9, score_8_up, score_7_up, ",
        quality_suffix=None,
        negative_required=True,
        default_negative="score_4, score_5, score_6",
    )


def test_pony_gets_score_prefix():
    pos, neg = adapt_prompt(
        user_prompt="a cat",
        negative_prompt="",
        syntax=_pony_syntax(),
        style_preset=None,
        family="pony",
        auto_negative=True,
    )
    assert pos.startswith("score_9")


def test_flux_strips_emphasis():
    pos, neg = adapt_prompt(
        user_prompt="a (beautiful:1.3) cat",
        negative_prompt="bad quality",
        syntax=_flux_syntax(),
        style_preset=None,
        family="flux",
        auto_negative=False,
    )
    assert "(beautiful:1.3)" not in pos
    assert "beautiful" in pos
    # Flux ignores negatives
    assert neg == ""


def test_sdxl_auto_negative_added():
    pos, neg = adapt_prompt(
        user_prompt="a cat",
        negative_prompt="",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=True,
    )
    assert len(neg) > 0


def test_user_negative_preserved():
    pos, neg = adapt_prompt(
        user_prompt="a cat",
        negative_prompt="blurry",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=False,
    )
    assert "blurry" in neg


def test_style_preset_photographic():
    preset = get_style_preset("photographic")
    assert preset is not None
    pos, neg = adapt_prompt(
        user_prompt="a portrait",
        negative_prompt="",
        syntax=_sdxl_syntax(),
        style_preset=preset,
        family="sdxl",
        auto_negative=False,
    )
    assert len(pos) > len("a portrait")


def test_unknown_style_preset_returns_none():
    preset = get_style_preset("nonexistent_style_xyz")
    assert preset is None


def test_no_style_preset_no_crash():
    pos, neg = adapt_prompt(
        user_prompt="a sunset",
        negative_prompt="",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=False,
    )
    assert "a sunset" in pos


def test_domain_pack_portrait_adds_positive_and_negative():
    pos, neg = adapt_prompt(
        user_prompt="close-up portrait of a woman",
        negative_prompt="",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=True,
    )
    assert "natural skin texture" in pos
    assert "catchlight in eyes" in pos
    assert "plastic skin" in neg
    assert "uncanny face" in neg


def test_domain_pack_text_suppression_for_flux_keeps_negative_empty():
    pos, neg = adapt_prompt(
        user_prompt="red golf ball product shot with no logo text",
        negative_prompt="",
        syntax=_flux_syntax(),
        style_preset=None,
        family="flux",
        auto_negative=True,
    )
    assert "clean unbranded surface" in pos
    assert neg == ""


def test_domain_pack_golf_ball_terms_are_not_duplicated():
    pos, neg = adapt_prompt(
        user_prompt="photorealistic golf ball, uniform circular dimples",
        negative_prompt="",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=True,
    )
    assert pos.lower().count("uniform circular dimples") == 1
    assert "even dimple spacing" in pos
    assert "irregular dimples" in neg


def test_domain_negative_added_when_user_negative_exists_even_if_auto_off():
    pos, neg = adapt_prompt(
        user_prompt="portrait headshot of a person",
        negative_prompt="blurry",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=False,
    )
    assert "natural skin texture" in pos
    assert "blurry" in neg
    assert "plastic skin" in neg


def test_domain_negative_not_added_when_auto_off_and_negative_blank():
    pos, neg = adapt_prompt(
        user_prompt="portrait of a person",
        negative_prompt="",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=False,
    )
    assert "natural skin texture" in pos
    assert neg == ""


def test_keyword_match_uses_word_boundaries_for_single_words():
    pos, neg = adapt_prompt(
        user_prompt="manual focus product photography",
        negative_prompt="",
        syntax=_sdxl_syntax(),
        style_preset=None,
        family="sdxl",
        auto_negative=True,
    )
    assert "natural skin texture" not in pos
    assert "plastic skin" not in neg
