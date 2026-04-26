"""Unit tests for planner/intent.py"""
from __future__ import annotations

import pytest
from ez_comfy.planner.intent import PipelineIntent, detect_intent


@pytest.mark.parametrize("prompt,has_ref,has_mask,expected", [
    ("a beautiful sunset",                  False, False, PipelineIntent.TXT2IMG),
    ("enhance this photo",                  True,  False, PipelineIntent.IMG2IMG),
    ("remove the person",                   True,  True,  PipelineIntent.INPAINT),
    ("upscale this image 4x",               True,  False, PipelineIntent.UPSCALE),
    ("animate this photo",                  True,  False, PipelineIntent.VIDEO),
    ("generate a scary sound effect",       False, False, PipelineIntent.AUDIO),
    ("create a video from this image",      True,  False, PipelineIntent.VIDEO),
    ("make this image high resolution",      True,  False, PipelineIntent.UPSCALE),
    ("paint this in watercolor style",      True,  False, PipelineIntent.IMG2IMG),
])
def test_detect_intent(prompt, has_ref, has_mask, expected):
    result = detect_intent(prompt, has_reference_image=has_ref, has_mask=has_mask)
    assert result == expected, f"detect_intent({prompt!r}) = {result}, expected {expected}"


def test_inpaint_requires_mask():
    # mask + ref → inpaint
    assert detect_intent("edit this", True, True) == PipelineIntent.INPAINT


def test_txt2img_no_image():
    # No image → txt2img regardless of prompt
    assert detect_intent("photo of a cat", False, False) == PipelineIntent.TXT2IMG


# ---------------------------------------------------------------------------
# Edge cases from spec/IMPROVEMENTS_SPEC.md §6.5
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,has_img,has_mask,expected", [
    # Baseline
    ("a cyberpunk city",                           False, False, PipelineIntent.TXT2IMG),
    # Upscale requires an image — without one falls to txt2img
    ("upscale and enhance this",                   True,  False, PipelineIntent.UPSCALE),
    ("upscale and enhance this",                   False, False, PipelineIntent.TXT2IMG),
    # Video signals
    ("animate a video of a dog",                   False, False, PipelineIntent.VIDEO),
    # "video game" must NOT trigger video intent (false positive guard)
    ("a video game character portrait",            False, False, PipelineIntent.TXT2IMG),
    # Inpaint requires explicit mask even when prompt implies it
    ("remove the person in the masked area",       True,  True,  PipelineIntent.INPAINT),
    # Replace-sky without mask → img2img, not inpaint
    ("replace the sky",                            True,  False, PipelineIntent.IMG2IMG),
    # Audio
    ("a song about robots",                        False, False, PipelineIntent.AUDIO),
    # "sound of waves" is an image description, not audio generation
    ("a portrait, sound of waves in background",   False, False, PipelineIntent.TXT2IMG),
    # "high resolution" without image stays txt2img (upscale keywords need image)
    ("high-resolution detailed render",            False, False, PipelineIntent.TXT2IMG),
])
def test_intent_edge_cases(prompt, has_img, has_mask, expected):
    result = detect_intent(prompt, has_reference_image=has_img, has_mask=has_mask)
    assert result == expected, (
        f"detect_intent({prompt!r}, img={has_img}, mask={has_mask}) "
        f"returned {result!r}, expected {expected!r}"
    )
