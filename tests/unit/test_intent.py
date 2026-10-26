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
