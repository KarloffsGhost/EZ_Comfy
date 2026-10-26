"""Unit tests for hardware/comfyui_inventory.py"""
from __future__ import annotations

import pytest
from ez_comfy.hardware.comfyui_inventory import NODE_CAPABILITY_MAP, has_capability


def test_has_controlnet_capability():
    class_types = {"ControlNetLoader", "ControlNetApply", "KSampler"}
    assert has_capability("controlnet", class_types)


def test_missing_capability():
    assert not has_capability("controlnet", {"KSampler", "CLIPTextEncode"})


def test_upscale_model_capability():
    class_types = {"UpscaleModelLoader", "ImageUpscaleWithModel"}
    assert has_capability("upscale_model", class_types)


def test_svd_capability():
    class_types = {"ImageOnlyCheckpointLoader", "SVD_img2vid_Conditioning"}
    assert has_capability("svd", class_types)


def test_stable_audio_capability():
    class_types = {"EmptyLatentAudio", "VAEDecodeAudio"}
    assert has_capability("stable_audio", class_types)


def test_unknown_capability_returns_false():
    assert not has_capability("nonexistent_cap_xyz", {"SomeNode"})


def test_all_capability_keys_are_strings():
    for key in NODE_CAPABILITY_MAP:
        assert isinstance(key, str)
        assert len(key) > 0


def test_partial_capability_match():
    """has_capability returns True if ANY node type from the list is present."""
    # Only one of the two ControlNet nodes present
    assert has_capability("controlnet", {"ControlNetLoader"})
