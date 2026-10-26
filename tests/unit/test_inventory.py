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


# ---------------------------------------------------------------------------
# Combo parsing: both /object_info shapes are live in ComfyUI 0.42
# ---------------------------------------------------------------------------
import logging

from ez_comfy.hardware.comfyui_inventory import _extract_list, scan_inventory

_OPTIONS = ["a.safetensors", "b.safetensors"]


def _info(node: str, key: str, entry) -> dict:
    return {node: {"input": {"required": {key: entry}}}}


def test_extract_list_legacy_shape():
    assert _extract_list(_info("N", "k", [_OPTIONS, {}]), "N", "k") == _OPTIONS


def test_extract_list_combo_shape():
    entry = ["COMBO", {"multiselect": False, "options": _OPTIONS}]
    assert _extract_list(_info("N", "k", entry), "N", "k") == _OPTIONS


def test_extract_list_both_shapes_agree():
    legacy = _extract_list(_info("N", "k", [_OPTIONS, {}]), "N", "k")
    combo = _extract_list(_info("N", "k", ["COMBO", {"options": _OPTIONS}]), "N", "k")
    assert legacy == combo


@pytest.mark.parametrize("entry", [
    ["INT", {"default": 1}],            # not a combo
    ["COMBO", {"no_options_here": 1}],
    ["COMBO"],
    "just-a-string",
    [],
    None,
])
def test_extract_list_unknown_shape_logs_and_returns_empty(entry, caplog):
    with caplog.at_level(logging.WARNING, logger="ez_comfy.hardware.comfyui_inventory"):
        assert _extract_list(_info("N", "k", entry), "N", "k") == []
    assert "N.k" in caplog.text


def test_extract_list_missing_node_or_input_is_quiet_empty(caplog):
    with caplog.at_level(logging.WARNING, logger="ez_comfy.hardware.comfyui_inventory"):
        assert _extract_list({}, "NoSuchNode", "k") == []
        assert _extract_list({"N": {"input": {"required": {}}}}, "N", "k") == []
    assert caplog.text == ""


class _FakeClient:
    def __init__(self, info: dict) -> None:
        self._info = info

    async def get_object_info(self, node_class=None) -> dict:
        return self._info


@pytest.mark.parametrize("wrap", [
    lambda options: [options, {}],
    lambda options: ["COMBO", {"options": options}],
], ids=["legacy", "combo"])
async def test_scan_inventory_reads_samplers_and_upscalers_in_both_shapes(wrap):
    info = {
        "KSampler": {"input": {"required": {
            "sampler_name": wrap(["euler", "dpmpp_2m"]),
            "scheduler": wrap(["normal", "karras"]),
        }}},
        "UpscaleModelLoader": {"input": {"required": {"model_name": wrap(["RealESRGAN_x4plus.safetensors"])}}},
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": wrap(["x.safetensors"])}}},
    }
    inv = await scan_inventory(_FakeClient(info))
    assert inv.samplers == ["euler", "dpmpp_2m"]
    assert inv.schedulers == ["normal", "karras"]
    assert inv.upscale_models == ["RealESRGAN_x4plus.safetensors"]
    assert [c.filename for c in inv.checkpoints] == ["x.safetensors"]
