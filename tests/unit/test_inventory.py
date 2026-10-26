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


# ---------------------------------------------------------------------------
# WI-2: multi-loader scanning
# ---------------------------------------------------------------------------

_WAN_FILES = [
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
]


def _multi_loader_info(wrap) -> dict:
    return {
        "CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": wrap(["DreamShaper_8_pruned.safetensors", "SDXL\\sd_xl_base_1.0.safetensors"])}}},
        "UNETLoader": {"input": {"required": {"unet_name": wrap(_WAN_FILES), "weight_dtype": wrap(["default"])}}},
        "CLIPVisionLoader": {"input": {"required": {"clip_name": wrap(["clip_vision_h.safetensors"])}}},
        "DualCLIPLoader": {"input": {"required": {
            "clip_name1": wrap(["clip_l.safetensors", "t5xxl_fp8.safetensors"]),
            "clip_name2": wrap(["t5xxl_fp8.safetensors", "clip_g.safetensors"]),
        }}},
    }


@pytest.mark.parametrize("wrap", [
    lambda options: [options, {}],
    lambda options: ["COMBO", {"options": options}],
], ids=["legacy", "combo"])
async def test_scan_inventory_lists_other_loaders_in_both_shapes(wrap):
    inv = await scan_inventory(_FakeClient(_multi_loader_info(wrap)))
    assert [m.filename for m in inv.diffusion_models] == _WAN_FILES
    assert all(m.loader == "diffusion_model" for m in inv.diffusion_models)
    assert inv.clip_vision_models == ["clip_vision_h.safetensors"]
    assert inv.dual_clip_models == ["clip_l.safetensors", "t5xxl_fp8.safetensors", "clip_g.safetensors"]


async def test_checkpoints_never_contain_diffusion_models():
    inv = await scan_inventory(_FakeClient(_multi_loader_info(lambda o: [o, {}])))
    assert all(m.loader == "checkpoint" for m in inv.checkpoints)
    assert not {m.filename for m in inv.checkpoints} & set(_WAN_FILES)


async def test_folder_is_recorded_for_files_in_a_subfolder():
    inv = await scan_inventory(_FakeClient(_multi_loader_info(lambda o: [o, {}])))
    by_name = {m.filename: m for m in inv.checkpoints}
    assert by_name["SDXL\\sd_xl_base_1.0.safetensors"].folder == "SDXL"
    assert by_name["DreamShaper_8_pruned.safetensors"].folder is None


async def test_missing_unetloader_is_not_fatal():
    info = _multi_loader_info(lambda o: [o, {}])
    del info["UNETLoader"], info["DualCLIPLoader"], info["CLIPVisionLoader"]
    inv = await scan_inventory(_FakeClient(info))
    assert inv.diffusion_models == [] and inv.dual_clip_models == [] and inv.clip_vision_models == []
    assert len(inv.checkpoints) == 2


async def test_all_models_orders_checkpoints_before_diffusion_models():
    inv = await scan_inventory(_FakeClient(_multi_loader_info(lambda o: [o, {}])))
    names = [m.filename for m in inv.all_models()]
    assert names == ["DreamShaper_8_pruned.safetensors", "SDXL\\sd_xl_base_1.0.safetensors", *_WAN_FILES]
    assert [m.filename for m in inv.models_for_loader("diffusion_model")] == _WAN_FILES


def test_inventory_response_includes_new_optional_fields_and_keeps_old_ones():
    from ez_comfy.api.routes import _inventory_response
    from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory, ModelInfo

    inv = ComfyUIInventory(
        checkpoints=[ModelInfo("a.safetensors", 0, "sd15")],
        diffusion_models=[ModelInfo("wan.safetensors", 0, "unknown", loader="diffusion_model")],
        clip_vision_models=["cv.safetensors"], dual_clip_models=["d.safetensors"],
    )
    response = _inventory_response(inv)
    assert response.diffusion_models == ["wan.safetensors"]
    assert response.clip_vision_models == ["cv.safetensors"] and response.dual_clip_models == ["d.safetensors"]
    assert [c["filename"] for c in response.checkpoints] == ["a.safetensors"]


async def test_overrides_plumbed_through_scan():
    info = _multi_loader_info(lambda o: [o, {}])
    info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0].append("someMergeV9.safetensors")
    overrides = {
        "someMergeV9.safetensors": "sdxl",
        _WAN_FILES[0]: "flux",          # diffusion models honour overrides too
    }
    inv = await scan_inventory(_FakeClient(info), overrides=overrides)
    by_name = {m.filename: m.family for m in inv.all_models()}
    assert by_name["someMergeV9.safetensors"] == "sdxl"
    assert by_name[_WAN_FILES[0]] == "flux"
    assert by_name[_WAN_FILES[1]] == "wan_video"          # not overridden


async def test_without_overrides_the_merge_stays_unknown():
    info = _multi_loader_info(lambda o: [o, {}])
    info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0].append("someMergeV9.safetensors")
    inv = await scan_inventory(_FakeClient(info))
    assert {m.filename: m.family for m in inv.checkpoints}["someMergeV9.safetensors"] == "unknown"


def test_settings_accept_known_override_families_and_default_to_empty():
    from ez_comfy.config.schema import Settings

    assert Settings().model_overrides == {}
    assert Settings.model_validate({"model_overrides": {"a.safetensors": "sdxl"}}).model_overrides == {
        "a.safetensors": "sdxl"
    }


def test_settings_reject_a_misspelled_override_family():
    import pydantic

    from ez_comfy.config.schema import Settings

    with pytest.raises(pydantic.ValidationError, match="unknown families"):
        Settings.model_validate({"model_overrides": {"a.safetensors": "sdxll"}})
