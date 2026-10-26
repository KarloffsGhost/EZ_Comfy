"""Wan 2.2 text-to-video: planning, recipe selection, builder vs the golden template graph (WI-5)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ez_comfy.comfyui.client import ComfyUIClient
from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory, ModelInfo
from ez_comfy.hardware.probe import HardwareProfile
from ez_comfy.planner.planner import GenerationRequest, plan_generation
from ez_comfy.workflows.composer import compose_workflow
from tests.golden import describe_difference, graphs_match, normalize_graph

GOLDEN = Path(__file__).parent.parent / "fixtures" / "workflows" / "wan22_t2v.api.json"

HIGH = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
LOW = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"


def _hardware() -> HardwareProfile:
    return HardwareProfile(gpu_name="Test GPU", gpu_vram_gb=16.0, system_ram_gb=64.0, cuda_version="13.0", platform="win32")


def _inventory(*, high=HIGH, low=LOW, encoder=ENCODER, vae=VAE, svd=False) -> ComfyUIInventory:
    diffusion = [
        ModelInfo(name, 0, "wan_video", loader="diffusion_model") for name in (high, low) if name
    ]
    inv = ComfyUIInventory(
        checkpoints=[ModelInfo("DreamShaper_8_pruned.safetensors", 0, "sd15")],
        diffusion_models=diffusion,
        clip_models=[encoder] if encoder else [],
        vaes=[vae] if vae else [],
        samplers=["euler"], schedulers=["simple"],
        discovered_class_types={"ImageOnlyCheckpointLoader", "SVD_img2vid_Conditioning"} if svd else set(),
    )
    return inv


def _plan(prompt="a red fox running through a snowy forest", inventory=None, **request):
    return plan_generation(
        GenerationRequest(prompt=prompt, intent_override="video", seed=1, **request),
        _hardware(), inventory or _inventory(),
    )


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Planning and recipe selection
# ---------------------------------------------------------------------------

def test_plan_uses_the_wan_recipe_and_fills_every_role():
    plan = _plan()
    assert plan.recipe.id == "video_wan22_t2v"
    assert plan.checkpoint_family == "wan_video"
    assert plan.models == {"diffusion_high": HIGH, "diffusion_low": LOW, "text_encoder": ENCODER, "vae": VAE}
    assert plan.summary()["models"] == plan.models
    assert (plan.video_frames, plan.video_fps) == (81, 16)
    assert (plan.params.width, plan.params.height, plan.params.steps, plan.params.cfg_scale) == (640, 640, 20, 3.5)
    assert not plan.warnings


def test_plan_records_why_svd_was_not_used():
    plan = _plan(inventory=_inventory(svd=True))
    rejected = {a.value: a.rejected_reason for d in plan.provenance.decisions if d.parameter == "recipe" for a in d.alternatives}
    assert "legacy recipe cannot run" in rejected["video_svd"]


@pytest.mark.parametrize("missing,role", [("high", "diffusion_high"), ("low", "diffusion_low"),
                                          ("encoder", "text_encoder"), ("vae", "vae")])
def test_incomplete_wan_set_is_reported_and_not_used(missing, role):
    from ez_comfy.planner.planner import _select_wan_t2v_models

    warnings: list[str] = []
    assert _select_wan_t2v_models(_inventory(**{missing: None}), warnings) is None
    if missing in ("high", "low"):
        assert any(f"missing: {role}" in w for w in warnings)


def test_no_wan_files_at_all_is_silent():
    from ez_comfy.planner.planner import _select_wan_t2v_models

    warnings: list[str] = []
    assert _select_wan_t2v_models(_inventory(high=None, low=None), warnings) is None
    assert warnings == []


def test_incomplete_wan_set_leaves_video_planning_with_nothing_it_can_run():
    with pytest.raises(RuntimeError):      # no SVD installed and no reference image, as before
        _plan(inventory=_inventory(low=None))


def test_i2v_files_are_not_mistaken_for_the_t2v_pair():
    from ez_comfy.planner.planner import _select_wan_t2v_models

    inv = _inventory(high="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                     low="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
    assert _select_wan_t2v_models(inv, []) is None


def test_reference_image_keeps_the_svd_path():
    inv = _inventory(svd=True)
    inv.checkpoints.append(ModelInfo("svd_xt.safetensors", 0, "svd"))
    plan = plan_generation(
        GenerationRequest(prompt="animate this", intent_override="video", reference_image=b"x", seed=1),
        _hardware(), inv,
    )
    assert plan.recipe.id == "video_svd" and plan.models == {}


def test_image_prompts_never_use_the_wan_recipe():
    plan = plan_generation(GenerationRequest(prompt="a fox", seed=1), _hardware(), _inventory())
    assert plan.recipe.id != "video_wan22_t2v" and plan.models == {}


@pytest.mark.parametrize("asked,expected", [(81, 81), (80, 81), (82, 81), (83, 85), (30, 29), (1, 1)])
def test_frame_count_snaps_to_one_plus_a_multiple_of_four(asked, expected):
    plan = _plan(video_frames=asked)
    assert plan.video_frames == expected
    assert (asked == expected) == (not any("must be 1 plus" in w for w in plan.warnings))


def test_user_fps_is_respected():
    assert _plan(video_fps=24).video_fps == 24


# ---------------------------------------------------------------------------
# Builder vs golden graph exported from ComfyUI's shipped template
# ---------------------------------------------------------------------------

def test_builder_matches_the_template_graph_for_a_default_plan():
    # Placeholder prompts let the comparison also check which text goes to which encoder.
    plan = _plan(prompt="PROMPT", negative_prompt="NEGATIVE")
    workflow = compose_workflow(plan)
    assert graphs_match(workflow, _golden(), compare_text=True), describe_difference(workflow, _golden(), True)


def test_builder_follows_the_plan_parameters():
    workflow = compose_workflow(_plan(width=832, height=480, steps=30, video_frames=33, video_fps=24))
    by_class = {}
    for node in workflow.values():
        by_class.setdefault(node["class_type"], []).append(node["inputs"])
    latent = by_class["EmptyHunyuanLatentVideo"][0]
    assert (latent["width"], latent["height"], latent["length"]) == (832, 480, 33)
    first, second = sorted(by_class["KSamplerAdvanced"], key=lambda i: i["start_at_step"])
    assert (first["start_at_step"], first["end_at_step"]) == (0, 15)
    assert (second["start_at_step"], second["end_at_step"]) == (15, 30)
    assert first["add_noise"] == "enable" and second["add_noise"] == "disable"
    assert by_class["CreateVideo"][0]["fps"] == 24


def test_builder_refuses_a_plan_without_the_model_roles():
    plan = _plan()
    plan.models = {"diffusion_high": HIGH}
    with pytest.raises(ValueError, match="needs model roles"):
        compose_workflow(plan)


def test_prompt_text_reaches_the_right_encoders():
    workflow = compose_workflow(_plan(prompt="a red fox"))
    positive_id = next(n for n in workflow.values() if n["class_type"] == "KSamplerAdvanced")["inputs"]["positive"][0]
    assert "red fox" in workflow[positive_id]["inputs"]["text"]


# ---------------------------------------------------------------------------
# Normalizer self-tests
# ---------------------------------------------------------------------------

def test_normalizer_ignores_ids_seeds_prompts_titles_and_prefixes():
    golden = _golden()
    relabeled = {str(int(k) + 1000): copy.deepcopy(v) for k, v in golden.items()}
    for node in relabeled.values():
        node["inputs"] = {
            k: ([str(int(v[0]) + 1000), v[1]] if isinstance(v, list) else v) for k, v in node["inputs"].items()
        }
        node["_meta"] = {"title": "whatever"}
        if "noise_seed" in node["inputs"]:
            node["inputs"]["noise_seed"] = 999
        if "text" in node["inputs"]:
            node["inputs"]["text"] = "something else entirely"
        if "filename_prefix" in node["inputs"]:
            node["inputs"]["filename_prefix"] = "other"
    assert normalize_graph(relabeled) == normalize_graph(golden)


def test_normalizer_detects_a_changed_widget():
    changed = copy.deepcopy(_golden())
    next(n for n in changed.values() if n["class_type"] == "KSamplerAdvanced")["inputs"]["cfg"] = 9.0
    assert not graphs_match(changed, _golden())


def test_swapped_positive_and_negative_is_invisible_to_structure_but_caught_with_text():
    swapped = copy.deepcopy(_golden())
    for node in swapped.values():
        if node["class_type"] == "KSamplerAdvanced":
            node["inputs"]["positive"], node["inputs"]["negative"] = node["inputs"]["negative"], node["inputs"]["positive"]
    assert graphs_match(swapped, _golden())                       # documented limitation
    assert not graphs_match(swapped, _golden(), compare_text=True)


def test_normalizer_detects_swapped_expert_models():
    swapped = copy.deepcopy(_golden())
    unets = [n for n in swapped.values() if n["class_type"] == "UNETLoader"]
    unets[0]["inputs"]["unet_name"], unets[1]["inputs"]["unet_name"] = unets[1]["inputs"]["unet_name"], unets[0]["inputs"]["unet_name"]
    assert not graphs_match(swapped, _golden())


def test_normalizer_detects_a_missing_node():
    trimmed = copy.deepcopy(_golden())
    del trimmed[next(k for k, n in trimmed.items() if n["class_type"] == "ModelSamplingSD3")]
    assert not graphs_match(trimmed, _golden())


# ---------------------------------------------------------------------------
# Video outputs
# ---------------------------------------------------------------------------

def _history(entry: dict) -> dict:
    return {"outputs": {"80": {"images": [entry]}}}


@pytest.mark.parametrize("animated,expected", [([True], "video"), (True, "video"), ([False], "image"), (False, "image"), (None, "image")])
def test_extract_outputs_marks_animated_images_as_video(animated, expected):
    entry = {"filename": "ezcomfy.mp4", "subfolder": "video", "type": "output"}
    if animated is not None:
        entry["animated"] = animated
    outputs = ComfyUIClient("http://comfy.test").extract_outputs(_history(entry))
    assert [o["media_type"] for o in outputs] == [expected]


def test_extract_outputs_keeps_gifs_audio_and_stills():
    history = {"outputs": {
        "1": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
        "2": {"audio": [{"filename": "a.flac", "subfolder": "", "type": "output"}]},
        "3": {"gifs": [{"filename": "a.gif", "subfolder": "", "type": "output"}]},
    }}
    types = sorted(o["media_type"] for o in ComfyUIClient("http://comfy.test").extract_outputs(history))
    assert types == ["audio", "image", "video"]
