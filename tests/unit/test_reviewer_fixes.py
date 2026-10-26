"""
Regression tests for the 8 issues identified by the external reviewer.
Each test is labelled with the issue number.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ez_comfy.planner.intent import PipelineIntent
from ez_comfy.workflows.recipes import select_recipe


# ---------------------------------------------------------------------------
# Issue 2: _get_upscale_model_name uses rec.entry.category (now .family)
# ---------------------------------------------------------------------------

def test_issue2_get_upscale_model_name_uses_family():
    """_get_upscale_model_name must read rec.entry.family, not .category."""
    from ez_comfy.workflows.upscale import _get_upscale_model_name

    rec = MagicMock()
    rec.entry.family = "upscaler"
    rec.entry.filename = "RealESRGAN_x4plus.pth"
    rec.installed = True

    plan = MagicMock()
    plan.recommendations = [rec]

    name = _get_upscale_model_name(plan)
    assert name == "RealESRGAN_x4plus.pth"


def test_issue2_get_upscale_model_name_default_when_none_installed():
    from ez_comfy.workflows.upscale import _get_upscale_model_name

    rec = MagicMock()
    rec.entry.family = "upscaler"
    rec.installed = False  # not installed

    plan = MagicMock()
    plan.recommendations = [rec]

    name = _get_upscale_model_name(plan)
    assert name == "RealESRGAN_x4plus.pth"


# ---------------------------------------------------------------------------
# Issue 3: select_recipe raises for non-txt2img with no capable recipes
# ---------------------------------------------------------------------------

def test_issue3_upscale_raises_without_capabilities():
    with pytest.raises(RuntimeError, match="intent='upscale'"):
        select_recipe(
            intent=PipelineIntent.UPSCALE,
            prompt="upscale this",
            has_reference_image=True,
            has_mask=False,
            discovered_class_types=set(),
        )


def test_issue3_video_raises_without_capabilities():
    with pytest.raises(RuntimeError, match="intent='video'"):
        select_recipe(
            intent=PipelineIntent.VIDEO,
            prompt="make a video",
            has_reference_image=True,
            has_mask=False,
            discovered_class_types=set(),
        )


def test_issue3_audio_raises_without_capabilities():
    with pytest.raises(RuntimeError, match="intent='audio'"):
        select_recipe(
            intent=PipelineIntent.AUDIO,
            prompt="make a sound",
            has_reference_image=False,
            has_mask=False,
            discovered_class_types=set(),
        )


def test_issue3_txt2img_still_falls_back():
    """txt2img always has a fallback; it must not raise."""
    recipe, _rejected = select_recipe(
        intent=PipelineIntent.TXT2IMG,
        prompt="a cat",
        has_reference_image=False,
        has_mask=False,
        discovered_class_types=set(),
    )
    assert recipe.intent == PipelineIntent.TXT2IMG


# ---------------------------------------------------------------------------
# Issue 4: recipe override emits warning for missing capabilities
# ---------------------------------------------------------------------------

def test_issue4_override_warns_on_missing_capabilities():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        recipe, _rejected = select_recipe(
            intent=PipelineIntent.IMG2IMG,
            prompt="test",
            has_reference_image=True,
            has_mask=False,
            discovered_class_types=set(),  # controlnet missing
            recipe_override="img2img_controlnet_canny",
        )
    assert recipe.id == "img2img_controlnet_canny"
    # Warning should have been emitted about missing capabilities
    capability_warnings = [x for x in w if "capabilities" in str(x.message).lower()]
    assert len(capability_warnings) > 0


def test_issue4_override_no_warning_when_caps_present():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        select_recipe(
            intent=PipelineIntent.IMG2IMG,
            prompt="test",
            has_reference_image=True,
            has_mask=False,
            discovered_class_types={"ControlNetLoader", "ControlNetApply"},
            recipe_override="img2img_controlnet_canny",
        )
    capability_warnings = [x for x in w if "capabilities" in str(x.message).lower()]
    assert len(capability_warnings) == 0


# ---------------------------------------------------------------------------
# Issue 5: queue_prompt + wait_for_completion share the same client_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_issue5_client_id_consistency():
    """queue_prompt and wait_for_completion must use the same client_id."""
    from ez_comfy.comfyui.client import ComfyUIClient

    captured_ids = {}

    async def fake_post(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        # Record the client_id sent in the prompt submission
        payload = kwargs.get("json", {})
        if "client_id" in payload:
            captured_ids["queue_client_id"] = payload["client_id"]
        resp.json = MagicMock(return_value={"prompt_id": "test-prompt-123"})
        return resp

    client = ComfyUIClient.__new__(ComfyUIClient)
    client._http = MagicMock()
    client._http.post = AsyncMock(side_effect=fake_post)

    prompt_id, returned_client_id = await client.queue_prompt({"node": "graph"})

    assert prompt_id == "test-prompt-123"
    assert returned_client_id == captured_ids["queue_client_id"], (
        "client_id returned by queue_prompt must match the one sent in the request"
    )


@pytest.mark.asyncio
async def test_issue5_wait_completion_passes_client_id_to_websocket():
    """wait_for_completion must forward client_id to _wait_websocket."""
    from ez_comfy.comfyui.client import ComfyUIClient

    captured = {}

    async def fake_wait_ws(prompt_id, client_id, timeout, on_progress):
        captured["client_id"] = client_id
        return {}

    client = ComfyUIClient.__new__(ComfyUIClient)
    client._wait_websocket = fake_wait_ws
    client.get_history = AsyncMock(return_value={"p": {}})

    the_client_id = str(uuid.uuid4())
    await client.wait_for_completion("p", client_id=the_client_id, timeout=5.0)

    assert captured["client_id"] == the_client_id


# ---------------------------------------------------------------------------
# Issue 6: SaveAnimatedWEBP uppercase casing
# ---------------------------------------------------------------------------

def test_issue6_video_uses_uppercase_webp_node():
    """Video builder must use SaveAnimatedWEBP (ComfyUI's built-in class name)."""
    from unittest.mock import MagicMock
    from ez_comfy.workflows.video import build_video_svd
    from ez_comfy.planner.param_resolver import ResolvedParams

    plan = MagicMock()
    plan.checkpoint = "svd_xt.safetensors"
    plan.params = ResolvedParams(
        steps=25, cfg_scale=2.5, sampler="euler", scheduler="karras",
        width=1024, height=576, clip_skip=1, denoise_strength=1.0,
        seed=42, batch_size=1, sources={},
    )
    plan.reference_image_path = "frame.png"

    nodes = build_video_svd(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "SaveAnimatedWEBP" in class_types
    assert "SaveAnimatedWebp" not in class_types  # wrong casing must be absent


# ---------------------------------------------------------------------------
# Issue 7: sampler/scheduler/checkpoint_override pass-through
# ---------------------------------------------------------------------------

def test_issue7_api_sampler_scheduler_passthrough():
    """_api_request_to_gen_request must not drop sampler or scheduler."""
    from ez_comfy.api.models import GenerateRequest
    from ez_comfy.api.routes import _api_request_to_gen_request

    req = GenerateRequest(
        prompt="test",
        sampler="dpmpp_2m",
        scheduler="karras",
        cfg_scale=9.0,
    )
    gen_req = _api_request_to_gen_request(req)
    assert gen_req.sampler == "dpmpp_2m"
    assert gen_req.scheduler == "karras"
    assert gen_req.cfg_scale == 9.0


def test_issue7_queue_request_checkpoint_override():
    """QueueRequest must expose checkpoint_override and it must pass through."""
    from ez_comfy.api.models import QueueRequest
    from ez_comfy.api.routes import _api_request_to_gen_request

    req = QueueRequest(
        prompt="test",
        checkpoint_override="mymodel.safetensors",
    )
    gen_req = _api_request_to_gen_request(req)
    assert gen_req.checkpoint_override == "mymodel.safetensors"


# ---------------------------------------------------------------------------
# Issue 8: CLI size_bytes (not size_gb)
# ---------------------------------------------------------------------------

def test_issue8_model_info_has_size_bytes_not_size_gb():
    """ModelInfo stores size_bytes; no size_gb attribute should exist."""
    from ez_comfy.hardware.comfyui_inventory import ModelInfo

    m = ModelInfo(filename="test.safetensors", size_bytes=4_000_000_000, family="sdxl")
    assert hasattr(m, "size_bytes")
    assert m.size_bytes == 4_000_000_000
    assert not hasattr(m, "size_gb"), "ModelInfo must not have size_gb; use size_bytes / 1e9"
    assert abs(m.size_bytes / 1e9 - 4.0) < 0.01


def test_issue9_checkpoint_resolution_uses_exact_installed_variant_filename():
    """Planner should submit the exact installed checkpoint filename, not catalog token."""
    from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory, ModelInfo
    from ez_comfy.planner.planner import GenerationRequest, plan_generation

    hardware = MagicMock()
    hardware.gpu_vram_gb = 15.9

    inventory = ComfyUIInventory(
        checkpoints=[
            ModelInfo(
                filename="realvisxlV50_v50LightningBakedvae.safetensors",
                size_bytes=0,
                family="sdxl",
            )
        ],
        discovered_class_types=set(),
    )

    req = GenerationRequest(prompt="photorealistic portrait, studio photo")
    plan = plan_generation(req, hardware, inventory, prefer_speed=True, auto_negative=True)
    assert plan.checkpoint == "realvisxlV50_v50LightningBakedvae.safetensors"
