"""Unit tests for workflow builders and composer."""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from ez_comfy.planner.intent import PipelineIntent
from ez_comfy.planner.param_resolver import ResolvedParams
from ez_comfy.planner.provenance import ProvenanceRecord
from ez_comfy.workflows.composer import compose_annotated_workflow, compose_workflow, list_builders
from ez_comfy.workflows.recipes import get_recipe


def _make_params(**overrides) -> ResolvedParams:
    defaults = dict(
        steps=20, cfg_scale=7.0, sampler="euler", scheduler="normal",
        width=1024, height=1024, clip_skip=1, denoise_strength=0.7,
        seed=42, batch_size=1, sources={},
    )
    defaults.update(overrides)
    return ResolvedParams(**defaults)


def _make_plan(recipe_id: str, intent: PipelineIntent = PipelineIntent.TXT2IMG, **kwargs):
    """Build a minimal GenerationPlan mock for workflow tests."""
    recipe = get_recipe(recipe_id)
    params = _make_params()
    plan = MagicMock()
    plan.recipe = recipe
    plan.intent = intent
    plan.prompt = "a beautiful landscape"
    plan.negative_prompt = "blurry, ugly"
    plan.checkpoint = "realvisxlV50_v50Lightning.safetensors"
    plan.checkpoint_family = "sdxl"
    plan.params = params
    plan.loras = []
    plan.vae_override = None
    plan.controlnet = None
    plan.controlnet_strength = 1.0
    plan.reference_image_path = "test_ref.png"
    plan.mask_image_path = "test_mask.png"
    plan.recommendations = []
    for k, v in kwargs.items():
        setattr(plan, k, v)
    return plan


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def test_list_builders_not_empty():
    builders = list_builders()
    assert len(builders) > 0


def test_compose_unknown_builder_raises():
    plan = _make_plan("txt2img_basic")
    plan.recipe = MagicMock()
    plan.recipe.builder = "build_nonexistent_xyz"
    with pytest.raises(ValueError, match="No workflow builder"):
        compose_workflow(plan)


# ---------------------------------------------------------------------------
# txt2img
# ---------------------------------------------------------------------------

def test_txt2img_basic_nodes():
    plan = _make_plan("txt2img_basic")
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "CheckpointLoaderSimple" in class_types
    assert "CLIPTextEncode" in class_types
    assert "EmptyLatentImage" in class_types
    assert "KSampler" in class_types
    assert "VAEDecode" in class_types
    assert "SaveImage" in class_types


def test_txt2img_basic_ksampler_denoise_is_1():
    plan = _make_plan("txt2img_basic")
    nodes = compose_workflow(plan)
    ksampler = next(n for n in nodes.values() if n["class_type"] == "KSampler")
    assert ksampler["inputs"]["denoise"] == 1.0


def test_txt2img_hires_fix_has_two_ksamplers():
    plan = _make_plan("txt2img_hires_fix")
    nodes = compose_workflow(plan)
    ksamplers = [n for n in nodes.values() if n["class_type"] == "KSampler"]
    assert len(ksamplers) == 2


def test_txt2img_hires_fix_has_latent_upscale():
    plan = _make_plan("txt2img_hires_fix")
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "LatentUpscale" in class_types


def test_txt2img_with_vae_override():
    plan = _make_plan("txt2img_basic")
    plan.vae_override = "sdxl_vae.safetensors"
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "VAELoader" in class_types


def test_txt2img_with_lora():
    plan = _make_plan("txt2img_basic")
    plan.loras = [("my_lora.safetensors", 0.8)]
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "LoraLoader" in class_types


# ---------------------------------------------------------------------------
# img2img
# ---------------------------------------------------------------------------

def test_img2img_basic_nodes():
    plan = _make_plan("img2img_basic", PipelineIntent.IMG2IMG)
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "LoadImage" in class_types
    assert "VAEEncode" in class_types
    assert "KSampler" in class_types


def test_img2img_controlnet_nodes():
    plan = _make_plan("img2img_controlnet_canny", PipelineIntent.IMG2IMG)
    plan.controlnet = "control_canny.safetensors"
    plan.controlnet_strength = 0.8
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "ControlNetLoader" in class_types
    assert "ControlNetApply" in class_types


# ---------------------------------------------------------------------------
# inpaint
# ---------------------------------------------------------------------------

def test_inpaint_basic_nodes():
    plan = _make_plan("inpaint_basic", PipelineIntent.INPAINT)
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "VAEEncodeForInpaint" in class_types
    assert "ImageToMask" in class_types


# ---------------------------------------------------------------------------
# upscale
# ---------------------------------------------------------------------------

def test_upscale_simple_nodes():
    plan = _make_plan("upscale_simple", PipelineIntent.UPSCALE)
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "UpscaleModelLoader" in class_types
    assert "ImageUpscaleWithModel" in class_types
    assert "CheckpointLoaderSimple" not in class_types  # no diffusion model needed


def test_upscale_refine_has_ksampler():
    plan = _make_plan("upscale_refine", PipelineIntent.UPSCALE)
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "UpscaleModelLoader" in class_types
    assert "KSampler" in class_types


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------

def test_video_svd_nodes():
    plan = _make_plan("video_svd", PipelineIntent.VIDEO)
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "ImageOnlyCheckpointLoader" in class_types
    assert "SVD_img2vid_Conditioning" in class_types
    assert "SaveAnimatedWEBP" in class_types


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def test_audio_stable_nodes():
    plan = _make_plan("audio_stable", PipelineIntent.AUDIO)
    nodes = compose_workflow(plan)
    class_types = {n["class_type"] for n in nodes.values()}
    assert "CLIPLoader" in class_types          # separate T5 encoder
    assert "EmptyLatentAudio" in class_types
    assert "VAEDecodeAudio" in class_types
    assert "SaveAudio" in class_types


def test_audio_stable_t5_clip_type():
    plan = _make_plan("audio_stable", PipelineIntent.AUDIO)
    nodes = compose_workflow(plan)
    clip_loader = next(n for n in nodes.values() if n["class_type"] == "CLIPLoader")
    assert clip_loader["inputs"]["type"] == "stable_audio"


def test_audio_does_not_use_checkpoint_clip():
    """Positive prompt must use the CLIPLoader T5 node, not the checkpoint."""
    plan = _make_plan("audio_stable", PipelineIntent.AUDIO)
    nodes = compose_workflow(plan)
    # CLIPLoader should be at node "30", positive prompt at node "2"
    clip_loader_id = next(k for k, n in nodes.items() if n["class_type"] == "CLIPLoader")
    pos_encode = next(n for n in nodes.values()
                      if n["class_type"] == "CLIPTextEncode"
                      and n.get("_meta", {}).get("title") == "Positive Prompt")
    clip_ref = pos_encode["inputs"]["clip"]
    assert clip_ref[0] == clip_loader_id, "Positive prompt must use CLIPLoader, not checkpoint CLIP"


# ---------------------------------------------------------------------------
# Annotated workflow (Note node injection)
# ---------------------------------------------------------------------------

def test_compose_annotated_workflow_injects_note():
    plan = _make_plan("txt2img_basic")
    plan.provenance = ProvenanceRecord(gpu_name="RTX Test", vram_available_gb=16.0)
    nodes = compose_annotated_workflow(plan)
    note_nodes = [n for n in nodes.values() if n.get("class_type") == "Note"]
    assert len(note_nodes) == 1


def test_note_node_class_type():
    plan = _make_plan("txt2img_basic")
    plan.provenance = ProvenanceRecord()
    nodes = compose_annotated_workflow(plan)
    note = next(n for n in nodes.values() if n.get("class_type") == "Note")
    assert note["class_type"] == "Note"


def test_note_node_id_no_collision():
    plan = _make_plan("txt2img_basic")
    plan.provenance = ProvenanceRecord()
    plain_nodes = compose_workflow(plan)
    annotated_nodes = compose_annotated_workflow(plan)
    # Note node should be one more than the max existing id
    max_plain_id = max(int(k) for k in plain_nodes if k.isdigit())
    assert str(max_plain_id + 1) in annotated_nodes
    # And the note should be new — plain nodes count + 1
    assert len(annotated_nodes) == len(plain_nodes) + 1


def test_note_node_contains_provenance_text():
    plan = _make_plan("txt2img_basic")
    plan.provenance = ProvenanceRecord(gpu_name="RTX Provenance Test", vram_available_gb=8.0)
    nodes = compose_annotated_workflow(plan)
    note = next(n for n in nodes.values() if n.get("class_type") == "Note")
    assert "EZ Comfy Provenance" in note["inputs"]["text"]
    assert "RTX Provenance Test" in note["inputs"]["text"]


def test_note_node_has_meta_title():
    plan = _make_plan("txt2img_basic")
    plan.provenance = ProvenanceRecord()
    nodes = compose_annotated_workflow(plan)
    note = next(n for n in nodes.values() if n.get("class_type") == "Note")
    assert note.get("_meta", {}).get("title") == "EZ Comfy Provenance"
