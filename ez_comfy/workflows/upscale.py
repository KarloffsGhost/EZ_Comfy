from __future__ import annotations

from ez_comfy.planner.planner import GenerationPlan
from ez_comfy.workflows.txt2img import _checkpoint_nodes, _lora_chain, _vae_node


def _get_upscale_model_name(plan: GenerationPlan) -> str:
    """Return the first installed upscale model from recommendations, or a safe default."""
    for rec in plan.recommendations:
        if rec.entry.family == "upscaler" and rec.installed:
            return rec.entry.filename
    return "RealESRGAN_x4plus.pth"


def build_upscale_simple(plan: GenerationPlan) -> dict:
    """Fast upscale using ESRGAN-style model — no refinement pass."""
    nodes: dict = {}

    ref_path = plan.reference_image_path or ""
    nodes["8"] = {
        "class_type": "LoadImage", "_meta": {"title": "Input Image"},
        "inputs": {"image": ref_path, "upload": "image"},
    }
    nodes["20"] = {
        "class_type": "UpscaleModelLoader", "_meta": {"title": "Upscale Model"},
        "inputs": {"model_name": _get_upscale_model_name(plan)},
    }
    nodes["21"] = {
        "class_type": "ImageUpscaleWithModel", "_meta": {"title": "Upscale Image"},
        "inputs": {"upscale_model": ["20", 0], "image": ["8", 0]},
    }
    nodes["7"] = {
        "class_type": "SaveImage", "_meta": {"title": "Save Image"},
        "inputs": {"images": ["21", 0], "filename_prefix": "ezcomfy_upscale"},
    }
    return nodes


def build_upscale_refine(plan: GenerationPlan) -> dict:
    """Upscale with ESRGAN then refine with a low-denoise img2img pass."""
    p = plan.params
    nodes: dict = {}

    nodes.update(_checkpoint_nodes(plan))
    lora_nodes, model_ref, clip_ref = _lora_chain(plan)
    nodes.update(lora_nodes)
    vae_node = _vae_node(plan)
    vae_ref = ["50", 0] if vae_node else ["1", 2]
    if vae_node:
        nodes.update(vae_node)

    nodes["2"] = {
        "class_type": "CLIPTextEncode", "_meta": {"title": "Positive"},
        "inputs": {"text": plan.prompt, "clip": clip_ref},
    }
    nodes["3"] = {
        "class_type": "CLIPTextEncode", "_meta": {"title": "Negative"},
        "inputs": {"text": plan.negative_prompt, "clip": clip_ref},
    }

    ref_path = plan.reference_image_path or ""
    nodes["8"] = {
        "class_type": "LoadImage", "_meta": {"title": "Input Image"},
        "inputs": {"image": ref_path, "upload": "image"},
    }
    # Pass 1: ESRGAN pixel-space upscale
    nodes["20"] = {
        "class_type": "UpscaleModelLoader", "_meta": {"title": "Upscale Model"},
        "inputs": {"model_name": _get_upscale_model_name(plan)},
    }
    nodes["21"] = {
        "class_type": "ImageUpscaleWithModel", "_meta": {"title": "Upscale Image"},
        "inputs": {"upscale_model": ["20", 0], "image": ["8", 0]},
    }
    # Pass 2: VAE encode upscaled image then refine with low denoise
    nodes["9"] = {
        "class_type": "VAEEncode", "_meta": {"title": "VAE Encode"},
        "inputs": {"pixels": ["21", 0], "vae": vae_ref},
    }
    nodes["5"] = {
        "class_type": "KSampler", "_meta": {"title": "KSampler (Refine)"},
        "inputs": {
            "model": model_ref, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["9", 0],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg_scale,
            "sampler_name": p.sampler, "scheduler": p.scheduler,
            "denoise": p.denoise_strength,  # recipe sets this to 0.3
        },
    }
    nodes["6"] = {
        "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"},
        "inputs": {"samples": ["5", 0], "vae": vae_ref},
    }
    nodes["7"] = {
        "class_type": "SaveImage", "_meta": {"title": "Save Image"},
        "inputs": {"images": ["6", 0], "filename_prefix": "ezcomfy_upscale"},
    }
    return nodes
