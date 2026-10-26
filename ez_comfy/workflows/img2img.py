from __future__ import annotations

from ez_comfy.planner.planner import GenerationPlan
from ez_comfy.workflows.txt2img import _checkpoint_nodes, _lora_chain, _vae_node


def build_img2img_basic(plan: GenerationPlan) -> dict:
    """Standard img2img: load reference image, VAE encode, denoise."""
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
    # Reference image
    ref_path = plan.reference_image_path or ""
    nodes["8"] = {
        "class_type": "LoadImage", "_meta": {"title": "Reference Image"},
        "inputs": {"image": ref_path, "upload": "image"},
    }
    # VAE encode
    nodes["9"] = {
        "class_type": "VAEEncode", "_meta": {"title": "VAE Encode"},
        "inputs": {"pixels": ["8", 0], "vae": vae_ref},
    }
    # KSampler
    nodes["5"] = {
        "class_type": "KSampler", "_meta": {"title": "KSampler"},
        "inputs": {
            "model": model_ref, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["9", 0],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg_scale,
            "sampler_name": p.sampler, "scheduler": p.scheduler,
            "denoise": p.denoise_strength,
        },
    }
    nodes["6"] = {
        "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"},
        "inputs": {"samples": ["5", 0], "vae": vae_ref},
    }
    nodes["7"] = {
        "class_type": "SaveImage", "_meta": {"title": "Save Image"},
        "inputs": {"images": ["6", 0], "filename_prefix": "ezcomfy"},
    }
    return nodes


def build_img2img_controlnet_canny(plan: GenerationPlan) -> dict:
    """img2img with ControlNet Canny for structure preservation."""
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
    # Load reference
    nodes["8"] = {
        "class_type": "LoadImage", "_meta": {"title": "Reference Image"},
        "inputs": {"image": ref_path, "upload": "image"},
    }
    # Load ControlNet model
    cn_model = plan.controlnet or ""
    nodes["10"] = {
        "class_type": "ControlNetLoader", "_meta": {"title": "ControlNet Loader"},
        "inputs": {"control_net_name": cn_model},
    }
    # Apply ControlNet
    nodes["12"] = {
        "class_type": "ControlNetApply", "_meta": {"title": "ControlNet Apply"},
        "inputs": {
            "conditioning": ["2", 0],
            "control_net":  ["10", 0],
            "image":        ["8", 0],
            "strength":     plan.controlnet_strength,
        },
    }
    # VAE encode reference
    nodes["9"] = {
        "class_type": "VAEEncode", "_meta": {"title": "VAE Encode"},
        "inputs": {"pixels": ["8", 0], "vae": vae_ref},
    }
    nodes["5"] = {
        "class_type": "KSampler", "_meta": {"title": "KSampler"},
        "inputs": {
            "model": model_ref, "positive": ["12", 0], "negative": ["3", 0],
            "latent_image": ["9", 0],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg_scale,
            "sampler_name": p.sampler, "scheduler": p.scheduler,
            "denoise": p.denoise_strength,
        },
    }
    nodes["6"] = {
        "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"},
        "inputs": {"samples": ["5", 0], "vae": vae_ref},
    }
    nodes["7"] = {
        "class_type": "SaveImage", "_meta": {"title": "Save Image"},
        "inputs": {"images": ["6", 0], "filename_prefix": "ezcomfy"},
    }
    return nodes
