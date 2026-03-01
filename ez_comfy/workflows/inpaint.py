from __future__ import annotations

from ez_comfy.planner.planner import GenerationPlan
from ez_comfy.workflows.txt2img import _checkpoint_nodes, _lora_chain, _vae_node


def build_inpaint_basic(plan: GenerationPlan) -> dict:
    """Standard inpaint: load image + mask, VAE encode, denoise masked area."""
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
    mask_path = plan.mask_image_path or ""

    # Load reference image
    nodes["8"] = {
        "class_type": "LoadImage", "_meta": {"title": "Reference Image"},
        "inputs": {"image": ref_path, "upload": "image"},
    }
    # Load mask image
    nodes["13"] = {
        "class_type": "LoadImage", "_meta": {"title": "Mask Image"},
        "inputs": {"image": mask_path, "upload": "image"},
    }
    # Convert mask image to mask tensor (use red channel)
    nodes["14"] = {
        "class_type": "ImageToMask", "_meta": {"title": "Image to Mask"},
        "inputs": {"image": ["13", 0], "channel": "red"},
    }
    # VAE encode with inpaint mask (grows mask by 6px to soften edges)
    nodes["9"] = {
        "class_type": "VAEEncodeForInpaint", "_meta": {"title": "VAE Encode (Inpaint)"},
        "inputs": {
            "pixels": ["8", 0],
            "vae": vae_ref,
            "mask": ["14", 0],
            "grow_mask_by": 6,
        },
    }
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
        "inputs": {"images": ["6", 0], "filename_prefix": "ezcomfy_inpaint"},
    }
    return nodes
