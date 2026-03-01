from __future__ import annotations

from ez_comfy.planner.planner import GenerationPlan


def _checkpoint_nodes(plan: GenerationPlan) -> dict:
    """Build CheckpointLoaderSimple node."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
            "inputs": {"ckpt_name": plan.checkpoint},
        }
    }


def _lora_chain(plan: GenerationPlan, start_node: int = 8) -> tuple[dict, str, str]:
    """
    Build a chain of LoraLoader nodes.
    Returns (nodes_dict, model_output_ref, clip_output_ref).
    model_output_ref / clip_output_ref are the node references for model/clip outputs.
    """
    nodes: dict = {}
    model_ref = ["1", 0]
    clip_ref  = ["1", 1]

    for i, (lora_name, strength) in enumerate(plan.loras):
        node_id = str(start_node + i)
        nodes[node_id] = {
            "class_type": "LoraLoader",
            "_meta": {"title": f"LoRA: {lora_name}"},
            "inputs": {
                "model": model_ref,
                "clip":  clip_ref,
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip":  strength,
            },
        }
        model_ref = [node_id, 0]
        clip_ref  = [node_id, 1]

    return nodes, model_ref, clip_ref


def _vae_node(plan: GenerationPlan, node_id: str = "50") -> dict | None:
    if not plan.vae_override:
        return None
    return {
        node_id: {
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
            "inputs": {"vae_name": plan.vae_override},
        }
    }


def build_txt2img_basic(plan: GenerationPlan) -> dict:
    """Standard text-to-image workflow with optional LoRA and VAE override."""
    p = plan.params
    nodes: dict = {}

    # Checkpoint
    nodes.update(_checkpoint_nodes(plan))

    # LoRA chain (if any)
    lora_nodes, model_ref, clip_ref = _lora_chain(plan)
    nodes.update(lora_nodes)

    # VAE override
    vae_node = _vae_node(plan)
    vae_ref = ["50", 0] if vae_node else ["1", 2]
    if vae_node:
        nodes.update(vae_node)

    # Positive CLIP encode
    nodes["2"] = {
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Positive Prompt"},
        "inputs": {"text": plan.prompt, "clip": clip_ref},
    }
    # Negative CLIP encode
    nodes["3"] = {
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Negative Prompt"},
        "inputs": {"text": plan.negative_prompt, "clip": clip_ref},
    }
    # Latent image
    nodes["4"] = {
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Empty Latent"},
        "inputs": {"width": p.width, "height": p.height, "batch_size": p.batch_size},
    }
    # KSampler
    nodes["5"] = {
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
        "inputs": {
            "model":        model_ref,
            "positive":     ["2", 0],
            "negative":     ["3", 0],
            "latent_image": ["4", 0],
            "seed":         p.seed,
            "steps":        p.steps,
            "cfg":          p.cfg_scale,
            "sampler_name": p.sampler,
            "scheduler":    p.scheduler,
            "denoise":      1.0,
        },
    }
    # VAE decode
    nodes["6"] = {
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
        "inputs": {"samples": ["5", 0], "vae": vae_ref},
    }
    # Save
    nodes["7"] = {
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
        "inputs": {"images": ["6", 0], "filename_prefix": "ezcomfy"},
    }
    return nodes


def build_photo_realism_v1(plan: GenerationPlan) -> dict:
    """
    Photo-realism two-pass pipeline:
      - Pass 1: generate at 65% target resolution
      - Pixel-space upscale to target via ImageScale (lanczos, sharper than latent)
      - Pass 2: img2img refine (denoise=0.45) for sharpness and fine detail
    No special capabilities required — works with stock ComfyUI.
    """
    p = plan.params
    nodes: dict = {}

    # Checkpoint
    nodes.update(_checkpoint_nodes(plan))

    # LoRA chain
    lora_nodes, model_ref, clip_ref = _lora_chain(plan)
    nodes.update(lora_nodes)

    # VAE override
    vae_node = _vae_node(plan)
    vae_ref = ["50", 0] if vae_node else ["1", 2]
    if vae_node:
        nodes.update(vae_node)

    # Pass 1 resolution: 65% of target, snapped to 64px grid
    p1_w = max(512, int(p.width * 0.65 / 64) * 64)
    p1_h = max(512, int(p.height * 0.65 / 64) * 64)

    # Prompts (shared between both passes)
    nodes["2"] = {
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Positive Prompt"},
        "inputs": {"text": plan.prompt, "clip": clip_ref},
    }
    nodes["3"] = {
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Negative Prompt"},
        "inputs": {"text": plan.negative_prompt, "clip": clip_ref},
    }

    # Pass 1: low-res latent generation
    nodes["4"] = {
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Latent (Pass 1)"},
        "inputs": {"width": p1_w, "height": p1_h, "batch_size": p.batch_size},
    }
    nodes["5"] = {
        "class_type": "KSampler",
        "_meta": {"title": "KSampler Pass 1"},
        "inputs": {
            "model": model_ref, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": p.seed, "steps": p.steps,
            "cfg": p.cfg_scale, "sampler_name": p.sampler,
            "scheduler": p.scheduler, "denoise": 1.0,
        },
    }
    nodes["6"] = {
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode (Pass 1)"},
        "inputs": {"samples": ["5", 0], "vae": vae_ref},
    }

    # Pixel-space upscale: lanczos gives sharper edges than latent upscale
    nodes["10"] = {
        "class_type": "ImageScale",
        "_meta": {"title": "Pixel Upscale (Lanczos)"},
        "inputs": {
            "image":          ["6", 0],
            "upscale_method": "lanczos",
            "width":          p.width,
            "height":         p.height,
            "crop":           "disabled",
        },
    }

    # Re-encode to latent for the refinement pass
    nodes["11"] = {
        "class_type": "VAEEncode",
        "_meta": {"title": "VAE Encode (Hires)"},
        "inputs": {"pixels": ["10", 0], "vae": vae_ref},
    }

    # Pass 2: high-res refinement (low denoise = preserve composition, add detail)
    refine_steps = max(10, p.steps // 2)
    nodes["12"] = {
        "class_type": "KSampler",
        "_meta": {"title": "KSampler Pass 2 (Refine)"},
        "inputs": {
            "model": model_ref, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["11", 0],
            "seed": p.seed + 1, "steps": refine_steps,
            "cfg": p.cfg_scale, "sampler_name": p.sampler,
            "scheduler": p.scheduler, "denoise": 0.45,
        },
    }
    nodes["13"] = {
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode (Final)"},
        "inputs": {"samples": ["12", 0], "vae": vae_ref},
    }

    # Save
    nodes["7"] = {
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
        "inputs": {"images": ["13", 0], "filename_prefix": "ezcomfy_photo"},
    }
    return nodes


def build_txt2img_hires_fix(plan: GenerationPlan) -> dict:
    """Two-pass hi-res fix: generate at reduced resolution, then upscale + refine."""
    p = plan.params
    nodes: dict = {}

    # Checkpoint
    nodes.update(_checkpoint_nodes(plan))

    # LoRA chain
    lora_nodes, model_ref, clip_ref = _lora_chain(plan)
    nodes.update(lora_nodes)

    # VAE override
    vae_node = _vae_node(plan)
    vae_ref = ["50", 0] if vae_node else ["1", 2]
    if vae_node:
        nodes.update(vae_node)

    # Pass 1 resolution (50-70% of target)
    p1_w = max(512, int(p.width * 0.65 / 64) * 64)
    p1_h = max(512, int(p.height * 0.65 / 64) * 64)

    # Prompts
    nodes["2"] = {
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Positive Prompt"},
        "inputs": {"text": plan.prompt, "clip": clip_ref},
    }
    nodes["3"] = {
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Negative Prompt"},
        "inputs": {"text": plan.negative_prompt, "clip": clip_ref},
    }

    # Pass 1: low-res latent
    nodes["4"] = {
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Latent (Pass 1)"},
        "inputs": {"width": p1_w, "height": p1_h, "batch_size": p.batch_size},
    }
    nodes["5"] = {
        "class_type": "KSampler",
        "_meta": {"title": "KSampler Pass 1"},
        "inputs": {
            "model": model_ref, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": p.seed, "steps": p.steps,
            "cfg": p.cfg_scale, "sampler_name": p.sampler,
            "scheduler": p.scheduler, "denoise": 1.0,
        },
    }

    # Upscale latent
    nodes["10"] = {
        "class_type": "LatentUpscale",
        "_meta": {"title": "Latent Upscale"},
        "inputs": {
            "samples":        ["5", 0],
            "upscale_method": "nearest-exact",
            "width":          p.width,
            "height":         p.height,
            "crop":           "disabled",
        },
    }

    # Pass 2: refine
    refine_steps = max(8, p.steps // 3)
    nodes["11"] = {
        "class_type": "KSampler",
        "_meta": {"title": "KSampler Pass 2 (Refine)"},
        "inputs": {
            "model": model_ref, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["10", 0],
            "seed": p.seed + 1, "steps": refine_steps,
            "cfg": p.cfg_scale, "sampler_name": p.sampler,
            "scheduler": p.scheduler, "denoise": 0.45,
        },
    }

    # Decode + save
    nodes["6"] = {
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
        "inputs": {"samples": ["11", 0], "vae": vae_ref},
    }
    nodes["7"] = {
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
        "inputs": {"images": ["6", 0], "filename_prefix": "ezcomfy"},
    }
    return nodes
