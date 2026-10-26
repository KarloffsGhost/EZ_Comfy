from __future__ import annotations

from ez_comfy.planner.planner import GenerationPlan

# Node ids follow the shipped template (video_wan2_2_14B_t2v.json, comfyui_workflow_templates_json
# 0.1.122) so a builder/template comparison is easy to read; the golden test relabels them anyway.


def build_video_wan22_t2v(plan: GenerationPlan) -> dict:
    """
    Wan 2.2 14B text-to-video: two expert models used in two sampling passes of one run.

    Node layout (template default, no distillation LoRAs):
      UNETLoader(high noise) -> ModelSamplingSD3 -> KSamplerAdvanced (steps 0..N/2, keeps leftover noise)
      UNETLoader(low noise)  -> ModelSamplingSD3 -> KSamplerAdvanced (steps N/2..N) -> VAEDecode
      CLIPLoader(umt5) -> CLIPTextEncode x2      EmptyHunyuanLatentVideo -> first sampler
      VAEDecode -> CreateVideo -> SaveVideo
    """
    models = plan.models
    missing = [role for role in ("diffusion_high", "diffusion_low", "text_encoder", "vae") if not models.get(role)]
    if missing:
        raise ValueError(f"Wan 2.2 recipe needs model roles {missing}; the plan has {sorted(models)}")

    p = plan.params
    frames = plan.video_frames
    fps = plan.video_fps
    if frames is None or fps is None:
        raise ValueError("Wan 2.2 recipe needs plan.video_frames and plan.video_fps")
    split = max(1, p.steps // 2)

    return {
        "71": {
            "class_type": "CLIPLoader", "_meta": {"title": "Load Text Encoder"},
            "inputs": {"clip_name": models["text_encoder"], "type": "wan", "device": "default"},
        },
        "73": {
            "class_type": "VAELoader", "_meta": {"title": "Load VAE"},
            "inputs": {"vae_name": models["vae"]},
        },
        "75": {
            "class_type": "UNETLoader", "_meta": {"title": "Load High-Noise Model"},
            "inputs": {"unet_name": models["diffusion_high"], "weight_dtype": "default"},
        },
        "76": {
            "class_type": "UNETLoader", "_meta": {"title": "Load Low-Noise Model"},
            "inputs": {"unet_name": models["diffusion_low"], "weight_dtype": "default"},
        },
        "82": {
            "class_type": "ModelSamplingSD3", "_meta": {"title": "Shift (high noise)"},
            "inputs": {"model": ["75", 0], "shift": 5.0},
        },
        "86": {
            "class_type": "ModelSamplingSD3", "_meta": {"title": "Shift (low noise)"},
            "inputs": {"model": ["76", 0], "shift": 5.0},
        },
        "89": {
            "class_type": "CLIPTextEncode", "_meta": {"title": "Positive Prompt"},
            "inputs": {"clip": ["71", 0], "text": plan.prompt},
        },
        "72": {
            "class_type": "CLIPTextEncode", "_meta": {"title": "Negative Prompt"},
            "inputs": {"clip": ["71", 0], "text": plan.negative_prompt},
        },
        "74": {
            "class_type": "EmptyHunyuanLatentVideo", "_meta": {"title": "Empty Video Latent"},
            "inputs": {"width": p.width, "height": p.height, "length": frames, "batch_size": 1},
        },
        "81": {
            "class_type": "KSamplerAdvanced", "_meta": {"title": "Sampler (high noise)"},
            "inputs": {
                "model": ["82", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["74", 0],
                "add_noise": "enable", "noise_seed": p.seed, "steps": p.steps, "cfg": p.cfg_scale,
                "sampler_name": p.sampler, "scheduler": p.scheduler,
                "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable",
            },
        },
        "78": {
            "class_type": "KSamplerAdvanced", "_meta": {"title": "Sampler (low noise)"},
            "inputs": {
                "model": ["86", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["81", 0],
                "add_noise": "disable", "noise_seed": 0, "steps": p.steps, "cfg": p.cfg_scale,
                "sampler_name": p.sampler, "scheduler": p.scheduler,
                "start_at_step": split, "end_at_step": p.steps, "return_with_leftover_noise": "disable",
            },
        },
        "87": {
            "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"},
            "inputs": {"samples": ["78", 0], "vae": ["73", 0]},
        },
        "88": {
            "class_type": "CreateVideo", "_meta": {"title": "Create Video"},
            "inputs": {"images": ["87", 0], "fps": fps},
        },
        "80": {
            "class_type": "SaveVideo", "_meta": {"title": "Save Video"},
            "inputs": {"video": ["88", 0], "filename_prefix": "video/ezcomfy_wan22", "format": "auto", "codec": "auto"},
        },
    }
