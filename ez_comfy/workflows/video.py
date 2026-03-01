from __future__ import annotations

from ez_comfy.planner.planner import GenerationPlan

# SVD default frame/fps settings (plan does not carry video-specific fields)
_SVD_DEFAULT_FRAMES = 25
_SVD_DEFAULT_FPS = 8


def build_video_svd(plan: GenerationPlan) -> dict:
    """
    Animate a still image using Stable Video Diffusion (SVD-XT).

    Node layout:
      ImageOnlyCheckpointLoader → SVD_img2vid_Conditioning
      LoadImage ──────────────────────────────────────────┘
      KSampler → VAEDecode → SaveAnimatedWebp
    """
    p = plan.params
    nodes: dict = {}

    # SVD uses ImageOnlyCheckpointLoader (model, clip_vision, vae)
    nodes["1"] = {
        "class_type": "ImageOnlyCheckpointLoader",
        "_meta": {"title": "Load SVD Checkpoint"},
        "inputs": {"ckpt_name": plan.checkpoint},
    }

    ref_path = plan.reference_image_path or ""
    nodes["8"] = {
        "class_type": "LoadImage", "_meta": {"title": "Reference Image"},
        "inputs": {"image": ref_path, "upload": "image"},
    }

    # SVD conditioning: encodes reference image into positive/negative/latent
    nodes["20"] = {
        "class_type": "SVD_img2vid_Conditioning",
        "_meta": {"title": "SVD Conditioning"},
        "inputs": {
            "clip_vision": ["1", 1],
            "init_image": ["8", 0],
            "vae": ["1", 2],
            "width": p.width,
            "height": p.height,
            "video_frames": _SVD_DEFAULT_FRAMES,
            "motion_bucket_id": 127,
            "fps": _SVD_DEFAULT_FPS,
            "augmentation_level": 0.0,
        },
    }

    nodes["5"] = {
        "class_type": "KSampler", "_meta": {"title": "KSampler"},
        "inputs": {
            "model": ["1", 0],
            "positive": ["20", 0],
            "negative": ["20", 1],
            "latent_image": ["20", 2],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg_scale,
            "sampler_name": p.sampler, "scheduler": p.scheduler,
            "denoise": 1.0,
        },
    }
    nodes["6"] = {
        "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"},
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    }
    # Save all decoded frames as animated WebP (built into ComfyUI — note WEBP uppercase)
    nodes["7"] = {
        "class_type": "SaveAnimatedWEBP",
        "_meta": {"title": "Save Animated WEBP"},
        "inputs": {
            "images": ["6", 0],
            "filename_prefix": "ezcomfy_video",
            "fps": float(_SVD_DEFAULT_FPS),
            "lossless": False,
            "quality": 80,
            "method": "default",
        },
    }
    return nodes
