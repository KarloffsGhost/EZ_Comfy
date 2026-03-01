from __future__ import annotations

import os

from ez_comfy.planner.planner import GenerationPlan

# Stable Audio Open requires a separate T5 text encoder.
# CheckpointLoaderSimple returns CLIP=None for this model.
# Standard Google T5-Base weights work (same architecture).
_DEFAULT_T5_CLIP = "t5_base_stable_audio.safetensors"
_DEFAULT_AUDIO_DURATION = 5.0


def build_audio_stable(plan: GenerationPlan) -> dict:
    """
    Generate audio using Stable Audio Open.

    Special wiring required: CheckpointLoaderSimple returns CLIP=None for
    stable_audio_open_1.0.safetensors. A separate CLIPLoader loads the T5
    text encoder (t5_base_stable_audio.safetensors). Both the model VAE
    (output 2 from checkpoint) and the T5 CLIP are wired separately.

    Node layout:
      CheckpointLoaderSimple → KSampler → VAEDecodeAudio → SaveAudio
      CLIPLoader ────────────→ CLIPTextEncode×2 ─────────────────────┘
      EmptyLatentAudio ────────────────────────────────────────────────┘
    """
    p = plan.params
    nodes: dict = {}

    # Audio checkpoint (CLIP output will be None — not used directly)
    nodes["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Audio Checkpoint"},
        "inputs": {"ckpt_name": plan.checkpoint},
    }

    # Separate T5 text encoder — required for stable_audio_open
    t5_clip_name = os.environ.get("EZCOMFY_AUDIO_T5_CLIP", _DEFAULT_T5_CLIP)
    nodes["30"] = {
        "class_type": "CLIPLoader",
        "_meta": {"title": "T5 Text Encoder"},
        "inputs": {"clip_name": t5_clip_name, "type": "stable_audio"},
    }

    # Encode prompts via T5 (not the checkpoint CLIP)
    nodes["2"] = {
        "class_type": "CLIPTextEncode", "_meta": {"title": "Positive Prompt"},
        "inputs": {"text": plan.prompt, "clip": ["30", 0]},
    }
    nodes["3"] = {
        "class_type": "CLIPTextEncode", "_meta": {"title": "Negative Prompt"},
        "inputs": {"text": plan.negative_prompt, "clip": ["30", 0]},
    }

    # Empty latent audio — duration in seconds at 44100Hz sample rate
    audio_duration = float(os.environ.get("EZCOMFY_AUDIO_DURATION", _DEFAULT_AUDIO_DURATION))
    nodes["4"] = {
        "class_type": "EmptyLatentAudio",
        "_meta": {"title": "Empty Latent Audio"},
        "inputs": {"seconds": audio_duration, "batch_size": p.batch_size},
    }

    nodes["5"] = {
        "class_type": "KSampler", "_meta": {"title": "KSampler"},
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg_scale,
            "sampler_name": p.sampler, "scheduler": p.scheduler,
            "denoise": 1.0,
        },
    }
    # VAE decode uses the checkpoint's VAE (output index 2)
    nodes["6"] = {
        "class_type": "VAEDecodeAudio",
        "_meta": {"title": "VAE Decode Audio"},
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    }
    nodes["7"] = {
        "class_type": "SaveAudio",
        "_meta": {"title": "Save Audio"},
        "inputs": {"audio": ["6", 0], "filename_prefix": "ezcomfy_audio"},
    }
    return nodes
