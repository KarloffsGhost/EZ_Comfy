from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ez_comfy.comfyui.client import ComfyUIClient
from ez_comfy.models.classifier import classify_checkpoint

logger = logging.getLogger(__name__)

# Maps capability names → ComfyUI class_types that prove the capability exists
NODE_CAPABILITY_MAP: dict[str, list[str]] = {
    "adetailer":          ["ADetailer", "ADetailerPipe"],
    "ipadapter":          ["IPAdapterApply", "IPAdapterModelLoader", "IPAdapter"],
    "controlnet":         ["ControlNetLoader", "ControlNetApply"],
    "regional_prompting": ["RegionalPrompt", "RegionalSampler"],
    "animatediff":        ["AnimateDiffLoader", "AnimateDiffSampler", "ADE_AnimateDiffLoaderWithContext"],
    "face_restore":       ["FaceRestoreWithModel", "FaceRestoreCFWithModel"],
    "upscale_model":      ["UpscaleModelLoader", "ImageUpscaleWithModel"],
    "svd":                ["ImageOnlyCheckpointLoader", "SVD_img2vid_Conditioning"],
    "stable_audio":       ["EmptyLatentAudio", "VAEDecodeAudio"],
}

# ComfyUI built-in class_types (used to distinguish custom nodes)
_BUILTIN_CLASS_TYPES = {
    "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode",
    "EmptyLatentImage", "SaveImage", "LoadImage", "LoraLoader", "VAELoader",
    "UpscaleModelLoader", "ImageUpscaleWithModel", "ControlNetLoader",
    "ControlNetApply", "SetLatentNoiseMask", "LatentUpscale", "ImageScale",
    "CLIPLoader", "VAEEncodeForInpaint", "VAEEncode", "ImageOnlyCheckpointLoader",
    "SVD_img2vid_Conditioning", "SaveAnimatedWEBP", "EmptyLatentAudio",
    "VAEDecodeAudio", "SaveAudio", "CLIPVisionEncode", "ConditioningStableAudio",
}


def has_capability(capability: str, discovered_class_types: set[str]) -> bool:
    """Returns True if ANY required class_type for this capability is present."""
    required = NODE_CAPABILITY_MAP.get(capability, [])
    return any(ct in discovered_class_types for ct in required)


@dataclass
class ModelInfo:
    filename: str
    size_bytes: int
    family: str
    variant: str | None = None


@dataclass
class LoRAInfo:
    filename: str
    size_bytes: int
    compatible_families: list[str] = field(default_factory=list)
    ambiguous: bool = False


@dataclass
class ComfyUIInventory:
    checkpoints: list[ModelInfo] = field(default_factory=list)
    loras: list[LoRAInfo] = field(default_factory=list)
    vaes: list[str] = field(default_factory=list)
    upscale_models: list[str] = field(default_factory=list)
    clip_models: list[str] = field(default_factory=list)
    controlnet_models: list[str] = field(default_factory=list)
    discovered_class_types: set[str] = field(default_factory=set)
    samplers: list[str] = field(default_factory=list)
    schedulers: list[str] = field(default_factory=list)


async def scan_inventory(client: ComfyUIClient) -> ComfyUIInventory:
    """Query ComfyUI API to build a full inventory of installed models and nodes."""
    inventory = ComfyUIInventory()

    # Discover all class_types from full /object_info dump
    try:
        all_info = await client.get_object_info()
        inventory.discovered_class_types = set(all_info.keys())
    except Exception as exc:
        logger.warning("Failed to fetch full object_info: %s", exc)
        all_info = {}

    # Checkpoints
    inventory.checkpoints = await _get_checkpoints(client, all_info)

    # LoRAs
    inventory.loras = await _get_loras(client, all_info)

    # VAEs
    inventory.vaes = _extract_list(all_info, "VAELoader", "vae_name")

    # Upscale models
    inventory.upscale_models = _extract_list(all_info, "UpscaleModelLoader", "model_name")

    # CLIP models
    inventory.clip_models = _extract_list(all_info, "CLIPLoader", "clip_name")

    # ControlNet models
    inventory.controlnet_models = _extract_list(all_info, "ControlNetLoader", "control_net_name")

    # Samplers and schedulers
    ksampler = all_info.get("KSampler", {})
    required = ksampler.get("input", {}).get("required", {})
    inventory.samplers = required.get("sampler_name", [[]])[0] if "sampler_name" in required else []
    inventory.schedulers = required.get("scheduler", [[]])[0] if "scheduler" in required else []

    logger.info(
        "Inventory: %d checkpoints, %d LoRAs, %d VAEs, %d upscalers, %d custom class_types",
        len(inventory.checkpoints),
        len(inventory.loras),
        len(inventory.vaes),
        len(inventory.upscale_models),
        len(inventory.discovered_class_types - _BUILTIN_CLASS_TYPES),
    )
    return inventory


async def _get_checkpoints(client: ComfyUIClient, all_info: dict) -> list[ModelInfo]:
    names = _extract_list(all_info, "CheckpointLoaderSimple", "ckpt_name")
    results = []
    for name in names:
        family, variant = classify_checkpoint(name)
        results.append(ModelInfo(filename=name, size_bytes=0, family=family, variant=variant))
    return results


async def _get_loras(client: ComfyUIClient, all_info: dict) -> list[LoRAInfo]:
    names = _extract_list(all_info, "LoraLoader", "lora_name")
    results = []
    for name in names:
        families, ambiguous = _infer_lora_family(name, size_bytes=0)
        results.append(LoRAInfo(filename=name, size_bytes=0, compatible_families=families, ambiguous=ambiguous))
    return results


def _extract_list(all_info: dict, node_class: str, input_key: str) -> list[str]:
    node = all_info.get(node_class, {})
    required = node.get("input", {}).get("required", {})
    entry = required.get(input_key, [[]])
    if entry and isinstance(entry[0], list):
        return entry[0]
    return []


def _infer_lora_family(filename: str, size_bytes: int) -> tuple[list[str], bool]:
    """Infer LoRA compatible families from filename/size. Returns (families, ambiguous)."""
    lower = filename.lower()

    # Explicit family keywords
    if any(k in lower for k in ("flux",)):
        return ["flux"], False
    if any(k in lower for k in ("xl", "sdxl", "pony", "illustrious")):
        return ["sdxl"], False
    if any(k in lower for k in ("sd3",)):
        return ["sd3"], False

    # Size-based inference (if available)
    if size_bytes > 300 * 1024 * 1024:
        return ["sd15"], False
    if 50 * 1024 * 1024 <= size_bytes <= 300 * 1024 * 1024:
        return ["sd15"], False  # most common size range for SD1.5

    # No clear signal
    return ["sd15", "sdxl"], True
