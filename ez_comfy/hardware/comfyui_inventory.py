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
    # Which ComfyUI loader node can load this file: "checkpoint" (CheckpointLoaderSimple) or
    # "diffusion_model" (UNETLoader). Sending a file to the wrong loader fails at run time.
    loader: str = "checkpoint"
    folder: str | None = None   # sub-folder inside the models folder, e.g. "SDXL"


CHECKPOINT_LOADER = "checkpoint"
DIFFUSION_MODEL_LOADER = "diffusion_model"


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
    # Files only UNETLoader / CLIPVisionLoader / DualCLIPLoader can load. `checkpoints` keeps
    # holding checkpoint-loader files only, so existing consumers never see these.
    diffusion_models: list[ModelInfo] = field(default_factory=list)
    clip_vision_models: list[str] = field(default_factory=list)
    dual_clip_models: list[str] = field(default_factory=list)

    def all_models(self) -> list[ModelInfo]:
        """Checkpoints then diffusion models, for display and matching (not for loader selection)."""
        return [*self.checkpoints, *self.diffusion_models]

    def models_for_loader(self, loader: str) -> list[ModelInfo]:
        """Only the files the given loader can load."""
        return [model for model in self.all_models() if model.loader == loader]


async def scan_inventory(
    client: ComfyUIClient, overrides: dict[str, str] | None = None
) -> ComfyUIInventory:
    """Query ComfyUI API to build a full inventory of installed models and nodes.

    `overrides` maps an installed file name to a model family and wins over the classifier.
    """
    inventory = ComfyUIInventory()

    # Discover all class_types from full /object_info dump
    try:
        all_info = await client.get_object_info()
        inventory.discovered_class_types = set(all_info.keys())
    except Exception as exc:
        logger.warning("Failed to fetch full object_info: %s", exc)
        all_info = {}

    # Checkpoints
    inventory.checkpoints = await _get_checkpoints(client, all_info, overrides)

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
    inventory.samplers = _extract_list(all_info, "KSampler", "sampler_name")
    inventory.schedulers = _extract_list(all_info, "KSampler", "scheduler")

    # Files that only other loaders can load (Wan/Flux/Qwen-style diffusion models, CLIP vision, dual CLIP)
    inventory.diffusion_models = _get_diffusion_models(all_info, overrides)
    inventory.clip_vision_models = _extract_list(all_info, "CLIPVisionLoader", "clip_name")
    inventory.dual_clip_models = _merge_unique(
        _extract_list(all_info, "DualCLIPLoader", "clip_name1"),
        _extract_list(all_info, "DualCLIPLoader", "clip_name2"),
    )

    logger.info(
        "Inventory: %d checkpoints, %d diffusion models, %d LoRAs, %d VAEs, %d upscalers, %d custom class_types",
        len(inventory.checkpoints),
        len(inventory.diffusion_models),
        len(inventory.loras),
        len(inventory.vaes),
        len(inventory.upscale_models),
        len(inventory.discovered_class_types - _BUILTIN_CLASS_TYPES),
    )
    return inventory


async def _get_checkpoints(
    client: ComfyUIClient, all_info: dict, overrides: dict[str, str] | None = None
) -> list[ModelInfo]:
    names = _extract_list(all_info, "CheckpointLoaderSimple", "ckpt_name")
    results = []
    for name in names:
        family, variant = classify_checkpoint(name, overrides=overrides)
        results.append(ModelInfo(
            filename=name, size_bytes=0, family=family, variant=variant,
            loader=CHECKPOINT_LOADER, folder=_folder_of(name),
        ))
    return results


def _get_diffusion_models(all_info: dict, overrides: dict[str, str] | None = None) -> list[ModelInfo]:
    results = []
    for name in _extract_list(all_info, "UNETLoader", "unet_name"):
        family, variant = classify_checkpoint(name, overrides=overrides)
        results.append(ModelInfo(
            filename=name, size_bytes=0, family=family, variant=variant,
            loader=DIFFUSION_MODEL_LOADER, folder=_folder_of(name),
        ))
    return results


def _folder_of(filename: str) -> str | None:
    """'SDXL\\base.safetensors' -> 'SDXL'; a bare file name -> None."""
    parts = filename.replace("\\", "/").rsplit("/", 1)
    return parts[0] if len(parts) == 2 else None


def _merge_unique(*lists: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for items in lists:
        for item in items:
            seen.setdefault(item, None)
    return list(seen)


async def _get_loras(client: ComfyUIClient, all_info: dict) -> list[LoRAInfo]:
    names = _extract_list(all_info, "LoraLoader", "lora_name")
    results = []
    for name in names:
        families, ambiguous = _infer_lora_family(name, size_bytes=0)
        results.append(LoRAInfo(filename=name, size_bytes=0, compatible_families=families, ambiguous=ambiguous))
    return results


def _extract_list(all_info: dict, node_class: str, input_key: str) -> list[str]:
    """Return a combo input's option list from an /object_info dump.

    ComfyUI emits two shapes for combo inputs and both are live in 0.42:
      legacy: [["a.safetensors", "b.safetensors"], {...}]
      COMBO:  ["COMBO", {"options": ["a.safetensors", "b.safetensors"], ...}]  (e.g. UpscaleModelLoader)
    A missing node or input yields []; an unrecognised shape is logged and yields [].
    """
    node = all_info.get(node_class)
    if not node:
        logger.debug("%s not in /object_info; no %s options", node_class, input_key)
        return []
    required = node.get("input", {}).get("required", {})
    if input_key not in required:
        logger.debug("%s has no required input %r; no options", node_class, input_key)
        return []

    entry = required[input_key]
    if isinstance(entry, list) and entry:
        first = entry[0]
        if isinstance(first, list):
            return first
        if first == "COMBO" and len(entry) > 1 and isinstance(entry[1], dict):
            options = entry[1].get("options")
            if isinstance(options, list):
                return options

    logger.warning(
        "Unrecognised combo shape for %s.%s: %.120r", node_class, input_key, entry
    )
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
