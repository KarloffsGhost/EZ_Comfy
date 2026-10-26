from __future__ import annotations

from dataclasses import dataclass, field

from ez_comfy.planner.intent import PipelineIntent
from ez_comfy.hardware.comfyui_inventory import has_capability


@dataclass
class Recipe:
    id: str
    name: str
    description: str
    intent: PipelineIntent
    priority: int
    when: str
    required_capabilities: list[str]
    requires_reference_image: bool
    requires_mask: bool
    supports_lora: bool
    supports_controlnet: bool
    builder: str                         # function name in workflow builders
    settings_overrides: dict | None = None


# ---------------------------------------------------------------------------
# v1 Recipe registry (10 core recipes)
# ---------------------------------------------------------------------------
RECIPES: list[Recipe] = [
    # --- txt2img ---
    Recipe(
        id="photo_realism_v1",
        name="Photo Realism (2-Pass)",
        description=(
            "Two-pass photorealistic pipeline: generate at 65% resolution, "
            "pixel-space lanczos upscale, then img2img refine (denoise=0.45) "
            "for maximum sharpness and skin/surface detail. "
            "No special capabilities required. "
            "Add ADetailer or Impact Pack for automatic face enhancement after generation."
        ),
        intent=PipelineIntent.TXT2IMG,
        priority=5,
        when="Portrait, photorealistic, cinematic, or face-focused prompts",
        required_capabilities=[],
        requires_reference_image=False,
        requires_mask=False,
        supports_lora=True,
        supports_controlnet=False,
        builder="build_photo_realism_v1",
    ),
    Recipe(
        id="txt2img_hires_fix",
        name="Hi-Res Fix",
        description="Generate at lower resolution then upscale and refine for higher detail",
        intent=PipelineIntent.TXT2IMG,
        priority=20,
        when="User requests high resolution, lots of detail, or quality",
        required_capabilities=[],
        requires_reference_image=False,
        requires_mask=False,
        supports_lora=True,
        supports_controlnet=False,
        builder="build_txt2img_hires_fix",
        settings_overrides=None,
    ),
    Recipe(
        id="txt2img_basic",
        name="Standard txt2img",
        description="Standard text-to-image generation",
        intent=PipelineIntent.TXT2IMG,
        priority=10,
        when="Default text-to-image",
        required_capabilities=[],
        requires_reference_image=False,
        requires_mask=False,
        supports_lora=True,
        supports_controlnet=False,
        builder="build_txt2img_basic",
    ),

    # --- img2img ---
    Recipe(
        id="img2img_controlnet_canny",
        name="ControlNet Canny",
        description="Keep image structure via Canny edge detection, change style/content",
        intent=PipelineIntent.IMG2IMG,
        priority=15,
        when="User wants to keep structure of reference image but change style",
        required_capabilities=["controlnet"],
        requires_reference_image=True,
        requires_mask=False,
        supports_lora=True,
        supports_controlnet=True,
        builder="build_img2img_controlnet_canny",
    ),
    Recipe(
        id="img2img_basic",
        name="Standard img2img",
        description="Modify an existing image with a text prompt",
        intent=PipelineIntent.IMG2IMG,
        priority=10,
        when="Modify existing image",
        required_capabilities=[],
        requires_reference_image=True,
        requires_mask=False,
        supports_lora=True,
        supports_controlnet=False,
        builder="build_img2img_basic",
    ),

    # --- inpaint ---
    Recipe(
        id="inpaint_basic",
        name="Standard Inpaint",
        description="Replace masked area of an image with new content",
        intent=PipelineIntent.INPAINT,
        priority=10,
        when="Replace masked area of image",
        required_capabilities=[],
        requires_reference_image=True,
        requires_mask=True,
        supports_lora=True,
        supports_controlnet=False,
        builder="build_inpaint_basic",
    ),

    # --- upscale ---
    Recipe(
        id="upscale_refine",
        name="Upscale + Refine",
        description="Upscale then img2img refine for sharpness",
        intent=PipelineIntent.UPSCALE,
        priority=20,
        when="Upscale and add detail with refinement pass",
        required_capabilities=["upscale_model"],
        requires_reference_image=True,
        requires_mask=False,
        supports_lora=False,
        supports_controlnet=False,
        builder="build_upscale_refine",
        settings_overrides={"steps": 12, "denoise_default": 0.3},
    ),
    Recipe(
        id="upscale_simple",
        name="Model Upscale",
        description="Fast upscale using ESRGAN-style model",
        intent=PipelineIntent.UPSCALE,
        priority=10,
        when="Fast upscale without quality refinement",
        required_capabilities=["upscale_model"],
        requires_reference_image=True,
        requires_mask=False,
        supports_lora=False,
        supports_controlnet=False,
        builder="build_upscale_simple",
    ),

    # --- video ---
    Recipe(
        id="video_svd",
        name="SVD img2vid",
        description="Animate a still image using Stable Video Diffusion",
        intent=PipelineIntent.VIDEO,
        priority=10,
        when="Animate a still image into a short video",
        required_capabilities=["svd"],
        requires_reference_image=True,
        requires_mask=False,
        supports_lora=False,
        supports_controlnet=False,
        builder="build_video_svd",
    ),

    # --- audio ---
    Recipe(
        id="audio_stable",
        name="Stable Audio Open",
        description="Generate sound effects or ambient audio",
        intent=PipelineIntent.AUDIO,
        priority=10,
        when="Generate audio from text description",
        required_capabilities=["stable_audio"],
        requires_reference_image=False,
        requires_mask=False,
        supports_lora=False,
        supports_controlnet=False,
        builder="build_audio_stable",
    ),
]

# Build lookup by id
_RECIPE_BY_ID: dict[str, Recipe] = {r.id: r for r in RECIPES}


def select_recipe(
    intent: PipelineIntent,
    prompt: str,
    has_reference_image: bool,
    has_mask: bool,
    discovered_class_types: set[str],
    recipe_override: str | None = None,
) -> Recipe:
    """Select the best recipe for the given intent and context."""
    if recipe_override:
        recipe = _RECIPE_BY_ID.get(recipe_override)
        if not recipe:
            raise ValueError(f"Unknown recipe: {recipe_override!r}")
        # Validate that required capabilities are present even for overrides;
        # caller can ignore the missing list but should not silently fail later.
        missing = [
            cap for cap in recipe.required_capabilities
            if not has_capability(cap, discovered_class_types)
        ]
        if missing:
            import warnings
            warnings.warn(
                f"Recipe {recipe_override!r} requires capabilities not detected in ComfyUI: "
                f"{missing}. Generation may fail.",
                stacklevel=2,
            )
        return recipe

    # Filter: intent match
    candidates = [r for r in RECIPES if r.intent == intent]

    # Filter: reference image + mask requirements
    if not has_reference_image:
        candidates = [r for r in candidates if not r.requires_reference_image]
    if not has_mask:
        candidates = [r for r in candidates if not r.requires_mask]

    # Filter: capability availability
    def caps_available(recipe: Recipe) -> bool:
        return all(has_capability(cap, discovered_class_types) for cap in recipe.required_capabilities)

    available = [r for r in candidates if caps_available(r)]

    # Fallback to recipes with no required capabilities if nothing else qualifies
    if not available:
        available = [r for r in candidates if not r.required_capabilities]

    if not available:
        # txt2img is the only intent with a capability-free fallback.
        # Other intents cannot meaningfully substitute — raise a clear error.
        if intent.value == "txt2img":
            return _RECIPE_BY_ID["txt2img_basic"]
        raise RuntimeError(
            f"No recipe available for intent={intent.value!r}. "
            "Required capabilities may be missing from your ComfyUI installation. "
            "Run 'ez-comfy check' to see installed capabilities, or install the "
            "required custom nodes."
        )

    prompt_lower = prompt.lower()

    # Photo realism: prefer 2-pass pixel pipeline for portrait/photorealism prompts
    _photo_keywords = {
        "portrait", "photorealistic", "realistic", "photo", "photograph",
        "person", "face", "woman", "man", "model", "cinematic", "headshot",
    }
    if any(kw in prompt_lower for kw in _photo_keywords):
        for r in available:
            if r.id == "photo_realism_v1":
                return r

    # Hires fix: boost priority if prompt suggests high quality/detail
    hires_keywords = {"detail", "detailed", "hires", "high res", "4k", "8k", "ultra", "sharp", "crisp"}
    if any(kw in prompt_lower for kw in hires_keywords):
        for r in available:
            if r.id == "txt2img_hires_fix":
                return r

    # Pick highest priority
    available.sort(key=lambda r: r.priority, reverse=True)
    return available[0]


def get_recipe(recipe_id: str) -> Recipe:
    recipe = _RECIPE_BY_ID.get(recipe_id)
    if not recipe:
        raise ValueError(f"Unknown recipe: {recipe_id!r}")
    return recipe


def list_recipes() -> list[Recipe]:
    return list(RECIPES)
