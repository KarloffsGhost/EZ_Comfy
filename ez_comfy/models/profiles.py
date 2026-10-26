from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSettings:
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    clip_skip: int = 1
    denoise_default: float = 0.7


@dataclass
class PromptSyntax:
    emphasis_format: str          # "weighted" = (word:1.3), "none" = Flux
    quality_prefix: str | None    # prepend to positive
    quality_suffix: str | None    # append to positive
    negative_required: bool       # False for Flux
    default_negative: str
    supports_break: bool = False


@dataclass
class ModelProfile:
    family: str
    native_resolution: tuple[int, int]
    resolution_buckets: list[tuple[int, int]]
    vram_requirement_gb: float
    default_settings: ModelSettings
    prompt_syntax: PromptSyntax
    supports_img2img: bool = True
    supports_inpaint: bool = True
    supports_controlnet: bool = True
    clip_type: str = "clip_l"
    vae_type: str = "built_in"


# ---------------------------------------------------------------------------
# Resolution buckets
# ---------------------------------------------------------------------------
SDXL_BUCKETS: list[tuple[int, int]] = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
]

SD15_BUCKETS: list[tuple[int, int]] = [
    (512, 512),
    (768, 512),
    (512, 768),
    (640, 448),
    (448, 640),
    (768, 432),
    (432, 768),
]

# Flux uses same buckets as SDXL
FLUX_BUCKETS = SDXL_BUCKETS

# ---------------------------------------------------------------------------
# Default negative prompts per family
# ---------------------------------------------------------------------------
_NEG_SDXL = (
    "worst quality, low quality, normal quality, lowres, watermark, "
    "text, signature, blurry, deformed"
)
_NEG_SD15 = (
    "worst quality, low quality, normal quality, lowres, bad anatomy, "
    "bad hands, extra digits, fewer digits, watermark, signature"
)
_NEG_ANIME = (
    "worst quality, low quality, lowres, bad anatomy, bad hands, "
    "text, error, ugly, duplicate, morbid"
)
_NEG_PONY = "score_4, score_3, score_2, score_1, worst quality, low quality"

# ---------------------------------------------------------------------------
# Prompt syntax per family
# ---------------------------------------------------------------------------
_SYNTAX_SDXL = PromptSyntax(
    emphasis_format="weighted",
    quality_prefix=None,
    quality_suffix=None,
    negative_required=True,
    default_negative=_NEG_SDXL,
    supports_break=True,
)
_SYNTAX_SD15 = PromptSyntax(
    emphasis_format="weighted",
    quality_prefix=None,
    quality_suffix=None,
    negative_required=True,
    default_negative=_NEG_SD15,
)
_SYNTAX_SD15_ANIME = PromptSyntax(
    emphasis_format="weighted",
    quality_prefix=None,
    quality_suffix=", masterpiece, best quality",
    negative_required=True,
    default_negative=_NEG_ANIME,
)
_SYNTAX_FLUX = PromptSyntax(
    emphasis_format="none",
    quality_prefix=None,
    quality_suffix=None,
    negative_required=False,
    default_negative="",
)
# Default negative prompt of the Wan 2.2 14B text-to-video template (node "CLIPTextEncode" in
# video_wan2_2_14B_t2v.json, comfyui_workflow_templates_json 0.1.122). Chinese, as shipped.
_NEG_WAN = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW"
_SYNTAX_WAN = PromptSyntax(
    emphasis_format="none",
    quality_prefix=None,
    quality_suffix=None,
    negative_required=True,
    default_negative=_NEG_WAN,
)
_SYNTAX_PONY = PromptSyntax(
    emphasis_format="weighted",
    quality_prefix="score_9, score_8_up, score_7_up, ",
    quality_suffix=None,
    negative_required=True,
    default_negative=_NEG_PONY,
    supports_break=True,
)

# ---------------------------------------------------------------------------
# Family profiles
# ---------------------------------------------------------------------------
PROFILES: dict[str, ModelProfile] = {
    "sd15": ModelProfile(
        family="sd15",
        native_resolution=(512, 512),
        resolution_buckets=SD15_BUCKETS,
        vram_requirement_gb=4.0,
        default_settings=ModelSettings(steps=20, cfg=7.0, sampler="euler", scheduler="normal"),
        prompt_syntax=_SYNTAX_SD15,
    ),
    "sd15_lightning": ModelProfile(
        family="sd15_lightning",
        native_resolution=(512, 512),
        resolution_buckets=SD15_BUCKETS,
        vram_requirement_gb=4.0,
        default_settings=ModelSettings(steps=8, cfg=1.5, sampler="euler", scheduler="sgm_uniform"),
        prompt_syntax=_SYNTAX_SD15,
    ),
    "sdxl": ModelProfile(
        family="sdxl",
        native_resolution=(1024, 1024),
        resolution_buckets=SDXL_BUCKETS,
        vram_requirement_gb=8.0,
        default_settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_SYNTAX_SDXL,
        clip_type="clip_l+clip_g",
    ),
    "sdxl_lightning": ModelProfile(
        family="sdxl_lightning",
        native_resolution=(1024, 1024),
        resolution_buckets=SDXL_BUCKETS,
        vram_requirement_gb=8.0,
        default_settings=ModelSettings(steps=6, cfg=1.5, sampler="euler", scheduler="sgm_uniform"),
        prompt_syntax=_SYNTAX_SDXL,
        clip_type="clip_l+clip_g",
    ),
    "sdxl_turbo": ModelProfile(
        family="sdxl_turbo",
        native_resolution=(1024, 1024),
        resolution_buckets=SDXL_BUCKETS,
        vram_requirement_gb=8.0,
        default_settings=ModelSettings(steps=4, cfg=1.0, sampler="euler", scheduler="sgm_uniform"),
        prompt_syntax=_SYNTAX_SDXL,
        clip_type="clip_l+clip_g",
    ),
    "pony": ModelProfile(
        family="pony",
        native_resolution=(1024, 1024),
        resolution_buckets=SDXL_BUCKETS,
        vram_requirement_gb=8.0,
        default_settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras", clip_skip=2),
        prompt_syntax=_SYNTAX_PONY,
        clip_type="clip_l+clip_g",
    ),
    "flux": ModelProfile(
        family="flux",
        native_resolution=(1024, 1024),
        resolution_buckets=FLUX_BUCKETS,
        vram_requirement_gb=12.0,
        default_settings=ModelSettings(steps=20, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_SYNTAX_FLUX,
        clip_type="t5xxl",
        vae_type="flux_ae",
    ),
    "flux_schnell": ModelProfile(
        family="flux_schnell",
        native_resolution=(1024, 1024),
        resolution_buckets=FLUX_BUCKETS,
        vram_requirement_gb=12.0,
        default_settings=ModelSettings(steps=4, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_SYNTAX_FLUX,
        clip_type="t5xxl",
        vae_type="flux_ae",
    ),
    "sd3": ModelProfile(
        family="sd3",
        native_resolution=(1024, 1024),
        resolution_buckets=SDXL_BUCKETS,
        vram_requirement_gb=10.0,
        default_settings=ModelSettings(steps=28, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_SYNTAX_SDXL,
        clip_type="triple_clip",
    ),
    "cascade": ModelProfile(
        family="cascade",
        native_resolution=(1024, 1024),
        resolution_buckets=SDXL_BUCKETS,
        vram_requirement_gb=10.0,
        default_settings=ModelSettings(steps=20, cfg=4.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_SYNTAX_SDXL,
    ),
    "svd": ModelProfile(
        family="svd",
        native_resolution=(1024, 576),
        resolution_buckets=[(1024, 576), (576, 1024), (768, 768)],
        vram_requirement_gb=12.0,
        default_settings=ModelSettings(steps=25, cfg=2.5, sampler="euler", scheduler="normal"),
        prompt_syntax=PromptSyntax(
            emphasis_format="none",
            quality_prefix=None,
            quality_suffix=None,
            negative_required=False,
            default_negative="",
        ),
        supports_controlnet=False,
    ),
    # --- Modern families (WI-3) -------------------------------------------------------------
    # Source for every value below: the workflow templates shipped with ComfyUI Desktop 1.1.8
    # (comfyui_workflow_templates_json 0.1.122, ComfyUI 0.42.0), read 2026-10-25; the file the
    # value came from is named per family. Sizes are the official Hugging Face file sizes of the
    # template's diffusion model (GiB, from the template's own download URL). That is a lower
    # bound (weights only, no activations or text encoder), not a measured peak; WI-5 measures.
    # `scheduler` names the node the template uses when it is not a KSampler scheduler name.
    # These families are classified and described but not selectable for generation until
    # WI-5/WI-6 add recipes for them.
    "wan_video": ModelProfile(
        # video_wan2_2_14B_t2v.json, template default (the "fast LoRA" switch off):
        # EmptyHunyuanLatentVideo 640x640x81, two KSamplerAdvanced passes of one 20-step run
        # (high-noise expert steps 0-10, low-noise expert 10-20), cfg 3.5, euler / simple,
        # ModelSamplingSD3 shift 5, CreateVideo 16 fps. With the switch on the template instead
        # uses 4 steps / cfg 1 plus the lightx2v 4-step LoRAs (not wired yet).
        # Other attested sizes: 832x480 (text_to_video_wan), 1280x704 (video_wan2_2_5B_ti2v),
        # 512x512 (image_to_video_wan). wan2.2_t2v_*_14B_fp8_scaled.safetensors: 13.31 GiB.
        family="wan_video",
        native_resolution=(640, 640),
        resolution_buckets=[(640, 640), (832, 480), (1280, 704), (512, 512)],
        vram_requirement_gb=13.4,
        default_settings=ModelSettings(steps=20, cfg=3.5, sampler="euler", scheduler="simple"),
        prompt_syntax=_SYNTAX_WAN,
        supports_img2img=False,
        supports_inpaint=False,
        supports_controlnet=False,
        clip_type="umt5_xxl",
        vae_type="wan",
    ),
    "ltx_video": ModelProfile(
        # video_ltx2_3_t2v.json: EmptyLTXVLatentVideo 768x512x97, euler, CFGGuider cfg 1,
        # ManualSigmas "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
        # (8 steps) plus the distilled LoRA at 0.5. ltx-2.3-22b-dev-fp8.safetensors: 27.14 GiB.
        family="ltx_video",
        native_resolution=(768, 512),
        resolution_buckets=[(768, 512)],
        vram_requirement_gb=27.2,
        default_settings=ModelSettings(steps=8, cfg=1.0, sampler="euler", scheduler="ManualSigmas"),
        prompt_syntax=_SYNTAX_FLUX,
        supports_img2img=False,
        supports_inpaint=False,
        supports_controlnet=False,
        clip_type="gemma3_12b",
    ),
    "qwen_image": ModelProfile(
        # image_qwen_image_2_1_t2i.json: EmptyLatentImage 1024x1024, KSampler 25 steps / cfg 1 /
        # euler / simple. image_qwen_image.json also attests 1328x1328.
        # qwen_image_2.1_int8_convrot.safetensors: 6.76 GiB (qwen_image_fp8_e4m3fn: 19.03 GiB).
        family="qwen_image",
        native_resolution=(1024, 1024),
        resolution_buckets=[(1024, 1024), (1328, 1328)],
        vram_requirement_gb=6.8,
        default_settings=ModelSettings(steps=25, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_SYNTAX_FLUX,
        supports_controlnet=False,
        clip_type="qwen_vl",
    ),
    "hidream": ModelProfile(
        # hidream_i1_dev.json: EmptySD3LatentImage 1024x1024, KSampler 28 steps / cfg 1 / lcm /
        # normal, ModelSamplingSD3 shift 6. hidream_i1_dev_fp8.safetensors: 15.93 GiB.
        family="hidream",
        native_resolution=(1024, 1024),
        resolution_buckets=[(1024, 1024)],
        vram_requirement_gb=16.0,
        default_settings=ModelSettings(steps=28, cfg=1.0, sampler="lcm", scheduler="normal"),
        prompt_syntax=_SYNTAX_FLUX,
        supports_controlnet=False,
        clip_type="clip_l+clip_g+t5xxl+llama",
    ),
    "flux2": ModelProfile(
        # image_flux2_text_to_image.json: EmptyFlux2LatentImage 1024x1024, Flux2Scheduler 20
        # steps, FluxGuidance 4 (the guidance value lives in that node, not in cfg), euler via
        # SamplerCustomAdvanced. flux2_dev_fp8mixed.safetensors: 33.02 GiB.
        # FLUX.3 names classify as "flux3" but have no profile yet: the FLUX.3 weights are
        # gated, so their size could not be confirmed.
        family="flux2",
        native_resolution=(1024, 1024),
        resolution_buckets=[(1024, 1024)],
        vram_requirement_gb=33.1,
        default_settings=ModelSettings(steps=20, cfg=1.0, sampler="euler", scheduler="Flux2Scheduler"),
        prompt_syntax=_SYNTAX_FLUX,
        supports_controlnet=False,
        clip_type="mistral",
        vae_type="flux2",
    ),
    "stable_audio": ModelProfile(
        family="stable_audio",
        native_resolution=(0, 0),
        resolution_buckets=[],
        vram_requirement_gb=8.0,
        default_settings=ModelSettings(steps=100, cfg=7.0, sampler="dpmpp_3m_sde", scheduler="exponential"),
        prompt_syntax=PromptSyntax(
            emphasis_format="none",
            quality_prefix=None,
            quality_suffix=None,
            negative_required=True,
            default_negative="",
        ),
        supports_img2img=False,
        supports_inpaint=False,
        supports_controlnet=False,
    ),
    "unknown": ModelProfile(
        family="unknown",
        native_resolution=(512, 512),
        resolution_buckets=SD15_BUCKETS,
        vram_requirement_gb=4.0,
        default_settings=ModelSettings(steps=20, cfg=7.0, sampler="euler", scheduler="normal"),
        prompt_syntax=_SYNTAX_SD15,
    ),
}


def get_profile(family: str, variant: str | None = None) -> ModelProfile:
    """Get the profile for a family, accounting for fast variants."""
    if variant in ("lightning", "hyper", "lcm"):
        fast_key = f"{family}_{variant}" if f"{family}_{variant}" in PROFILES else f"{family}_lightning"
        if fast_key in PROFILES:
            return PROFILES[fast_key]
    if variant == "turbo":
        turbo_key = f"{family}_turbo"
        if turbo_key in PROFILES:
            return PROFILES[turbo_key]
    if variant == "schnell":
        schnell_key = f"{family}_schnell"
        if schnell_key in PROFILES:
            return PROFILES[schnell_key]
    return PROFILES.get(family, PROFILES["unknown"])


def snap_to_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Snap (width, height) to the nearest trained resolution bucket."""
    if not buckets:
        return width, height

    target_area = width * height
    target_ratio = width / height if height > 0 else 1.0

    best = buckets[0]
    best_score = float("inf")

    for bw, bh in buckets:
        b_ratio = bw / bh if bh > 0 else 1.0
        # Score = combination of aspect ratio difference and area difference
        ratio_diff = abs(target_ratio - b_ratio) / max(target_ratio, b_ratio)
        area_diff = abs(target_area - bw * bh) / max(target_area, bw * bh)
        score = ratio_diff * 0.7 + area_diff * 0.3
        if score < best_score:
            best_score = score
            best = (bw, bh)

    return best


# Expose ModelSettings and PromptSyntax at module level for use by catalog
__all__ = [
    "ModelSettings",
    "PromptSyntax",
    "ModelProfile",
    "PROFILES",
    "SDXL_BUCKETS",
    "SD15_BUCKETS",
    "FLUX_BUCKETS",
    "get_profile",
    "snap_to_bucket",
]
