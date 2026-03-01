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
