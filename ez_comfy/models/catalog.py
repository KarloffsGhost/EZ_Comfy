from __future__ import annotations

import re
from dataclasses import dataclass, field

from ez_comfy.hardware.probe import HardwareProfile
from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory
from ez_comfy.models.profiles import ModelSettings, PromptSyntax, _NEG_SDXL, _NEG_SD15, _NEG_ANIME, _NEG_PONY


@dataclass
class LoRARecommendation:
    name: str
    filename: str
    source: str
    strength: float
    reason: str


@dataclass
class ModelCatalogEntry:
    id: str
    name: str
    family: str
    variant: str | None
    source: str
    filename: str                        # expected checkpoint filename (partial match)
    size_gb: float
    vram_min_gb: float
    vram_recommended_gb: float
    strengths: list[str]
    weaknesses: list[str]
    best_for: list[str]
    style_tags: list[str]
    settings: ModelSettings
    prompt_syntax: PromptSyntax
    recommended_loras: list[LoRARecommendation] = field(default_factory=list)
    recommended_vae: str | None = None
    requires_clip: str | None = None
    required_capabilities: list[str] = field(default_factory=list)
    download_command: str = ""
    license: str = "creativeml-openrail-m"
    gated: bool = False

    @property
    def effective_family(self) -> str:
        """Family string including variant for profile lookup."""
        if self.variant in ("lightning", "turbo", "schnell"):
            return f"{self.family}_{self.variant}"
        return self.family


@dataclass
class ModelRecommendation:
    entry: ModelCatalogEntry
    score: float
    installed: bool
    fits_vram: bool
    match_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt syntax helpers
# ---------------------------------------------------------------------------
def _sdxl_syntax() -> PromptSyntax:
    return PromptSyntax(emphasis_format="weighted", quality_prefix=None, quality_suffix=None,
                        negative_required=True, default_negative=_NEG_SDXL, supports_break=True)

def _sd15_syntax() -> PromptSyntax:
    return PromptSyntax(emphasis_format="weighted", quality_prefix=None, quality_suffix=None,
                        negative_required=True, default_negative=_NEG_SD15)

def _flux_syntax() -> PromptSyntax:
    return PromptSyntax(emphasis_format="none", quality_prefix=None, quality_suffix=None,
                        negative_required=False, default_negative="")

def _pony_syntax() -> PromptSyntax:
    return PromptSyntax(emphasis_format="weighted",
                        quality_prefix="score_9, score_8_up, score_7_up, ",
                        quality_suffix=None, negative_required=True,
                        default_negative=_NEG_PONY, supports_break=True)

def _anime_syntax() -> PromptSyntax:
    return PromptSyntax(emphasis_format="weighted", quality_prefix=None,
                        quality_suffix=", masterpiece, best quality",
                        negative_required=True, default_negative=_NEG_ANIME)


# ---------------------------------------------------------------------------
# v1 Model Catalog (~15 curated entries)
# ---------------------------------------------------------------------------
MODEL_CATALOG: list[ModelCatalogEntry] = [

    # -------------------------------------------------------------------------
    # Photorealism
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="realvis_xl_v50_lightning",
        name="RealVisXL V5.0 Lightning",
        family="sdxl", variant="lightning",
        source="SG161222/RealVisXL_V5.0_Lightning",
        filename="realvisxlV50_lightning",
        size_gb=6.5, vram_min_gb=6.0, vram_recommended_gb=8.0,
        strengths=["photorealism", "portraits", "speed", "cinematic"],
        weaknesses=["anime", "abstract", "text_rendering"],
        best_for=["photos", "portraits", "landscapes", "product shots"],
        style_tags=["realistic", "photographic", "cinematic", "editorial"],
        settings=ModelSettings(steps=6, cfg=1.5, sampler="euler", scheduler="sgm_uniform"),
        prompt_syntax=_sdxl_syntax(),
        download_command="huggingface-cli download SG161222/RealVisXL_V5.0_Lightning --include '*.safetensors'",
        recommended_loras=[
            LoRARecommendation("Detail Tweaker XL", "detail_tweaker_xl.safetensors",
                               "https://civitai.com/models/122359", 0.5,
                               "Sharpens fine detail without artifacts"),
        ],
    ),

    ModelCatalogEntry(
        id="juggernaut_xl_v9",
        name="Juggernaut XL v9",
        family="sdxl", variant=None,
        source="RunDiffusion/Juggernaut-XL-v9",
        filename="juggernautXL_v9",
        size_gb=6.6, vram_min_gb=8.0, vram_recommended_gb=12.0,
        strengths=["photorealism", "cinematic", "editorial", "portraits", "landscapes"],
        weaknesses=["anime", "text_rendering"],
        best_for=["editorial photography", "cinematic scenes", "high quality portraits"],
        style_tags=["realistic", "cinematic", "editorial", "dramatic"],
        settings=ModelSettings(steps=30, cfg=6.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="huggingface-cli download RunDiffusion/Juggernaut-XL-v9 --include '*.safetensors'",
    ),

    ModelCatalogEntry(
        id="realistic_vision_v51",
        name="Realistic Vision V5.1",
        family="sd15", variant=None,
        source="SG161222/Realistic_Vision_V5.1_noVAE",
        filename="realisticVisionV51",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["photorealism", "portraits", "cinematic"],
        weaknesses=["anime", "abstract"],
        best_for=["realistic photos", "portraits", "product photography"],
        style_tags=["realistic", "photographic"],
        settings=ModelSettings(steps=20, cfg=7.0, sampler="euler", scheduler="normal"),
        prompt_syntax=_sd15_syntax(),
        download_command="huggingface-cli download SG161222/Realistic_Vision_V5.1_noVAE --include '*.safetensors'",
        recommended_vae="vae-ft-mse-840000-ema-pruned.safetensors",
    ),

    # -------------------------------------------------------------------------
    # Stylized / Artistic
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="dreamshaper_xl_lightning",
        name="DreamShaper XL Lightning",
        family="sdxl", variant="lightning",
        source="Lykon/dreamshaper-xl-lightning",
        filename="dreamshaperXL_lightningDPMSDE",
        size_gb=6.5, vram_min_gb=6.0, vram_recommended_gb=8.0,
        strengths=["versatile", "artistic", "fantasy", "speed", "stylized"],
        weaknesses=["text_rendering", "architectural"],
        best_for=["creative scenes", "fantasy art", "concept art", "illustrations"],
        style_tags=["artistic", "stylized", "fantasy", "vibrant"],
        settings=ModelSettings(steps=6, cfg=2.0, sampler="dpmpp_sde", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="huggingface-cli download Lykon/dreamshaper-xl-lightning --include '*.safetensors'",
    ),

    ModelCatalogEntry(
        id="dreamshaper_8",
        name="DreamShaper 8",
        family="sd15", variant=None,
        source="Lykon/DreamShaper",
        filename="dreamshaper_8",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["versatile", "artistic", "fantasy", "photorealism", "stylized"],
        weaknesses=["text_rendering"],
        best_for=["general purpose", "fantasy", "concept art", "stylized photos"],
        style_tags=["artistic", "versatile", "fantasy", "stylized"],
        settings=ModelSettings(steps=20, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sd15_syntax(),
        download_command="huggingface-cli download Lykon/DreamShaper --include '*.safetensors'",
    ),

    # -------------------------------------------------------------------------
    # Anime / Illustration
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="pony_diffusion_v6_xl",
        name="Pony Diffusion V6 XL",
        family="pony", variant=None,
        source="https://civitai.com/models/257749",
        filename="ponyDiffusionV6XL",
        size_gb=6.5, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["anime", "characters", "stylized", "illustration"],
        weaknesses=["photorealism", "text_rendering", "requires score tags"],
        best_for=["anime characters", "stylized illustration", "fantasy characters"],
        style_tags=["anime", "illustration", "stylized", "vibrant"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras", clip_skip=2),
        prompt_syntax=_pony_syntax(),
        download_command="Download from: https://civitai.com/models/257749",
        license="creativeml-openrail-m",
    ),

    ModelCatalogEntry(
        id="anything_v5",
        name="Anything V5",
        family="sd15", variant=None,
        source="stablediffusionapi/anything-v5",
        filename="anything_v5",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["anime", "illustration", "characters"],
        weaknesses=["photorealism", "text_rendering"],
        best_for=["anime art", "manga style", "illustrated characters"],
        style_tags=["anime", "illustration", "manga"],
        settings=ModelSettings(steps=20, cfg=7.0, sampler="euler", scheduler="normal", clip_skip=2),
        prompt_syntax=_anime_syntax(),
        download_command="huggingface-cli download stablediffusionapi/anything-v5 --include '*.safetensors'",
    ),

    # -------------------------------------------------------------------------
    # General / Prompt-Adherent
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="flux_dev",
        name="Flux.1 Dev",
        family="flux", variant=None,
        source="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev",
        size_gb=23.8, vram_min_gb=12.0, vram_recommended_gb=16.0,
        strengths=["prompt_adherence", "text_rendering", "complex_scenes", "photorealism"],
        weaknesses=["speed", "vram_hungry", "anime"],
        best_for=["text in images", "complex compositions", "highest quality output"],
        style_tags=["photographic", "realistic", "artistic", "detailed"],
        settings=ModelSettings(steps=20, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_flux_syntax(),
        download_command="huggingface-cli download black-forest-labs/FLUX.1-dev --include '*.safetensors'",
        requires_clip="t5xxl_fp16.safetensors",
        gated=True,
        license="flux-1-dev-non-commercial-license",
    ),

    ModelCatalogEntry(
        id="flux_schnell",
        name="Flux.1 Schnell",
        family="flux", variant="schnell",
        source="black-forest-labs/FLUX.1-schnell",
        filename="flux1-schnell",
        size_gb=23.8, vram_min_gb=12.0, vram_recommended_gb=16.0,
        strengths=["speed", "prompt_adherence", "text_rendering"],
        weaknesses=["quality_vs_dev", "vram_hungry", "anime"],
        best_for=["fast high-quality generation", "text rendering", "rapid prototyping"],
        style_tags=["photographic", "realistic", "artistic"],
        settings=ModelSettings(steps=4, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_flux_syntax(),
        download_command="huggingface-cli download black-forest-labs/FLUX.1-schnell --include '*.safetensors'",
        requires_clip="t5xxl_fp16.safetensors",
        license="apache-2.0",
    ),

    ModelCatalogEntry(
        id="sdxl_base_10",
        name="SDXL Base 1.0",
        family="sdxl", variant=None,
        source="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0",
        size_gb=6.9, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["versatile", "quality", "photorealism", "artistic"],
        weaknesses=["speed", "text_rendering"],
        best_for=["general purpose", "high quality images", "versatile prompts"],
        style_tags=["versatile", "photographic", "artistic"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include '*.safetensors'",
        license="openrail++",
    ),

    # -------------------------------------------------------------------------
    # Video
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="svd_xt",
        name="Stable Video Diffusion XT",
        family="svd", variant=None,
        source="stabilityai/stable-video-diffusion-img2vid-xt",
        filename="svd_xt",
        size_gb=9.5, vram_min_gb=12.0, vram_recommended_gb=14.0,
        strengths=["video", "animation", "img2vid"],
        weaknesses=["text_to_video", "long_videos"],
        best_for=["animating still images", "short video clips"],
        style_tags=["video", "animation", "motion"],
        settings=ModelSettings(steps=25, cfg=2.5, sampler="euler", scheduler="normal"),
        prompt_syntax=PromptSyntax(emphasis_format="none", quality_prefix=None,
                                   quality_suffix=None, negative_required=False, default_negative=""),
        download_command="huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include '*.safetensors'",
        gated=True,
    ),

    # -------------------------------------------------------------------------
    # Audio
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="stable_audio_open_10",
        name="Stable Audio Open 1.0",
        family="stable_audio", variant=None,
        source="stabilityai/stable-audio-open-1.0",
        filename="stable_audio_open_1.0",
        size_gb=3.4, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["audio", "sound_effects", "ambient", "music_samples"],
        weaknesses=["vocals", "long_duration"],
        best_for=["sound effects", "ambient audio", "music stems", "foley"],
        style_tags=["audio", "sound", "ambient"],
        settings=ModelSettings(steps=100, cfg=7.0, sampler="dpmpp_3m_sde", scheduler="exponential"),
        prompt_syntax=PromptSyntax(emphasis_format="none", quality_prefix=None,
                                   quality_suffix=None, negative_required=True, default_negative=""),
        download_command="huggingface-cli download stabilityai/stable-audio-open-1.0 --include '*.safetensors'",
        requires_clip="t5_base_stable_audio.safetensors",
        gated=True,
    ),

    # -------------------------------------------------------------------------
    # Upscalers (not checkpoints — referenced by upscale recipes)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="realesrgan_x4plus",
        name="RealESRGAN x4plus",
        family="upscaler", variant=None,
        source="https://github.com/xinntao/Real-ESRGAN",
        filename="RealESRGAN_x4plus",
        size_gb=0.07, vram_min_gb=2.0, vram_recommended_gb=4.0,
        strengths=["upscaling", "general_purpose", "speed"],
        weaknesses=["anime_content"],
        best_for=["upscaling real photos", "general image upscaling"],
        style_tags=["upscaler"],
        settings=ModelSettings(steps=1, cfg=1.0, sampler="euler", scheduler="normal"),
        prompt_syntax=PromptSyntax(emphasis_format="none", quality_prefix=None,
                                   quality_suffix=None, negative_required=False, default_negative=""),
        download_command="Place RealESRGAN_x4plus.pth in ComfyUI/models/upscale_models/",
    ),

    ModelCatalogEntry(
        id="4x_ultrasharp",
        name="4x-UltraSharp",
        family="upscaler", variant=None,
        source="https://civitai.com/models/116225",
        filename="4x-UltraSharp",
        size_gb=0.07, vram_min_gb=2.0, vram_recommended_gb=4.0,
        strengths=["upscaling", "sharpness", "detail"],
        weaknesses=["can_oversharp"],
        best_for=["upscaling detailed images", "sharpening generated art"],
        style_tags=["upscaler"],
        settings=ModelSettings(steps=1, cfg=1.0, sampler="euler", scheduler="normal"),
        prompt_syntax=PromptSyntax(emphasis_format="none", quality_prefix=None,
                                   quality_suffix=None, negative_required=False, default_negative=""),
        download_command="Download from: https://civitai.com/models/116225 → place in ComfyUI/models/upscale_models/",
    ),

    # -------------------------------------------------------------------------
    # SDXL Photorealism (additional)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="crystal_clear_xl",
        name="CrystalClear XL",
        family="sdxl", variant=None,
        source="https://civitai.com/models/122822",
        filename="crystalClearXL",
        size_gb=6.7, vram_min_gb=8.0, vram_recommended_gb=12.0,
        strengths=["photorealism", "portraits", "skin_tones", "detail"],
        weaknesses=["anime", "text_rendering"],
        best_for=["portrait photography", "beauty shots", "skin detail"],
        style_tags=["realistic", "photographic", "detailed"],
        settings=ModelSettings(steps=25, cfg=6.5, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="Download from: https://civitai.com/models/122822",
    ),

    ModelCatalogEntry(
        id="zavychroma_xl",
        name="ZavyChromaXL",
        family="sdxl", variant=None,
        source="https://civitai.com/models/232867",
        filename="zavychromaXL",
        size_gb=6.6, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["photorealism", "cinematic", "color_grading", "landscapes"],
        weaknesses=["anime", "text_rendering"],
        best_for=["cinematic photography", "landscapes", "editorial"],
        style_tags=["cinematic", "realistic", "vibrant"],
        settings=ModelSettings(steps=28, cfg=6.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="Download from: https://civitai.com/models/232867",
    ),

    ModelCatalogEntry(
        id="photon_xl",
        name="Photon XL",
        family="sdxl", variant=None,
        source="https://civitai.com/models/790138",
        filename="photon_xl",
        size_gb=6.5, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["photorealism", "product_photography", "commercial", "clean"],
        weaknesses=["anime", "abstract"],
        best_for=["product shots", "commercial photography", "clean editorial"],
        style_tags=["realistic", "commercial", "clean"],
        settings=ModelSettings(steps=25, cfg=6.0, sampler="euler", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="Download from: https://civitai.com/models/790138",
    ),

    # -------------------------------------------------------------------------
    # SDXL Stylized / Artistic (additional)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="copax_timeless_xl",
        name="Copax TimeLessXL",
        family="sdxl", variant=None,
        source="https://civitai.com/models/118111",
        filename="copaxTimelessxlSDXL1",
        size_gb=6.6, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["versatile", "artistic", "painterly", "illustrations"],
        weaknesses=["text_rendering"],
        best_for=["artistic illustrations", "concept art", "stylized scenes"],
        style_tags=["artistic", "painterly", "versatile"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="Download from: https://civitai.com/models/118111",
    ),

    ModelCatalogEntry(
        id="albedobase_xl",
        name="AlbedoBase XL",
        family="sdxl", variant=None,
        source="https://civitai.com/models/140737",
        filename="albedobaseXL_v21",
        size_gb=6.6, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["artistic", "versatile", "fantasy", "concept_art"],
        weaknesses=["text_rendering"],
        best_for=["concept art", "fantasy scenes", "character illustration"],
        style_tags=["artistic", "fantasy", "concept_art"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="Download from: https://civitai.com/models/140737",
    ),

    # -------------------------------------------------------------------------
    # Pony / NoobAI (additional)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="pony_diffusion_v6_xl_turbo",
        name="Pony Diffusion V6 XL Turbo",
        family="pony", variant="turbo",
        source="https://civitai.com/models/257749",
        filename="ponyDiffusionV6XL_turbo",
        size_gb=6.5, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["anime", "characters", "stylized", "speed"],
        weaknesses=["photorealism", "text_rendering"],
        best_for=["fast anime generation", "characters", "fan art"],
        style_tags=["anime", "illustration", "stylized"],
        settings=ModelSettings(steps=8, cfg=2.0, sampler="euler", scheduler="sgm_uniform", clip_skip=2),
        prompt_syntax=_pony_syntax(),
        download_command="Download from: https://civitai.com/models/257749",
    ),

    ModelCatalogEntry(
        id="noob_epsilon_pred",
        name="NoobAI XL Epsilon Pred",
        family="pony", variant=None,
        source="https://civitai.com/models/833294",
        filename="noobaiXLNaiXL_epsilonPred10",
        size_gb=6.5, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["anime", "illustration", "characters", "detail"],
        weaknesses=["photorealism", "text_rendering"],
        best_for=["high quality anime art", "character illustration"],
        style_tags=["anime", "illustration", "detailed"],
        settings=ModelSettings(steps=28, cfg=5.0, sampler="dpmpp_2m", scheduler="karras", clip_skip=2),
        prompt_syntax=_pony_syntax(),
        download_command="Download from: https://civitai.com/models/833294",
    ),

    ModelCatalogEntry(
        id="autismmix_sdxl",
        name="AutismMix SDXL",
        family="pony", variant=None,
        source="https://civitai.com/models/288584",
        filename="autismmixSDXL",
        size_gb=6.5, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["anime", "stylized", "characters", "vibrant"],
        weaknesses=["photorealism"],
        best_for=["anime characters", "stylized art", "vibrant illustrations"],
        style_tags=["anime", "stylized", "vibrant"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras", clip_skip=2),
        prompt_syntax=_pony_syntax(),
        download_command="Download from: https://civitai.com/models/288584",
    ),

    # -------------------------------------------------------------------------
    # SD 1.5 (additional)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="deliberate_v5",
        name="Deliberate V5",
        family="sd15", variant=None,
        source="https://civitai.com/models/4823",
        filename="deliberate_v5",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["photorealism", "artistic", "versatile", "detail"],
        weaknesses=["text_rendering"],
        best_for=["detailed illustrations", "concept art", "photorealistic scenes"],
        style_tags=["realistic", "artistic", "detailed", "versatile"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="euler", scheduler="karras"),
        prompt_syntax=_sd15_syntax(),
        download_command="Download from: https://civitai.com/models/4823",
        recommended_vae="vae-ft-mse-840000-ema-pruned.safetensors",
    ),

    ModelCatalogEntry(
        id="rev_animated_v122",
        name="ReV Animated v1.2.2",
        family="sd15", variant=None,
        source="https://civitai.com/models/7371",
        filename="revAnimated_v122",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["anime", "photorealism", "versatile", "fantasy"],
        weaknesses=["text_rendering"],
        best_for=["semi-realistic anime", "fantasy scenes", "character art"],
        style_tags=["anime", "realistic", "fantasy", "versatile"],
        settings=ModelSettings(steps=20, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sd15_syntax(),
        download_command="Download from: https://civitai.com/models/7371",
        recommended_vae="vae-ft-mse-840000-ema-pruned.safetensors",
    ),

    ModelCatalogEntry(
        id="epicrealism_natural",
        name="epiCRealism Natural Sin RC1",
        family="sd15", variant=None,
        source="https://civitai.com/models/25694",
        filename="epicrealism_naturalSinRC1",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["photorealism", "portraits", "skin_texture", "natural"],
        weaknesses=["anime", "abstract"],
        best_for=["ultra-realistic portraits", "natural photography"],
        style_tags=["realistic", "photographic", "natural", "detailed"],
        settings=ModelSettings(steps=25, cfg=5.5, sampler="dpmpp_sde", scheduler="karras"),
        prompt_syntax=_sd15_syntax(),
        download_command="Download from: https://civitai.com/models/25694",
        recommended_vae="vae-ft-mse-840000-ema-pruned.safetensors",
    ),

    ModelCatalogEntry(
        id="toonyou_beta6",
        name="ToonYou Beta 6",
        family="sd15", variant=None,
        source="https://civitai.com/models/30240",
        filename="toonyou_beta6",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["cartoon", "anime", "illustration", "stylized"],
        weaknesses=["photorealism"],
        best_for=["cartoon characters", "stylized illustrations", "2D art"],
        style_tags=["cartoon", "anime", "illustration", "stylized"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_anime_syntax(),
        download_command="Download from: https://civitai.com/models/30240",
        recommended_vae="vae-ft-mse-840000-ema-pruned.safetensors",
    ),

    # -------------------------------------------------------------------------
    # Flux variants (additional)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="flux_dev_fp8",
        name="Flux.1 Dev FP8 (quantized)",
        family="flux", variant=None,
        source="Comfy-Org/flux1-dev",
        filename="flux1-dev-fp8",
        size_gb=12.0, vram_min_gb=8.0, vram_recommended_gb=12.0,
        strengths=["prompt_adherence", "text_rendering", "complex_scenes", "vram_efficient"],
        weaknesses=["speed", "anime", "slightly_lower_quality_vs_bf16"],
        best_for=["text in images", "complex scenes on mid-range GPUs", "prompt accuracy"],
        style_tags=["photographic", "realistic", "artistic", "detailed"],
        settings=ModelSettings(steps=20, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_flux_syntax(),
        download_command="huggingface-cli download Comfy-Org/flux1-dev --include '*.safetensors'",
        requires_clip="t5xxl_fp8_e4m3fn.safetensors",
        license="flux-1-dev-non-commercial-license",
    ),

    ModelCatalogEntry(
        id="flux_schnell_fp8",
        name="Flux.1 Schnell FP8 (quantized)",
        family="flux", variant="schnell",
        source="Comfy-Org/flux1-schnell",
        filename="flux1-schnell-fp8",
        size_gb=12.0, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["speed", "prompt_adherence", "text_rendering", "vram_efficient"],
        weaknesses=["quality_vs_dev", "anime"],
        best_for=["fast generation on mid-range GPUs", "rapid prototyping"],
        style_tags=["photographic", "realistic", "artistic"],
        settings=ModelSettings(steps=4, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_flux_syntax(),
        download_command="huggingface-cli download Comfy-Org/flux1-schnell --include '*.safetensors'",
        requires_clip="t5xxl_fp8_e4m3fn.safetensors",
        license="apache-2.0",
    ),

    ModelCatalogEntry(
        id="flux_hyper_8step",
        name="Flux Hyper 8-Step LoRA",
        family="flux", variant=None,
        source="ByteDance/Hyper-SD",
        filename="flux1-dev",
        size_gb=23.8, vram_min_gb=12.0, vram_recommended_gb=16.0,
        strengths=["speed", "prompt_adherence", "text_rendering"],
        weaknesses=["vram_hungry", "anime", "requires_lora"],
        best_for=["fast Flux generation", "text rendering in fewer steps"],
        style_tags=["photographic", "realistic", "artistic"],
        settings=ModelSettings(steps=8, cfg=3.5, sampler="euler", scheduler="simple"),
        prompt_syntax=_flux_syntax(),
        download_command="huggingface-cli download ByteDance/Hyper-SD --include '*flux*8steps*'",
        requires_clip="t5xxl_fp16.safetensors",
        gated=True,
        license="flux-1-dev-non-commercial-license",
        recommended_loras=[
            LoRARecommendation(
                "Hyper-SD Flux 8-Step", "Hyper-SD15-8steps-CFG-lora.safetensors",
                "https://huggingface.co/ByteDance/Hyper-SD", 0.125,
                "Accelerates Flux.1 Dev to 8 steps with minimal quality loss",
            )
        ],
    ),

    ModelCatalogEntry(
        id="flux_merged_gguf_q8",
        name="Flux.1 Dev GGUF Q8",
        family="flux", variant=None,
        source="city96/FLUX.1-dev-gguf",
        filename="flux1-dev-Q8_0",
        size_gb=12.5, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["prompt_adherence", "text_rendering", "vram_efficient", "cpu_offload"],
        weaknesses=["speed", "anime", "requires_gguf_nodes"],
        best_for=["high quality on 8GB VRAM", "text-heavy images"],
        style_tags=["photographic", "realistic", "detailed"],
        settings=ModelSettings(steps=20, cfg=1.0, sampler="euler", scheduler="simple"),
        prompt_syntax=_flux_syntax(),
        download_command="huggingface-cli download city96/FLUX.1-dev-gguf --include '*Q8*'",
        requires_clip="t5xxl_fp8_e4m3fn.safetensors",
        required_capabilities=["UnetLoaderGGUF"],
        license="flux-1-dev-non-commercial-license",
    ),

    # -------------------------------------------------------------------------
    # Video (additional)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="animatediff_lightning",
        name="AnimateDiff Lightning",
        family="animatediff", variant="lightning",
        source="ByteDance/AnimateDiff-Lightning",
        filename="animatediff_lightning_4step_comfyui",
        size_gb=1.7, vram_min_gb=8.0, vram_recommended_gb=12.0,
        strengths=["animation", "speed", "video", "img2vid"],
        weaknesses=["long_videos", "text_to_video_quality"],
        best_for=["animating generated images", "short animated clips"],
        style_tags=["video", "animation", "motion"],
        settings=ModelSettings(steps=4, cfg=1.0, sampler="euler", scheduler="sgm_uniform"),
        prompt_syntax=PromptSyntax(emphasis_format="weighted", quality_prefix=None,
                                   quality_suffix=None, negative_required=True,
                                   default_negative="bad quality, worst quality"),
        download_command="huggingface-cli download ByteDance/AnimateDiff-Lightning --include '*.safetensors'",
        required_capabilities=["animatediff"],
    ),

    ModelCatalogEntry(
        id="mochi_preview",
        name="Mochi Preview (Text-to-Video)",
        family="mochi", variant=None,
        source="genmo/mochi-1-preview",
        filename="mochi_preview",
        size_gb=10.0, vram_min_gb=12.0, vram_recommended_gb=16.0,
        strengths=["text_to_video", "motion_quality", "video"],
        weaknesses=["slow", "vram_hungry", "short_clips"],
        best_for=["text-to-video generation", "motion-rich video clips"],
        style_tags=["video", "animation", "motion"],
        settings=ModelSettings(steps=64, cfg=4.5, sampler="dpmpp_2m", scheduler="linear"),
        prompt_syntax=PromptSyntax(emphasis_format="none", quality_prefix=None,
                                   quality_suffix=None, negative_required=False, default_negative=""),
        download_command="huggingface-cli download genmo/mochi-1-preview --include '*.safetensors'",
        required_capabilities=["MochiVideoLatent"],
    ),

    ModelCatalogEntry(
        id="wan_video_14b",
        name="Wan Video 1.3B (Text-to-Video)",
        family="wan_video", variant=None,
        source="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        filename="wan_video_1.3b",
        size_gb=5.2, vram_min_gb=8.0, vram_recommended_gb=10.0,
        strengths=["text_to_video", "motion_quality", "vram_efficient", "video"],
        weaknesses=["resolution_limited", "clip_length"],
        best_for=["text-to-video on mid-range GPUs", "short video clips"],
        style_tags=["video", "animation", "motion"],
        settings=ModelSettings(steps=50, cfg=5.0, sampler="euler", scheduler="linear"),
        prompt_syntax=PromptSyntax(emphasis_format="none", quality_prefix=None,
                                   quality_suffix=None, negative_required=False, default_negative=""),
        download_command="huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include '*.safetensors'",
        required_capabilities=["WanVideoModelLoader"],
    ),

    # -------------------------------------------------------------------------
    # Inpaint-specific
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="sdxl_inpaint",
        name="SDXL Inpaint",
        family="sdxl", variant=None,
        source="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        filename="sdxl_inpaint",
        size_gb=6.9, vram_min_gb=8.0, vram_recommended_gb=12.0,
        strengths=["inpainting", "removal", "fill", "seamless"],
        weaknesses=["speed", "text_rendering"],
        best_for=["object removal", "background replacement", "inpainting tasks"],
        style_tags=["inpaint", "realistic"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
        prompt_syntax=_sdxl_syntax(),
        download_command="huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 --include '*.safetensors'",
        required_capabilities=["VAEEncodeForInpaint"],
    ),

    # -------------------------------------------------------------------------
    # Anime / Illustration (additional)
    # -------------------------------------------------------------------------
    ModelCatalogEntry(
        id="counterfeit_v30",
        name="Counterfeit V3.0",
        family="sd15", variant=None,
        source="gsdf/Counterfeit-V3.0",
        filename="CounterfeitV30",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["anime", "illustration", "soft_lighting", "characters"],
        weaknesses=["photorealism"],
        best_for=["soft anime art", "character illustration", "light novels"],
        style_tags=["anime", "illustration", "soft"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras", clip_skip=2),
        prompt_syntax=_anime_syntax(),
        download_command="huggingface-cli download gsdf/Counterfeit-V3.0 --include '*.safetensors'",
        recommended_vae="kl-f8-anime2.vae.safetensors",
    ),

    ModelCatalogEntry(
        id="meinamix_v11",
        name="MeinaMix V11",
        family="sd15", variant=None,
        source="https://civitai.com/models/7240",
        filename="meinamix_meinaV11",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["anime", "illustration", "vibrant", "characters", "detail"],
        weaknesses=["photorealism"],
        best_for=["detailed anime art", "vivid character illustrations"],
        style_tags=["anime", "illustration", "vibrant", "detailed"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="dpmpp_2m", scheduler="karras", clip_skip=2),
        prompt_syntax=_anime_syntax(),
        download_command="Download from: https://civitai.com/models/7240",
        recommended_vae="kl-f8-anime2.vae.safetensors",
    ),

    ModelCatalogEntry(
        id="pastel_mix",
        name="Pastel Mix",
        family="sd15", variant=None,
        source="andite/pastel-mix",
        filename="pastelmix-better-vae",
        size_gb=2.1, vram_min_gb=4.0, vram_recommended_gb=6.0,
        strengths=["anime", "pastel", "soft", "illustration"],
        weaknesses=["photorealism", "dark_themes"],
        best_for=["soft pastel anime art", "light watercolor style"],
        style_tags=["anime", "pastel", "soft", "watercolor"],
        settings=ModelSettings(steps=25, cfg=7.0, sampler="euler", scheduler="normal", clip_skip=2),
        prompt_syntax=_anime_syntax(),
        download_command="huggingface-cli download andite/pastel-mix --include '*.safetensors'",
    ),
]


# ---------------------------------------------------------------------------
# Keyword → strength mapping for prompt analysis
# ---------------------------------------------------------------------------
_KEYWORD_STRENGTH_MAP: dict[str, list[str]] = {
    "photo": ["photorealism"],
    "realistic": ["photorealism"],
    "portrait": ["portraits", "photorealism"],
    "face": ["portraits"],
    "person": ["portraits"],
    "people": ["portraits"],
    "anime": ["anime", "illustration"],
    "manga": ["anime", "illustration"],
    "cartoon": ["anime", "illustration"],
    "2d": ["anime", "illustration"],
    "illustration": ["illustration"],
    "cinematic": ["cinematic", "editorial"],
    "movie": ["cinematic"],
    "film": ["cinematic"],
    "dramatic": ["cinematic"],
    "fantasy": ["fantasy", "concept_art"],
    "dragon": ["fantasy"],
    "magic": ["fantasy"],
    "medieval": ["fantasy"],
    "architecture": ["architectural"],
    "building": ["architectural"],
    "interior": ["architectural"],
    "room": ["architectural"],
    "product": ["product_photography"],
    "packaging": ["product_photography"],
    "commercial": ["product_photography"],
    "abstract": ["artistic", "painterly"],
    "surreal": ["artistic"],
    "artistic": ["artistic"],
    "painting": ["painterly"],
    "text": ["text_rendering"],
    "logo": ["text_rendering"],
    "typography": ["text_rendering"],
    "words": ["text_rendering"],
    "sign": ["text_rendering"],
    "video": [],   # handled by intent
    "animate": [], # handled by intent
    "sound": [],   # handled by intent
    "audio": [],   # handled by intent
}


def _extract_strengths_from_prompt(prompt: str) -> set[str]:
    lower = prompt.lower()
    matched = set()
    for keyword, strengths in _KEYWORD_STRENGTH_MAP.items():
        if keyword in lower:
            matched.update(strengths)
    return matched


def _intent_compatible_families(intent_str: str) -> set[str]:
    from ez_comfy.planner.intent import PipelineIntent
    intent = PipelineIntent(intent_str) if isinstance(intent_str, str) else intent_str
    if intent == PipelineIntent.VIDEO:
        return {"svd"}
    if intent == PipelineIntent.AUDIO:
        return {"stable_audio"}
    if intent == PipelineIntent.UPSCALE:
        return {"upscaler"}
    # All others: image-generating families
    return {"sd15", "sdxl", "flux", "sd3", "cascade", "pony"}


def _is_installed(entry: ModelCatalogEntry, inventory: ComfyUIInventory) -> bool:
    """Check if catalog entry's checkpoint is installed."""
    if entry.family == "upscaler":
        return any(_model_name_matches(entry.filename, up) for up in inventory.upscale_models)
    return any(_model_name_matches(entry.filename, ck.filename) for ck in inventory.checkpoints)


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _model_signature_parts(name: str) -> list[str]:
    """
    Build stable signature parts from a model name for fuzzy install matching.
    Handles variants like:
      realvisxlV50_lightning
      realvisxlV50_v50LightningBakedvae.safetensors
    """
    stop = {"safetensors", "ckpt", "pth", "pt", "bin"}
    raw_parts = [p for p in re.split(r"[^a-z0-9]+", name.lower()) if p and p not in stop]
    combo_parts: list[str] = []
    for a, b in zip(raw_parts, raw_parts[1:]):
        if len(a) <= 3 or len(b) <= 3:
            combo_parts.append(a + b)
    parts = set(raw_parts + combo_parts)
    return [p for p in parts if len(p) >= 3]


def _model_name_matches(expected_name: str, installed_name: str) -> bool:
    expected_norm = _normalize_model_name(expected_name)
    installed_norm = _normalize_model_name(installed_name)

    if expected_norm and (
        expected_norm in installed_norm
        or installed_norm.startswith(expected_norm)
        or installed_norm in expected_norm
    ):
        return True

    parts = _model_signature_parts(expected_name)
    if not parts:
        return False
    matched = sum(1 for p in parts if p in installed_norm)
    return matched >= min(2, len(parts))


def recommend_models(
    prompt: str,
    intent: str,
    hardware: HardwareProfile,
    inventory: ComfyUIInventory,
    catalog: list[ModelCatalogEntry] | None = None,
    prefer_installed: bool = True,
    prefer_speed: bool = True,
    top_n: int = 3,
) -> list[ModelRecommendation]:
    """Rank catalog entries for a prompt + intent, returning top N recommendations."""
    if catalog is None:
        catalog = MODEL_CATALOG

    compatible_families = _intent_compatible_families(intent)
    prompt_strengths = _extract_strengths_from_prompt(prompt)
    prompt_lower = prompt.lower()

    candidates: list[ModelRecommendation] = []

    for entry in catalog:
        if entry.family not in compatible_families:
            continue

        fits_vram = entry.vram_min_gb <= hardware.gpu_vram_gb
        installed = _is_installed(entry, inventory)
        reasons: list[str] = []
        warnings: list[str] = []
        score = 0.0

        # Strength match (+30)
        strength_matches = prompt_strengths & set(entry.strengths)
        if strength_matches:
            score += 30.0
            reasons.append(f"matches: {', '.join(strength_matches)}")

        # Installed (+25)
        if installed:
            score += 25.0
            reasons.append("installed")

        # VRAM headroom (+20, scaled)
        if fits_vram:
            headroom = hardware.gpu_vram_gb - entry.vram_min_gb
            vram_score = min(20.0, headroom * 2.0)
            score += vram_score
        else:
            warnings.append(f"requires {entry.vram_min_gb}GB VRAM, you have {hardware.gpu_vram_gb}GB")

        # Speed variant (+15)
        if prefer_speed and entry.variant in ("lightning", "turbo", "schnell"):
            score += 15.0
            reasons.append("fast variant")

        # Style tag match (+10)
        style_matches = set(entry.style_tags) & _prompt_style_tags(prompt_lower)
        if style_matches:
            score += 10.0

        candidates.append(ModelRecommendation(
            entry=entry,
            score=score,
            installed=installed,
            fits_vram=fits_vram,
            match_reasons=reasons,
            warnings=warnings,
        ))

    # Sort by score, then prefer installed
    candidates.sort(key=lambda r: (r.score, r.installed), reverse=True)
    return candidates[:top_n]


def _prompt_style_tags(prompt_lower: str) -> set[str]:
    tags = set()
    tag_keywords = {
        "realistic": "realistic",
        "cinematic": "cinematic",
        "anime": "anime",
        "artistic": "artistic",
        "dark": "dark",
        "vibrant": "vibrant",
        "editorial": "editorial",
        "dramatic": "dramatic",
    }
    for kw, tag in tag_keywords.items():
        if kw in prompt_lower:
            tags.add(tag)
    return tags


def find_catalog_entry(filename: str, catalog: list[ModelCatalogEntry] | None = None) -> ModelCatalogEntry | None:
    """Find a catalog entry that matches an installed checkpoint filename."""
    if catalog is None:
        catalog = MODEL_CATALOG
    for entry in catalog:
        if _model_name_matches(entry.filename, filename):
            return entry
    return None


def resolve_installed_filename(entry: ModelCatalogEntry, inventory: ComfyUIInventory) -> str | None:
    """
    Resolve the concrete installed filename for a catalog entry.
    Returns the exact filename seen in ComfyUI inventory, or None if not found.
    """
    if entry.family == "upscaler":
        for up in inventory.upscale_models:
            if _model_name_matches(entry.filename, up):
                return up
        return None

    for ck in inventory.checkpoints:
        if _model_name_matches(entry.filename, ck.filename):
            return ck.filename
    return None
