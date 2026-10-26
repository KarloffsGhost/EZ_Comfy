from __future__ import annotations

import re
from dataclasses import dataclass

from ez_comfy.models.profiles import PromptSyntax


@dataclass
class StylePreset:
    id: str
    name: str
    positive_tokens: dict[str, str]   # family_group → tokens to append
    negative_tokens: dict[str, str]   # family_group → tokens to append to negative


@dataclass
class DomainPromptPack:
    id: str
    keywords: tuple[str, ...]
    positive_tokens: dict[str, tuple[str, ...]]
    negative_tokens: dict[str, tuple[str, ...]]


# ---------------------------------------------------------------------------
# Built-in style presets
# ---------------------------------------------------------------------------
STYLE_PRESETS: dict[str, StylePreset] = {
    "photographic": StylePreset(
        id="photographic", name="Photographic",
        positive_tokens={
            "sdxl":  ", professional photography, 8k, high detail, sharp focus",
            "sd15":  ", professional photograph, DSLR, sharp, high detail",
            "flux":  " as a professional photograph with sharp focus and high detail",
            "pony":  ", photography, sharp focus",
        },
        negative_tokens={},
    ),
    "cinematic": StylePreset(
        id="cinematic", name="Cinematic",
        positive_tokens={
            "sdxl":  ", cinematic lighting, film grain, color grading, bokeh, shallow depth of field, anamorphic",
            "sd15":  ", cinematic, dramatic lighting, film grain, 35mm photograph",
            "flux":  " in cinematic style with dramatic lighting and shallow depth of field",
            "pony":  ", cinematic, dramatic lighting, film grain",
        },
        negative_tokens={},
    ),
    "anime": StylePreset(
        id="anime", name="Anime",
        positive_tokens={
            "sdxl":  ", anime style, vibrant colors, detailed lineart",
            "sd15":  ", anime style, vibrant",
            "flux":  " in anime art style with vibrant colors and clean lineart",
            "pony":  ", anime style",
        },
        negative_tokens={},
    ),
    "digital_art": StylePreset(
        id="digital_art", name="Digital Art",
        positive_tokens={
            "sdxl":  ", digital art, concept art, artstation, highly detailed",
            "sd15":  ", digital painting, concept art, artstation",
            "flux":  " as detailed digital art in concept art style",
            "pony":  ", digital art, concept art",
        },
        negative_tokens={},
    ),
    "fantasy": StylePreset(
        id="fantasy", name="Fantasy",
        positive_tokens={
            "sdxl":  ", fantasy art, epic, magical, intricate, majestic",
            "sd15":  ", fantasy, magical, epic lighting, intricate details",
            "flux":  " in epic fantasy art style with magical lighting",
            "pony":  ", fantasy, magical, epic",
        },
        negative_tokens={},
    ),
    "pixel_art": StylePreset(
        id="pixel_art", name="Pixel Art",
        positive_tokens={
            "sdxl":  ", pixel art, 16-bit, retro game style",
            "sd15":  ", pixel art, 8-bit, retro",
            "flux":  " as pixel art with a retro game aesthetic",
            "pony":  ", pixel art style",
        },
        negative_tokens={},
    ),
    "watercolor": StylePreset(
        id="watercolor", name="Watercolor",
        positive_tokens={
            "sdxl":  ", watercolor painting, soft edges, flowing colors, artistic",
            "sd15":  ", watercolor, soft, flowing, artistic painting",
            "flux":  " painted in watercolor style with soft flowing colors",
            "pony":  ", watercolor style, soft colors",
        },
        negative_tokens={},
    ),
    "oil_painting": StylePreset(
        id="oil_painting", name="Oil Painting",
        positive_tokens={
            "sdxl":  ", oil painting, classical, textured brushstrokes, museum quality",
            "sd15":  ", oil painting, classical art style, textured",
            "flux":  " as a classical oil painting with visible brushstrokes",
            "pony":  ", oil painting, classical",
        },
        negative_tokens={},
    ),
    "3d_render": StylePreset(
        id="3d_render", name="3D Render",
        positive_tokens={
            "sdxl":  ", 3D render, octane render, ray tracing, photorealistic, 4k",
            "sd15":  ", 3D render, CGI, octane, ray traced",
            "flux":  " as a photorealistic 3D render with ray tracing",
            "pony":  ", 3D render, CGI",
        },
        negative_tokens={},
    ),
    "comic": StylePreset(
        id="comic", name="Comic Book",
        positive_tokens={
            "sdxl":  ", comic book style, bold outlines, dynamic composition, halftone",
            "sd15":  ", comic book, bold lines, graphic novel",
            "flux":  " in comic book style with bold outlines and dynamic composition",
            "pony":  ", comic style, bold outlines",
        },
        negative_tokens={},
    ),
    "minimalist": StylePreset(
        id="minimalist", name="Minimalist",
        positive_tokens={
            "sdxl":  ", minimalist design, clean composition, simple, elegant",
            "sd15":  ", minimalist, clean, simple design",
            "flux":  " in minimalist style with clean simple composition",
            "pony":  ", minimalist, clean",
        },
        negative_tokens={},
    ),
    "noir": StylePreset(
        id="noir", name="Noir",
        positive_tokens={
            "sdxl":  ", film noir, black and white, moody, dramatic shadows, high contrast",
            "sd15":  ", film noir, dark, moody, black and white, shadows",
            "flux":  " in film noir style with dramatic shadows and high contrast",
            "pony":  ", noir, dark moody atmosphere",
        },
        negative_tokens={},
    ),
    "cyberpunk": StylePreset(
        id="cyberpunk", name="Cyberpunk",
        positive_tokens={
            "sdxl":  ", cyberpunk aesthetic, neon lights, futuristic, rain-slicked streets, holographic",
            "sd15":  ", cyberpunk, neon, futuristic, dark city",
            "flux":  " in cyberpunk style with neon lights and futuristic technology",
            "pony":  ", cyberpunk, neon lights, futuristic",
        },
        negative_tokens={},
    ),
    "portrait": StylePreset(
        id="portrait", name="Portrait",
        positive_tokens={
            "sdxl":  ", professional portrait, studio lighting, sharp focus, bokeh background, detailed face",
            "sd15":  ", portrait photography, studio light, detailed face",
            "flux":  " as a professional portrait with studio lighting and sharp facial details",
            "pony":  ", portrait, detailed face, studio lighting",
        },
        negative_tokens={},
    ),
    "landscape": StylePreset(
        id="landscape", name="Landscape",
        positive_tokens={
            "sdxl":  ", epic landscape, golden hour, atmospheric, wide angle, stunning scenery",
            "sd15":  ", landscape photography, scenic, atmospheric, wide angle",
            "flux":  " as a stunning landscape photograph with atmospheric lighting",
            "pony":  ", epic landscape, scenic",
        },
        negative_tokens={},
    ),
    "product": StylePreset(
        id="product", name="Product Photography",
        positive_tokens={
            "sdxl":  ", product photography, clean background, professional lighting, commercial, sharp detail",
            "sd15":  ", product photo, clean background, commercial photography",
            "flux":  " as professional product photography with clean background and perfect lighting",
            "pony":  ", product photography, clean background",
        },
        negative_tokens={},
    ),
}


# ---------------------------------------------------------------------------
# Domain packs (automatic prompt enhancement)
# ---------------------------------------------------------------------------
DOMAIN_PACKS: tuple[DomainPromptPack, ...] = (
    DomainPromptPack(
        id="portrait_realism",
        keywords=("portrait", "headshot", "face", "man", "woman", "person", "human"),
        positive_tokens={
            "sdxl": ("natural skin texture", "realistic skin pores", "subtle facial asymmetry", "catchlight in eyes"),
            "sd15": ("natural skin texture", "realistic skin detail", "catchlight in eyes"),
            "flux": ("natural skin texture", "realistic skin detail", "subtle facial asymmetry"),
            "pony": ("detailed face", "natural skin texture"),
        },
        negative_tokens={
            "sdxl": ("plastic skin", "waxy skin", "over-smoothed skin", "doll-like face", "uncanny face"),
            "sd15": ("plastic skin", "waxy skin", "over-smoothed skin", "doll-like face", "uncanny face"),
            "pony": ("plastic skin", "over-smoothed skin", "doll-like face"),
        },
    ),
    DomainPromptPack(
        id="product_photo",
        keywords=("product", "packshot", "ecommerce", "catalog", "studio shot", "packaging"),
        positive_tokens={
            "sdxl": ("clean studio background", "controlled product lighting", "sharp edges", "true material texture"),
            "sd15": ("clean studio background", "controlled product lighting", "sharp edges"),
            "flux": ("clean studio background", "controlled product lighting", "true material texture"),
            "pony": ("clean studio background", "sharp edges"),
        },
        negative_tokens={
            "sdxl": ("logo distortion", "warped text", "label artifacts", "messy background"),
            "sd15": ("logo distortion", "warped text", "label artifacts", "messy background"),
            "pony": ("logo distortion", "warped text", "label artifacts"),
        },
    ),
    DomainPromptPack(
        id="text_suppression",
        keywords=("logo", "letters", "text", "typography", "brand", "label", "usga", "regulation"),
        positive_tokens={
            "sdxl": ("clean unbranded surface",),
            "sd15": ("clean unbranded surface",),
            "flux": ("clean unbranded surface",),
            "pony": ("clean unbranded surface",),
        },
        negative_tokens={
            "sdxl": ("gibberish text", "malformed letters", "random symbols", "misspelled words"),
            "sd15": ("gibberish text", "malformed letters", "random symbols", "misspelled words"),
            "pony": ("gibberish text", "malformed letters", "random symbols"),
        },
    ),
    DomainPromptPack(
        id="golf_ball_geometry",
        keywords=("golf ball",),
        positive_tokens={
            "sdxl": ("uniform circular dimples", "even dimple spacing", "consistent dimple depth", "perfectly spherical ball"),
            "sd15": ("uniform circular dimples", "even dimple spacing", "consistent dimple depth"),
            "flux": ("uniform circular dimples", "even dimple spacing", "consistent dimple depth"),
            "pony": ("uniform circular dimples", "even dimple spacing"),
        },
        negative_tokens={
            "sdxl": ("irregular dimples", "uneven spacing", "malformed dimples", "asymmetrical sphere"),
            "sd15": ("irregular dimples", "uneven spacing", "malformed dimples", "asymmetrical sphere"),
            "pony": ("irregular dimples", "uneven spacing", "malformed dimples"),
        },
    ),
)


def _family_group(family: str) -> str:
    """Map specific family to a style preset group key."""
    if family in ("pony",):
        return "pony"
    if family in ("flux", "flux_schnell"):
        return "flux"
    if family in ("sd15", "sd15_lightning"):
        return "sd15"
    # sdxl, sdxl_lightning, sdxl_turbo, sd3, cascade, unknown
    return "sdxl"


_WEIGHT_PATTERN = re.compile(r"\(([^()]+):[\d.]+\)")
_NAI_PLUS_PATTERN = re.compile(r"(\w[\w\s]*?)(\+{1,3})")


def adapt_prompt(
    user_prompt: str,
    negative_prompt: str,
    syntax: PromptSyntax,
    style_preset: StylePreset | None = None,
    family: str = "sdxl",
    auto_negative: bool = True,
) -> tuple[str, str]:
    """
    Adapt prompt and negative prompt for the selected model's syntax.
    Returns (adapted_positive, adapted_negative).
    """
    positive = user_prompt.strip()
    negative = negative_prompt.strip()

    # 1. Emphasis normalization
    positive = _normalize_emphasis(positive, syntax.emphasis_format)

    # 2. Quality prefix/suffix injection
    if syntax.quality_prefix and not positive.startswith(syntax.quality_prefix.strip(", ")):
        positive = syntax.quality_prefix + positive
    if syntax.quality_suffix and not positive.endswith(syntax.quality_suffix.strip(", ")):
        positive = positive + syntax.quality_suffix

    # 3. Style preset expansion
    if style_preset is not None:
        group = _family_group(family)
        pos_tokens = style_preset.positive_tokens.get(group, "")
        neg_tokens = style_preset.negative_tokens.get(group, "")
        if pos_tokens:
            if syntax.emphasis_format == "none":
                # Flux: append as natural language
                positive = positive.rstrip(".") + " " + pos_tokens.strip(", ")
            else:
                positive = positive.rstrip(", ") + pos_tokens
        if neg_tokens:
            negative = (negative.rstrip(", ") + neg_tokens) if negative else neg_tokens.strip(", ")

    # 4. Domain-aware enhancement packs (portrait/product/text-sensitive/etc)
    group = _family_group(family)
    active_packs = _active_domain_packs(user_prompt.lower())
    for pack in active_packs:
        pos_terms = pack.positive_tokens.get(group, ())
        if pos_terms:
            positive = _append_terms(positive, pos_terms)

        neg_terms = pack.negative_tokens.get(group, ())
        if neg_terms and syntax.negative_required and (auto_negative or negative):
            negative = _append_terms(negative, neg_terms)

    # 5. Auto-generate negative if blank
    if not negative and auto_negative and syntax.negative_required:
        negative = syntax.default_negative

    # 6. Flux: clear negative entirely (ignored by model)
    if not syntax.negative_required:
        negative = ""

    return positive, negative


def _normalize_emphasis(prompt: str, emphasis_format: str) -> str:
    """Normalize prompt emphasis syntax for the target model."""
    if emphasis_format == "none":
        # Strip (word:1.3) → word
        prompt = _WEIGHT_PATTERN.sub(r"\1", prompt)
    elif emphasis_format == "weighted":
        # Convert word+++ → (word:1.3) NAI format
        def nai_to_weighted(m: re.Match) -> str:
            word = m.group(1).strip()
            pluses = len(m.group(2))
            weight = 1.0 + pluses * 0.1
            return f"({word}:{weight:.1f})"
        prompt = _NAI_PLUS_PATTERN.sub(nai_to_weighted, prompt)
    return prompt


def _append_terms(text: str, terms: tuple[str, ...]) -> str:
    """Append comma-separated terms only if they don't already exist."""
    out = text.strip()
    out_lower = out.lower()
    for term in terms:
        t = term.strip()
        if not t:
            continue
        if t.lower() in out_lower:
            continue
        if out:
            out += ", " + t
        else:
            out = t
        out_lower = out.lower()
    return out


def _keyword_match(prompt_lower: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in prompt_lower
    return re.search(r"\b" + re.escape(keyword) + r"\b", prompt_lower) is not None


def _active_domain_packs(prompt_lower: str) -> list[DomainPromptPack]:
    packs: list[DomainPromptPack] = []
    for pack in DOMAIN_PACKS:
        if any(_keyword_match(prompt_lower, kw) for kw in pack.keywords):
            packs.append(pack)
    return packs


def get_style_preset(preset_id: str | None) -> StylePreset | None:
    if not preset_id:
        return None
    return STYLE_PRESETS.get(preset_id)
