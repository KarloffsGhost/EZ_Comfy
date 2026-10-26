from __future__ import annotations

import re

# Version-like context keeps "wan"/"ltx" from matching words such as "swan" or "wanderer".
# Names are matched on the lower-cased file name (folders included).
_WAN = re.compile(r"(?<![a-z])wan(?:[-_.]?\d|[-_])")
_LTX = re.compile(r"(?<![a-z])ltx(?:[-_.]?\d|[-_v])")
_FLUX2 = re.compile(r"flux[-_.]?2(?!\d)")
_FLUX3 = re.compile(r"flux[-_.]?3(?!\d)")
_QWEN_IMAGE = re.compile(r"qwen[-_ ]?image")

# Families the classifier can return, for validating `model_overrides`.
KNOWN_FAMILIES = frozenset({
    "sd15", "sdxl", "pony", "flux", "flux2", "flux3", "sd3", "cascade", "svd", "stable_audio",
    "wan_video", "ltx_video", "qwen_image", "hidream", "unknown",
})


def classify_checkpoint(
    filename: str,
    size_bytes: int | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """
    Classify a checkpoint by filename patterns and optional file size.
    Returns (family, variant).

    `overrides` maps an installed file name to a family and wins over every rule below.

    Families: sd15, sdxl, pony, flux, flux2, flux3, sd3, cascade, svd, stable_audio,
              wan_video, ltx_video, qwen_image, hidream, unknown
    Variants: lightning, turbo, lcm, hyper, schnell, None
    """
    lower = filename.lower()

    variant = _detect_variant(lower)

    if overrides and filename in overrides:
        return overrides[filename], variant

    # Audio models
    if any(k in lower for k in ("stable_audio", "stable-audio", "audio_open")):
        return "stable_audio", variant

    # Video models (before image families: their names can contain image-family words)
    if any(k in lower for k in ("svd", "stable-video", "stable_video")):
        return "svd", variant
    if _WAN.search(lower):
        return "wan_video", variant
    if _LTX.search(lower):
        return "ltx_video", variant

    # Flux generations: explicit tokens only; any other flux name stays the original family
    if "flux" in lower:
        if _FLUX3.search(lower):
            return "flux3", variant
        if _FLUX2.search(lower):
            return "flux2", variant
        return "flux", variant

    # Other modern UNET-loaded image families
    if "hidream" in lower:
        return "hidream", variant
    if _QWEN_IMAGE.search(lower):
        return "qwen_image", variant

    # Stable Cascade
    if "cascade" in lower:
        return "cascade", variant

    # SD3
    if "sd3" in lower or "stable-diffusion-3" in lower:
        return "sd3", variant

    # Pony — SDXL-based but separate family due to prompt requirements
    if "pony" in lower:
        return "pony", variant

    # SDXL signals (check before SD1.5 to avoid realvis/dreamshaper false matches)
    sdxl_keywords = ("xl", "sdxl", "realvis", "juggernaut", "proteus", "illustrious")
    if any(k in lower for k in sdxl_keywords):
        # Size sanity check: SDXL checkpoints are typically > 3GB
        if size_bytes is None or size_bytes > 2 * 1024**3:
            return "sdxl", variant

    # SD 1.5 signals
    sd15_keywords = ("v1-5", "v1_5", "sd-v1", "dreamshaper_8", "dreamshaper8", "realistic_vision",
                     "realisticvision", "epicrealism", "cyberrealistic", "anything", "meina")
    if any(k in lower for k in sd15_keywords):
        return "sd15", variant

    # Size-based fallback
    if size_bytes is not None:
        if size_bytes > 5 * 1024**3:
            return "sdxl", variant  # large → probably SDXL
        if size_bytes < 3 * 1024**3:
            return "sd15", variant  # small → probably SD1.5

    return "unknown", variant


def _detect_variant(lower: str) -> str | None:
    if "lightning" in lower:
        return "lightning"
    if "turbo" in lower:
        return "turbo"
    if "lcm" in lower:
        return "lcm"
    if "hyper" in lower:
        return "hyper"
    if "schnell" in lower:
        return "schnell"
    return None
