from __future__ import annotations


def classify_checkpoint(filename: str, size_bytes: int | None = None) -> tuple[str, str | None]:
    """
    Classify a checkpoint by filename patterns and optional file size.
    Returns (family, variant).

    Families: sd15, sdxl, flux, sd3, cascade, svd, stable_audio, unknown
    Variants: lightning, turbo, lcm, hyper, schnell, None
    """
    lower = filename.lower()

    variant = _detect_variant(lower)

    # Audio models
    if any(k in lower for k in ("stable_audio", "stable-audio", "audio_open")):
        return "stable_audio", variant

    # Video models
    if any(k in lower for k in ("svd", "stable-video", "stable_video")):
        return "svd", variant

    # Flux
    if "flux" in lower:
        return "flux", variant

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
