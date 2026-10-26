from __future__ import annotations

from enum import Enum


class PipelineIntent(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT  = "inpaint"
    UPSCALE  = "upscale"
    VIDEO    = "video"
    AUDIO    = "audio"


_VIDEO_KEYWORDS = {
    "video", "animate", "animation",
    "gif", "film clip", "film footage", "cinematic video", "moving image",
    "make it move", "bring to life",
}
# Compound words that contain a _VIDEO_KEYWORDS token but are not video-generation requests
_NOT_VIDEO_PHRASES = {"video game", "video games", "game video"}

_AUDIO_KEYWORDS = {
    # "sound" alone is too generic (appears in image descriptions like "sound of waves")
    # Use compound forms instead
    "sound effect", "sound effects", "sound design", "sound generation",
    "audio", "music", "song", "melody", "beat", "sfx",
    "noise", "ambient sound", "soundtrack",
}
_UPSCALE_KEYWORDS = {
    "upscale", "upscaling", "enhance resolution", "4x", "2x", "super resolution",
    "hd", "high resolution", "increase resolution",
}
_INPAINT_KEYWORDS = {
    "remove", "replace", "fill", "inpaint", "erase", "delete",
    "background removal", "change background",
}
_IMG2IMG_KEYWORDS = {
    "style", "painterly", "transform", "convert", "make it",
    "turn into", "change to", "modify", "edit",
}


def _matches(lower: str, keywords: set[str]) -> bool:
    """Check if any keyword appears as a whole word (or phrase) in the prompt."""
    import re
    for kw in keywords:
        if " " in kw:
            # Multi-word phrase: simple substring is fine
            if kw in lower:
                return True
        else:
            # Single token: require word boundary
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                return True
    return False


def detect_intent(
    prompt: str,
    has_reference_image: bool = False,
    has_mask: bool = False,
) -> PipelineIntent:
    """
    Heuristic intent detection from prompt text + image/mask presence.
    Fast, no LLM required.
    """
    lower = prompt.lower()

    # Audio signals (check before video to avoid false positives)
    if _matches(lower, _AUDIO_KEYWORDS):
        return PipelineIntent.AUDIO

    # Video signals — exclude compound phrases like "video game"
    if _matches(lower, _VIDEO_KEYWORDS) and not any(p in lower for p in _NOT_VIDEO_PHRASES):
        return PipelineIntent.VIDEO

    # Upscale signals (with or without image)
    if _matches(lower, _UPSCALE_KEYWORDS) and has_reference_image:
        return PipelineIntent.UPSCALE

    # Image-conditional intents
    if has_reference_image:
        # Inpaint requires an explicit mask; keywords alone without a mask are img2img
        if has_mask:
            return PipelineIntent.INPAINT
        if _matches(lower, _IMG2IMG_KEYWORDS) or _matches(lower, _INPAINT_KEYWORDS):
            return PipelineIntent.IMG2IMG
        # Default when image provided: img2img
        return PipelineIntent.IMG2IMG

    return PipelineIntent.TXT2IMG
