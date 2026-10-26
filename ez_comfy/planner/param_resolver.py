from __future__ import annotations

import random
from dataclasses import dataclass, field

from ez_comfy.models.profiles import ModelProfile, ModelSettings, snap_to_bucket


@dataclass
class ResolvedParams:
    steps: int
    cfg_scale: float
    sampler: str
    scheduler: str
    width: int
    height: int
    clip_skip: int
    denoise_strength: float
    seed: int
    batch_size: int
    # Transparency: track where each param came from
    sources: dict[str, str] = field(default_factory=dict)


_ASPECT_RATIO_HINTS: dict[str, tuple[float, float]] = {
    "square":    (1.0, 1.0),
    "landscape": (16.0, 9.0),
    "portrait":  (9.0, 16.0),
    "panoramic": (21.0, 9.0),
    "tall":      (9.0, 21.0),
    "3:2":       (3.0, 2.0),
    "2:3":       (2.0, 3.0),
    "4:3":       (4.0, 3.0),
    "3:4":       (3.0, 4.0),
    "16:9":      (16.0, 9.0),
    "9:16":      (9.0, 16.0),
    "1:1":       (1.0, 1.0),
    "21:9":      (21.0, 9.0),
}


def resolve_params(
    profile: ModelProfile,
    catalog_settings: ModelSettings | None = None,
    recipe_overrides: dict | None = None,
    user_overrides: dict | None = None,
    aspect_ratio: str | None = None,
) -> ResolvedParams:
    """
    Resolve final generation params using priority chain:
      user_overrides > recipe_overrides > catalog_settings > profile_defaults
    """
    sources: dict[str, str] = {}
    ro = recipe_overrides or {}
    uo = user_overrides or {}
    base = profile.default_settings
    cat = catalog_settings

    def pick(key: str, default, transform=None):
        if key in uo and uo[key] is not None:
            sources[key] = "user"
            v = uo[key]
        elif key in ro and ro[key] is not None:
            sources[key] = "recipe"
            v = ro[key]
        elif cat is not None and getattr(cat, key, None) is not None:
            sources[key] = "model_catalog"
            v = getattr(cat, key)
        else:
            sources[key] = "family_profile"
            v = getattr(base, key, default)
        return transform(v) if transform else v

    steps     = pick("steps",    20,    int)
    cfg_scale = pick("cfg",      7.0,   float)
    sampler   = pick("sampler",  "euler")
    scheduler = pick("scheduler","normal")
    clip_skip = pick("clip_skip", 1,    int)
    denoise   = pick("denoise_default", 0.7, float)
    batch     = int(uo.get("batch_size", 1))

    # Seed resolution
    raw_seed = uo.get("seed", -1)
    if raw_seed is None or raw_seed == -1:
        seed = random.randint(0, 2**32 - 1)
        sources["seed"] = "random"
    else:
        seed = int(raw_seed)
        sources["seed"] = "user"

    # Resolution resolution
    width, height, res_source = _resolve_resolution(profile, uo, aspect_ratio)
    sources["width"] = sources["height"] = res_source

    return ResolvedParams(
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        scheduler=scheduler,
        width=width,
        height=height,
        clip_skip=clip_skip,
        denoise_strength=denoise,
        seed=seed,
        batch_size=batch,
        sources=sources,
    )


def _resolve_resolution(
    profile: ModelProfile,
    user_overrides: dict,
    aspect_ratio: str | None,
) -> tuple[int, int, str]:
    """Returns (width, height, source_label)."""
    buckets = profile.resolution_buckets
    native_w, native_h = profile.native_resolution

    if not buckets:
        return native_w, native_h, "family_profile"

    user_w = user_overrides.get("width")
    user_h = user_overrides.get("height")

    if user_w and user_h:
        # Snap user-specified dimensions to nearest bucket
        snapped = snap_to_bucket(int(user_w), int(user_h), buckets)
        return snapped[0], snapped[1], "resolution_bucket"

    if aspect_ratio:
        hint = aspect_ratio.lower().strip()
        ratio = _ASPECT_RATIO_HINTS.get(hint)
        if ratio is None:
            # Try parsing "W:H"
            parts = hint.split(":")
            if len(parts) == 2:
                try:
                    ratio = (float(parts[0]), float(parts[1]))
                except ValueError:
                    ratio = None
        if ratio:
            target_area = native_w * native_h
            rw, rh = ratio
            # Derive target dims from ratio + native area
            h = int((target_area / (rw / rh)) ** 0.5)
            w = int(h * rw / rh)
            snapped = snap_to_bucket(w, h, buckets)
            return snapped[0], snapped[1], "resolution_bucket"

    return native_w, native_h, "family_profile"
