"""Unit tests for models/profiles.py"""
from __future__ import annotations

import pytest
from ez_comfy.models.profiles import SDXL_BUCKETS, get_profile, snap_to_bucket


def test_get_profile_sdxl():
    p = get_profile("sdxl")
    assert p.family == "sdxl"
    assert p.native_resolution == (1024, 1024)
    assert p.vram_requirement_gb > 0


def test_get_profile_sdxl_lightning():
    p = get_profile("sdxl", "lightning")
    assert p.default_settings.steps <= 10


def test_get_profile_flux():
    p = get_profile("flux")
    assert p.family == "flux"
    assert not p.prompt_syntax.negative_required


def test_get_profile_pony():
    p = get_profile("pony")
    assert p.family == "pony"
    assert "score_9" in (p.prompt_syntax.quality_prefix or "")


def test_get_profile_sd15():
    p = get_profile("sd15")
    assert p.native_resolution == (512, 512)


def test_get_profile_unknown_falls_back():
    p = get_profile("unknown_family_xyz")
    assert p is not None


@pytest.mark.parametrize("w,h,expected_w,expected_h", [
    (1024, 1024, 1024, 1024),   # exact match
    (1000, 1000, 1024, 1024),   # nearest square
    (1920, 1080, 1344, 768),    # 16:9 landscape
    (1080, 1920, 768, 1344),    # 9:16 portrait
    (800,  600,  1152,  896),   # small 4:3 — snaps to closest 4:3 bucket
])
def test_snap_to_bucket_sdxl(w, h, expected_w, expected_h):
    sw, sh = snap_to_bucket(w, h, SDXL_BUCKETS)
    assert (sw, sh) in SDXL_BUCKETS, f"snap_to_bucket({w},{h}) → ({sw},{sh}) not in SDXL_BUCKETS"
    assert sw == expected_w and sh == expected_h, (
        f"snap_to_bucket({w},{h}) = ({sw},{sh}), expected ({expected_w},{expected_h})"
    )
