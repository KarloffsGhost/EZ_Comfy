"""ComfyUI versions this release of EZ Comfy has been verified against.

Keep in step with the header line of COMPAT.md (tests/unit/test_compat.py enforces it).
Outside the range `check` prints a warning; nothing fails.
"""
from __future__ import annotations

import re

# Inclusive (lowest, highest) tested ComfyUI version.
TESTED_COMFYUI_RANGE: tuple[str, str] = ("0.42.0", "0.42.0")


def parse_version(text: str) -> tuple[int, ...] | None:
    """'0.42.0' -> (0, 42, 0). Ignores a leading 'v' and any suffix such as '-rc1' or '+cu130'."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", text or "")
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def version_status(version: str | None) -> str:
    """Return 'unknown', 'untested-older', 'untested-newer' or 'tested'."""
    parsed = parse_version(version or "")
    if parsed is None:
        return "unknown"
    low = parse_version(TESTED_COMFYUI_RANGE[0])
    high = parse_version(TESTED_COMFYUI_RANGE[1])
    assert low is not None and high is not None
    if parsed < low:
        return "untested-older"
    if parsed > high:
        return "untested-newer"
    return "tested"


def version_line(version: str | None) -> str:
    """The text `check` prints for the ComfyUI version, including any warning."""
    low, high = TESTED_COMFYUI_RANGE
    status = version_status(version)
    if status == "unknown":
        return "ComfyUI version: unavailable"
    if status == "tested":
        return f"ComfyUI version: {version}"
    return (
        f"ComfyUI version: {version}\n"
        f"[WARN] EZ Comfy was tested with ComfyUI {low} - {high}; "
        f"{version} is {'older' if status == 'untested-older' else 'newer'} and may behave differently."
    )
