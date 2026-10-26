"""EZ Comfy — deterministic ComfyUI orchestrator."""
from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("ez-comfy")
except metadata.PackageNotFoundError:
    # Running from a source checkout that was never installed.
    __version__ = "0.0.0+unknown"
