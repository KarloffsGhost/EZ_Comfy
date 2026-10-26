"""
Auto-detect a locally running ComfyUI instance and its model base path.

Resolution order (first hit wins for each field):
  1. Explicit value already in Settings (caller passes it through)
  2. ComfyUI Desktop app config  (%APPDATA%/ComfyUI/config.json `basePath`, then
     extra_models_config.yaml, in the same folder)
  3. Port probe — tries common ports until a ComfyUI HTML page responds
  4. Hard-coded defaults
"""
from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

_PROBE_PORTS = [8188, 8000, 8189, 8001]
_PROBE_TIMEOUT = 1.0
_COMFYUI_TITLE = "<title>ComfyUI</title>"


@dataclass
class DetectedConfig:
    base_url: str | None         # None if no ComfyUI is reachable
    model_base_path: str | None  # None if not derivable
    source: str                  # "config_file" | "desktop_app" | "probe" | "default"


def _desktop_app_config_path() -> Path | None:
    """Return the platform-appropriate path to ComfyUI Desktop's extra_models_config.yaml."""
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "ComfyUI" / "extra_models_config.yaml"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "ComfyUI" / "extra_models_config.yaml"
    else:
        config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(config_home) / "ComfyUI" / "extra_models_config.yaml"
    return None


def _parse_desktop_config_json(config_path: Path) -> str | None:
    """
    Read the models directory from ComfyUI Desktop's config.json (`basePath`).
    Returns None when the file is missing, unreadable, or points at a missing folder.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.debug("Could not read desktop app config %s: %s", config_path, exc)
        return None

    base = data.get("basePath") if isinstance(data, dict) else None
    if not isinstance(base, str) or not base:
        return None
    candidate = Path(base)
    models_dir = candidate / "models"
    if models_dir.exists():
        return str(models_dir)
    if candidate.exists():
        return str(candidate)
    return None


def desktop_model_path() -> str | None:
    """
    The models directory ComfyUI Desktop currently uses, ignoring any explicit setting.
    config.json (`basePath`) wins over extra_models_config.yaml, which is the older source.
    Used by `check` to suggest a replacement for a stale configured path.
    """
    yaml_path = _desktop_app_config_path()
    if yaml_path is None:
        return None
    json_path = yaml_path.parent / "config.json"
    if json_path.exists():
        found = _parse_desktop_config_json(json_path)
        if found:
            return found
    if yaml_path.exists():
        return _parse_desktop_config(yaml_path)[1]
    return None


def stale_model_path_warning(configured: str | None, suggestion: str | None) -> str | None:
    """
    Warning text when an explicitly configured model path does not exist, else None.
    Warn-only: callers must not change the configured value.
    """
    if not configured or Path(configured).exists():
        return None
    text = f"[WARN] comfyui.model_base_path '{configured}' does not exist."
    if suggestion and Path(suggestion) != Path(configured):
        text += f" ComfyUI Desktop currently uses '{suggestion}'; update config/settings.yaml if that is intended."
    return text


def _parse_desktop_config(config_path: Path) -> tuple[str | None, str | None]:
    """
    Parse ComfyUI Desktop's extra_models_config.yaml.
    Returns (base_url, model_base_path) — either may be None if not found.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.debug("Could not read desktop app config %s: %s", config_path, exc)
        return None, None

    model_path: str | None = None
    for section in data.values():
        if not isinstance(section, dict):
            continue
        base = section.get("base_path") or section.get("download_model_base")
        if base:
            candidate = Path(str(base))
            # Resolve "models" sub-directory if it exists
            models_dir = candidate / "models"
            if models_dir.exists():
                model_path = str(models_dir)
            elif candidate.exists():
                model_path = str(candidate)
            if model_path:
                break

    return None, model_path


def _probe_port(port: int) -> bool:
    """Return True if a ComfyUI instance is running on localhost:{port}."""
    base = f"http://127.0.0.1:{port}"
    try:
        resp = httpx.get(f"{base}/", timeout=_PROBE_TIMEOUT, follow_redirects=True)
        if resp.status_code == 200 and _COMFYUI_TITLE in resp.text:
            return True
    except httpx.HTTPError as exc:
        logger.debug("ComfyUI probe of %s/ failed: %s", base, exc)
        return False
    # Newer frontends may change the page title; /system_stats is the stable signal.
    try:
        resp = httpx.get(f"{base}/system_stats", timeout=_PROBE_TIMEOUT)
        body = resp.json() if resp.status_code == 200 else None
        return isinstance(body, dict) and isinstance(body.get("system"), dict)
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("ComfyUI probe of %s/system_stats failed: %s", base, exc)
        return False


def detect_comfyui(
    explicit_url: str | None = None,
    explicit_model_path: str | None = None,
) -> DetectedConfig:
    """
    Detect ComfyUI base URL and model path.

    If explicit values are provided (already set in settings.yaml), they win
    unconditionally — no probing is performed for that field.
    """
    # 1. Explicit overrides (from settings.yaml or env vars)
    if explicit_url and explicit_model_path:
        return DetectedConfig(
            base_url=explicit_url,
            model_base_path=explicit_model_path,
            source="config_file",
        )

    base_url = explicit_url
    model_path = explicit_model_path
    source = "config_file" if explicit_url else "default"

    # 2. Desktop app config — may provide the model path (config.json first, then the yaml)
    config_path = _desktop_app_config_path()
    if config_path and not model_path:
        json_path = config_path.parent / "config.json"
        found = _parse_desktop_config_json(json_path) if json_path.exists() else None
        if not found and config_path.exists():
            _, found = _parse_desktop_config(config_path)
        if found:
            model_path = found
            logger.debug("Model path from Desktop app config: %s", model_path)

    # 3. Port probe — only if no explicit URL given
    if not base_url:
        probed_ports: list[int] = []
        for port in _PROBE_PORTS:
            probed_ports.append(port)
            if _probe_port(port):
                base_url = f"http://127.0.0.1:{port}"
                source = "probe"
                logger.debug("Detected ComfyUI on port %d", port)
                break

        if not base_url:
            logger.warning(
                "ComfyUI not found on ports %s. "
                "Start ComfyUI or set comfyui.base_url in config/settings.yaml.",
                probed_ports,
            )
            source = "default"

    return DetectedConfig(
        base_url=base_url,
        model_base_path=model_path,
        source=source,
    )
