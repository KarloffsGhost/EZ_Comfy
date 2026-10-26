"""Unit tests for ez_comfy/comfyui/autodetect.py"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
import pytest
import yaml

from ez_comfy.comfyui.autodetect import (
    DetectedConfig,
    _parse_desktop_config,
    _probe_port,
    detect_comfyui,
)

_COMFYUI_HTML = "<html><head><title>ComfyUI</title></head><body></body></html>"
_OTHER_HTML = "<html><head><title>Other App</title></head><body></body></html>"


# ---------------------------------------------------------------------------
# _parse_desktop_config
# ---------------------------------------------------------------------------

def test_desktop_config_parses_base_path(tmp_path: Path):
    config = tmp_path / "extra_models_config.yaml"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    config.write_text(
        yaml.dump({
            "comfyui_desktop": {
                "base_path": str(tmp_path),
                "is_default": "true",
            }
        }),
        encoding="utf-8",
    )
    _, model_path = _parse_desktop_config(config)
    assert model_path == str(models_dir)


def test_desktop_config_missing_file_returns_none(tmp_path: Path):
    _, model_path = _parse_desktop_config(tmp_path / "nonexistent.yaml")
    assert model_path is None


def test_desktop_config_no_base_path_key(tmp_path: Path):
    config = tmp_path / "extra_models_config.yaml"
    config.write_text(yaml.dump({"comfyui_desktop": {"custom_nodes": "custom_nodes/"}}))
    _, model_path = _parse_desktop_config(config)
    assert model_path is None


# ---------------------------------------------------------------------------
# _probe_port
# ---------------------------------------------------------------------------

def test_probe_port_finds_comfyui():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_COMFYUI_HTML)

    with patch("ez_comfy.comfyui.autodetect.httpx.get") as mock_get:
        mock_get.return_value = httpx.Response(200, text=_COMFYUI_HTML)
        assert _probe_port(8000) is True


def test_probe_port_rejects_non_comfyui_server():
    with patch("ez_comfy.comfyui.autodetect.httpx.get") as mock_get:
        mock_get.return_value = httpx.Response(200, text=_OTHER_HTML)
        assert _probe_port(8000) is False


def test_probe_port_connection_refused():
    with patch("ez_comfy.comfyui.autodetect.httpx.get",
               side_effect=httpx.ConnectError("refused")):
        assert _probe_port(8188) is False


def test_probe_port_timeout():
    with patch("ez_comfy.comfyui.autodetect.httpx.get",
               side_effect=httpx.TimeoutException("timeout")):
        assert _probe_port(8188) is False


# ---------------------------------------------------------------------------
# detect_comfyui
# ---------------------------------------------------------------------------

def test_explicit_override_wins_no_probing():
    """If both explicit_url and explicit_model_path are set, no probing happens."""
    with patch("ez_comfy.comfyui.autodetect._probe_port") as mock_probe:
        result = detect_comfyui(
            explicit_url="http://127.0.0.1:9999",
            explicit_model_path="/my/models",
        )
    mock_probe.assert_not_called()
    assert result.base_url == "http://127.0.0.1:9999"
    assert result.model_base_path == "/my/models"
    assert result.source == "config_file"


def test_port_probe_finds_8000():
    """When port 8000 responds with ComfyUI HTML, it is selected."""
    def fake_probe(port: int) -> bool:
        return port == 8000

    with patch("ez_comfy.comfyui.autodetect._probe_port", side_effect=fake_probe), \
         patch("ez_comfy.comfyui.autodetect._desktop_app_config_path", return_value=None):
        result = detect_comfyui()

    assert result.base_url == "http://127.0.0.1:8000"
    assert result.source == "probe"


def test_port_probe_finds_8188():
    """When port 8188 responds with ComfyUI HTML, it is selected first."""
    def fake_probe(port: int) -> bool:
        return port == 8188

    with patch("ez_comfy.comfyui.autodetect._probe_port", side_effect=fake_probe), \
         patch("ez_comfy.comfyui.autodetect._desktop_app_config_path", return_value=None):
        result = detect_comfyui()

    assert result.base_url == "http://127.0.0.1:8188"
    assert result.source == "probe"


def test_no_comfyui_running_returns_none():
    """When no port responds, base_url is None."""
    with patch("ez_comfy.comfyui.autodetect._probe_port", return_value=False), \
         patch("ez_comfy.comfyui.autodetect._desktop_app_config_path", return_value=None):
        result = detect_comfyui()

    assert result.base_url is None
    assert result.source == "default"


def test_desktop_app_config_provides_model_path(tmp_path: Path):
    """Desktop app config is read and model path is extracted even during port probe."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    config_path = tmp_path / "extra_models_config.yaml"
    config_path.write_text(
        yaml.dump({
            "comfyui_desktop": {"base_path": str(tmp_path), "is_default": "true"}
        }),
        encoding="utf-8",
    )

    def fake_probe(port: int) -> bool:
        return port == 8000

    with patch("ez_comfy.comfyui.autodetect._probe_port", side_effect=fake_probe), \
         patch("ez_comfy.comfyui.autodetect._desktop_app_config_path",
               return_value=config_path):
        result = detect_comfyui()

    assert result.base_url == "http://127.0.0.1:8000"
    assert result.model_base_path == str(models_dir)


def test_explicit_model_path_not_overridden_by_desktop_config(tmp_path: Path):
    """Explicit model_base_path must win over desktop app config."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    config_path = tmp_path / "extra_models_config.yaml"
    config_path.write_text(
        yaml.dump({
            "comfyui_desktop": {"base_path": str(tmp_path)}
        }),
        encoding="utf-8",
    )

    with patch("ez_comfy.comfyui.autodetect._probe_port", return_value=False), \
         patch("ez_comfy.comfyui.autodetect._desktop_app_config_path",
               return_value=config_path):
        result = detect_comfyui(explicit_model_path="/explicit/models")

    assert result.model_base_path == "/explicit/models"
