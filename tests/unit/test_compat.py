"""Unit tests for ez_comfy/compat.py and its COMPAT.md header."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ez_comfy import __version__, compat
from ez_comfy.compat import TESTED_COMFYUI_RANGE, parse_version, version_line, version_status

COMPAT_MD = Path(__file__).parent.parent.parent / "COMPAT.md"


def test_compat_md_header_matches_constant():
    first_line = COMPAT_MD.read_text(encoding="utf-8").splitlines()[0]
    match = re.fullmatch(r"Tested ComfyUI: (\S+) - (\S+)", first_line)
    assert match, f"unexpected COMPAT.md header: {first_line!r}"
    assert (match.group(1), match.group(2)) == TESTED_COMFYUI_RANGE


@pytest.mark.parametrize("text,expected", [
    ("0.42.0", (0, 42, 0)),
    ("v0.3.1", (0, 3, 1)),
    ("0.42.0-rc1", (0, 42, 0)),
    ("1.1", (1, 1)),
    ("", None),
    ("unknown", None),
])
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_status_uses_numeric_not_string_order(monkeypatch):
    monkeypatch.setattr(compat, "TESTED_COMFYUI_RANGE", ("0.9.0", "0.10.0"))
    assert version_status("0.10.0") == "tested"      # a string compare would put 0.10 before 0.9
    assert version_status("0.8.9") == "untested-older"
    assert version_status("0.10.1") == "untested-newer"


def test_status_unknown_when_missing():
    assert version_status(None) == "unknown"
    assert version_line(None) == "ComfyUI version: unavailable"


def test_version_line_warns_outside_range_but_never_raises(monkeypatch):
    monkeypatch.setattr(compat, "TESTED_COMFYUI_RANGE", ("0.42.0", "0.42.0"))
    assert "[WARN]" not in version_line("0.42.0")
    assert "[WARN]" in version_line("0.43.0") and "newer" in version_line("0.43.0")
    assert "[WARN]" in version_line("0.41.9") and "older" in version_line("0.41.9")


def test_package_version_is_a_nonempty_string():
    assert isinstance(__version__, str) and __version__


def test_health_response_and_app_report_package_version():
    from ez_comfy.api.models import HealthResponse
    from ez_comfy.api.server import create_app
    from ez_comfy.config.schema import Settings

    assert HealthResponse.model_fields["version"].default == __version__
    assert create_app(Settings()).version == __version__
