"""Tests for the stale model path warning used by `check`."""
from __future__ import annotations

from pathlib import Path

from ez_comfy.comfyui.autodetect import stale_model_path_warning


def test_warns_and_suggests_when_configured_path_missing(tmp_path: Path):
    suggestion = tmp_path / "current"
    suggestion.mkdir()
    text = stale_model_path_warning(str(tmp_path / "gone"), str(suggestion))
    assert text is not None
    assert "does not exist" in text and str(tmp_path / "gone") in text and str(suggestion) in text


def test_warns_without_suggestion_when_none_found(tmp_path: Path):
    text = stale_model_path_warning(str(tmp_path / "gone"), None)
    assert text is not None and "currently uses" not in text


def test_no_warning_when_path_exists_or_unset(tmp_path: Path):
    assert stale_model_path_warning(str(tmp_path), "/elsewhere") is None
    assert stale_model_path_warning(None, "/elsewhere") is None
    assert stale_model_path_warning("", "/elsewhere") is None


def test_does_not_suggest_the_same_path(tmp_path: Path):
    text = stale_model_path_warning(str(tmp_path / "gone"), str(tmp_path / "gone"))
    assert text is not None and "currently uses" not in text
