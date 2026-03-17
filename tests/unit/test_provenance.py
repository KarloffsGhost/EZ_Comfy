"""Unit tests for planner/provenance.py"""
from __future__ import annotations

import pytest
from ez_comfy.planner.provenance import Alternative, Decision, ProvenanceRecord, _safe_value


# ---------------------------------------------------------------------------
# _safe_value
# ---------------------------------------------------------------------------

def test_safe_value_float_with_decimal():
    assert _safe_value(1.5) == "1.5"


def test_safe_value_float_whole_number():
    assert _safe_value(7.0) == "7"
    assert _safe_value(1.0) == "1"


def test_safe_value_int():
    assert _safe_value(42) == "42"


def test_safe_value_string():
    assert _safe_value("euler") == "euler"


def test_safe_value_none():
    assert _safe_value(None) == "None"


# ---------------------------------------------------------------------------
# Alternative
# ---------------------------------------------------------------------------

def test_alternative_fields():
    alt = Alternative(value="DreamShaper XL", rejected_reason="not installed")
    assert alt.value == "DreamShaper XL"
    assert alt.rejected_reason == "not installed"


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def test_decision_fields():
    d = Decision(
        parameter="checkpoint",
        chosen_value="realvisxl.safetensors",
        source="recommendation",
        reason="Top-scored installed model",
        alternatives=[Alternative("juggernaut.safetensors", "not installed")],
    )
    assert d.parameter == "checkpoint"
    assert d.chosen_value == "realvisxl.safetensors"
    assert d.source == "recommendation"
    assert len(d.alternatives) == 1


def test_decision_defaults_empty_alternatives():
    d = Decision(parameter="intent", chosen_value="txt2img", source="prompt_keyword", reason="no image provided")
    assert d.alternatives == []


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------

def test_provenance_add():
    rec = ProvenanceRecord()
    d = Decision(parameter="intent", chosen_value="txt2img", source="prompt_keyword", reason="default")
    rec.add(d)
    assert len(rec.decisions) == 1
    assert rec.decisions[0].parameter == "intent"


def test_empty_provenance_to_dict():
    rec = ProvenanceRecord()
    d = rec.to_dict()
    assert d["decisions"] == []
    assert d["ez_comfy_version"] == ""
    assert d["gpu_name"] == ""
    assert d["vram_available_gb"] == 0.0
    assert d["vram_estimated_gb"] == 0.0
    assert d["model_id"] == ""
    assert d["model_family"] == ""
    assert d["recipe_id"] == ""


def test_provenance_record_to_dict():
    rec = ProvenanceRecord(
        gpu_name="RTX 4090",
        vram_available_gb=24.0,
        vram_estimated_gb=6.5,
        model_id="realvis_xl_v50_lightning",
        model_family="sdxl_lightning",
        recipe_id="txt2img_basic",
        ez_comfy_version="0.1.0",
    )
    rec.add(Decision(
        parameter="checkpoint",
        chosen_value="realvisxl.safetensors",
        source="recommendation",
        reason="Top-scored installed model",
        alternatives=[Alternative("juggernaut.safetensors", "not installed")],
    ))
    rec.add(Decision(
        parameter="steps",
        chosen_value=6,
        source="model_catalog",
        reason="Model catalog entry specifies steps=6",
    ))

    d = rec.to_dict()
    assert d["gpu_name"] == "RTX 4090"
    assert d["vram_available_gb"] == 24.0
    assert d["vram_estimated_gb"] == 6.5
    assert d["model_id"] == "realvis_xl_v50_lightning"
    assert d["recipe_id"] == "txt2img_basic"
    assert d["ez_comfy_version"] == "0.1.0"
    assert len(d["decisions"]) == 2

    ckpt = d["decisions"][0]
    assert ckpt["parameter"] == "checkpoint"
    assert ckpt["chosen_value"] == "realvisxl.safetensors"
    assert ckpt["source"] == "recommendation"
    assert len(ckpt["alternatives"]) == 1
    assert ckpt["alternatives"][0]["value"] == "juggernaut.safetensors"
    assert ckpt["alternatives"][0]["rejected_reason"] == "not installed"

    steps = d["decisions"][1]
    assert steps["chosen_value"] == "6"
    assert steps["source"] == "model_catalog"


def test_provenance_to_human_readable():
    rec = ProvenanceRecord(
        gpu_name="RTX 3090",
        vram_available_gb=24.0,
        vram_estimated_gb=8.0,
    )
    rec.add(Decision(
        parameter="checkpoint",
        chosen_value="realvisxl.safetensors",
        source="recommendation",
        reason="Top-scored installed model",
        alternatives=[Alternative("juggernaut.safetensors", "not installed")],
    ))
    text = rec.to_human_readable()
    assert "EZ Comfy Provenance" in text
    assert "RTX 3090" in text
    assert "24.0GB VRAM" in text
    assert "8.0GB" in text
    assert "checkpoint" in text
    assert "realvisxl.safetensors" in text
    assert "[recommendation]" in text
    assert "why: Top-scored installed model" in text
    assert "rejected: juggernaut.safetensors" in text
    assert "not installed" in text


def test_provenance_float_formatting_in_dict():
    rec = ProvenanceRecord()
    rec.add(Decision(parameter="cfg", chosen_value=1.5, source="model_catalog", reason="catalog"))
    rec.add(Decision(parameter="cfg_whole", chosen_value=7.0, source="family_profile", reason="profile"))
    d = rec.to_dict()
    assert d["decisions"][0]["chosen_value"] == "1.5"
    assert d["decisions"][1]["chosen_value"] == "7"


def test_provenance_multiple_alternatives():
    rec = ProvenanceRecord()
    rec.add(Decision(
        parameter="checkpoint",
        chosen_value="model_c.safetensors",
        source="recommendation",
        reason="Only installed model",
        alternatives=[
            Alternative("model_a.safetensors", "requires 12GB VRAM, only 8GB available"),
            Alternative("model_b.safetensors", "not installed"),
        ],
    ))
    text = rec.to_human_readable()
    assert "model_a.safetensors" in text
    assert "requires 12GB VRAM" in text
    assert "model_b.safetensors" in text
    assert "not installed" in text
