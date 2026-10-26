"""Tests for scripts/export_template_graph.py: folding of helper nodes when flattening a template."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "export_template_graph.py"
_spec = importlib.util.spec_from_file_location("export_template_graph", SCRIPT)
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["a.safetensors"], {}]}},
        "input_order": {"required": ["ckpt_name"]},
    },
    "KSampler": {
        "input": {"required": {
            "model": ["MODEL", {}], "steps": ["INT", {}], "cfg": ["FLOAT", {}],
            "seed": ["INT", {}], "sampler_name": [["euler"], {}],
        }},
        "input_order": {"required": ["model", "seed", "steps", "cfg", "sampler_name"]},
    },
    "SaveImage": {
        "input": {"required": {"images": ["IMAGE", {}], "filename_prefix": ["STRING", {}]}},
        "input_order": {"required": ["images", "filename_prefix"]},
    },
}


def _link(link_id, origin, slot, target, target_slot, kind):
    return {"id": link_id, "origin_id": origin, "origin_slot": slot, "target_id": target,
            "target_slot": target_slot, "type": kind}


def _template(switch_on: bool) -> dict:
    """A tiny template: two step counts behind a boolean switch, as in the Wan 2.2 template."""
    nodes = [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["a.safetensors"], "inputs": []},
        {"id": 2, "type": "PrimitiveInt", "widgets_values": [20, "fixed"], "inputs": []},
        {"id": 3, "type": "PrimitiveInt", "widgets_values": [4, "fixed"], "inputs": []},
        {"id": 4, "type": "PrimitiveBoolean", "widgets_values": [switch_on], "inputs": []},
        {"id": 5, "type": "ComfySwitchNode", "widgets_values": [False], "inputs": [
            {"name": "on_false", "link": 12}, {"name": "on_true", "link": 13}, {"name": "switch", "link": 14}]},
        {"id": 6, "type": "KSampler", "widgets_values": [7, "randomize", 99, 3.5, "euler"], "inputs": [
            {"name": "model", "link": 11}, {"name": "steps", "link": 15}]},
        {"id": 7, "type": "MarkdownNote", "widgets_values": ["note"], "inputs": []},
        {"id": 8, "type": "SaveImage", "widgets_values": ["out"], "inputs": [{"name": "images", "link": None}]},
    ]
    links = [
        _link(11, 1, 0, 6, 0, "MODEL"),
        _link(12, 2, 0, 5, 0, "INT"), _link(13, 3, 0, 5, 1, "INT"), _link(14, 4, 0, 5, 2, "BOOLEAN"),
        _link(15, 5, 0, 6, 1, "INT"),
    ]
    return {"nodes": nodes, "links": links}


def test_switch_off_selects_the_on_false_branch():
    graph = exporter.Exporter(**{"nodes": _template(False)["nodes"], "links": _template(False)["links"],
                                 "object_info": OBJECT_INFO}).build()
    assert graph["6"]["inputs"]["steps"] == 20


def test_switch_on_selects_the_on_true_branch():
    template = _template(True)
    graph = exporter.Exporter(template["nodes"], template["links"], OBJECT_INFO).build()
    assert graph["6"]["inputs"]["steps"] == 4


def test_helper_and_note_nodes_are_not_exported_and_links_become_references():
    template = _template(False)
    graph = exporter.Exporter(template["nodes"], template["links"], OBJECT_INFO).build()
    assert set(graph) == {"1", "6", "8"}
    assert graph["6"]["inputs"]["model"] == ["1", 0]


def test_seed_widget_skips_the_control_after_generate_value():
    template = _template(False)
    graph = exporter.Exporter(template["nodes"], template["links"], OBJECT_INFO).build()
    inputs = graph["6"]["inputs"]
    assert (inputs["seed"], inputs["cfg"], inputs["sampler_name"]) == (7, 3.5, "euler")
