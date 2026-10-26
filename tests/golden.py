"""Normalized comparison of ComfyUI API graphs, for golden-file tests.

Two graphs are equal when they have the same nodes, the same links, and the same values for a
whitelist of widgets, regardless of node ids. Seeds, prompt text, titles and filename prefixes are
ignored. Nodes are matched by a structural key (what feeds them and how their output is used), so
two identical-looking nodes with different roles (positive vs negative prompt) stay distinct.
Floats are compared to 6 decimals (the UI stores 5.0 as 5.000000000000001). Structure alone cannot
tell the positive and negative prompt apart (swapping them gives an isomorphic graph), so callers
that care pass compare_text=True and plan with the placeholder prompts PROMPT / NEGATIVE.
"""
from __future__ import annotations

import hashlib
import json

# Widgets whose values define the model/graph behaviour. Everything else is ignored.
COMPARED_WIDGETS = frozenset({
    "ckpt_name", "unet_name", "clip_name", "clip_name1", "clip_name2", "vae_name", "lora_name", "type",
    "weight_dtype", "device", "width", "height", "length", "batch_size", "steps", "cfg", "sampler_name",
    "scheduler", "shift", "fps", "add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise",
    "denoise", "strength_model", "format", "codec",
})


def _is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def _digest(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def _widgets(node: dict, compare_text: bool) -> dict:
    names = COMPARED_WIDGETS | ({"text"} if compare_text else set())
    return {
        k: (round(v, 6) if isinstance(v, float) else v)
        for k, v in node["inputs"].items() if not _is_link(v) and k in names
    }


def normalize_graph(graph: dict, compare_text: bool = False) -> list:
    """Return a canonical, id-free description of the graph (a sorted list)."""
    upstream: dict[str, str] = {}
    dangling = "<missing node>"

    def up(node_id: str) -> str:
        if node_id in upstream:
            return upstream[node_id]
        node = graph[node_id]
        links = sorted(
            (name, up(v[0]) if v[0] in graph else dangling, v[1])
            for name, v in node["inputs"].items() if _is_link(v)
        )
        upstream[node_id] = _digest([node["class_type"], _widgets(node, compare_text), links])
        return upstream[node_id]

    consumers: dict[str, set] = {node_id: set() for node_id in graph}
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if _is_link(value) and value[0] in graph:
                consumers[value[0]].add((node["class_type"], name))

    key = {node_id: _digest([up(node_id), sorted(consumers[node_id])]) for node_id in graph}

    description = []
    for node_id, node in graph.items():
        links = sorted(
            (name, key.get(v[0], dangling), v[1]) for name, v in node["inputs"].items() if _is_link(v)
        )
        description.append([key[node_id], node["class_type"], _widgets(node, compare_text), links])
    return sorted(description, key=lambda entry: json.dumps(entry, sort_keys=True, default=str))


def graphs_match(actual: dict, expected: dict, compare_text: bool = False) -> bool:
    return normalize_graph(actual, compare_text) == normalize_graph(expected, compare_text)


def describe_difference(actual: dict, expected: dict, compare_text: bool = False) -> str:
    """Human-readable summary for a failed assertion."""
    a, e = normalize_graph(actual, compare_text), normalize_graph(expected, compare_text)
    only_actual = [x for x in a if x not in e]
    only_expected = [x for x in e if x not in a]
    return f"only in actual: {only_actual}\nonly in expected: {only_expected}"
