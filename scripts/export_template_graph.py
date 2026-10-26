"""Flatten a ComfyUI workflow template (UI format, optionally with one subgraph) into API format.

Used to produce the golden graphs under tests/fixtures/workflows/. Primitive, switch and math
helper nodes are folded into constants, so the output contains only real ComfyUI nodes. Widget
order comes from a live /object_info, so run it against a ComfyUI of the version being recorded.

    python scripts/export_template_graph.py TEMPLATE.json --url http://127.0.0.1:8000 \
        --out tests/fixtures/workflows/wan22_t2v.api.json --placeholder-prompts
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import httpx

# Helper nodes folded into constants while exporting.
_PRIMITIVES = {"PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean", "PrimitiveString", "PrimitiveStringMultiline"}
_SKIP = {"MarkdownNote", "Note", "Reroute"}
_SEED_WIDGETS = {"seed", "noise_seed"}   # followed by a UI-only control_after_generate value
_LINK_TYPES = {"MODEL", "CLIP", "VAE", "LATENT", "IMAGE", "CONDITIONING", "CLIP_VISION", "MASK", "VIDEO", "AUDIO",
               "CONTROL_NET", "SIGMAS", "SAMPLER", "NOISE", "GUIDER", "CLIP_VISION_OUTPUT"}


class Exporter:
    def __init__(self, nodes: list[dict], links: list[dict], object_info: dict) -> None:
        self.nodes = {n["id"]: n for n in nodes}
        self.links = {l["id"]: l for l in links}
        self.info = object_info
        self.api: dict[str, dict] = {}

    # -- constants ---------------------------------------------------------------------------
    def _const(self, node_id: int, slot: int = 0):
        """Value produced by a folded helper node (or a subgraph input falling back to its inner default)."""
        node = self.nodes[node_id]
        kind = node["type"]
        values = node.get("widgets_values") or []
        if kind in _PRIMITIVES:
            link = self._incoming(node, "value")
            # Fed by a subgraph input (origin -10) or unconnected: the node's own widget is the default.
            if link is None or link["origin_id"] == -10:
                return values[0]
            return self._value(link)
        if kind == "ComfySwitchNode":
            switch = self._value(self._incoming(node, "switch"))
            return self._value(self._incoming(node, "on_true" if switch else "on_false"))
        if kind == "ComfyMathExpression":
            a = self._value(self._incoming(node, "values.a"))
            b = self._value(self._incoming(node, "values.b"))
            result = math.floor(a * b) + 1          # the only expression the Wan templates use
            if values[0] != "floor(a * b) + 1":
                raise ValueError(f"unsupported expression {values[0]!r}")
            return result
        raise ValueError(f"cannot fold node type {kind}")

    def _incoming(self, node: dict, name: str):
        for entry in node.get("inputs", []):
            if entry["name"] == name and entry.get("link") is not None:
                return self.links[entry["link"]]
        return None

    def _is_foldable(self, node: dict) -> bool:
        return node["type"] in _PRIMITIVES or node["type"] in ("ComfySwitchNode", "ComfyMathExpression")

    def _value(self, link: dict | None):
        """Resolve a link to a constant, or to a [node_id, slot] reference when it ends in a real node."""
        if link is None:
            raise ValueError("expected a connected value")
        origin = link["origin_id"]
        if origin == -10:
            raise ValueError("subgraph input reached without an inner default")
        if self._is_foldable(self.nodes[origin]):
            return self._const(origin, link["origin_slot"])
        return [str(origin), link["origin_slot"]]

    # -- graph -------------------------------------------------------------------------------
    def _widget_names(self, node_type: str) -> list[str]:
        spec = self.info[node_type]["input"]
        order = self.info[node_type].get("input_order", {})
        names = []
        for section in ("required", "optional"):
            for name in order.get(section, list(spec.get(section, {}))):
                entry = spec.get(section, {}).get(name)
                if entry is None:
                    continue
                kind = entry[0]
                if isinstance(kind, list) or kind in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                    names.append(name)
                elif kind not in _LINK_TYPES:
                    names.append(name)
        return names

    def build(self) -> dict:
        for node_id, node in self.nodes.items():
            if node["type"] in _SKIP or self._is_foldable(node) or node.get("mode", 0) in (2, 4):
                continue
            self.api[str(node_id)] = self._api_node(node)
        return self.api

    def _api_node(self, node: dict) -> dict:
        widget_names = self._widget_names(node["type"])
        values = list(node.get("widgets_values") or [])
        widgets: dict[str, object] = {}
        index = 0
        for name in widget_names:
            if index >= len(values):
                break
            widgets[name] = values[index]
            index += 1
            if name in _SEED_WIDGETS:
                index += 1                      # skip control_after_generate
        inputs: dict[str, object] = {}
        for name in self._all_input_names(node):
            link = self._incoming(node, name)
            if link is not None:
                if link["origin_id"] == -10:
                    if name in widgets:
                        inputs[name] = widgets[name]   # subgraph input: keep the inner default
                    continue
                inputs[name] = self._value(link)
            elif name in widgets:
                inputs[name] = widgets[name]
        for name, value in widgets.items():
            inputs.setdefault(name, value)
        return {"class_type": node["type"], "inputs": inputs}

    def _all_input_names(self, node: dict) -> list[str]:
        return [entry["name"] for entry in node.get("inputs", [])]


def _attach_top_level_savers(template: dict, subgraph: dict, graph: dict, object_info: dict) -> None:
    """Add Save*/Preview* nodes that live outside the subgraph, wired to its (single) output producer."""
    instance_ids = {n["id"] for n in template["nodes"] if n["type"] == subgraph["id"]}
    out_link = next((l for l in subgraph["links"] if l["target_id"] == -20), None)
    if out_link is None or len(instance_ids) != 1:
        raise ValueError("expected exactly one subgraph instance with one output")
    producer = [str(out_link["origin_id"]), out_link["origin_slot"]]
    top_links = {l[0]: l for l in template["links"]}
    for node in template["nodes"]:
        if not node["type"].startswith(("Save", "Preview")):
            continue
        # Build from the widgets only, then wire the inputs that came from the subgraph instance.
        stripped = {**node, "inputs": [{**entry, "link": None} for entry in node.get("inputs", [])]}
        api = Exporter([stripped], [], object_info)._api_node(stripped)
        for entry in node.get("inputs", []):
            link = top_links.get(entry.get("link"))
            if link is not None and link[1] in instance_ids:
                api["inputs"][entry["name"]] = producer
        graph[str(node["id"])] = api


def export(template: dict, object_info: dict) -> dict:
    subgraphs = (template.get("definitions") or {}).get("subgraphs", [])
    if len(subgraphs) > 1:
        raise ValueError("templates with more than one subgraph are not supported")
    source = subgraphs[0] if subgraphs else template
    graph = Exporter(source["nodes"], source["links"], object_info).build()
    if subgraphs:
        _attach_top_level_savers(template, subgraphs[0], graph, object_info)
    # Keep only nodes reachable from a Save* / Preview* output.
    outputs = [nid for nid, node in graph.items() if node["class_type"].startswith(("Save", "Preview"))]
    keep: set[str] = set()
    stack = list(outputs)
    while stack:
        nid = stack.pop()
        if nid in keep:
            continue
        keep.add(nid)
        for value in graph[nid]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph:
                stack.append(value[0])
    return {nid: graph[nid] for nid in sorted(keep, key=int)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("template", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--placeholder-prompts", action="store_true",
                        help="replace CLIPTextEncode text with PROMPT / NEGATIVE placeholders")
    args = parser.parse_args(argv)

    template = json.loads(args.template.read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.url, timeout=60.0) as http:
        response = http.get("/object_info")
        response.raise_for_status()
        info = response.json()
    graph = export(template, info)
    if args.placeholder_prompts:
        for node in graph.values():
            for role, placeholder in (("positive", "PROMPT"), ("negative", "NEGATIVE")):
                link = node["inputs"].get(role)
                if isinstance(link, list) and graph[link[0]]["class_type"] == "CLIPTextEncode":
                    graph[link[0]]["inputs"]["text"] = placeholder
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graph, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(graph)} nodes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
