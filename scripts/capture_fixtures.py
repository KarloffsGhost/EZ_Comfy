"""Record scrubbed ComfyUI responses for use as test fixtures.

The repository is public, so nothing is written next to the tests. The script writes to a
staging folder and prints a summary of what was kept and dropped; the maintainer reviews the
staged files and copies them to tests/fixtures/comfy_<version>/ by hand.

    python scripts/capture_fixtures.py --url http://127.0.0.1:8000 --staging ./fixture_staging \
        --keep RealESRGAN_x4plus.safetensors --keep sd_xl_base_1.0.safetensors

Scrubbing is allow-list based: unknown fields are dropped rather than redacted.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("capture_fixtures")

# Node classes EZ Comfy reads or builds with (loaders, sampler, video/audio).
NODE_CLASSES = (
    "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader", "CLIPVisionLoader",
    "VAELoader", "LoraLoader", "UpscaleModelLoader", "ControlNetLoader", "KSampler",
    "ImageOnlyCheckpointLoader", "SaveAudio", "EmptyLatentAudio", "VAEDecodeAudio",
    "SaveAnimatedWEBP", "SVD_img2vid_Conditioning",
)

# Inputs whose options are the user's own file names. Everything else (sampler_name,
# scheduler, weight_dtype, ...) is a fixed ComfyUI vocabulary and is kept whole.
MODEL_FILE_INPUTS = frozenset({
    "ckpt_name", "unet_name", "clip_name", "clip_name1", "clip_name2", "vae_name",
    "lora_name", "model_name", "control_net_name",
})

# system_stats fields that are safe to publish.
_SYSTEM_KEYS = ("os", "comfyui_version", "python_version", "pytorch_version", "embedded_python",
                "ram_total", "ram_free")
_DEVICE_KEYS = ("name", "type", "index", "vram_total", "vram_free", "torch_vram_total", "torch_vram_free")

_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"',;]*")
_UNIX_HOME_PATH = re.compile(r"/(?:home|Users|root)/[^\s\"',;]*")
_UNC_PATH = re.compile(r"\\\\[^\s\"',;]+")


def scrub_text(text: str) -> str:
    """Replace absolute paths (and so user names in them) with a placeholder."""
    for pattern in (_UNC_PATH, _WINDOWS_PATH, _UNIX_HOME_PATH):
        text = pattern.sub("<path>", text)
    return text


def _scrub_strings(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [_scrub_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _scrub_strings(item) for key, item in value.items()}
    return value


def scrub_system_stats(stats: dict) -> dict:
    """Keep versions, OS and GPU memory; drop argv, host names, paths and anything unknown."""
    system = stats.get("system") if isinstance(stats.get("system"), dict) else {}
    devices = stats.get("devices") if isinstance(stats.get("devices"), list) else []
    return {
        "system": {key: system[key] for key in _SYSTEM_KEYS if key in system},
        "devices": [
            _scrub_strings({key: device[key] for key in _DEVICE_KEYS if key in device})
            for device in devices if isinstance(device, dict)
        ],
    }


def _filter_options(options: list, keep: set[str]) -> list:
    return [option for option in options if isinstance(option, str) and option in keep]


def scrub_node_info(node: dict, keep: set[str]) -> dict:
    """Keep a node's schema; reduce user file-name options to the hand-chosen `keep` set."""
    cleaned = {
        "input": {}, "input_order": node.get("input_order"),
        "output": node.get("output"), "output_name": node.get("output_name"),
        "name": node.get("name"), "category": node.get("category"),
    }
    for section, inputs in (node.get("input") or {}).items():
        if not isinstance(inputs, dict):
            continue
        cleaned["input"][section] = {}
        for input_name, spec in inputs.items():
            cleaned["input"][section][input_name] = (
                _scrub_combo(spec, keep) if input_name in MODEL_FILE_INPUTS else spec
            )
    return _scrub_strings({key: value for key, value in cleaned.items() if value is not None})


def _scrub_combo(spec: Any, keep: set[str]) -> Any:
    """Handle both combo shapes: [[options], {...}] and ["COMBO", {"options": [...]}]."""
    if not isinstance(spec, list) or not spec:
        return spec
    if isinstance(spec[0], list):
        return [_filter_options(spec[0], keep), *spec[1:]]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        extras = dict(spec[1])
        extras["options"] = _filter_options(extras.get("options", []), keep)
        return ["COMBO", extras]
    return spec


def build_fixture_set(stats: dict, object_info: dict[str, dict], keep: set[str]) -> dict[str, Any]:
    """Return {filename: json-able} for the fixture folder."""
    files: dict[str, Any] = {"system_stats.json": scrub_system_stats(stats)}
    for class_name in NODE_CLASSES:
        node = object_info.get(class_name)
        if isinstance(node, dict):
            files[f"object_info/{class_name}.json"] = {class_name: scrub_node_info(node, keep)}
    return files


def fetch(url: str) -> tuple[dict, dict[str, dict]]:
    with httpx.Client(base_url=url, timeout=30.0) as http:
        stats = http.get("/system_stats")
        stats.raise_for_status()
        object_info: dict[str, dict] = {}
        for class_name in NODE_CLASSES:
            response = http.get(f"/object_info/{class_name}")
            if response.status_code == 404:
                logger.warning("Node %s not present on this server; skipped", class_name)
                continue
            response.raise_for_status()
            object_info.update(response.json())
    return stats.json(), object_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--staging", type=Path, default=Path("fixture_staging"))
    parser.add_argument("--keep", action="append", default=[],
                        help="model file name to keep in loader options (repeatable); all others are dropped")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    keep = set(args.keep)
    stats, object_info = fetch(args.url)
    files = build_fixture_set(stats, object_info, keep)

    version = files["system_stats.json"]["system"].get("comfyui_version", "unknown")
    target = args.staging / f"comfy_{version}"
    seen_names = {
        option
        for node in object_info.values() if isinstance(node, dict)
        for section in (node.get("input") or {}).values() if isinstance(section, dict)
        for name, spec in section.items() if name in MODEL_FILE_INPUTS
        for option in _combo_options(spec)
    }
    for relative, payload in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Staged {len(files)} files in {target}")
    print(f"  kept model file names : {sorted(keep & seen_names) or 'none'}")
    print(f"  dropped model names   : {len(seen_names - keep)} (not recorded)")
    print("  dropped system fields : argv and everything outside the allow-list")
    print("Review the staged files, then copy them to tests/fixtures/ by hand. Nothing was committed.")
    return 0


def _combo_options(spec: Any) -> list[str]:
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return [option for option in spec[0] if isinstance(option, str)]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return [option for option in spec[1].get("options", []) if isinstance(option, str)]
    return []


if __name__ == "__main__":
    sys.exit(main())
