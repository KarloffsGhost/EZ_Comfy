"""Privacy tests for scripts/capture_fixtures.py: the repo is public."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "capture_fixtures.py"
_spec = importlib.util.spec_from_file_location("capture_fixtures", SCRIPT)
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)

SAMPLE_STATS = {
    "system": {
        "os": "nt",
        "comfyui_version": "0.42.0",
        "python_version": "3.12.11",
        "pytorch_version": "2.9.1+cu130",
        "embedded_python": False,
        "ram_total": 1, "ram_free": 1,
        "argv": [r"C:\Users\someone\AppData\Local\Comfy\main.py", "--base-directory", r"J:\Documents\someone\ComfyUI"],
        "hostname": "SOMEONE-PC",
    },
    "devices": [{
        "name": "cuda:0 NVIDIA GeForce RTX 5070 Ti : cudaMallocAsync",
        "type": "cuda", "index": 0, "vram_total": 17066033152, "vram_free": 1, "torch_vram_total": 1,
        "torch_vram_free": 1, "debug_path": r"C:\Users\someone\secret",
    }],
}


def test_stats_scrub_drops_argv_paths_and_user_names():
    dumped = json.dumps(capture.scrub_system_stats(SAMPLE_STATS))
    for leaked in ("someone", "SOMEONE", "argv", "Users", "J:", "hostname", "debug_path"):
        assert leaked not in dumped
    assert "0.42.0" in dumped and "RTX 5070 Ti" in dumped


def test_stats_scrub_tolerates_missing_sections():
    assert capture.scrub_system_stats({}) == {"system": {}, "devices": []}


def test_scrub_text_removes_windows_unc_and_unix_home_paths():
    text = r"see C:\Users\someone\x.txt and \\server\share\y and /home/someone/z and D:/models/a.safetensors"
    cleaned = capture.scrub_text(text)
    assert "someone" not in cleaned and "server" not in cleaned and "models" not in cleaned


def _node(shape: str, files: list[str]) -> dict:
    ckpt = [files, {}] if shape == "legacy" else ["COMBO", {"options": files}]
    return {
        "input": {"required": {"ckpt_name": ckpt, "sampler_name": [["euler", "dpmpp_2m"], {}]}},
        "input_order": {"required": ["ckpt_name", "sampler_name"]},
        "output": ["MODEL"], "output_name": ["MODEL"], "name": "CheckpointLoaderSimple",
        "category": "loaders", "description": r"loads from C:\Users\someone\models",
        "python_module": "custom_nodes.some-private-pack",
    }


def test_user_model_names_dropped_except_hand_chosen_both_shapes():
    files = ["private_finetune_v3.safetensors", "sd_xl_base_1.0.safetensors"]
    for shape in ("legacy", "combo"):
        cleaned = capture.scrub_node_info(_node(shape, files), keep={"sd_xl_base_1.0.safetensors"})
        dumped = json.dumps(cleaned)
        assert "private_finetune_v3" not in dumped
        assert "sd_xl_base_1.0.safetensors" in dumped
        assert "euler" in dumped, "fixed vocabularies such as samplers are kept whole"
        assert "someone" not in dumped and "some-private-pack" not in dumped


def test_build_fixture_set_only_includes_known_nodes():
    object_info = {"CheckpointLoaderSimple": _node("legacy", ["a.safetensors"]),
                   "SomeUsersCustomNode": {"input": {}, "name": "SomeUsersCustomNode"}}
    result = capture.build_fixture_set(SAMPLE_STATS, object_info, keep=set())
    assert set(result) == {"system_stats.json", "object_info/CheckpointLoaderSimple.json"}
