# Contributing to EZ Comfy

Contributions are welcome. Here's how the project is structured and where different kinds of changes belong.

## Getting started

```bash
git clone https://github.com/yourusername/ez-comfy
cd ez-comfy
pip install -e ".[dev]"
pytest tests/unit/ -v   # should be 135 tests, all green, ~1 second, no ComfyUI needed
```

---

## Where things live

| You want to… | Edit this |
|---|---|
| Add a model to the catalog | `ez_comfy/models/catalog.py` |
| Change model family defaults (steps, CFG, sampler) | `ez_comfy/models/profiles.py` |
| Add a resolution bucket | `ez_comfy/models/profiles.py` → `RESOLUTION_BUCKETS` |
| Add a workflow recipe | `ez_comfy/workflows/` + register in `recipes.py` and `composer.py` |
| Change prompt adaptation rules | `ez_comfy/planner/prompt_adapter.py` |
| Change intent detection keywords | `ez_comfy/planner/intent.py` |
| Add a ComfyUI capability mapping | `ez_comfy/hardware/comfyui_inventory.py` → `NODE_CAPABILITY_MAP` |
| Change API models / request shapes | `ez_comfy/api/models.py` |
| Add a UI feature | `ez_comfy/api/routes.py` → `_UI_HTML` |
| Change CLI commands | `ez_comfy/__main__.py` |

---

## Adding a model to the catalog

Edit `ez_comfy/models/catalog.py` and add a `ModelCatalogEntry` to `MODEL_CATALOG`:

```python
ModelCatalogEntry(
    id="my-model-id",             # unique slug, kebab-case
    name="My Model Name v1",      # human-readable
    family="sdxl",                # sdxl | sd15 | flux | pony | svd | stable_audio | sd3
    variant=None,                 # lightning | turbo | schnell | None
    filename="mymodel_v1.safetensors",   # exact filename as it appears in ComfyUI
    alt_filenames=[],             # other filenames this model might be installed as
    vram_min_gb=6.0,
    size_bytes=6_600_000_000,
    tasks=["txt2img", "img2img"],
    strengths=["photorealism", "portraits"],
    weaknesses=["anime"],
    prompt_syntax=PromptSyntax.STANDARD,
    source="org/repo-on-huggingface",
    download_command="huggingface-cli download org/repo mymodel_v1.safetensors",
    recommended_vae=None,
    settings=None,               # ModelSettings override, or None to use profile defaults
)
```

Then add a test in `tests/unit/test_catalog.py` covering the new entry.

---

## Adding a workflow recipe

1. Write a builder function in `ez_comfy/workflows/` (e.g. `txt2img.py`):

```python
def build_my_recipe(plan: GenerationPlan) -> dict:
    """Returns a ComfyUI API-format node graph dict."""
    workflow = {}
    workflow["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": plan.checkpoint},
    }
    # ... add more nodes ...
    return workflow
```

2. Register it in `ez_comfy/workflows/recipes.py` — add a `Recipe` entry to `RECIPE_REGISTRY`.

3. Register the builder in `ez_comfy/workflows/composer.py` → `_BUILDERS` dict.

4. Add tests in `tests/unit/test_workflows.py` verifying the node graph structure.

**Workflow rules:**
- All workflows are Python dicts, not JSON files
- Node IDs are string integers: `"1"`, `"2"`, etc.
- Node references use `["node_id", output_index]` arrays
- Never hardcode a checkpoint name — always use `plan.checkpoint`
- Always use `plan.params.*` for sampler settings

---

## Adding a capability mapping

If a new recipe requires a custom node, add the capability to `NODE_CAPABILITY_MAP` in `ez_comfy/hardware/comfyui_inventory.py`:

```python
NODE_CAPABILITY_MAP: dict[str, list[str]] = {
    # capability_name: [list of class_type strings that provide it]
    "my_feature": ["MyCustomNode", "MyCustomNodeAlt"],
}
```

Then declare it in your recipe's `required_capabilities` list.

---

## Code conventions

- Python 3.11+ — use `X | Y` union types, `match` where appropriate
- Pydantic v2 for all data models
- `httpx` for HTTP (not `requests`)
- `websockets` for WebSocket client
- `pytest-asyncio` with `asyncio_mode = "auto"` — all async tests work automatically
- No external frontend frameworks — UI is inline HTML/JS in `routes.py`
- `argparse` for CLI (not click/typer)

---

## Tests

Unit tests live in `tests/unit/`. They require no running ComfyUI instance and no GPU.

```bash
pytest tests/unit/ -v              # run all
pytest tests/unit/test_catalog.py  # run one module
```

Integration tests (if you add any) belong in `tests/integration/` and should be skipped by default unless `EZCOMFY_INTEGRATION=1` is set.

---

## Pull requests

- Keep changes focused — one feature or fix per PR
- Add or update tests for any changed logic
- Run `pytest tests/unit/` before submitting — all 135 should pass
- Follow the existing code style (no type annotations on unchanged code, no docstrings on simple functions)
