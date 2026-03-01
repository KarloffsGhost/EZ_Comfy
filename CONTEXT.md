# EZ_Comfy — Build Context for Implementation

## What This Is
EZ_Comfy is a standalone ComfyUI orchestrator that solves three problems:
1. **Which model?** — curated catalog of ~50 models with task-matching, download recommendations
2. **Which nodes?** — recipe library of proven workflow patterns, auto-selected by intent
3. **Which settings?** — model-specific + recipe-specific optimal params with resolution bucketing

Read `SPEC.md` for the full architecture. Build in phase order (1→10). Each phase should be fully tested before moving to the next.

## Critical Rules

### 1. ComfyUI API Patterns
- **Base URL:** `http://127.0.0.1:8188` (default ComfyUI port; configurable in settings.yaml)
- **Workflow format:** Pure Python dicts, NOT JSON files. Each key is a string node ID.
- **Node references:** `["node_id", output_index]` arrays.
- **Submit:** `POST /prompt` with `{"prompt": workflow_dict}` → returns `{"prompt_id": "..."}`
- **Poll:** `GET /history/{prompt_id}` — check `status.status_str == "success"`
- **WebSocket:** `ws://{base_url}/ws?clientId={uuid}` — real-time progress events
- **Download:** `GET /view?filename=X&subfolder=Y&type=output`
- **Upload image:** `POST /upload/image` (multipart form)
- **Free VRAM:** `POST /free` with `{"unload_models": true, "free_memory": true}`
- **Model lists:** `GET /object_info/CheckpointLoaderSimple` → `.CheckpointLoaderSimple.input.required.ckpt_name[0]`

### 2. VRAM Handoff (MUST implement)
Before any ComfyUI generation:
```python
# Unload Ollama models to free VRAM
GET {ollama_url}/api/ps → list loaded models
POST {ollama_url}/api/generate {"model": name, "keep_alive": 0} for each
```
After generation (in finally block):
```python
POST {comfyui_url}/free {"unload_models": true, "free_memory": true}
```
Use `vram_guard()` async context manager to wrap all generation calls.

### 3. Checkpoint Classification
Use filename patterns + file size to classify:
- `lightning`, `turbo`, `lcm`, `hyper` → fast variant (low steps, low CFG)
- `schnell` → Flux fast variant
- `xl`, `sdxl`, `realvis` + size > 3GB → SDXL family
- `flux` → Flux family
- `sd3` → SD3 family
- `v1-5`, `dreamshaper` + size < 3GB → SD 1.5 family
- `stable_audio`, `audio_open` → audio
- `svd`, `stable-video` → video
- `cascade` → Stable Cascade
- `pony` → Pony (SDXL-based but needs score tags)

### 4. Sampler Presets (from working MultiModel config)
| Family | Steps | CFG | Sampler | Scheduler |
|--------|-------|-----|---------|-----------|
| sdxl_lightning | 6 | 1.5 | euler | sgm_uniform |
| sdxl_turbo | 4 | 1.0 | euler | sgm_uniform |
| sd15_lightning | 8 | 1.5 | euler | sgm_uniform |
| sdxl (standard) | 25 | 7.0 | dpmpp_2m | karras |
| sd15 (standard) | 20 | 7.0 | euler | normal |
| flux_schnell | 4 | 1.0 | euler | simple |
| flux_dev | 20 | 1.0 | euler | simple |
| sd3 | 28 | 7.0 | dpmpp_2m | karras |
| cascade | 20 | 4.0 | euler | simple |
| svd | 25 | 2.5 | euler | normal |
| stable_audio | 100 | 7.0 | dpmpp_3m_sde | exponential |

### 5. Resolution Buckets (CRITICAL)
**Never generate at arbitrary resolutions.** Always snap to the nearest trained bucket.

SDXL buckets: `1024x1024, 1152x896, 896x1152, 1216x832, 832x1216, 1344x768, 768x1344, 1536x640, 640x1536`
SD 1.5 buckets: `512x512, 768x512, 512x768, 640x448, 448x640, 768x432, 432x768`
Flux buckets: Same as SDXL (trained on 1024x1024 base)

Implement `snap_to_bucket(w, h, buckets)` — find bucket with smallest area difference to (w*h) that matches closest aspect ratio.

### 6. Prompt Syntax by Model Family (CRITICAL)
| Family | Quality Prefix | Quality Suffix | Negative | Emphasis | Notes |
|--------|---------------|----------------|----------|----------|-------|
| SD 1.5 | — | `, masterpiece, best quality` (anime models) | Standard negatives | `(word:1.3)` | — |
| SDXL | — | — | Standard negatives | `(word:1.3)`, BREAK | — |
| Pony | `score_9, score_8_up, score_7_up, ` | — | `score_4, score_3, ...` | `(word:1.3)` | MUST have score tags |
| Flux | — | — | **IGNORED** (don't send) | **NONE** (strip weights) | Natural language only |
| SD3 | — | — | Standard negatives | `(word:1.3)` | — |

### 7. Hardware Notes
- EZ Comfy detects GPU and VRAM at runtime via `nvidia-smi`
- ComfyUI models path is set in `config/settings.yaml` → `comfyui.model_base_path`
- Ollama is assumed to run at `localhost:11434` (configurable); set `ollama.enabled: false` to disable the handoff
- The T5 encoder for Stable Audio (`t5_base_stable_audio.safetensors`) must be in ComfyUI's `clip/` folder

### 8. Project Conventions
- Python 3.11+, async throughout
- Pydantic v2 for all data models
- httpx for all HTTP (not requests)
- websockets library for WebSocket client
- pytest-asyncio with `asyncio_mode = "auto"`
- FastAPI with lifespan context manager
- Inline HTML/JS for UI (no framework, no build step) — same pattern as MultiModel
- All file paths use forward slashes in code
- Windows platform (win32)

### 9. Test Strategy
- Mock httpx for ComfyUI client tests
- Mock nvidia-smi subprocess for hardware probe tests
- Use FastAPI TestClient with mocked engine for API tests
- Unit test each workflow recipe builder — verify node graph structure is valid
- Unit test classifier with known checkpoint filenames
- Unit test prompt adapter — verify Pony gets score tags, Flux strips emphasis
- Unit test resolution bucket snapping
- Unit test param resolver priority chain
- Unit test model recommendation scoring

### 10. Package Setup
```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "ez-comfy"
version = "0.1.0"
requires-python = ">=3.11"
```

### 11. CLI Entry Point
`python -m ez_comfy` dispatches to:
- `check` — hardware + ComfyUI health + installed models
- `generate "prompt"` — CLI generation
- `plan "prompt"` — show plan without generating
- `recommend "prompt"` — show model recommendations
- `serve --port 8088` — start web server

Use `argparse` (not click/typer) to keep dependencies minimal.

### 12. Audio Workflow Gotcha
Stable Audio Open's `CheckpointLoaderSimple` returns `CLIP=None`. You MUST use a separate `CLIPLoader` node with `type="stable_audio"` loading `t5_base_stable_audio.safetensors`. This is a known issue — see MultiModel's fix.

### 13. Error Handling Patterns
- ComfyUI unreachable → clear error message with URL
- No compatible checkpoint found → list what IS installed, recommend what to download from catalog
- VRAM insufficient → suggest smaller model or lower resolution, show VRAM estimates
- Generation timeout → report timeout, suggest reducing steps/resolution
- Ollama unreachable → warn but continue (VRAM handoff is best-effort, non-fatal)
- WebSocket fails → fall back to HTTP polling silently
- Missing capabilities for recipe → fall back to simpler recipe that uses built-in nodes only, warn user

### 14. Four Architectural Contracts (MUST enforce)

**14a. Recommendation-only downloads:** The app NEVER downloads models. It shows download instructions (HuggingFace CLI commands, Civitai links) in the UI. The user downloads models themselves.

**14b. Capability-based node detection:** Recipes declare `required_capabilities` (e.g., `["adetailer"]`), NOT package names. These map to ComfyUI class_types via `NODE_CAPABILITY_MAP` in `comfyui_inventory.py`. The inventory stores `discovered_class_types: set[str]` from `GET /object_info`. Recipe selection calls `has_capability(cap, discovered_class_types)`.

**14c. Single-runner GPU lock:** `GenerationEngine._gpu_lock = asyncio.Lock()`. Every code path that submits to ComfyUI (`generate()`, `compare()`, queue `_process_loop()`) acquires this lock. No concurrent GPU jobs from any endpoint.

**14d. Inline WebSocket previews:** Latent preview images are sent as base64 data URIs in WebSocket messages (`preview_b64` field). No separate `/v1/preview/*` endpoint.

### 15. Model Catalog Maintenance (v1: ~15 entries)
The curated catalog (`models/catalog.py`) is a Python data structure, not a YAML file. This is intentional — it's code that ships with the package. To add a new model:
1. Add a `ModelCatalogEntry` to the `MODEL_CATALOG` list
2. Include: strengths, weaknesses, optimal settings, prompt syntax, VRAM requirements
3. Test with `test_catalog.py`

### 15. Output Sidecar Metadata
Every generation saves two files:
- `ezcomfy_00001.png` (or `.mp3`, `.webp`) — the output
- `ezcomfy_00001.json` — sidecar with full plan: prompt, model, all settings, seed, generation time

This enables:
- History reload (click thumbnail → restore all settings)
- Reproducibility (same seed + same settings = same output)
- Workflow export (sidecar contains the raw ComfyUI workflow dict)

### 16. WebSocket Progress Events
ComfyUI sends these event types via WebSocket:
```json
{"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}}
{"type": "execution_start", "data": {"prompt_id": "..."}}
{"type": "executing", "data": {"node": "5"}}
{"type": "progress", "data": {"value": 3, "max": 6, "prompt_id": "..."}}
{"type": "executed", "data": {"node": "7", "output": {"images": [...]}}}
{"type": "execution_error", "data": {"prompt_id": "...", "exception_message": "..."}}
```
Parse these to drive the UI progress bar. The `progress` type gives step-level updates during KSampler.
