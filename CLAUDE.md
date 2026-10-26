# EZ_Comfy — Project Instructions

## What This Project Is
EZ_Comfy is a standalone ComfyUI orchestrator that solves three problems:
1. **Which model?** — curated catalog of ~50 models with task-matching + download recommendations
2. **Which nodes?** — recipe library of proven workflow patterns, auto-selected by intent
3. **Which settings?** — model-specific optimal params, prompt syntax adaptation, resolution bucketing

## Build Instructions
- Read `SPEC.md` for full architecture and component specs
- Read `CONTEXT.md` for implementation patterns, gotchas, and hardware details
- Build in phase order (Phase 1 → 10) as defined in SPEC.md Section 6
- Each phase must be fully tested before starting the next

## Key Conventions
- Python 3.11+, fully async
- Pydantic v2 for all data models
- httpx for HTTP, websockets for WebSocket (not requests)
- pytest-asyncio with asyncio_mode="auto"
- FastAPI with lifespan
- Inline HTML/JS UI (no framework)
- argparse for CLI (not click/typer)
- All ComfyUI workflows are Python dicts, not JSON template files

## Project Structure
```
ez_comfy/           # main package
  api/              # FastAPI server + routes + UI
  hardware/         # GPU probe + ComfyUI inventory scanner
  models/           # curated model catalog + family profiles + classifier
  planner/          # intent detection + prompt adapter + param resolver + plan generation
  workflows/        # recipe registry + ComfyUI node graph composers
  comfyui/          # ComfyUI HTTP/WebSocket client + VRAM management
  config/           # settings schema
  engine.py         # top-level orchestrator + generation queue
  __main__.py       # CLI entry point
config/             # settings.yaml
tests/              # unit + integration tests
```

## Running
```bash
pip install -e ".[dev]"
python -m ez_comfy check              # health check
python -m ez_comfy serve --port 8088  # web UI at http://127.0.0.1:8088/v1/ui
python -m ez_comfy generate "prompt"  # CLI generation
python -m ez_comfy plan "prompt"      # preview plan
python -m ez_comfy recommend "prompt" # model recommendations
```

## ComfyUI Connection
- Default URL: http://127.0.0.1:8188 (standard ComfyUI port; override in config/settings.yaml)
- Models path: set in config/settings.yaml → comfyui.model_base_path
- VRAM handoff with Ollama required before/after generation

## Critical Behaviors
- ALWAYS snap resolution to trained buckets (never arbitrary sizes)
- ALWAYS adapt prompts for model syntax (Pony score tags, Flux no-emphasis)
- ALWAYS auto-generate negative prompts per model family (except Flux)
- ALWAYS check VRAM fit before selecting a model
- WebSocket for progress, HTTP polling as fallback
