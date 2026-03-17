from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ez_comfy.api.models import (
    CompareRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    InstallPlanResponse,
    InventoryResponse,
    JobStatusResponse,
    OutputFile,
    PlanRequest,
    PlanResponse,
    QueueRequest,
    RecommendationsResponse,
)
from ez_comfy.engine import GenerationEngine, GenerationQueue
from ez_comfy.planner.intent import detect_intent
from ez_comfy.planner.planner import GenerationRequest as _GenReq
from ez_comfy.workflows.recipes import list_recipes

logger = logging.getLogger(__name__)
router = APIRouter()

_CAPABILITY_GUIDANCE: dict[str, dict[str, str]] = {
    "upscale_model": {
        "title": "Install an upscaler model",
        "how_to": "Place RealESRGAN_x4plus.pth or 4x-UltraSharp.pth in ComfyUI/models/upscale_models/",
    },
    "controlnet": {
        "title": "Install ControlNet support",
        "how_to": "Install ControlNet nodes and place models under ComfyUI/models/controlnet/",
    },
    "adetailer": {
        "title": "Install ADetailer custom node",
        "how_to": "Install ADetailer via ComfyUI-Manager and restart ComfyUI.",
    },
    "ipadapter": {
        "title": "Install IPAdapter custom node",
        "how_to": "Install ComfyUI-IPAdapter via ComfyUI-Manager and add required encoder weights.",
    },
    "animatediff": {
        "title": "Install AnimateDiff custom node",
        "how_to": "Install ComfyUI-AnimateDiff via ComfyUI-Manager and required motion modules.",
    },
    "svd": {
        "title": "Install Stable Video Diffusion model",
        "how_to": "Download SVD checkpoint and place it in ComfyUI/models/checkpoints/",
    },
    "stable_audio": {
        "title": "Install Stable Audio model + T5 clip",
        "how_to": "Install stable_audio_open_1.0 checkpoint and t5_base_stable_audio.safetensors in clip/",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(request: Request) -> GenerationEngine:
    return request.app.state.engine


def _queue(request: Request) -> GenerationQueue:
    return request.app.state.queue


def _api_request_to_gen_request(r: GenerateRequest | QueueRequest | PlanRequest) -> _GenReq:
    return _GenReq(
        prompt=r.prompt,
        negative_prompt=r.negative_prompt,
        intent_override=r.intent_override,
        checkpoint_override=getattr(r, "checkpoint_override", None),
        recipe_override=r.recipe_override,
        style_preset=r.style_preset,
        width=r.width,
        height=r.height,
        aspect_ratio=r.aspect_ratio,
        steps=r.steps,
        cfg_scale=getattr(r, "cfg_scale", None),
        sampler=getattr(r, "sampler", None),
        scheduler=getattr(r, "scheduler", None),
        seed=r.seed,
        batch_size=r.batch_size,
        denoise_strength=r.denoise_strength,
        loras=getattr(r, "loras", None),
    )


def _build_generate_response(result, base_url: str) -> GenerateResponse:
    outputs = []
    for out in result.outputs:
        fn = out.get("filename", "")
        sf = out.get("subfolder", "")
        tp = out.get("type", "output")   # ComfyUI's folder type — needed for /view URL
        url = f"{base_url}/view?filename={fn}&subfolder={sf}&type={tp}"
        outputs.append(OutputFile(
            type=out.get("media_type", "image"),  # our semantic type — image/audio/video
            filename=fn,
            subfolder=sf,
            url=url,
        ))
    return GenerateResponse(
        prompt_id=result.prompt_id,
        recipe=result.plan.recipe.id,
        checkpoint=result.plan.checkpoint,
        family=result.plan.checkpoint_family,
        duration_seconds=round(result.duration_seconds, 2),
        outputs=outputs,
        warnings=result.plan.warnings,
        plan_summary=result.plan.summary(),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/v1/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    engine = _engine(request)
    queue = _queue(request)
    hw = request.app.state.hardware
    inv = request.app.state.inventory
    comfyui_ok = await engine._client.health_check()
    return HealthResponse(
        status="ok" if comfyui_ok else "degraded",
        comfyui=comfyui_ok,
        comfyui_url=request.app.state.settings.comfyui.base_url,
        gpu_name=hw.gpu_name,
        gpu_vram_gb=hw.gpu_vram_gb,
        checkpoints_loaded=len(inv.checkpoints),
        queue_depth=queue.queue_depth(),
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@router.post("/v1/plan", response_model=PlanResponse)
async def plan(request: Request, body: PlanRequest) -> PlanResponse:
    """Plan a generation without running it. Returns full parameter breakdown."""
    engine = _engine(request)
    gen_req = _api_request_to_gen_request(body)
    plan = await engine.plan_only(gen_req)
    return PlanResponse(
        intent=plan.intent.value,
        recipe=plan.recipe.id,
        checkpoint=plan.checkpoint,
        family=plan.checkpoint_family,
        profile=plan.profile.family,
        estimated_vram_gb=plan.estimated_vram_gb,
        estimated_time_seconds=plan.estimated_time_seconds,
        warnings=plan.warnings,
        missing_capabilities=plan.missing_capabilities,
        params={
            "width": plan.params.width,
            "height": plan.params.height,
            "steps": plan.params.steps,
            "cfg_scale": plan.params.cfg_scale,
            "sampler": plan.params.sampler,
            "scheduler": plan.params.scheduler,
            "seed": plan.params.seed,
            "batch_size": plan.params.batch_size,
            "denoise_strength": plan.params.denoise_strength,
            "clip_skip": plan.params.clip_skip,
        },
        param_sources=plan.param_sources,
        prompt=plan.prompt,
        negative_prompt=plan.negative_prompt,
        recommendations=[
            {
                "model": r.entry.name,
                "filename": r.entry.filename,
                "score": r.score,
                "installed": r.installed,
                "reason": "; ".join(r.match_reasons),
                "vram_min_gb": r.entry.vram_min_gb,
                "source": r.entry.source,
                "download_command": r.entry.download_command,
            }
            for r in plan.recommendations
        ],
    )


# ---------------------------------------------------------------------------
# Workflow export (download the ComfyUI node graph JSON)
# ---------------------------------------------------------------------------

@router.post("/v1/plan/workflow")
async def export_workflow(
    request: Request,
    body: PlanRequest,
    provenance: str = Query(
        "summary",
        description=(
            "Provenance inclusion level: "
            "'summary' injects a Note node with human-readable provenance (default), "
            "'full' adds Note node + machine-readable JSON key, "
            "'none' returns raw workflow without provenance."
        ),
    ),
) -> JSONResponse:
    """Plan and compose the ComfyUI workflow, returning it as a downloadable JSON file."""
    import json
    from fastapi.responses import Response as _Resp
    from ez_comfy.workflows.composer import compose_annotated_workflow, compose_workflow

    if provenance not in ("summary", "full", "none"):
        raise HTTPException(status_code=400, detail="provenance must be 'summary', 'full', or 'none'")

    engine = _engine(request)
    gen_req = _api_request_to_gen_request(body)
    plan = await engine.plan_only(gen_req)

    if provenance == "none":
        workflow = compose_workflow(plan)
    elif provenance == "full":
        workflow = compose_annotated_workflow(plan)
        workflow["_ez_comfy_provenance"] = plan.provenance.to_dict()
    else:  # "summary" (default)
        workflow = compose_annotated_workflow(plan)

    filename = f"{plan.recipe.id}_{plan.checkpoint_family}_workflow.json"
    return _Resp(
        content=json.dumps(workflow, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/v1/history/{prompt_id}/provenance")
async def get_provenance(request: Request, prompt_id: str) -> JSONResponse:
    """Return provenance record for a completed generation, read from sidecar JSON."""
    import json
    from pathlib import Path

    settings = request.app.state.settings
    meta_dir = Path(settings.comfyui.output_dir) / "ez_comfy_meta"
    sidecar_path = meta_dir / f"{prompt_id}.json"

    if not sidecar_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No provenance record found for prompt_id={prompt_id!r}. "
                   "Sidecar metadata is written after generation completes.",
        )

    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return JSONResponse(content=data.get("provenance", {}))


# ---------------------------------------------------------------------------
# Generate (JSON, no image upload)
# ---------------------------------------------------------------------------

@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    """Submit a generation request. Blocks until complete."""
    engine = _engine(request)
    gen_req = _api_request_to_gen_request(body)
    base_url = str(request.app.state.settings.comfyui.base_url)
    try:
        result = await engine.generate(gen_req, timeout=body.timeout)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    return _build_generate_response(result, base_url)


# ---------------------------------------------------------------------------
# Generate with file upload (multipart form)
# ---------------------------------------------------------------------------

@router.post("/v1/generate/form", response_model=GenerateResponse)
async def generate_form(
    request: Request,
    prompt: Annotated[str, Form()],
    negative_prompt: Annotated[str, Form()] = "",
    style_preset: Annotated[str | None, Form()] = None,
    width: Annotated[int | None, Form()] = None,
    height: Annotated[int | None, Form()] = None,
    aspect_ratio: Annotated[str | None, Form()] = None,
    steps: Annotated[int | None, Form()] = None,
    cfg_scale: Annotated[float | None, Form()] = None,
    sampler: Annotated[str | None, Form()] = None,
    scheduler: Annotated[str | None, Form()] = None,
    seed: Annotated[int, Form()] = -1,
    batch_size: Annotated[int, Form()] = 1,
    denoise_strength: Annotated[float, Form()] = 0.7,
    intent_override: Annotated[str | None, Form()] = None,
    checkpoint_override: Annotated[str | None, Form()] = None,
    recipe_override: Annotated[str | None, Form()] = None,
    timeout: Annotated[float, Form()] = 300.0,
    reference_image: Annotated[UploadFile | None, File()] = None,
    mask_image: Annotated[UploadFile | None, File()] = None,
) -> GenerateResponse:
    """Submit a generation with optional image uploads (img2img / inpaint)."""
    engine = _engine(request)
    base_url = str(request.app.state.settings.comfyui.base_url)

    ref_bytes = await reference_image.read() if reference_image else None
    mask_bytes = await mask_image.read() if mask_image else None

    gen_req = _GenReq(
        prompt=prompt,
        negative_prompt=negative_prompt,
        reference_image=ref_bytes,
        mask_image=mask_bytes,
        intent_override=intent_override,
        checkpoint_override=checkpoint_override,
        recipe_override=recipe_override,
        style_preset=style_preset,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        scheduler=scheduler,
        seed=seed,
        batch_size=batch_size,
        denoise_strength=denoise_strength,
    )

    try:
        result = await engine.generate(gen_req, timeout=timeout)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    return _build_generate_response(result, base_url)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

@router.post("/v1/compare")
async def compare(request: Request, body: CompareRequest) -> list[GenerateResponse]:
    """Run multiple generations sequentially for comparison."""
    engine = _engine(request)
    base_url = str(request.app.state.settings.comfyui.base_url)
    gen_reqs = [_api_request_to_gen_request(r) for r in body.requests]
    try:
        results = await engine.compare(gen_reqs, timeout=body.timeout)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return [_build_generate_response(r, base_url) for r in results]


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

@router.post("/v1/queue")
async def queue_job(request: Request, body: QueueRequest) -> dict:
    """Enqueue a generation job. Returns job_id immediately."""
    q = _queue(request)
    gen_req = _api_request_to_gen_request(body)
    job_id = q.enqueue(gen_req)
    return {"job_id": job_id, "status": "queued", "queue_depth": q.queue_depth()}


@router.post("/v1/queue/form")
async def queue_form(
    request: Request,
    prompt: Annotated[str, Form()],
    negative_prompt: Annotated[str, Form()] = "",
    style_preset: Annotated[str | None, Form()] = None,
    width: Annotated[int | None, Form()] = None,
    height: Annotated[int | None, Form()] = None,
    aspect_ratio: Annotated[str | None, Form()] = None,
    steps: Annotated[int | None, Form()] = None,
    cfg_scale: Annotated[float | None, Form()] = None,
    sampler: Annotated[str | None, Form()] = None,
    scheduler: Annotated[str | None, Form()] = None,
    seed: Annotated[int, Form()] = -1,
    batch_size: Annotated[int, Form()] = 1,
    denoise_strength: Annotated[float, Form()] = 0.7,
    intent_override: Annotated[str | None, Form()] = None,
    checkpoint_override: Annotated[str | None, Form()] = None,
    recipe_override: Annotated[str | None, Form()] = None,
    reference_image: Annotated[UploadFile | None, File()] = None,
    mask_image: Annotated[UploadFile | None, File()] = None,
) -> dict:
    """Enqueue a generation job with optional image uploads."""
    q = _queue(request)
    ref_bytes = await reference_image.read() if reference_image else None
    mask_bytes = await mask_image.read() if mask_image else None

    gen_req = _GenReq(
        prompt=prompt,
        negative_prompt=negative_prompt,
        reference_image=ref_bytes,
        mask_image=mask_bytes,
        intent_override=intent_override,
        checkpoint_override=checkpoint_override,
        recipe_override=recipe_override,
        style_preset=style_preset,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        scheduler=scheduler,
        seed=seed,
        batch_size=batch_size,
        denoise_strength=denoise_strength,
    )
    job_id = q.enqueue(gen_req)
    return {"job_id": job_id, "status": "queued", "queue_depth": q.queue_depth()}


@router.get("/v1/queue", response_model=list[JobStatusResponse])
async def list_queue(request: Request) -> list[JobStatusResponse]:
    q = _queue(request)
    base_url = str(request.app.state.settings.comfyui.base_url)
    responses = []
    for job in q.list_jobs():
        result_resp = None
        if job.result:
            result_resp = _build_generate_response(job.result, base_url)
        responses.append(JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            result=result_resp,
            error=job.error,
            queued_at=job.queued_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        ))
    return responses


@router.get("/v1/queue/{job_id}", response_model=JobStatusResponse)
async def job_status(request: Request, job_id: str) -> JobStatusResponse:
    q = _queue(request)
    job = q.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    base_url = str(request.app.state.settings.comfyui.base_url)
    result_resp = None
    if job.result:
        result_resp = _build_generate_response(job.result, base_url)
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=result_resp,
        error=job.error,
        queued_at=job.queued_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.delete("/v1/queue/{job_id}")
async def cancel_job(request: Request, job_id: str) -> dict:
    q = _queue(request)
    cancelled = q.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found or not cancellable")
    return {"job_id": job_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@router.get("/v1/inventory", response_model=InventoryResponse)
async def inventory(request: Request) -> InventoryResponse:
    inv = request.app.state.inventory
    return InventoryResponse(
        checkpoints=[{"filename": c.filename, "family": c.family, "size_gb": round(c.size_bytes / 1e9, 2)} for c in inv.checkpoints],
        loras=[{"filename": l.filename, "families": l.compatible_families, "ambiguous": l.ambiguous} for l in inv.loras],
        vaes=inv.vaes,
        upscale_models=inv.upscale_models,
        samplers=inv.samplers,
        schedulers=inv.schedulers,
        discovered_capabilities=sorted(inv.discovered_class_types),
    )


@router.post("/v1/inventory/refresh", response_model=InventoryResponse)
async def refresh_inventory(request: Request) -> InventoryResponse:
    engine = _engine(request)
    inv = await engine.refresh_inventory()
    request.app.state.inventory = inv
    return InventoryResponse(
        checkpoints=[{"filename": c.filename, "family": c.family, "size_gb": round(c.size_bytes / 1e9, 2)} for c in inv.checkpoints],
        loras=[{"filename": l.filename, "families": l.compatible_families, "ambiguous": l.ambiguous} for l in inv.loras],
        vaes=inv.vaes,
        upscale_models=inv.upscale_models,
        samplers=inv.samplers,
        schedulers=inv.schedulers,
        discovered_capabilities=sorted(inv.discovered_class_types),
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@router.get("/v1/recommendations", response_model=RecommendationsResponse)
async def recommendations(
    request: Request,
    prompt: str,
    intent: str | None = None,
    top_n: int = 8,
) -> RecommendationsResponse:
    engine = _engine(request)
    recs = engine.get_recommendations(prompt, intent, top_n=top_n)
    return RecommendationsResponse(
        prompt=prompt,
        intent=intent or "auto",
        recommendations=[
            {
                "model": r.entry.name,
                "filename": r.entry.filename,
                "score": r.score,
                "installed": r.installed,
                "reason": "; ".join(r.match_reasons),
                "vram_min_gb": r.entry.vram_min_gb,
                "source": r.entry.source,
                "download_command": r.entry.download_command,
            }
            for r in recs
        ],
    )


# ---------------------------------------------------------------------------
# Install Assistant
# ---------------------------------------------------------------------------

@router.get("/v1/install/plan", response_model=InstallPlanResponse)
async def install_plan(
    request: Request,
    prompt: str,
    intent: str | None = None,
) -> InstallPlanResponse:
    """
    Suggest what to install (models/upscalers/capabilities) for a given prompt.
    Recommendation-only: does not install anything automatically.
    """
    engine = _engine(request)
    inventory = request.app.state.inventory

    resolved_intent = intent or detect_intent(prompt, False, False).value
    recs = engine.get_recommendations(prompt, resolved_intent, top_n=12)

    recommended_installed: list[dict] = []
    recommended_to_install: list[dict] = []
    for r in recs[:5]:
        row = {
            "model": r.entry.name,
            "filename": r.entry.filename,
            "family": r.entry.family,
            "installed": r.installed,
            "score": r.score,
            "reason": "; ".join(r.match_reasons),
            "source": r.entry.source,
            "download_command": r.entry.download_command,
            "vram_min_gb": r.entry.vram_min_gb,
        }
        if r.installed:
            recommended_installed.append(row)
        else:
            recommended_to_install.append(row)

    # If user asks for upscale and none are installed, explicitly suggest top upscalers.
    if resolved_intent == "upscale" and not inventory.upscale_models:
        upscale_recs = engine.get_recommendations(prompt, "upscale", top_n=6)
        seen = {x["filename"] for x in recommended_to_install}
        for r in upscale_recs[:3]:
            if r.entry.filename in seen:
                continue
            recommended_to_install.append({
                "model": r.entry.name,
                "filename": r.entry.filename,
                "family": r.entry.family,
                "installed": r.installed,
                "score": r.score,
                "reason": "; ".join(r.match_reasons) or "recommended upscaler",
                "source": r.entry.source,
                "download_command": r.entry.download_command,
                "vram_min_gb": r.entry.vram_min_gb,
            })
            seen.add(r.entry.filename)

    missing_capabilities: list[str] = []
    notes: list[str] = []
    try:
        plan = await engine.plan_only(_GenReq(prompt=prompt, intent_override=resolved_intent))
        missing_capabilities = plan.missing_capabilities
        notes.extend(plan.warnings[:3])
    except RuntimeError as exc:
        notes.append(str(exc))
        if resolved_intent == "upscale":
            missing_capabilities = ["upscale_model"]
        elif resolved_intent == "video":
            missing_capabilities = ["svd"]
        elif resolved_intent == "audio":
            missing_capabilities = ["stable_audio"]

    capability_guidance = []
    for cap in missing_capabilities:
        meta = _CAPABILITY_GUIDANCE.get(cap, {"title": cap, "how_to": "Install required ComfyUI nodes/models."})
        capability_guidance.append({
            "capability": cap,
            "title": meta["title"],
            "how_to": meta["how_to"],
        })

    notes.append("EZ Comfy is recommendation-only for installs: review commands and install manually.")

    return InstallPlanResponse(
        prompt=prompt,
        intent=resolved_intent,
        recommended_installed=recommended_installed,
        recommended_to_install=recommended_to_install,
        missing_capabilities=missing_capabilities,
        capability_guidance=capability_guidance,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

@router.get("/v1/recipes")
async def recipes() -> list[dict]:
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "intent": r.intent.value,
            "priority": r.priority,
            "when": r.when,
            "requires_reference_image": r.requires_reference_image,
            "requires_mask": r.requires_mask,
            "required_capabilities": r.required_capabilities,
        }
        for r in list_recipes()
    ]


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_UI_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EZ Comfy</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }
  h1 { font-size: 1.4rem; font-weight: 700; color: #fff; }
  h2 { font-size: 1rem; font-weight: 600; color: #ccc; margin-bottom: .5rem; }
  header { padding: .75rem 1.5rem; background: #1a1a2e; display: flex; align-items: center; gap: 1rem; border-bottom: 1px solid #333; }
  #status-dot { width: 10px; height: 10px; border-radius: 50%; background: #555; flex-shrink: 0; }
  #status-dot.ok { background: #4caf50; }
  #status-dot.degraded { background: #f44336; }
  #status-text { font-size: .8rem; color: #888; }
  main { display: flex; gap: 0; height: calc(100vh - 48px); }
  /* Left panel */
  #left { width: 380px; flex-shrink: 0; overflow-y: auto; padding: 1rem; border-right: 1px solid #333; display: flex; flex-direction: column; gap: 1rem; }
  label { display: block; font-size: .75rem; color: #aaa; margin-bottom: .25rem; }
  textarea, input[type=text], input[type=number], select {
    width: 100%; background: #1e1e1e; color: #e0e0e0; border: 1px solid #444; border-radius: 6px;
    padding: .4rem .6rem; font-size: .85rem; outline: none;
  }
  textarea:focus, input:focus, select:focus { border-color: #6c63ff; }
  textarea { resize: vertical; min-height: 80px; }
  .row { display: flex; gap: .5rem; }
  .row > * { flex: 1; }
  button {
    padding: .5rem 1rem; border: none; border-radius: 6px; cursor: pointer;
    font-size: .85rem; font-weight: 600; transition: background .15s;
  }
  #btn-generate { background: #6c63ff; color: #fff; }
  #btn-generate:hover { background: #5a52d5; }
  #btn-generate:disabled { background: #444; cursor: not-allowed; }
  #btn-plan { background: #333; color: #ccc; }
  #btn-plan:hover { background: #444; }
  #btn-install { background: #2f3b52; color: #d8e3ff; }
  #btn-install:hover { background: #3a4966; }
  #btn-clear { background: #2a2a2a; color: #888; }
  #btn-download-wf { background: #1a3a2a; color: #6ddc9a; }
  #btn-download-wf:hover { background: #1f4a34; }
  #btn-open-comfyui { background: #1a2a3a; color: #6ab4dc; }
  #btn-open-comfyui:hover { background: #1f3a4a; }
  .wf-toast { display:none; background:#1a2a1a; border:1px solid #3a5a3a; border-radius:6px; padding:.5rem .75rem; font-size:.75rem; color:#8dc; margin-top:.5rem; }
  .section { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: .75rem; }
  .collapsible-header { cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
  .collapsible-header::after { content: "▾"; font-size: .75rem; color: #666; }
  .collapsible-body { margin-top: .5rem; }
  input[type=file] { font-size: .8rem; color: #aaa; }
  /* Right panel */
  #right { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
  /* Progress */
  #progress-bar-wrap { background: #222; border-radius: 4px; height: 6px; overflow: hidden; }
  #progress-bar { height: 100%; background: #6c63ff; width: 0%; transition: width .3s; }
  #progress-text { font-size: .75rem; color: #777; margin-top: .25rem; }
  /* Outputs */
  #outputs { display: flex; flex-wrap: wrap; gap: .75rem; }
  .output-card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; overflow: hidden; max-width: 512px; }
  .output-card img { width: 100%; display: block; }
  .output-card audio { width: 100%; }
  .output-meta { padding: .4rem .6rem; font-size: .7rem; color: #666; }
  /* Plan / warnings */
  #plan-section { display: none; background: #111; border: 1px solid #2a2a2a; border-radius: 8px; padding: .75rem; }
  #plan-section pre { font-size: .72rem; color: #8a8; overflow-x: auto; white-space: pre-wrap; }
  #warnings { display: none; background: #1a1200; border: 1px solid #443300; border-radius: 8px; padding: .6rem .75rem; }
  #warnings ul { padding-left: 1rem; font-size: .8rem; color: #ffcc44; }
  /* Recommendations */
  #recs-list { display: flex; flex-direction: column; gap: .5rem; }
  .rec-card { background: #161616; border: 1px solid #2a2a2a; border-radius: 6px; padding: .5rem .75rem; font-size: .78rem; }
  .rec-card .rec-name { font-weight: 600; color: #ccc; }
  .rec-card .rec-detail { color: #666; font-size: .7rem; margin-top: .15rem; }
  .rec-card.installed .rec-name::after { content: " \u2713"; color: #4caf50; }
  .rec-card.not-installed .rec-name::after { content: " (not installed)"; color: #888; font-weight: 400; font-size: .7rem; }
  /* Provenance panel */
  #provenance-section { display: none; margin-top: .75rem; }
  #provenance-section details { background: #0d1a0d; border: 1px solid #1a3a1a; border-radius: 8px; padding: .6rem .75rem; }
  #provenance-section summary { cursor: pointer; font-size: .82rem; color: #5c9; font-weight: 600; user-select: none; }
  #provenance-section summary::marker { color: #5c9; }
  .prov-table { width: 100%; border-collapse: collapse; margin-top: .5rem; font-size: .74rem; }
  .prov-table td { padding: .2rem .4rem; vertical-align: top; }
  .prov-table tr:nth-child(even) td { background: #111f11; }
  .prov-param { color: #8c8; font-weight: 600; white-space: nowrap; }
  .prov-value { color: #ddd; }
  .prov-source { color: #555; font-size: .68rem; white-space: nowrap; }
  .prov-reason { color: #666; font-size: .68rem; }
  .prov-rejected { margin-top: .4rem; }
  .prov-rejected summary { font-size: .72rem; color: #666; cursor: pointer; }
  .prov-rejected ul { margin: .3rem 0 0 1rem; padding: 0; font-size: .68rem; color: #664; }
  .how-to-get { display: inline-block; margin-top: .3rem; font-size: .7rem; color: #6c63ff; cursor: pointer; text-decoration: underline; }
  #err-msg { display: none; background: #200; border: 1px solid #500; border-radius: 6px; padding: .5rem .75rem; color: #f66; font-size: .8rem; }
</style>
</head>
<body>

<header>
  <h1>⚡ EZ Comfy</h1>
  <div id="status-dot"></div>
  <span id="status-text">checking…</span>
</header>

<main>
<div id="left">

  <!-- Prompt -->
  <div class="section">
    <h2>Prompt</h2>
    <label for="prompt">Positive prompt</label>
    <textarea id="prompt" rows="4" placeholder="A photo of…"></textarea>
    <label for="neg-prompt" style="margin-top:.5rem">Negative prompt (optional)</label>
    <textarea id="neg-prompt" rows="2" placeholder="blurry, low quality…"></textarea>
  </div>

  <!-- Style -->
  <div class="section">
    <h2>Style &amp; Mode</h2>
    <div class="row">
      <div>
        <label for="style">Style preset</label>
        <select id="style">
          <option value="">None</option>
          <option value="photographic">Photographic</option>
          <option value="cinematic">Cinematic</option>
          <option value="anime">Anime</option>
          <option value="digital_art">Digital Art</option>
          <option value="fantasy">Fantasy</option>
          <option value="pixel_art">Pixel Art</option>
          <option value="watercolor">Watercolor</option>
          <option value="oil_painting">Oil Painting</option>
          <option value="3d_render">3D Render</option>
          <option value="comic">Comic</option>
          <option value="minimalist">Minimalist</option>
          <option value="noir">Noir</option>
          <option value="cyberpunk">Cyberpunk</option>
          <option value="portrait">Portrait</option>
          <option value="landscape">Landscape</option>
          <option value="product">Product</option>
        </select>
      </div>
      <div>
        <label for="intent">Intent</label>
        <select id="intent">
          <option value="">Auto-detect</option>
          <option value="txt2img">Text to Image</option>
          <option value="img2img">Image to Image</option>
          <option value="inpaint">Inpaint</option>
          <option value="upscale">Upscale</option>
          <option value="video">Video (SVD)</option>
          <option value="audio">Audio</option>
        </select>
      </div>
    </div>
  </div>

  <!-- Resolution -->
  <div class="section">
    <div class="collapsible-header" onclick="toggle('res-body')"><h2>Resolution</h2></div>
    <div class="collapsible-body" id="res-body">
      <div class="row">
        <div><label for="width">Width</label><input type="number" id="width" placeholder="auto" min="64" max="2048" step="64"></div>
        <div><label for="height">Height</label><input type="number" id="height" placeholder="auto" min="64" max="2048" step="64"></div>
      </div>
      <label style="margin-top:.4rem" for="aspect">Or aspect ratio</label>
      <select id="aspect">
        <option value="">Use width/height above</option>
        <option value="1:1">1:1 Square</option>
        <option value="16:9">16:9 Landscape</option>
        <option value="9:16">9:16 Portrait</option>
        <option value="4:3">4:3</option>
        <option value="3:4">3:4</option>
        <option value="21:9">21:9 Panoramic</option>
      </select>
    </div>
  </div>

  <!-- Sampler -->
  <div class="section">
    <div class="collapsible-header" onclick="toggle('samp-body')"><h2>Sampler Settings</h2></div>
    <div class="collapsible-body" id="samp-body">
      <div class="row">
        <div><label for="steps">Steps</label><input type="number" id="steps" placeholder="auto" min="1" max="150"></div>
        <div><label for="cfg">CFG Scale</label><input type="number" id="cfg" placeholder="auto" min="1" max="30" step="0.5"></div>
      </div>
      <div class="row" style="margin-top:.4rem">
        <div><label for="seed">Seed (-1 = random)</label><input type="number" id="seed" value="-1"></div>
        <div><label for="denoise">Denoise</label><input type="number" id="denoise" value="0.7" min="0" max="1" step="0.05"></div>
      </div>
    </div>
  </div>

  <!-- Reference image -->
  <div class="section">
    <div class="collapsible-header" onclick="toggle('img-body')"><h2>Reference Image</h2></div>
    <div class="collapsible-body" id="img-body">
      <label for="ref-img">Reference image (img2img / inpaint / video)</label>
      <input type="file" id="ref-img" accept="image/*">
      <label for="mask-img" style="margin-top:.4rem">Mask image (inpaint — white = replace)</label>
      <input type="file" id="mask-img" accept="image/*">
    </div>
  </div>

  <!-- Model -->
  <div class="section">
    <div class="collapsible-header" onclick="toggle('model-body')"><h2>Model Override</h2></div>
    <div class="collapsible-body" id="model-body">
      <label for="checkpoint">Checkpoint (leave blank = auto)</label>
      <input type="text" id="checkpoint" placeholder="realvisxlV50_v50Lightning.safetensors">
      <label for="recipe" style="margin-top:.4rem">Recipe override</label>
      <select id="recipe">
        <option value="">Auto-select</option>
      </select>
    </div>
  </div>

  <!-- Buttons -->
  <div class="row">
    <button id="btn-generate" onclick="doGenerate()">⚡ Generate</button>
    <button id="btn-plan" onclick="doPlan()">📋 Plan Only</button>
    <button id="btn-install" onclick="doInstallPlan()">🧰 Install Plan</button>
    <button id="btn-clear" onclick="doClear()">✕ Clear</button>
  </div>
  <div class="row" style="margin-top:.35rem">
    <button id="btn-download-wf" onclick="doDownloadWorkflow()">⬇ Download Workflow</button>
    <button id="btn-open-comfyui" onclick="doOpenInComfyUI()">🔗 Open in ComfyUI</button>
  </div>
  <div class="wf-toast" id="wf-toast">Workflow downloaded — drag the <strong>.json</strong> file into the ComfyUI canvas to load it.</div>

</div><!-- /left -->

<div id="right">

  <div id="err-msg"></div>

  <!-- Progress -->
  <div id="progress-section" style="display:none">
    <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
    <div id="progress-text">Initializing…</div>
  </div>

  <!-- Warnings -->
  <div id="warnings"><ul id="warnings-list"></ul></div>

  <!-- Plan output -->
  <div id="plan-section"><h2>Generation Plan</h2><pre id="plan-pre"></pre></div>

  <!-- Provenance panel -->
  <div id="provenance-section">
    <details>
      <summary>What EZ Comfy decided for you</summary>
      <table class="prov-table" id="prov-table"></table>
      <details class="prov-rejected" id="prov-rejected-wrap" style="display:none">
        <summary>Alternatives considered</summary>
        <ul id="prov-rejected-list"></ul>
      </details>
    </details>
  </div>

  <!-- Install assistant output -->
  <div id="install-section" style="display:none" class="section">
    <h2>Install Assistant</h2>
    <pre id="install-pre" style="font-size:.72rem; color:#d6d6d6; white-space:pre-wrap;"></pre>
  </div>

  <!-- Outputs -->
  <div id="outputs"></div>

  <!-- Recommendations -->
  <div id="recs-section" style="display:none" class="section">
    <h2>Model Recommendations</h2>
    <div id="recs-list"></div>
  </div>

</div><!-- /right -->
</main>

<script>
// ---- State ----
let _polling = null;
let _comfyuiUrl = 'http://127.0.0.1:8188';

// ---- Startup ----
window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadRecipes();
  setInterval(checkHealth, 30000);
});

async function checkHealth() {
  try {
    const r = await fetch('/v1/health');
    const d = await r.json();
    const dot = document.getElementById('status-dot');
    dot.className = d.status;
    if (d.comfyui_url) _comfyuiUrl = d.comfyui_url;
    document.getElementById('status-text').textContent =
      `${d.gpu_name} · ${d.gpu_vram_gb}GB VRAM · ${d.checkpoints_loaded} checkpoints · ComfyUI ${d.comfyui ? 'online' : 'OFFLINE'}`;
  } catch (e) {
    document.getElementById('status-text').textContent = 'Cannot reach server';
  }
}

async function loadRecipes() {
  try {
    const r = await fetch('/v1/recipes');
    const recipes = await r.json();
    const sel = document.getElementById('recipe');
    recipes.forEach(rec => {
      const opt = document.createElement('option');
      opt.value = rec.id;
      opt.textContent = `${rec.name} — ${rec.description}`;
      sel.appendChild(opt);
    });
  } catch (e) { /* non-fatal */ }
}

// ---- Helpers ----
function toggle(id) {
  const el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? '' : 'none';
}

function buildFormData() {
  const fd = new FormData();
  const add = (k, v) => { if (v !== null && v !== undefined && v !== '') fd.append(k, v); };
  add('prompt', document.getElementById('prompt').value.trim());
  add('negative_prompt', document.getElementById('neg-prompt').value.trim());
  add('style_preset', document.getElementById('style').value);
  add('intent_override', document.getElementById('intent').value);
  const w = document.getElementById('width').value;
  const h = document.getElementById('height').value;
  if (w) add('width', w);
  if (h) add('height', h);
  add('aspect_ratio', document.getElementById('aspect').value);
  add('steps', document.getElementById('steps').value);
  add('cfg_scale', document.getElementById('cfg').value);
  add('seed', document.getElementById('seed').value);
  add('denoise_strength', document.getElementById('denoise').value);
  add('checkpoint_override', document.getElementById('checkpoint').value);
  add('recipe_override', document.getElementById('recipe').value);
  const refImg = document.getElementById('ref-img').files[0];
  if (refImg) fd.append('reference_image', refImg);
  const maskImg = document.getElementById('mask-img').files[0];
  if (maskImg) fd.append('mask_image', maskImg);
  return fd;
}

function setGenerating(on) {
  document.getElementById('btn-generate').disabled = on;
  document.getElementById('btn-generate').textContent = on ? '⏳ Generating…' : '⚡ Generate';
  document.getElementById('progress-section').style.display = on ? 'block' : 'none';
  if (on) {
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = 'Submitting…';
  }
}

function showError(msg) {
  const el = document.getElementById('err-msg');
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

function showWarnings(warnings) {
  const wrap = document.getElementById('warnings');
  const list = document.getElementById('warnings-list');
  list.innerHTML = '';
  if (warnings && warnings.length) {
    warnings.forEach(w => {
      const li = document.createElement('li');
      li.textContent = w;
      list.appendChild(li);
    });
    wrap.style.display = 'block';
  } else {
    wrap.style.display = 'none';
  }
}

function renderOutput(result, baseUrl) {
  const container = document.getElementById('outputs');
  container.innerHTML = '';
  if (!result.outputs || !result.outputs.length) {
    container.innerHTML = '<p style="color:#666">No outputs returned.</p>';
    return;
  }
  result.outputs.forEach(out => {
    const card = document.createElement('div');
    card.className = 'output-card';
    if (out.type === 'image') {
      const img = document.createElement('img');
      img.src = out.url;
      img.alt = out.filename;
      card.appendChild(img);
    } else if (out.type === 'audio') {
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.src = out.url;
      card.appendChild(audio);
    } else {
      const link = document.createElement('a');
      link.href = out.url;
      link.textContent = out.filename;
      link.style.padding = '.5rem';
      link.style.display = 'block';
      card.appendChild(link);
    }
    const meta = document.createElement('div');
    meta.className = 'output-meta';
    meta.textContent = `${result.recipe} · ${result.checkpoint} · ${result.duration_seconds}s · seed ${result.plan_summary.seed}`;
    card.appendChild(meta);
    container.appendChild(card);
  });
}

function renderProvenance(provenance) {
  const sec = document.getElementById('provenance-section');
  if (!provenance || !provenance.decisions || !provenance.decisions.length) {
    sec.style.display = 'none'; return;
  }
  sec.style.display = 'block';
  const table = document.getElementById('prov-table');
  table.innerHTML = '';
  const allAlternatives = [];
  provenance.decisions.forEach(d => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="prov-param">${d.parameter}</td>
      <td class="prov-value">${d.chosen_value}</td>
      <td class="prov-source">[${d.source}]</td>
      <td class="prov-reason">${d.reason}</td>`;
    table.appendChild(tr);
    (d.alternatives || []).forEach(a => allAlternatives.push(a));
  });
  const rejWrap = document.getElementById('prov-rejected-wrap');
  const rejList = document.getElementById('prov-rejected-list');
  if (allAlternatives.length) {
    rejWrap.style.display = '';
    rejList.innerHTML = '';
    allAlternatives.forEach(a => {
      const li = document.createElement('li');
      li.textContent = `${a.value} \u2014 ${a.rejected_reason}`;
      rejList.appendChild(li);
    });
  } else {
    rejWrap.style.display = 'none';
  }
}

function renderRecs(recs) {
  const sec = document.getElementById('recs-section');
  const list = document.getElementById('recs-list');
  if (!recs || !recs.length) { sec.style.display = 'none'; return; }
  sec.style.display = 'block';
  list.innerHTML = '';
  recs.forEach(r => {
    const card = document.createElement('div');
    card.className = 'rec-card ' + (r.installed ? 'installed' : 'not-installed');
    card.innerHTML = `<div class="rec-name">${r.model}</div>
      <div class="rec-detail">${r.reason} · Score: ${r.score} · ${r.vram_min_gb}GB VRAM min</div>`;
    if (!r.installed && r.source) {
      const link = document.createElement('span');
      link.className = 'how-to-get';
      link.textContent = 'How to get this model ↗';
      const url = r.source.startsWith('http') ? r.source : 'https://huggingface.co/' + r.source;
      link.onclick = () => window.open(url, '_blank');
      card.appendChild(link);
    }
    list.appendChild(card);
  });
}

// ---- Actions ----
async function doGenerate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) { showError('Please enter a prompt.'); return; }
  showError('');
  setGenerating(true);
  document.getElementById('outputs').innerHTML = '';
  document.getElementById('plan-section').style.display = 'none';
  document.getElementById('provenance-section').style.display = 'none';
  showWarnings([]);

  try {
    const fd = buildFormData();
    const r = await fetch('/v1/queue/form', { method: 'POST', body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const { job_id } = await r.json();
    document.getElementById('progress-text').textContent = 'Queued — waiting for GPU…';
    pollJob(job_id);
  } catch (e) {
    showError(e.message);
    setGenerating(false);
  }
}

function pollJob(jobId) {
  if (_polling) clearInterval(_polling);
  let dots = 0;
  _polling = setInterval(async () => {
    try {
      const r = await fetch(`/v1/queue/${jobId}`);
      const job = await r.json();

      if (job.status === 'running') {
        dots = (dots + 1) % 4;
        document.getElementById('progress-text').textContent = 'Generating' + '.'.repeat(dots + 1);
        document.getElementById('progress-bar').style.width = '40%';
      } else if (job.status === 'done') {
        clearInterval(_polling);
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-text').textContent = `Done in ${job.result.duration_seconds}s`;
        setGenerating(false);
        const comfyBase = await getComfyBase();
        renderOutput(job.result, comfyBase);
        showWarnings(job.result.warnings);
        renderRecs(null);
        renderProvenance(job.result.plan_summary && job.result.plan_summary.provenance);
      } else if (job.status === 'error') {
        clearInterval(_polling);
        showError('Generation failed: ' + (job.error || 'Unknown error'));
        setGenerating(false);
      } else if (job.status === 'cancelled') {
        clearInterval(_polling);
        showError('Job was cancelled.');
        setGenerating(false);
      }
    } catch (e) {
      clearInterval(_polling);
      showError('Polling error: ' + e.message);
      setGenerating(false);
    }
  }, 1500);
}

async function getComfyBase() {
  try {
    const r = await fetch('/v1/health');
    const d = await r.json();
    // We can't get the ComfyUI URL from health directly, so use the output URLs as-is
    return '';
  } catch { return ''; }
}

function buildPlanBody() {
  return {
    prompt: document.getElementById('prompt').value.trim(),
    negative_prompt: document.getElementById('neg-prompt').value.trim(),
    style_preset: document.getElementById('style').value || null,
    intent_override: document.getElementById('intent').value || null,
    width: parseInt(document.getElementById('width').value) || null,
    height: parseInt(document.getElementById('height').value) || null,
    aspect_ratio: document.getElementById('aspect').value || null,
    steps: parseInt(document.getElementById('steps').value) || null,
    cfg_scale: parseFloat(document.getElementById('cfg').value) || null,
    seed: parseInt(document.getElementById('seed').value),
    denoise_strength: parseFloat(document.getElementById('denoise').value),
    checkpoint_override: document.getElementById('checkpoint').value || null,
    recipe_override: document.getElementById('recipe').value || null,
  };
}

async function doDownloadWorkflow(openComfyUI = false) {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) { showError('Please enter a prompt.'); return; }
  showError('');
  try {
    const r = await fetch('/v1/plan/workflow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPlanBody()),
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || `HTTP ${r.status}`); }
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    const filename = (cd.match(/filename="([^"]+)"/) || [])[1] || 'workflow.json';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
    const toast = document.getElementById('wf-toast');
    toast.style.display = 'block';
    if (openComfyUI) window.open(_comfyuiUrl, '_blank');
    setTimeout(() => { toast.style.display = 'none'; }, 8000);
  } catch (e) { showError(e.message); }
}

async function doOpenInComfyUI() {
  await doDownloadWorkflow(true);
}

async function doPlan() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) { showError('Please enter a prompt.'); return; }
  showError('');

  const body = buildPlanBody();

  try {
    const r = await fetch('/v1/plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || `HTTP ${r.status}`); }
    const plan = await r.json();
    const sec = document.getElementById('plan-section');
    sec.style.display = 'block';
    document.getElementById('plan-pre').textContent = JSON.stringify(plan, null, 2);
    showWarnings(plan.warnings);
    renderRecs(plan.recommendations);
    renderProvenance(plan.provenance);
  } catch (e) {
    showError(e.message);
  }
}

async function doInstallPlan() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) { showError('Please enter a prompt.'); return; }
  showError('');

  const intent = document.getElementById('intent').value || '';
  const qs = new URLSearchParams({ prompt });
  if (intent) qs.set('intent', intent);

  try {
    const r = await fetch('/v1/install/plan?' + qs.toString());
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    const sec = document.getElementById('install-section');
    const pre = document.getElementById('install-pre');
    sec.style.display = 'block';
    pre.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    showError(e.message);
  }
}

function doClear() {
  if (_polling) clearInterval(_polling);
  document.getElementById('prompt').value = '';
  document.getElementById('neg-prompt').value = '';
  document.getElementById('outputs').innerHTML = '';
  document.getElementById('plan-section').style.display = 'none';
  document.getElementById('install-section').style.display = 'none';
  document.getElementById('recs-section').style.display = 'none';
  document.getElementById('progress-section').style.display = 'none';
  document.getElementById('provenance-section').style.display = 'none';
  showWarnings([]);
  showError('');
  setGenerating(false);
}
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    return HTMLResponse(content=_UI_HTML)
