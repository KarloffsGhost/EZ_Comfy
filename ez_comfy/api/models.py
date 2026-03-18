from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for POST /v1/generate (JSON-only path, no image uploads)."""
    prompt: str
    negative_prompt: str = ""
    style_preset: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    seed: int = -1
    batch_size: int = Field(default=1, ge=1, le=8)
    denoise_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    loras: list[tuple[str, float]] | None = None
    intent_override: str | None = None
    checkpoint_override: str | None = None
    recipe_override: str | None = None
    enhance_prompt: bool = False
    timeout: float = Field(default=300.0, ge=10.0, le=1800.0)


class PlanRequest(BaseModel):
    """Request body for POST /v1/plan."""
    prompt: str
    negative_prompt: str = ""
    style_preset: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int = -1
    batch_size: int = Field(default=1, ge=1, le=8)
    denoise_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    intent_override: str | None = None
    checkpoint_override: str | None = None
    recipe_override: str | None = None


class CompareRequest(BaseModel):
    """Request body for POST /v1/compare."""
    requests: list[GenerateRequest]
    timeout: float = Field(default=300.0, ge=10.0, le=1800.0)


class QueueRequest(BaseModel):
    """Request body for POST /v1/queue."""
    prompt: str
    negative_prompt: str = ""
    style_preset: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int = -1
    batch_size: int = Field(default=1, ge=1, le=8)
    denoise_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    loras: list[tuple[str, float]] | None = None
    intent_override: str | None = None
    checkpoint_override: str | None = None
    recipe_override: str | None = None


class OutputFile(BaseModel):
    type: str       # image | audio | video
    filename: str
    subfolder: str = ""
    url: str = ""


class GenerateResponse(BaseModel):
    prompt_id: str
    recipe: str
    checkpoint: str
    family: str
    duration_seconds: float
    outputs: list[OutputFile]
    warnings: list[str]
    plan_summary: dict


class PlanResponse(BaseModel):
    intent: str
    recipe: str
    checkpoint: str
    family: str
    profile: str
    estimated_vram_gb: float
    estimated_time_seconds: float
    warnings: list[str]
    missing_capabilities: list[str]
    params: dict
    param_sources: dict
    prompt: str
    negative_prompt: str
    recommendations: list[dict]
    provenance: dict | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                         # queued | running | done | error | cancelled
    result: GenerateResponse | None = None
    error: str | None = None
    queued_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None


class RecommendationsResponse(BaseModel):
    prompt: str
    intent: str
    recommendations: list[dict]


class InstallPlanResponse(BaseModel):
    prompt: str
    intent: str
    recommended_installed: list[dict]
    recommended_to_install: list[dict]
    missing_capabilities: list[str]
    capability_guidance: list[dict]
    notes: list[str]


class InventoryResponse(BaseModel):
    checkpoints: list[dict]
    loras: list[dict]
    vaes: list[str]
    upscale_models: list[str]
    samplers: list[str]
    schedulers: list[str]
    discovered_capabilities: list[str]


class HealthResponse(BaseModel):
    status: str                 # ok | degraded
    comfyui: bool
    comfyui_url: str = ""
    gpu_name: str
    gpu_vram_gb: float
    checkpoints_loaded: int
    queue_depth: int
    version: str = "1.0.0"
