from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from ez_comfy.comfyui.client import ComfyUIClient, ProgressEvent
from ez_comfy.comfyui.vram import vram_guard
from ez_comfy.config.schema import Settings
from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory, scan_inventory
from ez_comfy.hardware.probe import HardwareProfile
from ez_comfy.models.catalog import ModelRecommendation, recommend_models
from ez_comfy.planner.intent import PipelineIntent, detect_intent
from ez_comfy.planner.planner import GenerationPlan, GenerationRequest, plan_generation
from ez_comfy.workflows.composer import compose_workflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    plan: GenerationPlan
    prompt_id: str
    outputs: list[dict]              # [{type, filename, subfolder, ...}, ...]
    duration_seconds: float
    progress_events: list[ProgressEvent]
    error: str | None = None

    def output_urls(self, base_url: str) -> list[str]:
        """Return view URLs for all outputs."""
        urls = []
        for out in self.outputs:
            fn = out.get("filename", "")
            sf = out.get("subfolder", "")
            tp = out.get("type", "output")
            params = f"filename={fn}&subfolder={sf}&type={tp}"
            urls.append(f"{base_url.rstrip('/')}/view?{params}")
        return urls


# ---------------------------------------------------------------------------
# Queue job tracking
# ---------------------------------------------------------------------------

@dataclass
class QueuedJob:
    job_id: str
    request: GenerationRequest
    status: str = "queued"           # queued | running | done | error | cancelled
    result: GenerationResult | None = None
    error: str | None = None
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GenerationEngine:
    """
    Central orchestrator for ComfyUI generation.

    Holds a single asyncio.Lock so only one generation runs on the GPU at a time,
    across direct generate(), compare(), and queue-driven jobs.
    """

    def __init__(
        self,
        client: ComfyUIClient,
        settings: Settings,
        hardware: HardwareProfile,
        inventory: ComfyUIInventory,
    ) -> None:
        self._client = client
        self._settings = settings
        self._hardware = hardware
        self._inventory = inventory
        self._gpu_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refresh_inventory(self) -> ComfyUIInventory:
        """Re-scan ComfyUI for installed models and capabilities."""
        self._inventory = await scan_inventory(self._client)
        logger.info(
            "Inventory refreshed: %d checkpoints, %d LoRAs, %d capabilities detected",
            len(self._inventory.checkpoints),
            len(self._inventory.loras),
            len(self._inventory.discovered_class_types),
        )
        return self._inventory

    async def plan_only(
        self,
        request: GenerationRequest,
        prefer_speed: bool | None = None,
        auto_negative: bool | None = None,
    ) -> GenerationPlan:
        """Plan a generation without touching the GPU."""
        return plan_generation(
            request=request,
            hardware=self._hardware,
            inventory=self._inventory,
            prefer_speed=(
                prefer_speed if prefer_speed is not None
                else self._settings.preferences.prefer_speed
            ),
            auto_negative=(
                auto_negative if auto_negative is not None
                else self._settings.preferences.auto_negative_prompt
            ),
        )

    async def generate(
        self,
        request: GenerationRequest,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        prefer_speed: bool | None = None,
        auto_negative: bool | None = None,
        timeout: float = 300.0,
    ) -> GenerationResult:
        """
        Full generation pipeline:
          plan → upload images → compose → submit → wait → extract outputs.
        Acquires GPU lock for the duration.
        """
        async with self._gpu_lock:
            return await self._run_generation(
                request=request,
                on_progress=on_progress,
                prefer_speed=prefer_speed,
                auto_negative=auto_negative,
                timeout=timeout,
            )

    async def compare(
        self,
        requests: list[GenerationRequest],
        on_progress: Callable[[ProgressEvent], None] | None = None,
        timeout: float = 300.0,
    ) -> list[GenerationResult]:
        """
        Run multiple generations sequentially for A/B comparison.
        Each acquires the GPU lock individually so they don't interleave.
        """
        results = []
        for req in requests:
            result = await self.generate(req, on_progress=on_progress, timeout=timeout)
            results.append(result)
        return results

    def get_recommendations(
        self,
        prompt: str,
        intent: str | None = None,
        top_n: int = 3,
    ) -> list[ModelRecommendation]:
        """Return model recommendations without touching the GPU."""
        if intent is None:
            intent = detect_intent(prompt, False, False).value
        return recommend_models(
            prompt=prompt,
            intent=intent,
            hardware=self._hardware,
            inventory=self._inventory,
            prefer_speed=self._settings.preferences.prefer_speed,
            top_n=top_n,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_generation(
        self,
        request: GenerationRequest,
        on_progress: Callable[[ProgressEvent], None] | None,
        prefer_speed: bool | None,
        auto_negative: bool | None,
        timeout: float,
    ) -> GenerationResult:
        progress_events: list[ProgressEvent] = []

        def _collect(event: ProgressEvent) -> None:
            progress_events.append(event)
            if on_progress:
                on_progress(event)

        # Unload Ollama models before generation, free ComfyUI VRAM after
        ollama_url = self._settings.ollama.base_url if self._settings.ollama.enabled else None
        async with vram_guard(self._client, ollama_url):
            # 1. Plan
            plan = plan_generation(
                request=request,
                hardware=self._hardware,
                inventory=self._inventory,
                prefer_speed=(
                    prefer_speed if prefer_speed is not None
                    else self._settings.preferences.prefer_speed
                ),
                auto_negative=(
                    auto_negative if auto_negative is not None
                    else self._settings.preferences.auto_negative_prompt
                ),
            )

            # 2. Upload reference / mask images → get ComfyUI filenames
            if request.reference_image:
                resp = await self._client.upload_image(
                    request.reference_image, "reference.png"
                )
                plan.reference_image_path = resp.get("name", "reference.png")
                logger.debug("Uploaded reference image: %s", plan.reference_image_path)

            if request.mask_image:
                resp = await self._client.upload_image(
                    request.mask_image, "mask.png"
                )
                plan.mask_image_path = resp.get("name", "mask.png")
                logger.debug("Uploaded mask image: %s", plan.mask_image_path)

            # 3. Compose ComfyUI node graph
            workflow = compose_workflow(plan)
            logger.debug("Composed workflow with %d nodes for recipe %s", len(workflow), plan.recipe.id)

            # 4. Submit — keep client_id so the WS listener sees the same session
            prompt_id, client_id = await self._client.queue_prompt(workflow)
            logger.info("Submitted prompt %s (recipe=%s, checkpoint=%s)", prompt_id, plan.recipe.id, plan.checkpoint)

            # 5. Wait for completion using the same client_id
            t0 = time.monotonic()
            history_entry = await self._client.wait_for_completion(
                prompt_id=prompt_id,
                client_id=client_id,
                timeout=timeout,
                on_progress=_collect,
            )
            duration = time.monotonic() - t0

            # 6. Extract outputs
            outputs = self._client.extract_outputs(history_entry)
            logger.info("Generation done in %.1fs — %d outputs", duration, len(outputs))

            # 7. Write provenance sidecar (best-effort; never fails the generation)
            try:
                await self._write_sidecar(plan, prompt_id, outputs, duration)
            except Exception as exc:
                logger.warning("Failed to write provenance sidecar: %s", exc)

            return GenerationResult(
                plan=plan,
                prompt_id=prompt_id,
                outputs=outputs,
                duration_seconds=duration,
                progress_events=progress_events,
            )

    async def _write_sidecar(
        self,
        plan,
        prompt_id: str,
        outputs: list[dict],
        duration: float,
    ) -> None:
        """Write provenance sidecar JSON alongside generation outputs."""
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        meta_dir = Path(self._settings.comfyui.output_dir) / "ez_comfy_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        sidecar = {
            "schema_version": "1.0",
            "prompt_id": prompt_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 2),
            "provenance": plan.provenance.to_dict(),
            "plan_summary": plan.summary(),
            "outputs": outputs,
        }

        path = meta_dir / f"{prompt_id}.json"
        path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
        logger.debug("Wrote provenance sidecar: %s", path)


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

class GenerationQueue:
    """
    Background FIFO queue for fire-and-forget generation requests.
    Delegates to GenerationEngine.generate() (which holds the GPU lock),
    so queue jobs and direct API calls never race on the GPU.
    """

    def __init__(self, engine: GenerationEngine) -> None:
        self._engine = engine
        self._jobs: dict[str, QueuedJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._process_loop(), name="ez_comfy_queue")
            logger.info("Generation queue started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Generation queue stopped")

    def enqueue(self, request: GenerationRequest) -> str:
        job_id = str(uuid.uuid4())
        job = QueuedJob(job_id=job_id, request=request)
        self._jobs[job_id] = job
        self._queue.put_nowait(job_id)
        logger.info("Enqueued job %s (queue depth=%d)", job_id, self._queue.qsize())
        return job_id

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status != "queued":
            return False
        job.status = "cancelled"
        logger.info("Cancelled queued job %s", job_id)
        return True

    def get_job(self, job_id: str) -> QueuedJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[QueuedJob]:
        return list(self._jobs.values())

    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def _process_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self._jobs.get(job_id)
                if not job or job.status == "cancelled":
                    continue

                job.status = "running"
                job.started_at = time.monotonic()
                logger.info("Processing job %s", job_id)

                try:
                    result = await self._engine.generate(job.request)
                    job.result = result
                    job.status = "done"
                    logger.info("Job %s done in %.1fs", job_id, result.duration_seconds)
                except Exception as exc:
                    job.error = str(exc)
                    job.status = "error"
                    logger.error("Job %s failed: %s", job_id, exc)
                finally:
                    job.finished_at = time.monotonic()
            finally:
                self._queue.task_done()
