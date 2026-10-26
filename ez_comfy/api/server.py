from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ez_comfy.comfyui.client import ComfyUIClient
from ez_comfy.config.schema import Settings
from ez_comfy.engine import GenerationEngine, GenerationQueue
from ez_comfy.hardware.comfyui_inventory import scan_inventory
from ez_comfy.hardware.probe import probe_hardware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: probe hardware, scan ComfyUI inventory, start queue. Shutdown: close client."""
    settings: Settings = app.state.settings
    client = ComfyUIClient(base_url=settings.comfyui.base_url)
    app.state.client = client

    logger.info("Probing hardware …")
    hardware = probe_hardware()
    logger.info("Hardware: %s  %.1f GB VRAM", hardware.gpu_name, hardware.gpu_vram_gb)
    app.state.hardware = hardware

    logger.info("Scanning ComfyUI inventory at %s …", settings.comfyui.base_url)
    try:
        inventory = await scan_inventory(client)
        logger.info(
            "Inventory: %d checkpoints, %d LoRAs, capabilities=%s",
            len(inventory.checkpoints),
            len(inventory.loras),
            sorted(inventory.discovered_class_types)[:10],
        )
    except Exception as exc:
        logger.warning("Could not scan inventory (ComfyUI may be offline): %s", exc)
        from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory
        inventory = ComfyUIInventory()
    app.state.inventory = inventory

    engine = GenerationEngine(
        client=client,
        settings=settings,
        hardware=hardware,
        inventory=inventory,
    )
    app.state.engine = engine

    queue = GenerationQueue(engine=engine)
    queue.start()
    app.state.queue = queue

    logger.info("EZ Comfy ready")
    yield

    logger.info("Shutting down …")
    await queue.stop()
    await client.close()


def create_app(settings: Settings) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="EZ Comfy",
        description="Hardware-aware ComfyUI orchestrator with intelligent model and workflow selection",
        version="1.0.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Register routes
    from ez_comfy.api.routes import router
    app.include_router(router)

    return app
