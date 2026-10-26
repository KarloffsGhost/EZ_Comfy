from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

import httpx

if TYPE_CHECKING:
    from ez_comfy.comfyui.client import ComfyUIClient

logger = logging.getLogger(__name__)


async def unload_ollama_models(ollama_url: str) -> None:
    """Unload all currently-loaded Ollama models to free VRAM for ComfyUI."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{ollama_url}/api/ps")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            for m in models:
                name = m.get("name", "")
                if not name:
                    continue
                try:
                    await client.post(
                        f"{ollama_url}/api/generate",
                        json={"model": name, "keep_alive": 0},
                        timeout=10,
                    )
                    logger.debug("Unloaded Ollama model: %s", name)
                except Exception as exc:
                    logger.warning("Failed to unload Ollama model %s: %s", name, exc)
    except Exception as exc:
        logger.warning("Could not unload Ollama models (non-fatal): %s", exc)


async def free_comfyui_vram(client: "ComfyUIClient") -> None:
    """Tell ComfyUI to unload models and free GPU memory."""
    try:
        await client._http.post(
            "/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
        logger.debug("Freed ComfyUI VRAM")
    except Exception as exc:
        logger.warning("Could not free ComfyUI VRAM (non-fatal): %s", exc)


@asynccontextmanager
async def vram_guard(
    client: "ComfyUIClient",
    ollama_url: str | None,
) -> AsyncGenerator[None, None]:
    """Context manager: unloads Ollama before generation, frees ComfyUI after."""
    if ollama_url:
        await unload_ollama_models(ollama_url)
    try:
        yield
    finally:
        await free_comfyui_vram(client)
