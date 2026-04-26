"""
Integration tests for GenerationEngine.

Tests cover the full orchestration pipeline (plan → compose → submit → wait →
extract) against a mocked ComfyUIClient. No real ComfyUI or Ollama process
is required; all HTTP calls are intercepted via unittest.mock.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ez_comfy.comfyui.client import ComfyUIClient, ProgressEvent
from ez_comfy.config.schema import Settings, ComfyUIConfig, OllamaConfig, PreferencesConfig
from ez_comfy.engine import GenerationEngine, GenerationResult
from ez_comfy.hardware.comfyui_inventory import ComfyUIInventory, ModelInfo
from ez_comfy.hardware.probe import HardwareProfile
from ez_comfy.planner.planner import GenerationRequest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _hardware() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA GeForce RTX 4070",
        gpu_vram_gb=12.0,
        system_ram_gb=32.0,
        cuda_version="12.4",
        platform="win32",
    )


def _inventory() -> ComfyUIInventory:
    """Minimal inventory: one installed checkpoint matching the catalog entry dreamshaper_8."""
    return ComfyUIInventory(
        checkpoints=[
            ModelInfo(
                filename="dreamshaper_8.safetensors",
                size_bytes=int(2.1 * 1e9),
                family="sd15",
                variant=None,
            )
        ],
        loras=[],
        vaes=[],
        upscale_models=[],
        discovered_class_types={
            "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler",
            "VAEDecode", "EmptyLatentImage", "SaveImage",
        },
    )


def _settings(ollama_enabled: bool = True) -> Settings:
    s = Settings()
    s.comfyui = ComfyUIConfig(base_url="http://127.0.0.1:8188", output_dir="output")
    s.ollama = OllamaConfig(base_url="http://localhost:11434", enabled=ollama_enabled)
    s.preferences = PreferencesConfig(prefer_speed=True, auto_negative_prompt=True)
    return s


def _mock_client(
    prompt_id: str = "test-prompt-id",
    outputs: list[dict] | None = None,
    queue_side_effect=None,
) -> ComfyUIClient:
    """Build a mock ComfyUIClient that simulates successful submission."""
    if outputs is None:
        outputs = [{"filename": "output_00001.png", "subfolder": "", "type": "output"}]

    client = MagicMock(spec=ComfyUIClient)
    if queue_side_effect:
        client.queue_prompt = AsyncMock(side_effect=queue_side_effect)
    else:
        client.queue_prompt = AsyncMock(return_value=(prompt_id, "fake-client-id"))
    client.wait_for_completion = AsyncMock(return_value={
        "outputs": {
            "9": {"images": [{"filename": o["filename"], "subfolder": o["subfolder"], "type": o["type"]}]}
        }
        for o in outputs
    })
    client.extract_outputs = MagicMock(return_value=outputs)
    client.upload_image = AsyncMock(return_value={"name": "reference.png"})
    client.close = AsyncMock()
    return client


def _engine(client=None, ollama_enabled: bool = True) -> GenerationEngine:
    return GenerationEngine(
        client=client or _mock_client(),
        settings=_settings(ollama_enabled=ollama_enabled),
        hardware=_hardware(),
        inventory=_inventory(),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_txt2img(tmp_path):
    """Full txt2img generation: plan → compose → submit → extract → sidecar."""
    client = _mock_client()
    engine = _engine(client=client)

    # Redirect sidecar writes to tmp_path
    with patch("ez_comfy.comfyui.vram.unload_ollama_models", AsyncMock()), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", AsyncMock()), \
         patch.object(engine._settings.comfyui, "output_dir", str(tmp_path)):

        request = GenerationRequest(prompt="a cyberpunk city at sunset, cinematic lighting")
        result = await engine.generate(request)

    assert isinstance(result, GenerationResult)
    assert result.prompt_id == "test-prompt-id"
    assert len(result.outputs) == 1
    assert result.outputs[0]["filename"] == "output_00001.png"
    assert result.duration_seconds >= 0

    # Submission must have happened
    client.queue_prompt.assert_called_once()

    # Workflow passed to ComfyUI must be a non-empty dict
    submitted_workflow = client.queue_prompt.call_args[0][0]
    assert isinstance(submitted_workflow, dict)
    assert len(submitted_workflow) > 0


@pytest.mark.asyncio
async def test_ollama_eviction_called_before_generation():
    """VRAM guard must call Ollama unload before queueing the prompt."""
    call_order = []

    async def fake_unload(url):
        call_order.append("unload")

    async def fake_free(client):
        call_order.append("free")

    client = _mock_client()

    async def mock_queue(*args, **kwargs):
        call_order.append("submit")
        return ("pid", "cid")

    client.queue_prompt = AsyncMock(side_effect=mock_queue)

    engine = _engine(client=client)

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", fake_unload), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", fake_free), \
         patch.object(engine._settings.comfyui, "output_dir", "/tmp"):
        request = GenerationRequest(prompt="a landscape painting")
        await engine.generate(request)

    assert call_order[0] == "unload", "Ollama must be unloaded before ComfyUI submission"
    assert "submit" in call_order
    assert call_order[-1] == "free", "ComfyUI VRAM must be freed last"


@pytest.mark.asyncio
async def test_comfyui_vram_freed_when_ollama_disabled():
    """With Ollama disabled, free_comfyui_vram must still be called after generation."""
    free_called = []

    async def fake_free(client):
        free_called.append(True)

    engine = _engine(ollama_enabled=False)

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", AsyncMock()) as mock_unload, \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", fake_free), \
         patch.object(engine._settings.comfyui, "output_dir", "/tmp"):
        request = GenerationRequest(prompt="a mountain lake")
        await engine.generate(request)

    # unload is skipped when ollama disabled (None url)
    # free must still happen
    assert free_called == [True]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_down_does_not_block_generation():
    """
    If Ollama is unreachable, unload_ollama_models logs and returns — generation
    must continue. We simulate this by letting the real function run against a
    mock httpx that raises ConnectError, proving the internal error handling works
    end-to-end through the engine.
    """
    from unittest.mock import MagicMock

    # Make httpx.AsyncClient raise ConnectError on __aenter__ (Ollama unreachable)
    mock_http_cls = MagicMock()
    mock_http_cls.return_value.__aenter__ = AsyncMock(
        side_effect=__import__("httpx").ConnectError("Ollama is down")
    )
    mock_http_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    client = _mock_client()
    engine = _engine(client=client)

    with patch("ez_comfy.comfyui.vram.httpx.AsyncClient", mock_http_cls), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", AsyncMock()), \
         patch.object(engine._settings.comfyui, "output_dir", "/tmp"):
        request = GenerationRequest(prompt="a forest path in autumn")
        result = await engine.generate(request)

    assert result is not None
    client.queue_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_comfyui_submission_failure_propagates():
    """When ComfyUI queue_prompt fails with 500, the engine must raise."""
    client = _mock_client(
        queue_side_effect=RuntimeError("ComfyUI returned 500")
    )
    engine = _engine(client=client)

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", AsyncMock()), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", AsyncMock()), \
         patch.object(engine._settings.comfyui, "output_dir", "/tmp"):
        with pytest.raises(RuntimeError, match="ComfyUI returned 500"):
            await engine.generate(GenerationRequest(prompt="a space station"))


@pytest.mark.asyncio
async def test_comfyui_vram_freed_even_on_submission_failure():
    """free_comfyui_vram must be called even when queue_prompt raises."""
    free_called = []

    async def fake_free(client):
        free_called.append(True)

    client = _mock_client(queue_side_effect=RuntimeError("submission failed"))
    engine = _engine(client=client)

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", AsyncMock()), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", fake_free), \
         patch.object(engine._settings.comfyui, "output_dir", "/tmp"):
        with pytest.raises(RuntimeError):
            await engine.generate(GenerationRequest(prompt="a robot"))

    assert free_called == [True], "VRAM must be freed even after submission failure"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_jobs_serialize_on_gpu_lock():
    """
    Two concurrent generate() calls must not interleave — the GPU lock
    ensures they run sequentially.
    """
    execution_log: list[str] = []

    original_run = GenerationEngine._run_generation

    async def logged_run(self, *args, **kwargs):
        execution_log.append("start")
        result = await original_run(self, *args, **kwargs)
        execution_log.append("end")
        return result

    client = _mock_client()
    engine = _engine(client=client)

    with patch.object(GenerationEngine, "_run_generation", logged_run), \
         patch("ez_comfy.comfyui.vram.unload_ollama_models", AsyncMock()), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", AsyncMock()), \
         patch.object(engine._settings.comfyui, "output_dir", "/tmp"):
        req = GenerationRequest(prompt="test prompt")
        await asyncio.gather(
            engine.generate(req),
            engine.generate(req),
        )

    # Serialized: start1, end1, start2, end2 — never start1, start2 together
    assert execution_log == ["start", "end", "start", "end"], (
        f"Expected sequential execution but got: {execution_log}"
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provenance_record_attached_to_result(tmp_path):
    """The generation result's plan must include a non-empty ProvenanceRecord."""
    engine = _engine()

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", AsyncMock()), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", AsyncMock()), \
         patch.object(engine._settings.comfyui, "output_dir", str(tmp_path)):
        result = await engine.generate(
            GenerationRequest(prompt="a portrait of an astronaut")
        )

    assert hasattr(result.plan, "provenance"), "Plan must have a provenance attribute"
    provenance = result.plan.provenance
    assert provenance is not None
    assert len(provenance.decisions) > 0, "Provenance must record at least one decision"
