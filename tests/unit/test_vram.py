"""Unit tests for ez_comfy/comfyui/vram.py"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ez_comfy.comfyui.vram import free_comfyui_vram, unload_ollama_models, vram_guard

# httpx requires a request object on responses when raise_for_status() is called
_GET_REQ = httpx.Request("GET", "http://ollama/api/ps")
_POST_REQ = httpx.Request("POST", "http://ollama/api/generate")
_FREE_REQ = httpx.Request("POST", "http://comfy/free")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_client_mock(
    ps_status: int = 200,
    ps_body: dict | None = None,
    post_side_effect: Exception | None = None,
    post_status: int = 200,
):
    """
    Build a mock that replaces httpx.AsyncClient for unload_ollama_models.
    Wraps into an async context manager as the real code uses `async with httpx.AsyncClient(...)`.
    """
    if ps_body is None:
        ps_body = {"models": []}

    get_resp = httpx.Response(ps_status, json=ps_body, request=_GET_REQ)

    inner = AsyncMock()
    inner.get = AsyncMock(return_value=get_resp)
    if post_side_effect is not None:
        inner.post = AsyncMock(side_effect=post_side_effect)
    else:
        inner.post = AsyncMock(
            return_value=httpx.Response(post_status, json={}, request=_POST_REQ)
        )

    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=inner)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_cls, inner


def _comfyui_client(handler) -> MagicMock:
    """Return a mock ComfyUIClient whose _http uses the given transport handler."""
    transport = httpx.MockTransport(handler)
    client = MagicMock()
    client._http = httpx.AsyncClient(transport=transport, base_url="http://comfy")
    return client


# ---------------------------------------------------------------------------
# unload_ollama_models
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unload_ollama_happy_path(caplog):
    mock_cls, inner = _ollama_client_mock(
        ps_body={"models": [{"name": "llama3:8b"}]},
    )
    with patch("ez_comfy.comfyui.vram.httpx.AsyncClient", mock_cls):
        with caplog.at_level(logging.DEBUG, logger="ez_comfy.comfyui.vram"):
            await unload_ollama_models("http://ollama")
    # Should log the eviction
    assert "llama3:8b" in caplog.text
    inner.post.assert_called_once()


@pytest.mark.asyncio
async def test_unload_ollama_skips_when_ollama_down(caplog):
    """Connection refused must not raise — just log a warning."""
    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch("ez_comfy.comfyui.vram.httpx.AsyncClient", mock_cls):
        with caplog.at_level(logging.WARNING, logger="ez_comfy.comfyui.vram"):
            await unload_ollama_models("http://ollama")
    assert caplog.records, "Expected at least one log record"
    assert any("non-fatal" in r.message.lower() or "could not" in r.message.lower()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_unload_ollama_handles_ps_500(caplog):
    """A 500 from /api/ps must not raise — outer except logs it."""
    mock_cls, _ = _ollama_client_mock(ps_status=500, ps_body={})
    with patch("ez_comfy.comfyui.vram.httpx.AsyncClient", mock_cls):
        with caplog.at_level(logging.WARNING, logger="ez_comfy.comfyui.vram"):
            await unload_ollama_models("http://ollama")
    assert caplog.records, "Expected a warning log"


@pytest.mark.asyncio
async def test_unload_ollama_no_models_running():
    """When /api/ps returns no models, no eviction POST calls should be made."""
    mock_cls, inner = _ollama_client_mock(ps_body={"models": []})
    with patch("ez_comfy.comfyui.vram.httpx.AsyncClient", mock_cls):
        await unload_ollama_models("http://ollama")
    inner.post.assert_not_called()


@pytest.mark.asyncio
async def test_unload_ollama_model_eviction_failure_non_fatal(caplog):
    """POST /api/generate raises a network error — must log and not re-raise."""
    mock_cls, _ = _ollama_client_mock(
        ps_body={"models": [{"name": "mistral:7b"}]},
        post_side_effect=httpx.TimeoutException("eviction timed out"),
    )
    with patch("ez_comfy.comfyui.vram.httpx.AsyncClient", mock_cls):
        with caplog.at_level(logging.WARNING, logger="ez_comfy.comfyui.vram"):
            await unload_ollama_models("http://ollama")
    assert "mistral:7b" in caplog.text


# ---------------------------------------------------------------------------
# free_comfyui_vram
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_free_comfyui_vram_happy_path(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    client = _comfyui_client(handler)
    with caplog.at_level(logging.DEBUG, logger="ez_comfy.comfyui.vram"):
        await free_comfyui_vram(client)
    assert any("vram" in r.message.lower() or "free" in r.message.lower()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_free_comfyui_vram_handles_timeout(caplog):
    """A timeout on /free must not raise."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = _comfyui_client(handler)
    with caplog.at_level(logging.WARNING, logger="ez_comfy.comfyui.vram"):
        await free_comfyui_vram(client)
    assert caplog.records, "Expected a warning log"


@pytest.mark.asyncio
async def test_free_comfyui_vram_handles_connection_error(caplog):
    """A connection error on /free must not raise."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _comfyui_client(handler)
    with caplog.at_level(logging.WARNING, logger="ez_comfy.comfyui.vram"):
        await free_comfyui_vram(client)
    assert any("non-fatal" in r.message.lower() or "could not" in r.message.lower()
               for r in caplog.records)


# ---------------------------------------------------------------------------
# vram_guard context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vram_guard_calls_unload_and_free():
    """Full vram_guard: unloads Ollama before, frees ComfyUI after."""
    unload_called = []
    free_called = []

    async def fake_unload(url: str) -> None:
        unload_called.append(url)

    async def fake_free(client) -> None:
        free_called.append(True)

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", fake_unload), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", fake_free):
        async with vram_guard(client=MagicMock(), ollama_url="http://ollama"):
            pass

    assert unload_called == ["http://ollama"]
    assert free_called == [True]


@pytest.mark.asyncio
async def test_vram_guard_skips_unload_when_ollama_disabled():
    """vram_guard with ollama_url=None must not call unload_ollama_models."""
    unload_called = []
    free_called = []

    async def fake_unload(url: str) -> None:
        unload_called.append(url)

    async def fake_free(client) -> None:
        free_called.append(True)

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", fake_unload), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", fake_free):
        async with vram_guard(client=MagicMock(), ollama_url=None):
            pass

    assert unload_called == []
    assert free_called == [True]


@pytest.mark.asyncio
async def test_vram_guard_frees_comfyui_even_on_generation_error():
    """Even if the body of the context raises, ComfyUI VRAM must still be freed."""
    free_called = []

    async def fake_unload(url: str) -> None:
        pass

    async def fake_free(client) -> None:
        free_called.append(True)

    with patch("ez_comfy.comfyui.vram.unload_ollama_models", fake_unload), \
         patch("ez_comfy.comfyui.vram.free_comfyui_vram", fake_free):
        with pytest.raises(RuntimeError, match="generation failed"):
            async with vram_guard(client=MagicMock(), ollama_url="http://ollama"):
                raise RuntimeError("generation failed")

    assert free_called == [True]
