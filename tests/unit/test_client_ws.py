"""Unit tests for comfyui/client.py: binary frames, completion signals, cancel (WI-4)."""
from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from ez_comfy.comfyui.client import (
    ComfyUIClient,
    GenerationCancelled,
    GenerationError,
    ProgressEvent,
    _parse_binary_frame,
)

JPEG = b"\xff\xd8\xff\xe0fakejpegbytes"
PNG = b"\x89PNG\r\n\x1a\nfakepngbytes"


def _frame(event: int, *parts: bytes) -> bytes:
    return struct.pack(">I", event) + b"".join(parts)


# ---------------------------------------------------------------------------
# Binary frame parser (layout: ComfyUI protocol.py / server.py send_image*)
# ---------------------------------------------------------------------------

def test_parse_type1_jpeg():
    frame = _parse_binary_frame(_frame(1, struct.pack(">I", 1), JPEG))
    assert frame is not None and frame.mime_type == "image/jpeg" and frame.image_bytes == JPEG


def test_parse_type1_png():
    frame = _parse_binary_frame(_frame(1, struct.pack(">I", 2), PNG))
    assert frame is not None and frame.mime_type == "image/png" and frame.image_bytes == PNG


def test_parse_type1_unknown_image_type_defaults_to_jpeg():
    frame = _parse_binary_frame(_frame(1, struct.pack(">I", 99), JPEG))
    assert frame is not None and frame.mime_type == "image/jpeg"


def test_parse_type4_with_metadata():
    meta = json.dumps({"image_type": "image/png", "node_id": "5"}).encode()
    frame = _parse_binary_frame(_frame(4, struct.pack(">I", len(meta)), meta, PNG))
    assert frame is not None
    assert frame.mime_type == "image/png"
    assert frame.metadata == {"image_type": "image/png", "node_id": "5"}
    assert frame.image_bytes == PNG


def test_parse_type3_text_returns_none():
    assert _parse_binary_frame(_frame(3, struct.pack(">I", 0), b"some progress text")) is None


def test_parse_short_frame_returns_none():
    assert _parse_binary_frame(b"") is None
    assert _parse_binary_frame(b"\x00\x00\x00\x01") is None


def test_parse_unknown_type_returns_none():
    assert _parse_binary_frame(_frame(77, struct.pack(">I", 1), JPEG)) is None


@pytest.mark.parametrize("raw", [
    _frame(1, struct.pack(">I", 1)),                                    # no image bytes
    _frame(4, struct.pack(">I", 500), b"{}", JPEG),                     # metadata length past the end
    _frame(4, struct.pack(">I", 5), b"not{j", JPEG),                    # metadata is not JSON
    _frame(4, struct.pack(">I", 2), b"[]", JPEG),                       # metadata is not an object
])
def test_parse_malformed_frames_return_none_without_raising(raw):
    assert _parse_binary_frame(raw) is None


# ---------------------------------------------------------------------------
# Websocket waiting
# ---------------------------------------------------------------------------

class FakeWS:
    """Async-iterable stand-in for a websocket: yields queued messages, then blocks forever."""

    def __init__(self, messages: list, then: str = "block") -> None:
        self._messages = list(messages)
        self._then = then

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._messages:
            item = self._messages.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self._then == "close":
            raise StopAsyncIteration
        await asyncio.Event().wait()   # block until cancelled / timed out


def _msg(msg_type: str, **data) -> str:
    return json.dumps({"type": msg_type, "data": data})


def _client(history: dict | None = None) -> ComfyUIClient:
    client = ComfyUIClient("http://comfy.test")
    client.get_history = AsyncMock(return_value=history if history is not None else {})
    return client


_SUCCESS = {"p": {"status": {"status_str": "success"}, "outputs": {}}}


async def test_finished_before_connect_returns_quickly_via_history_check():
    client = _client(_SUCCESS)
    started = time.monotonic()
    entry = await client.wait_for_completion("p", "cid", timeout=5.0, ws=FakeWS([]))
    assert entry == _SUCCESS["p"]
    assert time.monotonic() - started < 1.0


async def test_single_deadline_not_doubled():
    client = _client({})                      # never finishes
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await client.wait_for_completion("p", "cid", timeout=0.3, ws=FakeWS([]))
    assert time.monotonic() - started < 0.9    # one deadline, not websocket timeout + polling timeout


async def test_polling_fallback_gets_only_the_remaining_time(monkeypatch):
    client = _client({})
    seen = {}

    async def fake_poll(prompt_id, timeout):
        seen["timeout"] = timeout
        return {"polled": True}

    monkeypatch.setattr(client, "_wait_polling", fake_poll)
    result = await client.wait_for_completion("p", "cid", timeout=10.0, ws=FakeWS([ConnectionError("boom")]))
    assert result == {"polled": True}
    assert 0 < seen["timeout"] <= 10.0


async def test_execution_interrupted_raises_cancelled():
    client = _client({})
    ws = FakeWS([_msg("execution_interrupted", prompt_id="p", node_id="3")])
    with pytest.raises(GenerationCancelled):
        await client.wait_for_completion("p", "cid", timeout=5.0, ws=ws)


async def test_interrupt_of_a_different_prompt_is_ignored():
    client = _client({})
    client.get_history = AsyncMock(side_effect=[{}, _SUCCESS])   # pre-check, then final fetch
    ws = FakeWS([
        _msg("execution_interrupted", prompt_id="someone-else"),
        _msg("executing", node=None, prompt_id="p"),
    ])
    entry = await client.wait_for_completion("p", "cid", timeout=5.0, ws=ws)
    assert entry == _SUCCESS["p"]


async def test_execution_error_raises_generation_error_without_polling(monkeypatch):
    client = _client({})
    monkeypatch.setattr(client, "_wait_polling", AsyncMock(side_effect=AssertionError("must not poll")))
    ws = FakeWS([_msg("execution_error", prompt_id="p", exception_message="out of memory")])
    with pytest.raises(GenerationError, match="out of memory"):
        await client.wait_for_completion("p", "cid", timeout=5.0, ws=ws)


async def test_generation_error_is_still_a_runtime_error():
    assert issubclass(GenerationError, RuntimeError)
    assert issubclass(GenerationCancelled, GenerationError)


async def test_text_frame_does_not_produce_a_preview_event():
    client = _client({})
    client.get_history = AsyncMock(side_effect=[{}, _SUCCESS])
    events: list[ProgressEvent] = []
    ws = FakeWS([
        _frame(3, struct.pack(">I", 0), b"progress text"),
        _frame(1, struct.pack(">I", 2), PNG),
        _msg("execution_success", prompt_id="p"),
    ])
    await client.wait_for_completion("p", "cid", timeout=5.0, on_progress=events.append, ws=ws)
    previews = [e for e in events if e.event_type == "preview"]
    assert len(previews) == 1
    assert previews[0].preview_b64.startswith("data:image/png;base64,")


async def test_history_precheck_reports_interrupted_prompt_as_cancelled():
    interrupted = {"p": {"status": {"status_str": "error", "messages": [["execution_interrupted", {}]]}}}
    client = _client(interrupted)
    with pytest.raises(GenerationCancelled):
        await client.wait_for_completion("p", "cid", timeout=5.0, ws=FakeWS([]))


async def test_websocket_closing_early_falls_back_to_polling(monkeypatch):
    client = _client({})
    monkeypatch.setattr(client, "_wait_polling", AsyncMock(return_value={"polled": True}))
    assert await client.wait_for_completion("p", "cid", timeout=5.0, ws=FakeWS([], then="close")) == {"polled": True}


# ---------------------------------------------------------------------------
# cancel_prompt: scoped to one prompt, never an unscoped /interrupt
# ---------------------------------------------------------------------------

def _client_with_transport(handler) -> tuple[ComfyUIClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = ComfyUIClient("http://comfy.test")
    client._http = httpx.AsyncClient(base_url="http://comfy.test", transport=httpx.MockTransport(recording))
    return client, seen


async def test_cancel_uses_job_api_when_available():
    client, seen = _client_with_transport(lambda r: httpx.Response(200, json={"cancelled": True}))
    result = await client.cancel_prompt("abc")
    assert result.cancelled and result.method == "job_api"
    assert [r.url.path for r in seen] == ["/api/jobs/abc/cancel"]


async def test_cancel_of_finished_or_unknown_prompt_reports_not_found():
    client, _ = _client_with_transport(lambda r: httpx.Response(200, json={"cancelled": False}))
    result = await client.cancel_prompt("abc")
    assert not result.cancelled and result.method == "not_found"


def _legacy_server(queue: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/jobs/"):
            return httpx.Response(404)
        if request.url.path == "/queue" and request.method == "GET":
            return httpx.Response(200, json=queue)
        return httpx.Response(200, json={})
    return handler


async def test_cancel_scoped_to_prompt_id_when_running_on_legacy_server():
    queue = {"queue_running": [[0, "abc", {}]], "queue_pending": [[1, "other", {}]]}
    client, seen = _client_with_transport(_legacy_server(queue))
    result = await client.cancel_prompt("abc")
    assert result.cancelled and result.method == "interrupt"
    interrupt = [r for r in seen if r.url.path == "/interrupt"]
    assert len(interrupt) == 1 and json.loads(interrupt[0].content) == {"prompt_id": "abc"}


async def test_cancel_deletes_pending_prompt_from_queue_on_legacy_server():
    queue = {"queue_running": [[0, "other", {}]], "queue_pending": [[1, "abc", {}]]}
    client, seen = _client_with_transport(_legacy_server(queue))
    result = await client.cancel_prompt("abc")
    assert result.cancelled and result.method == "queue_delete"
    delete = [r for r in seen if r.url.path == "/queue" and r.method == "POST"]
    assert json.loads(delete[0].content) == {"delete": ["abc"]}
    assert not [r for r in seen if r.url.path == "/interrupt"], "must not interrupt the running prompt"


async def test_cancel_unknown_prompt_on_legacy_server_sends_nothing_destructive():
    queue = {"queue_running": [[0, "other", {}]], "queue_pending": []}
    client, seen = _client_with_transport(_legacy_server(queue))
    result = await client.cancel_prompt("abc")
    assert not result.cancelled and result.method == "not_found"
    assert not [r for r in seen if r.method == "POST" and not r.url.path.startswith("/api/jobs/")]


async def test_cancel_network_failure_returns_failed_without_raising():
    def boom(request):
        raise httpx.ConnectError("refused")

    client, _ = _client_with_transport(boom)
    result = await client.cancel_prompt("abc")
    assert not result.cancelled and result.method == "failed"


def test_real_comfyui_0_42_preview_header_parses_as_jpeg():
    """Header bytes captured from a live ComfyUI 0.42.0 preview frame (see fixture note)."""
    fixture = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "ws_frames" / "live_comfyui_0_42_0.json").read_text()
    )
    for entry in fixture["frames"]:
        raw = bytes.fromhex(entry["header16_hex"]) + b"\x00" * 32   # stand-in for the rest of the JPEG
        frame = _parse_binary_frame(raw)
        assert frame is not None
        assert (frame.event_type, frame.mime_type) == (1, "image/jpeg")
        assert frame.image_bytes.startswith(b"\xff\xd8\xff")
