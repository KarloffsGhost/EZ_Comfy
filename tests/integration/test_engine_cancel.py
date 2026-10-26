"""Integration tests for cancelling queued and running generations (WI-4)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ez_comfy.comfyui.client import CancelResult, ComfyUIClient, GenerationCancelled
from ez_comfy.engine import GenerationQueue
from ez_comfy.planner.planner import GenerationRequest
from tests.integration.test_engine import _engine


def _client(*, blocking_prompts: set[str], cancel_result: CancelResult | None = None) -> MagicMock:
    """Mock client whose prompts in `blocking_prompts` never finish until cancelled."""
    client = MagicMock(spec=ComfyUIClient)
    submitted: list[str] = []

    async def queue_prompt(workflow, client_id=None):
        prompt_id = f"prompt-{len(submitted) + 1}"
        submitted.append(prompt_id)
        return prompt_id, client_id or "cid"

    async def wait_for_completion(prompt_id, client_id, timeout=300.0, on_progress=None, ws=None):
        if prompt_id in blocking_prompts:
            await asyncio.Event().wait()
        return {"outputs": {"9": {"images": [{"filename": "o.png", "subfolder": "", "type": "output"}]}}}

    @asynccontextmanager
    async def progress_socket(client_id):
        yield None

    client.queue_prompt = queue_prompt
    client.wait_for_completion = wait_for_completion
    client.progress_socket = progress_socket
    client.cancel_prompt = AsyncMock(return_value=cancel_result or CancelResult(True, "job_api"))
    client.extract_outputs = MagicMock(return_value=[{"filename": "o.png", "subfolder": "", "type": "output"}])
    client.close = AsyncMock()
    client.submitted = submitted
    return client


def _patched(tmp_path, engine):
    return (
        patch("ez_comfy.comfyui.vram.unload_ollama_models", AsyncMock()),
        patch("ez_comfy.comfyui.vram.free_comfyui_vram", AsyncMock()),
        patch.object(engine._settings.comfyui, "output_dir", str(tmp_path)),
    )


async def _until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


async def test_cancel_running_prompt_raises_cancelled_and_releases_gpu_lock(tmp_path):
    client = _client(blocking_prompts={"prompt-1"})
    engine = _engine(client=client)
    p1, p2, p3 = _patched(tmp_path, engine)
    with p1, p2, p3:
        task = asyncio.create_task(engine.generate(GenerationRequest(prompt="a cat")))
        await _until(lambda: engine._current_prompt_id == "prompt-1")

        result = await engine.cancel_prompt("prompt-1")
        assert result.cancelled
        with pytest.raises(GenerationCancelled):
            await task

        assert not engine._gpu_lock.locked()
        assert engine._current_prompt_id is None and engine._current_waiter is None

        # a following job runs normally
        follow_up = await engine.generate(GenerationRequest(prompt="a dog"))
        assert follow_up.prompt_id == "prompt-2"


async def test_cancel_is_scoped_to_the_prompt_id():
    client = _client(blocking_prompts={"prompt-1"})
    engine = _engine(client=client)
    await engine.cancel_prompt("some-prompt")
    client.cancel_prompt.assert_awaited_once_with("some-prompt")


async def test_cancelling_an_unknown_prompt_leaves_the_running_one_alone(tmp_path):
    client = _client(blocking_prompts={"prompt-1"}, cancel_result=CancelResult(False, "not_found"))
    engine = _engine(client=client)
    p1, p2, p3 = _patched(tmp_path, engine)
    with p1, p2, p3:
        task = asyncio.create_task(engine.generate(GenerationRequest(prompt="a cat")))
        await _until(lambda: engine._current_prompt_id == "prompt-1")
        result = await engine.cancel_prompt("not-the-running-one")
        assert not result.cancelled
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_cancel_event_set_before_submission_cancels_right_after_submit(tmp_path):
    client = _client(blocking_prompts={"prompt-1"})
    engine = _engine(client=client)
    event = asyncio.Event()
    event.set()
    p1, p2, p3 = _patched(tmp_path, engine)
    with p1, p2, p3:
        with pytest.raises(GenerationCancelled):
            await engine.generate(GenerationRequest(prompt="a cat"), cancel_event=event)
    client.cancel_prompt.assert_awaited_once_with("prompt-1")
    assert not engine._gpu_lock.locked()


async def test_queue_cancel_running_job_ends_cancelled_and_next_job_runs(tmp_path):
    client = _client(blocking_prompts={"prompt-1"})
    engine = _engine(client=client)
    queue = GenerationQueue(engine)
    p1, p2, p3 = _patched(tmp_path, engine)
    with p1, p2, p3:
        queue.start()
        try:
            first = queue.enqueue(GenerationRequest(prompt="first"))
            second = queue.enqueue(GenerationRequest(prompt="second"))
            await _until(lambda: queue.get_job(first).prompt_id == "prompt-1")

            assert queue.cancel(first) is False            # not queued any more: sync path unchanged
            assert await queue.cancel_running(first) is True

            await _until(lambda: queue.get_job(first).status == "cancelled")
            await _until(lambda: queue.get_job(second).status == "done")

            assert queue.get_job(first).status == "cancelled"      # not overwritten by done/error
            assert queue.get_job(first).error is None
            assert not engine._gpu_lock.locked()
        finally:
            await queue.stop()


async def test_cancel_running_is_false_for_unknown_queued_and_finished_jobs(tmp_path):
    client = _client(blocking_prompts=set())
    engine = _engine(client=client)
    queue = GenerationQueue(engine)
    assert await queue.cancel_running("nope") is False

    queued_id = queue.enqueue(GenerationRequest(prompt="never started"))
    assert await queue.cancel_running(queued_id) is False       # queued, not running
    assert queue.cancel(queued_id) is True                      # existing sync behaviour

    p1, p2, p3 = _patched(tmp_path, engine)
    with p1, p2, p3:
        queue.start()
        try:
            done_id = queue.enqueue(GenerationRequest(prompt="quick"))
            await _until(lambda: queue.get_job(done_id).status == "done")
            assert await queue.cancel_running(done_id) is False
        finally:
            await queue.stop()
