from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    event_type: str          # "progress", "executing", "executed", "execution_error"
    node_id: str | None
    step: int | None
    total_steps: int | None
    preview_b64: str | None  # latent preview as base64 data URI, or None


class GenerationError(RuntimeError):
    """ComfyUI reported that a prompt failed."""


class GenerationCancelled(GenerationError):
    """A prompt was interrupted or cancelled before it finished."""


@dataclass
class CancelResult:
    cancelled: bool
    method: str          # "job_api" | "queue_delete" | "interrupt" | "not_found" | "failed"
    detail: str = ""


# ComfyUI binary websocket frames (protocol.py BinaryEventTypes / server.py send_image*):
#   every frame starts with a 4-byte big-endian event type.
#   1 PREVIEW_IMAGE: then a 4-byte image type (1 JPEG, 2 PNG), then the image bytes.
#   4 PREVIEW_IMAGE_WITH_METADATA: then a 4-byte JSON-metadata length, the metadata
#     (its "image_type" is the MIME type), then the image bytes.
#   2 UNENCODED_PREVIEW_IMAGE and 3 TEXT carry no displayable image for us.
_EVENT_PREVIEW_IMAGE = 1
_EVENT_PREVIEW_IMAGE_WITH_METADATA = 4
_IMAGE_TYPE_MIME = {1: "image/jpeg", 2: "image/png"}


@dataclass
class BinaryFrame:
    event_type: int
    mime_type: str
    image_bytes: bytes
    metadata: dict | None = None


def _parse_binary_frame(raw: bytes) -> BinaryFrame | None:
    """Decode a binary websocket frame into a preview image, or None if it is not one.

    Never raises: short, malformed, TEXT and unknown frames return None.
    """
    if len(raw) < 8:
        logger.debug("Ignoring short binary websocket frame (%d bytes)", len(raw))
        return None

    event_type = int.from_bytes(raw[:4], "big")

    if event_type == _EVENT_PREVIEW_IMAGE:
        image_type = int.from_bytes(raw[4:8], "big")
        mime = _IMAGE_TYPE_MIME.get(image_type)
        if mime is None:
            logger.debug("Unknown preview image type %d; assuming JPEG", image_type)
            mime = "image/jpeg"
        image = raw[8:]
        return BinaryFrame(event_type, mime, image) if image else None

    if event_type == _EVENT_PREVIEW_IMAGE_WITH_METADATA:
        metadata_end = 8 + int.from_bytes(raw[4:8], "big")
        if metadata_end > len(raw):
            logger.warning("Ignoring preview frame: metadata length %d exceeds frame size %d",
                           metadata_end - 8, len(raw))
            return None
        try:
            metadata = json.loads(raw[8:metadata_end])
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("Ignoring preview frame with unreadable metadata: %s", exc)
            return None
        if not isinstance(metadata, dict):
            logger.warning("Ignoring preview frame: metadata is %s, not an object", type(metadata).__name__)
            return None
        mime = metadata.get("image_type")
        image = raw[metadata_end:]
        if not image:
            return None
        return BinaryFrame(event_type, mime if isinstance(mime, str) and mime else "image/jpeg",
                           image, metadata)

    logger.debug("Ignoring binary websocket frame of event type %d", event_type)
    return None



_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi")


def _is_video_output(entry: dict, animated, index: int) -> bool:
    """SaveVideo/PreviewVideo list their files under "images". ComfyUI 0.42.0 marks them with
    metadata.kind == "video" on the entry and an `animated` list on the node output (parallel to
    `images`); either signal, or a video file extension, is enough."""
    metadata = entry.get("metadata")
    if isinstance(metadata, dict) and metadata.get("kind") == "video":
        return True
    if isinstance(animated, list):
        if index < len(animated) and animated[index] is True:
            return True
    elif animated is True:
        return True
    return str(entry.get("filename", "")).lower().endswith(_VIDEO_EXTENSIONS)


class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self._base_url = base_url.rstrip("/")
        self._ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._http.get("/system_stats", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def system_stats(self) -> dict:
        resp = await self._http.get("/system_stats")
        resp.raise_for_status()
        return resp.json()

    async def get_object_info(self, node_class: str | None = None) -> dict:
        url = f"/object_info/{node_class}" if node_class else "/object_info"
        resp = await self._http.get(url, timeout=60.0)
        resp.raise_for_status()
        return resp.json()

    async def queue_prompt(self, workflow: dict, client_id: str | None = None) -> tuple[str, str]:
        """
        Submit a workflow to ComfyUI.
        Returns (prompt_id, client_id) — both must be kept together so the
        WebSocket listener connects with the same client_id that was registered.
        """
        cid = client_id or str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": cid}
        resp = await self._http.post("/prompt", json=payload, timeout=30.0)
        if resp.status_code != 200:
            # Surface ComfyUI's validation details instead of a generic 400
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:500]
            logger.error("ComfyUI rejected workflow (HTTP %d): %s", resp.status_code, err_body)
            resp.raise_for_status()
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {data}")
        return prompt_id, cid

    async def get_history(self, prompt_id: str) -> dict:
        resp = await self._http.get(f"/history/{prompt_id}", timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    @asynccontextmanager
    async def progress_socket(self, client_id: str) -> AsyncIterator[object | None]:
        """Open the progress websocket. Open it BEFORE queue_prompt so a fast or fully cached
        prompt cannot finish before we are listening. Yields None if the connection fails;
        wait_for_completion then falls back to HTTP polling.
        """
        import websockets

        ws_url = f"{self._ws_url}?clientId={client_id}"
        try:
            ws = await websockets.connect(ws_url, ping_interval=20)
        except Exception as exc:
            logger.warning("Could not open progress websocket %s (%s); will poll instead", ws_url, exc)
            yield None
            return
        try:
            yield ws
        finally:
            await ws.close()

    async def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str,
        timeout: float = 300.0,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        ws: object | None = None,
    ) -> dict:
        """
        Wait for a prompt to complete.
        client_id must be the same one passed to queue_prompt() so the WebSocket
        receives events for this specific prompt. Pass an already-open `ws` (from
        progress_socket, opened before submitting) to avoid missing early events.
        `timeout` is one shared deadline for the websocket and the polling fallback.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            if ws is None:
                return await self._wait_websocket(prompt_id, client_id, timeout, on_progress)
            return await self._wait_websocket(prompt_id, client_id, timeout, on_progress, ws=ws)
        except (GenerationError, TimeoutError):
            raise
        except Exception as exc:
            remaining = deadline - loop.time()
            logger.warning("WebSocket progress failed for prompt %s (%s), falling back to HTTP polling",
                           prompt_id, exc)
            if remaining <= 0:
                raise TimeoutError(f"ComfyUI generation timed out after {timeout}s") from exc
            return await self._wait_polling(prompt_id, remaining)

    async def _finished_history_entry(self, prompt_id: str) -> dict | None:
        """Return the history entry if the prompt already finished, None if still running.

        Raises GenerationError / GenerationCancelled if it finished unsuccessfully.
        """
        entry = (await self.get_history(prompt_id)).get(prompt_id, {})
        status = entry.get("status", {})
        status_str = status.get("status_str", "")
        if status_str == "success":
            return entry
        if status_str == "error":
            messages = status.get("messages", [])
            if any(isinstance(m, (list, tuple)) and m and m[0] == "execution_interrupted" for m in messages):
                raise GenerationCancelled(f"Prompt {prompt_id} was interrupted")
            raise GenerationError(f"ComfyUI generation failed: {messages}")
        return None

    async def _wait_websocket(
        self,
        prompt_id: str,
        client_id: str,
        timeout: float,
        on_progress: Callable[[ProgressEvent], None] | None,
        ws: object | None = None,
    ) -> dict:
        async with asyncio.timeout(timeout):
            if ws is not None:
                await self._listen(ws, prompt_id, on_progress)
            else:
                async with self.progress_socket(client_id) as own_ws:
                    if own_ws is None:
                        raise ConnectionError("progress websocket unavailable")
                    await self._listen(own_ws, prompt_id, on_progress)

        # Fetch and return final history
        history = await self.get_history(prompt_id)
        return history.get(prompt_id, {})

    async def _listen(
        self,
        ws: object,
        prompt_id: str,
        on_progress: Callable[[ProgressEvent], None] | None,
    ) -> None:
        # The prompt may have finished before we were listening (cached, or socket opened late).
        try:
            if await self._finished_history_entry(prompt_id) is not None:
                return
        except GenerationError:
            raise
        except Exception as exc:
            logger.warning("History pre-check failed for prompt %s: %s", prompt_id, exc)

        async for raw in ws:
            if isinstance(raw, bytes):
                frame = _parse_binary_frame(raw)
                if frame is not None and on_progress:
                    b64 = f"data:{frame.mime_type};base64," + base64.b64encode(frame.image_bytes).decode()
                    on_progress(ProgressEvent(
                        event_type="preview", node_id=None, step=None, total_steps=None, preview_b64=b64,
                    ))
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type", "")
            data = msg.get("data", {})

            if msg_type == "progress":
                if on_progress:
                    on_progress(ProgressEvent(
                        event_type="progress", node_id=None,
                        step=data.get("value"), total_steps=data.get("max"), preview_b64=None,
                    ))

            elif msg_type == "executing":
                node = data.get("node")
                if on_progress:
                    on_progress(ProgressEvent(
                        event_type="executing", node_id=node, step=None, total_steps=None, preview_b64=None,
                    ))
                # node=None means execution finished (also sent after an interrupt)
                if node is None and data.get("prompt_id") == prompt_id:
                    return

            elif msg_type == "executed":
                if data.get("prompt_id") == prompt_id and on_progress:
                    on_progress(ProgressEvent(
                        event_type="executed", node_id=data.get("node"), step=None, total_steps=None,
                        preview_b64=None,
                    ))

            elif msg_type == "execution_success":
                if data.get("prompt_id") == prompt_id:
                    return

            elif msg_type == "execution_interrupted":
                if data.get("prompt_id") == prompt_id:
                    raise GenerationCancelled(f"Prompt {prompt_id} was interrupted")

            elif msg_type == "execution_error":
                if data.get("prompt_id") == prompt_id:
                    err = data.get("exception_message", "Unknown error")
                    raise GenerationError(f"ComfyUI execution error: {err}")

        raise ConnectionError("progress websocket closed before the prompt finished")

    async def _wait_polling(self, prompt_id: str, timeout: float) -> dict:
        """HTTP polling fallback."""
        elapsed = 0.0
        poll_interval = 1.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            entry = await self._finished_history_entry(prompt_id)
            if entry is not None:
                return entry
        raise TimeoutError(f"ComfyUI generation timed out after {timeout}s")

    async def download_output(self, filename: str, subfolder: str = "", output_type: str = "output") -> bytes:
        resp = await self._http.get(
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": output_type},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.content

    async def upload_image(self, image_bytes: bytes, filename: str) -> dict:
        files = {"image": (filename, image_bytes, "image/png")}
        resp = await self._http.post("/upload/image", files=files, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    async def get_queue(self) -> dict:
        resp = await self._http.get("/queue", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    async def cancel_prompt(self, prompt_id: str) -> CancelResult:
        """Cancel exactly one prompt, whether it is queued or running. Never sends an unscoped interrupt.

        Prefers ComfyUI's idempotent POST /api/jobs/{id}/cancel (no check-then-act race). Older
        servers without it get /queue delete (pending) or /interrupt scoped to the prompt id (running).
        """
        try:
            resp = await self._http.post(f"/api/jobs/{prompt_id}/cancel", timeout=10.0)
        except httpx.HTTPError as exc:
            logger.error("Cancel request for prompt %s failed: %s", prompt_id, exc)
            return CancelResult(False, "failed", str(exc))

        if resp.status_code == 200:
            try:
                cancelled = bool(resp.json().get("cancelled"))
            except ValueError:
                logger.error("Cancel response for prompt %s was not JSON: %.200s", prompt_id, resp.text)
                return CancelResult(False, "failed", "unreadable response")
            return CancelResult(cancelled, "job_api" if cancelled else "not_found")

        if resp.status_code in (404, 405):
            logger.info("Job-cancel API unavailable (HTTP %d); falling back to /queue + scoped /interrupt",
                        resp.status_code)
            return await self._cancel_via_queue(prompt_id)

        logger.error("Cancel for prompt %s rejected (HTTP %d): %.200s", prompt_id, resp.status_code, resp.text)
        return CancelResult(False, "failed", f"HTTP {resp.status_code}")

    async def _cancel_via_queue(self, prompt_id: str) -> CancelResult:
        try:
            queue = await self.get_queue()
            is_running = any(len(i) > 1 and i[1] == prompt_id for i in queue.get("queue_running", []))
            is_pending = any(len(i) > 1 and i[1] == prompt_id for i in queue.get("queue_pending", []))
            if is_pending:
                resp = await self._http.post("/queue", json={"delete": [prompt_id]}, timeout=10.0)
                resp.raise_for_status()
                return CancelResult(True, "queue_delete")
            if is_running:
                resp = await self._http.post("/interrupt", json={"prompt_id": prompt_id}, timeout=10.0)
                resp.raise_for_status()
                return CancelResult(True, "interrupt")
        except httpx.HTTPError as exc:
            logger.error("Queue-based cancel for prompt %s failed: %s", prompt_id, exc)
            return CancelResult(False, "failed", str(exc))
        return CancelResult(False, "not_found")

    async def free_vram(self) -> None:
        await self._http.post(
            "/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10.0,
        )

    def extract_outputs(self, history_entry: dict) -> list[dict]:
        """Extract output file references from a history entry."""
        outputs = []
        for node_id, node_output in history_entry.get("outputs", {}).items():
            animated = node_output.get("animated")
            for index, img in enumerate(node_output.get("images", [])):
                outputs.append({**img, "media_type": "video" if _is_video_output(img, animated, index) else "image"})
            for audio in node_output.get("audio", []):
                outputs.append({**audio, "media_type": "audio"})
            for video in node_output.get("gifs", []):
                outputs.append({**video, "media_type": "video"})
        return outputs

    async def close(self) -> None:
        await self._http.aclose()
