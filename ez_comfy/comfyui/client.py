from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    event_type: str          # "progress", "executing", "executed", "execution_error"
    node_id: str | None
    step: int | None
    total_steps: int | None
    preview_b64: str | None  # latent preview as base64 data URI, or None


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

    async def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str,
        timeout: float = 300.0,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> dict:
        """
        Wait for a prompt to complete.
        client_id must be the same one passed to queue_prompt() so the WebSocket
        receives events for this specific prompt.
        """
        try:
            return await self._wait_websocket(prompt_id, client_id, timeout, on_progress)
        except Exception as exc:
            logger.warning("WebSocket progress failed (%s), falling back to HTTP polling", exc)
            return await self._wait_polling(prompt_id, timeout)

    async def _wait_websocket(
        self,
        prompt_id: str,
        client_id: str,
        timeout: float,
        on_progress: Callable[[ProgressEvent], None] | None,
    ) -> dict:
        import websockets

        ws_url = f"{self._ws_url}?clientId={client_id}"

        async with asyncio.timeout(timeout):
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        # Binary message = latent preview image
                        if on_progress and len(raw) > 8:
                            # ComfyUI binary preview: first 8 bytes are header
                            img_bytes = raw[8:]
                            b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()
                            event = ProgressEvent(
                                event_type="preview",
                                node_id=None,
                                step=None,
                                total_steps=None,
                                preview_b64=b64,
                            )
                            on_progress(event)
                        continue

                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")
                    data = msg.get("data", {})

                    if msg_type == "progress":
                        step = data.get("value")
                        total = data.get("max")
                        event = ProgressEvent(
                            event_type="progress",
                            node_id=None,
                            step=step,
                            total_steps=total,
                            preview_b64=None,
                        )
                        if on_progress:
                            on_progress(event)

                    elif msg_type == "executing":
                        node = data.get("node")
                        event = ProgressEvent(
                            event_type="executing",
                            node_id=node,
                            step=None,
                            total_steps=None,
                            preview_b64=None,
                        )
                        if on_progress:
                            on_progress(event)
                        # node=None means execution finished
                        if node is None and data.get("prompt_id") == prompt_id:
                            break

                    elif msg_type == "executed":
                        if data.get("prompt_id") == prompt_id:
                            event = ProgressEvent(
                                event_type="executed",
                                node_id=data.get("node"),
                                step=None,
                                total_steps=None,
                                preview_b64=None,
                            )
                            if on_progress:
                                on_progress(event)

                    elif msg_type == "execution_error":
                        if data.get("prompt_id") == prompt_id:
                            err = data.get("exception_message", "Unknown error")
                            raise RuntimeError(f"ComfyUI execution error: {err}")

        # Fetch and return final history
        history = await self.get_history(prompt_id)
        return history.get(prompt_id, {})

    async def _wait_polling(self, prompt_id: str, timeout: float) -> dict:
        """HTTP polling fallback."""
        elapsed = 0.0
        poll_interval = 1.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            history = await self.get_history(prompt_id)
            entry = history.get(prompt_id, {})
            status = entry.get("status", {})
            status_str = status.get("status_str", "")
            if status_str in ("success", "error"):
                if status_str == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI generation failed: {msgs}")
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

    async def cancel_prompt(self, prompt_id: str) -> None:
        await self._http.post("/interrupt", timeout=10.0)

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
            for img in node_output.get("images", []):
                outputs.append({**img, "media_type": "image"})
            for audio in node_output.get("audio", []):
                outputs.append({**audio, "media_type": "audio"})
            for video in node_output.get("gifs", []):
                outputs.append({**video, "media_type": "video"})
        return outputs

    async def close(self) -> None:
        await self._http.aclose()
