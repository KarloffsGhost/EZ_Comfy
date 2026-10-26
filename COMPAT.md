Tested ComfyUI: 0.42.0 - 0.42.0

# ComfyUI compatibility

The first line is read by `tests/unit/test_compat.py` and must match `TESTED_COMFYUI_RANGE`
in `ez_comfy/compat.py`. `python -m ez_comfy check` warns (never fails) outside that range.

| ComfyUI | Frontend / package | Checked | Notes |
|---------|--------------------|---------|-------|
| 0.42.0 (Desktop 1.1.8) | `comfyui_workflow_templates` 0.11.96 | 2026-10-24 | Websocket binary frames, scoped cancel (`/api/jobs/{id}/cancel`), both combo shapes. See `DECISIONS.md` (V1-V5). |

To widen the range: run EZ Comfy against the new version, record the row above, then change the
constant and this header together.
