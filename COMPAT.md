Tested ComfyUI: 0.42.0 - 0.42.0

# ComfyUI compatibility

The first line is read by `tests/unit/test_compat.py` and must match `TESTED_COMFYUI_RANGE`
in `ez_comfy/compat.py`. `python -m ez_comfy check` warns (never fails) outside that range.

| ComfyUI | Frontend / package | Checked | Notes |
|---------|--------------------|---------|-------|
| 0.42.0 (Desktop 1.1.8) | `comfyui_workflow_templates` 0.11.96 | 2026-10-24 | Websocket binary frames, scoped cancel (`/api/jobs/{id}/cancel`), both combo shapes. See `DECISIONS.md` (V1-V5). |

To widen the range: run EZ Comfy against the new version, record the row above, then change the
constant and this header together.

## End-to-end runs

| Date | Recipe | Settings | Result |
|------|--------|----------|--------|
| 2026-10-25 | `video_wan22_t2v` (Wan 2.2 14B fp8, plain mode) | 512x512, 17 frames, 10 steps, 16 fps, RTX 5070 Ti 16 GB, ComfyUI 0.42.0 | 128 s including model load; one MP4 (17 frames, 1.06 s, 377 KB); VRAM peaked at about 15.6 GiB (the card was full; the models offload to RAM), GPU 100 % and 58 C, 256 W peak |
| 2026-10-25 | `video_wan22_t2v` (default size) | 640x640, 81 frames (5.06 s), 20 steps, 16 fps, same machine | 639 s (10.7 min) including a warm-ish load; VRAM peaked at about 14.3 GiB, GPU 93 % average and 100 % peak, 65 C, 285 W peak |
| 2026-10-25 | `video_wan22_t2v` (default size, seed 4242) on both cards at once, one ComfyUI instance per card | 640x640, 81 frames, 20 steps, 16 fps; RTX 5070 Ti 16 GB (instance on port 8000) and RTX 5060 Ti 16 GB (second instance, `--cuda-device 1`, port 8001, OCuLink dock at PCIe x4), ComfyUI 0.42.0 | 5070 Ti: 693 s (about 8 % slower than the 639 s solo run), 268 W and 66 C peak, link Gen 5 x4. 5060 Ti: 1277 s, 179 W (its power limit) and 77 C peak, link Gen 4 x4. Combined GPU draw about 430 W for the 11 minutes both ran; no throttling or crash. Both MP4s have 81 frames at 640x640 and show the same scene. The 5060 Ti was not timed alone, so how much of its slower time is the x4 link is not measured |
