# Decisions

Non-obvious choices, newest first. Each entry: date, decision, why.

## 2026-10-24 — Verify-first checks against live ComfyUI 0.42.0

Run read-only against `http://127.0.0.1:8000` (ComfyUI Desktop 1.1.8).

| Check | Command | Result | Conclusion |
|-------|---------|--------|------------|
| V1 | `GET /object_info/{CheckpointLoaderSimple,UNETLoader,CLIPLoader,DualCLIPLoader,VAELoader}` | every combo input is the legacy `[[options], {...}]` shape | WI-2 parser must still accept both shapes, but no `COMBO` shape was seen on these loaders. `sampler_name`/`scheduler` were **not** checked. |
| V2 | `GET /` | `<title>ComfyUI</title>` | existing title probe still matches; the `/system_stats` fallback in WI-1 is belt-and-braces |
| V5 | `GET /system_stats` | `comfyui_version 0.42.0`, one device (`cuda:0 RTX 5070 Ti`, 17066033152 bytes), pytorch 2.9.1+cu130 | `vram_total` is in bytes. The spec expected a second GPU (RTX 5060 Ti); ComfyUI does not report one. |
| V7 | listed `comfyui_workflow_templates*` in the venv | packages present (`comfyui_workflow_templates` 0.11.96); a filename filter returned mostly cloud `api_*` templates | local Wan 2.2 / LTX / Flux / Qwen templates still to be located before WI-5/WI-6 |

Not yet run (need a real generation, so they occupy the GPU): V3 (binary frames), V4 (cancel), V6 (Stable Audio).

Open decisions D1–D5 are proceeding on the spec's recommended defaults until the maintainer says otherwise: D1 allow-list + `model_overrides`, D3 start with whichever LTX mode the shipped template supports, D4 `qwen_image`, D5 no metadata previews.

## 2026-10-24 — Catalog matching resolves per installed file (WI-7a)

**Decision.** `match_installed_file(filename, entries)` picks the single most specific entry for an installed file. Ranking: exact (ignoring extension) > catalog token contained in the installed name (longer token wins) > installed name contained in the token > fuzzy signature parts. A tie returns `None` and logs both entry ids.

**Why.** The old rule asked "does this entry match this file?" per entry, so `flux_dev_fp8` also claimed `flux1-dev.safetensors`, and `flux_dev` claimed an fp8 file. Truthiness of `_model_name_matches` is unchanged, so names that never collided behave as before.

**Count.** The spec reports 18 cross-matches; re-running the old rule over the catalog gave 17 here. The test asserts every one resolves to its owner rather than pinning the number.

**Tradeoff.** `resolve_installed_filename` now takes an optional `catalog` argument (default `MODEL_CATALOG`) because an entry has to be compared against its siblings.

## 2026-10-24 — `flux_hyper_8step` gets its own filename

It is Flux dev plus the Hyper-SD LoRA, not a separate checkpoint, so it shared `flux1-dev` with `flux_dev`. It is now `flux1-dev-hyper-8step`. Consequence: a plain `flux1-dev.safetensors` maps only to `flux_dev`; the hyper entry matches only if a file with that name is installed. Rejected alternative: keep the shared name and let the tie return `None`, which would have broken `flux_dev` itself.

## 2026-10-24 — `wan_video_14b` renamed to `wan_video_1_3b`

The entry's name, source and filename all describe Wan **1.3B**. The id said 14B. The maintainer's installed Wan 2.2 14B files are not this entry; they arrive with WI-7b.

## 2026-10-24 — Entries without a workflow builder are `execution_supported=False`

`animatediff_lightning`, `mochi_preview`, `wan_video_1_3b`. `recommend_models(..., runnable_only=True)` (used by the planner) drops them; the default listing keeps them with a "not yet runnable" warning. Today these families are not in any intent's compatible set, so the visible behavior change is nil until WI-5 extends the video families.
