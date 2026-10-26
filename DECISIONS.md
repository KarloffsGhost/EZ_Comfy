# Decisions

Non-obvious choices, newest first. Each entry: date, decision, why.

## 2026-10-24 — Tolerant combo parsing and exact upscaler filenames (WI-2 item 1, pulled forward)

**Decision.** `_extract_list` accepts both combo shapes (`[[options], {...}]` and `["COMBO", {"options": [...]}]`), warns on any other shape, and is used for samplers/schedulers too. `ModelRecommendation` gained `installed_filename` (defaulted), and the upscale builder sends that exact name when present.

**Why.** On ComfyUI 0.42 `UpscaleModelLoader` is the COMBO shape, so the old parser returned `[]` and EZ_Comfy reported no upscalers installed; the builder also sent an extensionless catalog token while the installed file is `RealESRGAN_x4plus.safetensors`. This corrects V1 (earlier "all legacy" covered only core V1 loaders). `KSampler` `sampler_name`/`scheduler` are legacy shape (45 samplers, 9 schedulers parse live).

**Verified live (read-only).** `scan_inventory` against `http://127.0.0.1:8000` now lists `RealESRGAN_x4plus.safetensors`.

**Compatibility.** The upscale builder still keys on `rec.installed` and falls back to the catalog token when no exact name is recorded, so existing tests and mock-based callers are unchanged. The `.pth` fallback when nothing is installed is left as is.

## 2026-10-24 — Independent review of the spec and WI-7a (Opus reviewer), and what changed

Fixed in the WI-7a follow-up commit:

- **Override-alias regression.** Tier 2 ("installed name contained in catalog token") ranked the *longest* token first, so `--checkpoint flux` reached `flux_hyper_8step`, `pony` the turbo entry and `sdxl` `copax_timeless_xl`. Tier 2 now prefers the *shortest* containing token; tests pin `flux`, `pony`, `sdxl` to the entries master returned.
- **Generic-word false match.** `ltx-2.3-22b-dev-fp8` matched `flux_dev_fp8` through the fuzzy tier on "dev"/"fp8" and the combined part "devfp8", and a txt2img plan then chose the LTX file with Flux settings. The fuzzy tier now requires the token's leading word (`flux1`, `realvisxlv50`, …) to appear. A stoplist would have needed to cover combined parts too. Kept in WI-7a because 7a.1's goal is unambiguous mapping and this was a live wrong pick.
- **Slowdown (~45x at 300 files).** `resolve_installed_filename` re-ranked every file for every entry. The file-to-owner map is now built once per (files, competitor set) and cached (`_installed_owners`); 300 checkpoints went from ~1 s to ~0.05 s, and a tie warns once instead of ~37 times.
- Exact-tier matching also ignores a folder prefix (`SDXL\sd_xl_base_1.0.safetensors` is now an exact match for `sd_xl_base_1.0`); other tiers still see the full path, so non-colliding names keep matching as before.
- Tests: the cross-match test now feeds variant filenames (the old version was a subset of self-resolution); added override-alias, real-world-name, upscaler, performance-bound, tie-warns-once and planner-level tests (`runnable_only` is passed; the LTX file is not planned as Flux).

Spec corrections to apply (spec file is untracked and not edited here):

- **Upscalers are invisible on 0.42 (live issue, not in the spec).** `/object_info/UpscaleModelLoader` returns `["COMBO", {"options": [...]}]`, so the current `_extract_list` returns `[]` and the upscale intent reports nothing installed. The upscale builder also sends a `.pth`/extensionless name while the installed file is `.safetensors`. The tolerant combo parser (WI-2 item 1) and a `resolve_installed_filename`-based upscale name move ahead of WI-4 as a small standalone item.
- **V1 coverage.** The first V1 run only covered core V1 loaders (legacy shape). Add `UpscaleModelLoader` and the other `comfy_extras` loaders to Section V.
- **WI-8 is premise-stale.** ComfyUI and `nvidia-smi` both report a single GPU (RTX 5070 Ti, 16303 MiB). `/system_stats` already lists every torch device, primary first. Defer WI-8 until a second GPU is present and use a synthetic two-device fixture. "bytes → GB" must be GiB (`/1024**3`; 15.9 vs 17.1 changes which entries fit). Drop or date the "[observed] cuda:1 = 5060 Ti" claim.
- **WI-4 cancel race.** Reading `/queue` then acting is check-then-act. 0.42 has `POST /api/jobs/{id}/cancel` (idempotent whatever the job state); prefer it, fall back to `/queue` delete plus a scoped `/interrupt`.
- **WI-2 evidence is stale.** `UNETLoader` lists 6 files (the 4 Wan 2.2 files plus `xl-inpaint-0.1\diffusion_pytorch_model{,.fp16}.safetensors`), so "`models/unet/` empty" is wrong; `checkpoints/SDXL/` holds `sd_xl_base_1.0`. Update the 2.5 acceptance.
- **Privacy (public repo).** WI-3's "real filenames from the audit" would publish the maintainer's inventory. Use filenames from ComfyUI's shipped templates; keep the spec untracked or scrubbed.
- **Port.** 8188 is the correct default for standalone ComfyUI and the gitignored `settings.yaml` already sets 8000; no default change (it would break the additive rule). Quirk: `__main__.py:41` treats an explicitly configured 8188 as "not explicit".
- **Cross-match count.** 18 with bare filenames, 17 with `.safetensors` appended; both are right.

Reviewer-verified as correct in the spec: scoped `/interrupt`, `execution_interrupted`, binary frame layout (4-byte event + 4-byte image type, 1 = JPEG, 2 = PNG), `executing` with `node: null` going to the owning client only.

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
