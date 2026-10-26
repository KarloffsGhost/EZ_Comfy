# Decisions

Non-obvious choices, newest first. Each entry: date, decision, why.

## 2026-10-25 — Wan 2.2 text-to-video recipe (WI-5, part 1: planning, builder, outputs)

**Run live on 2026-10-25** (RTX 5070 Ti, ComfyUI 0.42.0, through `python -m ez_comfy generate`): 512x512, 17 frames, 10 steps finished in 128 s including the cold model load and produced one MP4 (17 frames at 16 fps). VRAM peaked at about 15.6 GiB (the card was full, so the models offload to RAM), GPU 100 %, 58 C, 256 W peak. Recorded in `COMPAT.md`.

**Live correction.** The spec (and my first implementation) said the video marker is `animated` on each image entry. In the real history entry `animated` is a list on the *node output*, parallel to `images`, and each file carries `metadata.kind == "video"`. `extract_outputs` was labelling the MP4 an image until the live check. It now accepts the metadata marker, the node-level `animated` list (or bool) and a video file extension, and is pinned by a fixture recorded from this run (`tests/fixtures/history/`).

- **Golden graph.** `scripts/export_template_graph.py` flattens the shipped `video_wan2_2_14B_t2v.json` (a subgraph with a boolean "fast LoRA" switch) into API format by folding the primitive, switch and math helper nodes, using the live `/object_info` for widget order. The result, `tests/fixtures/workflows/wan22_t2v.api.json`, has 14 nodes and contains only public template file names and `PROMPT`/`NEGATIVE` placeholders. The builder is compared to it with `tests/golden.py`.
- **Template default = plain mode, not the LoRA mode.** With the switch off the template runs one 20-step pass split at step 10 (high-noise expert 0-10, low-noise expert 10-20), cfg 3.5, shift 5, 16 fps, 81 frames, no extra LoRAs. That is what the builder and the `wan_video` profile now use (the earlier WI-3 profile had the 4-step LoRA numbers). The installed lightx2v 4-step LoRAs would make it about five times fewer steps; that is a follow-up variant, not wired.
- **Normalized comparison.** Nodes are matched by a structural key (what feeds them and how their output is consumed), floats compare to 6 decimals (the template stores 5.0 as `5.000000000000001`; the first comparison failed on exactly that), and seeds/titles/prefixes are ignored. Structure cannot distinguish positive from negative prompt, so the golden test plans with the placeholder prompts and compares text too; a self-test pins that limitation.
- **Multi-file plan.** `GenerationPlan.models` maps roles (`diffusion_high`, `diffusion_low`, `text_encoder`, `vae`) to files; `checkpoint` stays the high-noise file for display. `Recipe.families` and `select_recipe(..., family=)` reject recipes for the wrong family: a legacy recipe (no families) can never run a modern family, and the Wan recipe is never used with another family. SVD is still chosen with a reference image and rejected, with a recorded reason, otherwise.
- **All four roles are required.** If the pair, encoder or VAE is missing, Wan is not used; a warning names what is missing when some Wan t2v files exist. i2v files are never taken for the t2v pair.
- **Frames and fps.** `GenerationRequest.video_frames/video_fps` were declared but unused (default 25/8) and are now `None` = family default (81 at 16 fps for Wan). Wan length must be 1 + 4n (the latent node's step is 4, read from live `/object_info`), so a request is snapped to the nearest valid value (ties up) with a warning. SVD ignores both, as before.
- **Prompt group.** Modern natural-language families (`wan_video`, `ltx_video`, `qwen_image`, `hidream`, `flux2`, `flux3`) now share the `flux` style group so SDXL-style domain packs are not injected into video prompts.
- **Default negative prompt** for Wan is the template's own (Chinese) text, stored as `_NEG_WAN` with its source cited.
- **Outputs.** See the live correction above; still images, audio and `gifs` are unchanged.
- **Full-size run measured (2026-10-25):** the default 640x640, 81 frames, 20 steps took 639 s (10.7 min), 5.06 s of video, GPU 93 % average, 65 C and 285 W peak, VRAM peak about 14.3 GiB. The plan carries that as a warning and tells the user to raise `--timeout` (default 300 s) rather than changing the default; the planner's 120 s video estimate is still a placeholder because two points do not make a model.
- **Still open:** the 4-step LoRA variant, image-to-video and LTX; catalog entries and `_intent_compatible_families` for video are untouched (WI-7b).

## 2026-10-25 — Classifier: modern families and `model_overrides` (WI-3)

**V7 resolved.** The templates ComfyUI Desktop 1.1.8 ships locally (`comfyui_workflow_templates_json` 0.1.122, under the install's `.venv`) include Wan 2.1/2.2, LTX 2/2.3/2.5, Qwen-Image 1/2.1, HiDream, Flux 1/2/3, Krea. File names in the new classifier tests and every profile value come from them; the files and sizes are cited in `profiles.py`.

- **Families added:** `wan_video`, `ltx_video`, `qwen_image`, `hidream`, `flux2`, `flux3`. Tokens: `wan` and `ltx` need a version-like context and a non-letter before them (so `swan2`, `wanderer`, `multx` stay unknown); video families are checked before image families; `qwen` needs `image` after it, so Qwen *text encoders* are not image models. Previously unrecognised `flux1-*` names keep the `flux` family (D2: explicit tokens only, else the single family).
- **Profiles for five of the six.** `flux3` is classification-only: its weights are gated on Hugging Face (HTTP 401), so the file size could not be confirmed, and a guessed profile would be worse than none. All profiles state in comments that `vram_requirement_gb` is the official weight-file size in GiB (a lower bound, not a measured peak; WI-5 measures) and that settings that depend on LoRAs or custom samplers (Wan 4-step, LTX manual sigmas, Flux2 scheduler) are named after the template's node. None of these families is selectable for generation until WI-5/6.
- **D1 resolved as (b): no SDXL allow-list.** The only candidate was a popular merge whose newer releases are no longer SDXL-based, so a name-only rule would misclassify them. It stays `unknown`; `check` now lists unclassified models and names `model_overrides`.
- **`model_overrides`** (`Settings.model_overrides`, file name -> family): wins over every classifier rule, reaches checkpoints and diffusion models through `scan_inventory(client, overrides=...)`, and is handed over by the server lifespan, the engine's refresh and the CLI commands. Matching is by the exact name ComfyUI reports. A misspelled family fails at startup with the allowed list rather than being ignored.
- **Fix to WI-2's fallback.** "Prefer a known family" would have chosen `stable_audio` or the now-classified `ltx_video` checkpoint for an image prompt. The fallback now prefers a family compatible with the intent, then an unclassified file, then the first.
- Not done: LoRA family inference (spec non-goal); the Wan/LTX/Qwen/Flux3 catalog entries (WI-7b).

## 2026-10-25 — Multi-loader scanning and the loader guard (WI-2, rest)

- **Verified live:** `check` lists 6 `UNETLoader` files (four Wan 2.2 14B files plus two `xl-inpaint-0.1\diffusion_pytorch_model*` files) under diffusion models; checkpoints stay at 7. The Wan files classify as `unknown` until WI-3.
- **`checkpoints` stays checkpoint-loader only.** New lists (`diffusion_models`, `clip_vision_models`, `dual_clip_models`) are separate, so existing consumers (API, engine, catalog) see no change.
- **The guard is enforced at selection, not only by list placement.** The planner and `resolve_installed_filename` take `inventory.models_for_loader("checkpoint")`, so a UNET file that ended up in `checkpoints` through a future scan bug would still not be selected (tested). An explicit `--checkpoint` naming a diffusion-model file is not treated as installed.
- **Selection happens before recipe choice** (planner steps 3 and 5), so the guard uses the one loader every current builder uses. `Recipe.model_loader` exists now and a recipe that needs another loader adds a warning; recipe-aware model selection is deferred to WI-5/6, which are the first to add such recipes.
- **Fallback prefers a known family** over the first file; if every file is unknown it still takes the first.
- **Not done here:** `ModelInfo.size_bytes` is still 0 (no API for it), so `check` prints `0.0GB`. The two `xl-inpaint` files are classified `sdxl` by name; that is WI-3's call.

## 2026-10-25 — Config diagnostics, version tracking, fixture capture (WI-1)

- **Warn, don't override.** `check` warns when an explicit `model_base_path` is missing and suggests ComfyUI Desktop's current folder (live: `G:/Documents/ComfyUI/models` missing, suggests `J:\Documents\ComfyUI\models`). `detect_comfyui` keeps its explicit-input semantics, so the existing tests are unchanged.
- **config.json is derived from the yaml's folder**, not read from `%APPDATA%` directly, so tests that patch the yaml path stay hermetic and never read the maintainer's real config.
- **`TESTED_COMFYUI_RANGE` is exactly `0.42.0 - 0.42.0`.** Only 0.42.0 has been verified; a later patch release warns until the range is widened with a COMPAT.md row. Chosen over a `0.42.x` claim we have not tested.
- **`pyproject.toml` stays at 0.1.0** until WI-2 and WI-3 merge (spec 1.4.5). `__version__` falls back to `0.0.0+unknown` when the package is not installed.
- **Fixture capture is allow-list based** (unknown fields are dropped, not redacted) and stages outside `tests/`. No fixtures are committed yet: the capture was run against the live server into a scratch folder for review; nothing in it matched `Users`, drive letters, user name or host name. Open: the maintainer chooses which model file names to keep.
- Observed, not fixed: `check` prints `0.0GB` for every checkpoint (the size is not populated).

## 2026-10-24 — Client hardening: frames, completion, cancel (WI-4)

**Live checks (V3, V4) on ComfyUI 0.42.0**, tiny SD1.5 jobs (512x512, `PreviewImage` so nothing landed in `output/`), GPU at 0-8% beforehand:

- **V3.** This server runs with previews off, so a plain prompt sends **no** binary frames (the first run saw only JSON messages). Previews can be requested per prompt with `extra_data.preview_method` (`execution.py:734`; it resets after each prompt), which is how frames were captured without changing any server setting. 25 frames, all event type 1, header `00000001 00000001 ffd8ffe000104a46` (type 1, image type 1 = JPEG, then JPEG data), ~44-50 KB each. Matches the parser. The old `raw[8:]` slice was correct for type 1; its defect was treating TEXT/other frames as images. Only the 16-byte header is stored as a fixture (`tests/fixtures/ws_frames/`), not images. EZ_Comfy does not send `extra_data.preview_method`, so on a server with previews off the UI simply gets no previews; left as is (non-goal).
- **V4.** `POST /api/jobs/{id}/cancel` verified: cancels a running prompt (`{"cancelled": true}`, the waiter ended with `GenerationCancelled`), cancels a pending prompt while the running one is untouched, and returns `{"cancelled": false}` (HTTP 200) for a finished or unknown id. Queue empty at the end. Source confirms `execution_interrupted` carries `prompt_id` and `execution_success` exists (`execution.py:702,828`).

**Decisions.**

- Cancel prefers the job API (idempotent, no check-then-act race); falls back to `/queue` delete (pending) or `/interrupt` with `{"prompt_id": ...}` (running) when it returns 404/405. An unscoped interrupt is never sent.
- A prompt deleted while pending produces no websocket message, so the engine also cancels its in-flight wait after a successful cancel instead of waiting for a message that never comes.
- The progress socket is opened before `queue_prompt` (`ComfyUIClient.progress_socket`; yields `None` and falls back to polling if it cannot connect), the history is checked once right after listening starts, and the websocket and polling fallback share one deadline.
- `execution_success` is accepted as a completion signal alongside `executing` with `node: null`.
- New `GenerationError(RuntimeError)` and `GenerationCancelled(GenerationError)`: existing handlers that catch `RuntimeError` keep working, and an execution error no longer falls through to a pointless polling pass.
- `GenerationQueue.cancel` keeps its synchronous `bool` signature; `cancel_running` is new. `DELETE /v1/queue/{job_id}` tries queued, then running, then returns the same 404 as before.
- `_wait_websocket(prompt_id, client_id, timeout, on_progress)` keeps its original call shape (ws is passed as a keyword only when given) so the existing test that stubs it is unchanged.
- Opting in to metadata previews (`feature_flags`) is not done (D5 = no); the parser still tolerates type 4.

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
