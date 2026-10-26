# EZ_Comfy Improvements — Implementation Spec

> **Goal:** Address the highest-leverage gaps surfaced by the 2026-04-25 audit
> of the v0.1.0 codebase. The repo has been quiescent since 2026-03-17. This
> spec defines six discrete, independently shippable work items so an
> implementing agent can land each one without coordinating with the others.

**Audit reference date:** 2026-04-25
**Last code commit before this spec:** `0b6ed3f` (2026-03-17)

---

## 0. Conventions

- Each work item is **independently mergeable**. No item depends on another.
- Each item lists: **Goal**, **Files** (NEW vs MODIFIED), **Behavior contract**,
  **Acceptance criteria**, **Test plan**, **Non-goals**.
- An item is *done* only when its acceptance criteria pass and its test plan
  is implemented and green (Commandment II).
- Implementer must follow the existing code style in each touched file before
  writing new code (Commandment VII): read at least 50 lines first.
- No item changes the public API contract in a breaking way. Endpoints,
  config keys, and CLI flags retain backward compatibility unless an item
  explicitly says otherwise.

---

## Work Item 1 — Auto-Detect ComfyUI Setup on First Run

### 1.1 Goal

A fresh user should be able to run `python -m ez_comfy check` immediately
after `pip install -e .` and have it succeed against a local ComfyUI Desktop
install — with no manual editing of `config/settings.yaml`.

Today, two things block this:
1. `comfyui.base_url` defaults to `http://127.0.0.1:8000`, but the standalone
   ComfyUI server defaults to `8188`. The Desktop app uses `8000`. Users hit
   "ComfyUI is not reachable" without obvious cause.
2. `comfyui.model_base_path` is hardcoded to a previous developer's
   `G:/Documents/ComfyUI/models` and must be hand-edited.

### 1.2 Files

- **MODIFIED:** [ez_comfy/config/schema.py](../ez_comfy/config/schema.py) — add `auto_detect: bool = True` to the
  `comfyui` config block (default true).
- **NEW:** `ez_comfy/comfyui/autodetect.py` — port + models-path resolution.
- **MODIFIED:** [ez_comfy/__main__.py](../ez_comfy/__main__.py) — invoke autodetect when `check` runs and
  `auto_detect: true`. Print what was detected.
- **MODIFIED:** [config/settings.example.yaml](../config/settings.example.yaml) — set `auto_detect: true` and add a
  comment explaining override behavior.
- **MODIFIED:** [README.md](../README.md) — replace the "Configure" section's hand-edit step with
  "configuration is auto-detected; see `config/settings.example.yaml` to
  override".

### 1.3 Behavior Contract

`autodetect.detect_comfyui()` returns a `DetectedConfig` with:

```python
@dataclass
class DetectedConfig:
    base_url: str | None         # None if no ComfyUI reachable
    model_base_path: str | None  # None if not derivable
    source: str                  # "config_file", "desktop_app", "probe", "default"
```

Resolution order (first hit wins for each field):

1. **Explicit override** in `config/settings.yaml` — never overridden, returned as-is.
2. **ComfyUI Desktop app config** — read
   `%APPDATA%/ComfyUI/extra_models_config.yaml` on Windows,
   `~/Library/Application Support/ComfyUI/extra_models_config.yaml` on macOS,
   `~/.config/ComfyUI/extra_models_config.yaml` on Linux. If present, parse
   for `comfyui.base_path` to derive the models path.
3. **Port probe** — for `base_url`, GET `http://127.0.0.1:{port}/` for ports
   `[8188, 8000, 8189, 8001]` with a 1s timeout each; first 200 response with
   `<title>ComfyUI</title>` in the body wins.
4. **Default fallback** — `http://127.0.0.1:8188`, no models path.

`source` reflects which step succeeded for `base_url`.

### 1.4 Acceptance Criteria

- [ ] Running `python -m ez_comfy check` on a system with only ComfyUI Desktop
  installed (port 8000, models in `%APPDATA%/...`) succeeds without editing
  any config file.
- [ ] Running on a system with stock ComfyUI on port 8188 also succeeds.
- [ ] If a user *has* set `base_url` in `settings.yaml`, that value is used
  unconditionally — autodetect does not override it.
- [ ] `check` prints one line: `Detected ComfyUI at <url> (source: <source>)`.
- [ ] If nothing is reachable, error message names the four ports tried.

### 1.5 Test Plan

`tests/unit/test_autodetect.py`:

- `test_explicit_override_wins` — settings.yaml value short-circuits probing.
- `test_desktop_app_config_parsed` — given a fixture
  `extra_models_config.yaml`, returns expected models path.
- `test_port_probe_finds_8000` — mock httpx so port 8000 returns the ComfyUI
  HTML; assert detected.
- `test_port_probe_finds_8188` — same for 8188.
- `test_no_comfyui_running_returns_none` — all probes fail; `base_url` is None.
- `test_non_comfyui_server_rejected` — port returns 200 but body is not
  ComfyUI's HTML; not selected.

### 1.6 Non-Goals

- Auto-launching ComfyUI if it isn't running.
- Cross-network discovery (mDNS, etc.). Localhost only.
- Writing detected values back to `settings.yaml` (we just use them in memory).

---

## Work Item 2 — Refresh the Model Catalog

### 2.1 Goal

[ez_comfy/models/catalog.py](../ez_comfy/models/catalog.py) currently contains **14** `ModelCatalogEntry` instances.
[README.md:36](../README.md) advertises "a curated catalog of ~50 popular checkpoints". Close
this gap, and bring the catalog up to date with model families that have
shipped since 2026-03-17.

### 2.2 Files

- **MODIFIED:** [ez_comfy/models/catalog.py](../ez_comfy/models/catalog.py) — add new entries.
- **MODIFIED:** [ez_comfy/models/profiles.py](../ez_comfy/models/profiles.py) — add new family settings/syntax
  records only if a new entry needs a family not yet present.
- **MODIFIED:** [tests/unit/test_catalog.py](../tests/unit/test_catalog.py) — extend coverage. Create the file
  if absent.

### 2.3 Behavior Contract

Catalog must contain **at least 35** entries, distributed across these
buckets (minimum counts):

| Family / use case                | Min entries |
|----------------------------------|-------------|
| SDXL photorealism                | 4           |
| SDXL stylized / anime            | 3           |
| Pony / NoobAI (SDXL-derived)     | 3           |
| SD 1.5                           | 4           |
| Flux dev / schnell / variants    | 4           |
| Video (SVD, AnimateDiff, others) | 3           |
| Audio                            | 1           |
| Specialty (inpaint, upscale)     | 2           |

Each entry must satisfy:
- `vram_min_gb` and `vram_recommended_gb` populated (no zeros) and pass
  sanity check (`min <= recommended`).
- `download_command` is non-empty and points to a resolvable HuggingFace or
  CivitAI URL. Implementer must NOT verify URLs at import time (no network on
  import); URL format only.
- `prompt_syntax` set correctly: e.g., Pony entries enable score-tag
  prefixing; Flux entries strip emphasis weights.
- `required_capabilities` lists any non-default ComfyUI custom nodes the
  recipe will need (e.g., `"ControlNetLoader"`).

### 2.4 Acceptance Criteria

- [ ] `len(MODEL_CATALOG) >= 35`.
- [ ] No regression: every entry that was in the catalog before this change
  is still present (filename match — entries may be re-tuned but not deleted
  unless the model is known broken/abandoned, in which case the deletion
  must be justified in the commit message).
- [ ] Bucket counts above all met.
- [ ] All entries pass new validation tests (see 2.5).

### 2.5 Test Plan

`tests/unit/test_catalog.py`:

- `test_minimum_entry_count` — `len(MODEL_CATALOG) >= 35`.
- `test_bucket_distribution` — assert each bucket meets minimum.
- `test_vram_consistency` — for every entry, `vram_min_gb <= vram_recommended_gb`
  and both are positive.
- `test_download_command_format` — every entry has a non-empty
  `download_command` matching a regex for hf/civitai URLs.
- `test_unique_ids_and_filenames` — no duplicate `id`, no duplicate
  `filename`.
- `test_prompt_syntax_pony_has_score_tags` — every Pony-family entry's
  `prompt_syntax` enables score-tag injection.
- `test_prompt_syntax_flux_strips_weights` — every Flux-family entry strips
  emphasis weights.

### 2.6 Non-Goals

- Live catalog updates from a remote source. Static Python file is fine.
- Model quality benchmarking or auto-ranking.
- License classification beyond what's already in the dataclass.

---

## Work Item 3 — Test the VRAM Guard / Ollama Handoff

### 3.1 Goal

[ez_comfy/comfyui/vram.py](../ez_comfy/comfyui/vram.py) implements the marquee "no VRAM surprises" feature
described in [README.md:71-84](../README.md). It currently has zero unit tests. Pin the
contract so future refactors can't silently regress it.

### 3.2 Files

- **NEW:** `tests/unit/test_vram.py`.
- **MODIFIED:** [ez_comfy/comfyui/vram.py](../ez_comfy/comfyui/vram.py) — only if needed to make the module
  testable (e.g., split a hard-coded URL into a parameter). No behavior
  change.

### 3.3 Behavior Contract

The two functions under test:

- `unload_ollama_models(base_url, timeout)` — POSTs `keep_alive=0` to
  `{base_url}/api/generate` for each running model. Per Commandment V, on
  network or non-2xx response, must **log with context** (which URL, which
  model, what status) and return without raising. Generation must continue
  even if Ollama is down.
- `free_comfyui_vram(base_url, timeout)` — POSTs to `{base_url}/free`. Same
  error handling contract.

### 3.4 Acceptance Criteria

- [ ] Coverage of `vram.py` reaches **>= 90%** (measured by
  `pytest --cov=ez_comfy.comfyui.vram`).
- [ ] Every `except` block in `vram.py` is exercised by at least one test.
- [ ] No silent `except: pass` blocks remain — all log with operation
  context (Commandment V).

### 3.5 Test Plan

`tests/unit/test_vram.py`, using `respx` or `httpx.MockTransport`:

- `test_unload_ollama_happy_path` — one model running, eviction call
  returns 200, function completes, captured log shows the eviction.
- `test_unload_ollama_skips_when_ollama_down` — connection refused; log
  records the error, function returns normally.
- `test_unload_ollama_handles_500` — Ollama returns 500; log records, no
  raise.
- `test_unload_ollama_no_models_running` — `/api/tags` returns empty list;
  no eviction calls made.
- `test_free_comfyui_vram_happy_path` — POST `/free` returns 200; logged.
- `test_free_comfyui_vram_handles_timeout` — request times out; logged, no
  raise.

### 3.6 Non-Goals

- Testing the *integration* with the engine (covered in Work Item 4).
- Adding retry logic. Current single-attempt behavior is intentional.

---

## Work Item 4 — Engine + Queue Integration Tests

### 4.1 Goal

`GenerationEngine` ([ez_comfy/engine.py](../ez_comfy/engine.py)) and the queue together orchestrate the
GPU lock, VRAM handoff, ComfyUI submission, and result handling. Today they
have only indirect coverage via per-component unit tests. Add tests that
exercise the full happy path and three failure modes against a mocked
ComfyUI client.

### 4.2 Files

- **NEW:** `tests/integration/__init__.py`, `tests/integration/conftest.py`,
  `tests/integration/test_engine.py`.
- **MODIFIED:** [ez_comfy/engine.py](../ez_comfy/engine.py) — only if needed for testability (constructor
  injection of the ComfyUI client and Ollama URL). No behavior change.
- **MODIFIED:** [pyproject.toml](../pyproject.toml) — add `respx` to `[project.optional-dependencies] dev`.

### 4.3 Behavior Contract

Tests run a full `engine.generate(request)` call against:
- A mocked `ComfyUIClient` that simulates submission, websocket progress
  events, and image retrieval.
- A mocked Ollama HTTP endpoint via `respx`.
- An in-memory hardware profile fixture (RTX 4070, 12GB VRAM).

The tests must not require a real ComfyUI or Ollama process.

### 4.4 Acceptance Criteria

- [ ] `pytest tests/integration/` passes in under 10 seconds total.
- [ ] Tests run on Windows and Linux CI (no OS-specific assumptions).
- [ ] No real network calls (verified by leaving `respx` in strict mode).

### 4.5 Test Plan

`tests/integration/test_engine.py`:

- `test_happy_path_txt2img` — submit a prompt, mock returns one image,
  assert: Ollama eviction was called, ComfyUI submission was called,
  workflow JSON matches expected recipe, result includes image path,
  ComfyUI `/free` was called after.
- `test_ollama_down_does_not_block_generation` — Ollama refused;
  generation still completes (Commandment V).
- `test_comfyui_submission_failure_propagates` — ComfyUI returns 500;
  engine raises a typed error and releases the GPU lock.
- `test_concurrent_jobs_serialize_on_lock` — two jobs submitted; second
  blocks until first releases the GPU lock; queue state reflects
  `running` then `queued`.
- `test_provenance_record_attached` — result includes a non-empty
  `ProvenanceRecord` (per the existing PROVENANCE_SPEC).

### 4.6 Non-Goals

- Performance/benchmark tests.
- Real GPU testing (out of scope for unit/integration suite).

---

## Work Item 5 — Reconcile SPEC.md with Implemented Routes

### 5.1 Goal

[SPEC.md:22-23,1110-1116,1143](../SPEC.md) reference endpoints that don't exist in the code:
`/v1/ui`, `/v1/hardware`, `/v1/catalog`, `/v1/catalog/recommend`. The
implemented endpoints serve the same functions under different names
(`/`, `/v1/inventory`, `/v1/recommendations`). README.md is correct;
SPEC.md is stale.

Per Commandment X, the spec is the contract. Either the code is wrong or
the spec is wrong. **Decision: the code is the source of truth here** — it
ships, was tested, and the rename was deliberate (per commit `9436eb2`
"Fix image rendering and intent detection false positives" timeframe).
Update SPEC.md to match.

### 5.2 Files

- **MODIFIED:** [SPEC.md](../SPEC.md) — replace stale endpoint names. No code changes.

### 5.3 Behavior Contract

After this change, every endpoint mentioned in SPEC.md must correspond to a
real route in [ez_comfy/api/routes.py](../ez_comfy/api/routes.py). Verified by a one-shot grep at review
time, not by an automated test (it's documentation).

### 5.4 Acceptance Criteria

- [ ] `grep -n "v1/ui\|v1/hardware\|v1/catalog" SPEC.md` returns no matches
  (other than in a "renamed from..." migration note, if added).
- [ ] Architecture diagram at [SPEC.md:19-61](../SPEC.md) lists only real endpoints.
- [ ] Section "Web UI (`GET /v1/ui`)" retitled to "Web UI (`GET /`)".

### 5.5 Test Plan

Documentation-only change; no test code. Reviewer must spot-check the
listed grep returns clean.

### 5.6 Non-Goals

- Renaming any actual route. The current paths stay.
- Adding redirect routes from old paths. They were never shipped.

---

## Work Item 6 — Intent Detection Edge Cases

### 6.1 Goal

[tests/unit/test_intent.py](../tests/unit/test_intent.py) covers the basic intents but not the ambiguous
cases that motivated commit `9436eb2` ("Fix image rendering and intent
detection false positives"). Pin the fix with parametrized tests so the
classifier can be refactored later without regressing.

### 6.2 Files

- **MODIFIED:** [tests/unit/test_intent.py](../tests/unit/test_intent.py) — add parametrized cases.
- **MODIFIED:** [ez_comfy/planner/](../ez_comfy/planner/) (intent module) — only if a missed case
  reveals a real classification bug. Document any code change in the commit
  body.

### 6.3 Behavior Contract

The classifier under test takes `(prompt: str, has_input_image: bool,
has_mask: bool) -> Intent`. Test cases in section 6.5 are the contract.

### 6.4 Acceptance Criteria

- [ ] All cases in 6.5 pass.
- [ ] If any case fails on first run, the implementer either (a) fixes the
  classifier and notes it in the commit, or (b) escalates to the user
  with the specific case before changing the test expectation.

### 6.5 Test Plan

Parametrized cases (each is `(prompt, has_image, has_mask, expected_intent)`):

| Prompt                                         | img | mask | Expected   |
|------------------------------------------------|-----|------|------------|
| `"a cyberpunk city"`                           |  N  |  N   | `txt2img`  |
| `"upscale and enhance this"`                   |  Y  |  N   | `upscale`  |
| `"upscale and enhance this"`                   |  N  |  N   | `txt2img`  |
| `"animate a video of a dog"`                   |  N  |  N   | `video`    |
| `"a video game character portrait"`            |  N  |  N   | `txt2img`  |
| `"remove the person in the masked area"`       |  Y  |  Y   | `inpaint`  |
| `"replace the sky"`                            |  Y  |  N   | `img2img`  |
| `"a song about robots"`                        |  N  |  N   | `audio`    |
| `"a portrait, sound of waves in background"`   |  N  |  N   | `txt2img`  |
| `"high-resolution detailed render"`            |  N  |  N   | `txt2img`  |

### 6.6 Non-Goals

- Rewriting the intent classifier. Only test, plus minimal fix if a case
  exposes a true bug.

---

## 7. Out of Scope (Deferred)

The following audit findings are deferred. Listed here so they aren't
re-discovered in the next audit:

- **Dependency upper bounds in pyproject.toml.** Defer until a real upgrade
  break occurs.
- **Web UI Playwright tests.** The inline HTML/JS surface is small and
  changes rarely; the cost/benefit doesn't pencil today.
- **Custom-named-checkpoint classifier fallback.** Edge case; no user has
  hit it.
- **Hardware-tier config profiles.** Useful but premature without user
  signal.

---

## 8. Implementation Order

If the implementing agent works items sequentially, this order minimizes
merge risk:

1. **Work Item 5** (SPEC.md doc fix — no code, lowest risk warmup).
2. **Work Item 3** (VRAM tests — small, isolated, builds confidence).
3. **Work Item 6** (intent tests — small, isolated).
4. **Work Item 1** (auto-detect — touches config, CLI, README).
5. **Work Item 4** (engine integration tests — depends on stable VRAM
   contract from #3).
6. **Work Item 2** (catalog refresh — largest content change, easiest to
   review last).

Each item is one commit (or PR). No item is a prerequisite for any other,
so they can also be parallelized across agents if desired.
