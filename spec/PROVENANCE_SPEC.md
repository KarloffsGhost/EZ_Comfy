# EZ_Comfy Provenance System — Implementation Spec

> **Goal:** Every automated decision EZ_Comfy makes (model, workflow, VRAM,
> parameters) must be explicitly traceable so that the handoff back to
> ComfyUI's node graph stays trustworthy. Users should never wonder "why did
> it pick that model?" or "where did 6 steps come from?"

---

## 1. Design Principles

1. **Provenance is structural, not reconstructed.** Decisions are recorded at
   the moment they happen inside the planner, not derived after the fact.
2. **It travels with the workflow.** Exported ComfyUI JSON includes a Note
   node so provenance survives import into the canvas.
3. **Rejected alternatives are explicit.** Users see what was passed over and
   why — not just what was chosen.
4. **Three granularity tiers.** Casual users see a summary panel, power users
   get full JSON, CI pipelines get sidecar files.

---

## 2. Data Model

### File: `ez_comfy/planner/provenance.py` (NEW)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Alternative:
    """A candidate that was considered but not selected."""
    value: str                # display-friendly value (model name, recipe id, etc.)
    rejected_reason: str      # human-readable: "requires 12GB VRAM, only 8GB available"


@dataclass
class Decision:
    """One automated decision made during planning."""
    parameter: str            # "checkpoint", "recipe", "steps", "width", "sampler", "seed", etc.
    chosen_value: Any         # the value that was selected
    source: str               # "user", "vram_constraint", "capability_fallback",
                              # "model_catalog", "family_profile", "recipe",
                              # "resolution_bucket", "random", "prompt_keyword"
    reason: str               # human-readable explanation of WHY this value was chosen
    alternatives: list[Alternative] = field(default_factory=list)


@dataclass
class ProvenanceRecord:
    """Complete audit trail for one planning run."""
    decisions: list[Decision] = field(default_factory=list)
    ez_comfy_version: str = ""

    # Hardware context at decision time
    gpu_name: str = ""
    vram_available_gb: float = 0.0
    vram_estimated_gb: float = 0.0

    # Resolved identity
    model_id: str = ""               # catalog id or filename
    model_family: str = ""
    recipe_id: str = ""

    def add(self, decision: Decision) -> None:
        """Append a decision to the record."""
        self.decisions.append(decision)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            "ez_comfy_version": self.ez_comfy_version,
            "gpu_name": self.gpu_name,
            "vram_available_gb": self.vram_available_gb,
            "vram_estimated_gb": self.vram_estimated_gb,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "recipe_id": self.recipe_id,
            "decisions": [
                {
                    "parameter": d.parameter,
                    "chosen_value": _safe_value(d.chosen_value),
                    "source": d.source,
                    "reason": d.reason,
                    "alternatives": [
                        {"value": a.value, "rejected_reason": a.rejected_reason}
                        for a in d.alternatives
                    ],
                }
                for d in self.decisions
            ],
        }

    def to_human_readable(self) -> str:
        """Format as a readable multi-line string for ComfyUI Note nodes."""
        lines = [
            "=== EZ Comfy Provenance ===",
            f"GPU: {self.gpu_name} ({self.vram_available_gb}GB VRAM)",
            f"Est. VRAM usage: {self.vram_estimated_gb}GB",
            "",
        ]
        for d in self.decisions:
            source_tag = f"[{d.source}]"
            lines.append(f"{d.parameter}: {_safe_value(d.chosen_value)}  {source_tag}")
            lines.append(f"  why: {d.reason}")
            for alt in d.alternatives:
                lines.append(f"  rejected: {alt.value} — {alt.rejected_reason}")
        return "\n".join(lines)


def _safe_value(v: Any) -> str:
    """Convert any value to a display-safe string."""
    if isinstance(v, float):
        return f"{v:.2f}" if v != int(v) else str(int(v))
    return str(v)
```

### Constraints

- All fields use stdlib types only (dataclasses, not Pydantic) — matches
  existing planner convention.
- `Alternative` captures the **rejected candidate** and **why**, not scoring
  internals. Keep reasons user-facing.
- `_safe_value()` prevents float formatting noise (e.g. `1.5` not `1.500000`).

---

## 3. Integration Points

### 3.1 Attach ProvenanceRecord to GenerationPlan

**File:** `ez_comfy/planner/planner.py`

Add field to `GenerationPlan`:

```python
provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
```

The `plan_generation()` function builds the record incrementally as it makes
decisions. Each numbered step in the function corresponds to one or more
`Decision` entries.

### 3.2 Decision Recording Locations

Each decision must be recorded at the exact code location where the choice is
made. Here is the mapping:

#### 3.2.1 Intent Detection (step 1)

**Where:** `plan_generation()`, after `detect_intent()` or
`intent_override` is applied.

```
Decision(
    parameter="intent",
    chosen_value=intent.value,
    source="user" if request.intent_override else "prompt_keyword",
    reason="User override" if request.intent_override
           else f"Detected from prompt keywords",
    alternatives=[]  # intent detection is single-pass, no alternatives
)
```

#### 3.2.2 Model Selection (step 3)

**Where:** `_select_checkpoint()` — this function needs to be modified to
return rejection reasons alongside the selected checkpoint.

Current signature:
```python
def _select_checkpoint(...) -> tuple[str, ModelCatalogEntry | None]
```

New signature:
```python
def _select_checkpoint(...) -> tuple[str, ModelCatalogEntry | None, list[Decision]]
```

The function must emit **one** `Decision` with:
- `parameter="checkpoint"`
- `source=` one of: `"user"`, `"recommendation"`, `"fallback"`
- `reason=` human-readable (e.g., "Top-scored installed model for
  photorealism intent")
- `alternatives=` list of `Alternative` for every candidate that was
  passed over, with rejection reasons:
  - `"not installed"` — catalog entry not found in inventory
  - `f"requires {vram_min}GB VRAM, only {available}GB available"` — VRAM
    constraint
  - `"lower relevance score ({score})"` — scored lower

**Data source for alternatives:** The `recommendations` list from
`recommend_models()` already contains `installed`, `fits_vram`,
`match_reasons`, and `warnings` — these feed directly into `Alternative`
entries.

#### 3.2.3 Recipe Selection (step 5)

**Where:** `_select_recipe()` — must emit a `Decision`.

```
Decision(
    parameter="recipe",
    chosen_value=recipe.id,
    source="user" if request.recipe_override
           else "capability_fallback" if any_capability_filtered
           else "prompt_keyword" if keyword_matched
           else "default",
    reason=<why this recipe won>,
    alternatives=[
        Alternative(r.id, "requires {caps} not installed")
        for r in capability_filtered_recipes
    ]
)
```

The key scenario: user's prompt matches `photo_realism_v1` but a required
capability is missing → falls back to `txt2img_basic`. The `Decision` must
capture this fallback chain.

**Implementation note:** `select_recipe()` in `workflows/recipes.py`
currently does not return which recipes were filtered out or why. Modify it
to return that information:

New return type:
```python
def select_recipe(...) -> tuple[Recipe, list[tuple[Recipe, str]]]
```

Where the second element is a list of `(rejected_recipe, reason)` tuples.

#### 3.2.4 Parameter Resolution (step 7)

**Where:** `resolve_params()` in `planner/param_resolver.py`.

Currently, `ResolvedParams.sources` is a `dict[str, str]` mapping param name
to source label (e.g., `{"steps": "model_catalog"}`). This stays — it's
already useful. But `plan_generation()` must also emit `Decision` objects for
the **non-obvious** resolved params.

Emit a `Decision` for each of these parameters:
- `steps`
- `cfg_scale` (display as `cfg`)
- `sampler`
- `scheduler`
- `width` / `height` (combined as one `"resolution"` decision)
- `seed`

The `reason` field should reference the source layer:
- `"user"` → `"User explicitly set {param}={value}"`
- `"recipe"` → `"Recipe '{recipe_id}' overrides {param} to {value}"`
- `"model_catalog"` → `"Model catalog entry for {model_name} specifies {param}={value}"`
- `"family_profile"` → `"Default for {family} family"`
- `"random"` → `"Random seed (no user seed specified)"`
- `"resolution_bucket"` → `"Snapped from {original}x{original} to nearest {family} bucket"`

**No alternatives needed for params** — the priority chain is deterministic
and not a ranking. The `source` and `reason` are sufficient.

#### 3.2.5 VRAM Estimation (step 8)

**Where:** `plan_generation()`, after `_estimate_vram()`.

This is not a decision per se but context. Record it on the
`ProvenanceRecord` directly:

```python
provenance.vram_available_gb = hardware.gpu_vram_gb
provenance.vram_estimated_gb = vram_est
provenance.gpu_name = hardware.gpu_name
```

#### 3.2.6 Prompt Adaptation

**Where:** `plan_generation()`, after `adapt_prompt()`.

Emit a `Decision` only when the prompt was materially changed:

```
Decision(
    parameter="prompt_adaptation",
    chosen_value="<summary of changes>",
    source="family_syntax",
    reason="Applied {family}-specific prompt syntax: {changes_description}",
    alternatives=[]
)
```

Changes to summarize:
- Quality prefix/suffix added (e.g., Pony score tags)
- Emphasis stripped (Flux)
- Style preset tokens injected
- Domain pack tokens injected
- Auto-negative prompt generated

Only emit this decision if at least one change was made. If the prompt
passes through unchanged, skip it.

**Implementation:** `adapt_prompt()` currently returns
`(adapted_prompt, adapted_negative)`. Add a third return value:

```python
def adapt_prompt(...) -> tuple[str, str, list[str]]
```

Where the third element is a list of human-readable change descriptions
(e.g., `["Added Pony score prefix", "Generated SDXL default negative"]`).
Empty list if no changes.

---

## 4. Workflow Export with Provenance

### 4.1 Note Node Injection

**File:** `ez_comfy/workflows/composer.py`

Add function:

```python
def compose_annotated_workflow(plan: GenerationPlan) -> dict:
    """Compose workflow and inject a Note node with provenance."""
    workflow = compose_workflow(plan)

    # Find a free node ID (max existing + 1)
    max_id = max((int(k) for k in workflow if k.isdigit()), default=0)
    note_id = str(max_id + 1)

    workflow[note_id] = {
        "class_type": "Note",
        "inputs": {
            "text": plan.provenance.to_human_readable()
        },
        "_meta": {"title": "EZ Comfy Provenance"}
    }
    return workflow
```

**Why a Note node:** ComfyUI's built-in `Note` node is always available (no
custom nodes needed). It renders as a sticky note on the canvas. When a user
imports the exported workflow, they immediately see what was automated.

### 4.2 Export Endpoint Changes

**File:** `ez_comfy/api/routes.py`

Modify `POST /v1/plan/workflow`:

```python
@router.post("/v1/plan/workflow")
async def export_workflow(
    request: Request,
    body: PlanRequest,
    provenance: str = Query("summary", enum=["summary", "full", "none"]),
) -> Response:
```

| `provenance` value | Behavior |
|---|---|
| `"summary"` (default) | Inject Note node with `to_human_readable()` |
| `"full"` | Inject Note node AND add top-level `"_ez_comfy_provenance"` key with full `to_dict()` |
| `"none"` | Raw workflow dict, no provenance (for users who want clean export) |

Response format stays the same (JSON download with Content-Disposition).

### 4.3 New Endpoint: Provenance for Completed Generations

**File:** `ez_comfy/api/routes.py`

```
GET /v1/history/{generation_id}/provenance
```

Returns `ProvenanceRecord.to_dict()` for a completed generation. Reads from
the sidecar JSON (see Section 5).

Returns 404 if no sidecar found for that generation ID.

---

## 5. Sidecar Metadata File

### 5.1 When Written

**File:** `ez_comfy/engine.py`, in `_run_generation()`, after outputs are
extracted and before returning `GenerationResult`.

### 5.2 File Location

```
{comfyui_output_dir}/ez_comfy_meta/{prompt_id}.json
```

Use a subdirectory to avoid cluttering ComfyUI's output folder. Create the
directory on first write.

### 5.3 Schema

```json
{
    "schema_version": "1.0",
    "prompt_id": "<comfyui prompt id>",
    "generated_at": "<ISO 8601 timestamp>",
    "duration_seconds": 12.3,
    "provenance": { "<ProvenanceRecord.to_dict() output>" },
    "plan_summary": {
        "intent": "txt2img",
        "recipe": "photo_realism_v1",
        "checkpoint": "realvisxlV50_lightning.safetensors",
        "family": "sdxl_lightning",
        "prompt": "<adapted prompt>",
        "original_prompt": "<user's original prompt>",
        "negative_prompt": "<adapted negative>",
        "width": 1024,
        "height": 1024,
        "steps": 6,
        "cfg": 1.5,
        "sampler": "euler",
        "scheduler": "sgm_uniform",
        "seed": 8291047,
        "loras": [],
        "vae_override": null
    },
    "outputs": [
        {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
    ]
}
```

### 5.4 Implementation

```python
async def _write_sidecar(
    self,
    plan: GenerationPlan,
    prompt_id: str,
    outputs: list[dict],
    duration: float,
) -> None:
    """Write provenance sidecar JSON alongside generation outputs."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    meta_dir = Path(self._settings.comfyui.output_dir) / "ez_comfy_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    sidecar = {
        "schema_version": "1.0",
        "prompt_id": prompt_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 2),
        "provenance": plan.provenance.to_dict(),
        "plan_summary": plan.summary(),
        "outputs": outputs,
    }

    path = meta_dir / f"{prompt_id}.json"
    path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
```

**Config requirement:** `Settings` must expose `comfyui.output_dir`. If not
already present, add it to `config/schema.py` with default
`"output"` (ComfyUI's default output directory).

### 5.5 Settings Addition

**File:** `ez_comfy/config/schema.py`

Add to `ComfyUIConfig`:

```python
output_dir: str = "output"  # ComfyUI output directory (for sidecar metadata)
```

---

## 6. UI Provenance Panel

### 6.1 Location

**File:** `ez_comfy/api/routes.py`, in the inline HTML/JS UI.

### 6.2 Behavior

After a generation completes (or after a plan-only request), render a
collapsible `<details>` element below the result:

```html
<details>
  <summary>What EZ Comfy decided for you</summary>
  <table>
    <tr><td>Model</td><td>RealVisXL V5.0 Lightning (6.5GB)</td>
        <td class="source">[recommendation]</td></tr>
    <tr><td>Recipe</td><td>photo_realism_v1</td>
        <td class="source">[prompt_keyword]</td></tr>
    <tr><td>Steps</td><td>6</td>
        <td class="source">[model_catalog]</td></tr>
    <!-- ... -->
  </table>
  <div class="rejected" id="rejected-toggle">
    <h4>Alternatives considered</h4>
    <ul>
      <li>Juggernaut XL v9 — not installed</li>
      <li>DreamShaper XL — lower relevance score</li>
    </ul>
  </div>
</details>
```

### 6.3 Data Source

The `/v1/plan` endpoint already returns `plan.summary()`. Extend `summary()`
to include `provenance: plan.provenance.to_dict()`. The UI JavaScript reads
this from the plan response and renders the table.

---

## 7. Changes by File (Implementation Checklist)

| # | File | Change | Type |
|---|------|--------|------|
| 1 | `ez_comfy/planner/provenance.py` | New file. `Alternative`, `Decision`, `ProvenanceRecord` dataclasses | NEW |
| 2 | `ez_comfy/planner/planner.py` | Add `provenance: ProvenanceRecord` field to `GenerationPlan`. Build record in `plan_generation()`. Modify `_select_checkpoint()` return type to include decisions. | MODIFY |
| 3 | `ez_comfy/workflows/recipes.py` | Modify `select_recipe()` to return `tuple[Recipe, list[tuple[Recipe, str]]]` — rejected recipes with reasons. | MODIFY |
| 4 | `ez_comfy/planner/prompt_adapter.py` | Modify `adapt_prompt()` to return `tuple[str, str, list[str]]` — third element is list of change descriptions. | MODIFY |
| 5 | `ez_comfy/workflows/composer.py` | Add `compose_annotated_workflow()`. | MODIFY |
| 6 | `ez_comfy/api/routes.py` | Modify `POST /v1/plan/workflow` with `provenance` query param. Add `GET /v1/history/{id}/provenance`. Add provenance panel to inline UI. Update `summary()` response to include provenance. | MODIFY |
| 7 | `ez_comfy/engine.py` | Call `_write_sidecar()` after generation. Add `_write_sidecar()` method. | MODIFY |
| 8 | `ez_comfy/config/schema.py` | Add `output_dir` to `ComfyUIConfig`. | MODIFY |
| 9 | `tests/test_provenance.py` | Unit tests for `ProvenanceRecord`, `Decision`, serialization. | NEW |
| 10 | `tests/test_planner.py` | Update existing planner tests to verify provenance is populated. | MODIFY |
| 11 | `tests/test_composer.py` | Test `compose_annotated_workflow()` injects Note node. | MODIFY |
| 12 | `tests/test_recipes.py` | Update to handle new `select_recipe()` return type. | MODIFY |
| 13 | `tests/test_prompt_adapter.py` | Update to handle new `adapt_prompt()` return type. | MODIFY |

---

## 8. Implementation Order

Execute in this order. Each step must pass tests before proceeding.

### Step 1: Data Model
- Create `ez_comfy/planner/provenance.py`
- Write `tests/test_provenance.py`
  - Test `Decision` and `Alternative` construction
  - Test `ProvenanceRecord.to_dict()` serialization
  - Test `ProvenanceRecord.to_human_readable()` formatting
  - Test `_safe_value()` edge cases (floats, ints, strings)

### Step 2: Prompt Adapter Return Type
- Modify `adapt_prompt()` to return `tuple[str, str, list[str]]`
- Update all callers (only `plan_generation()` in `planner.py`)
- Update `tests/test_prompt_adapter.py` to unpack the third element
- All existing prompt adapter tests must still pass

### Step 3: Recipe Selector Return Type
- Modify `select_recipe()` to return `tuple[Recipe, list[tuple[Recipe, str]]]`
- Update all callers (`_select_recipe()` in `planner.py`)
- Update `tests/test_recipes.py` to unpack the second element
- All existing recipe tests must still pass

### Step 4: Planner Integration
- Add `provenance` field to `GenerationPlan`
- Build `ProvenanceRecord` inside `plan_generation()`:
  - After intent detection → add intent Decision
  - After `_select_checkpoint()` → add checkpoint Decision with alternatives
  - After `_select_recipe()` → add recipe Decision with alternatives
  - After `adapt_prompt()` → add prompt_adaptation Decision (if changes made)
  - After `resolve_params()` → add param Decisions
  - After VRAM estimation → set hardware context fields
- Modify `_select_checkpoint()` to collect alternatives from recommendations
- Update `summary()` to include `provenance.to_dict()`
- Update `tests/test_planner.py`

### Step 5: Annotated Workflow Export
- Add `compose_annotated_workflow()` to `composer.py`
- Test that Note node is injected with correct content
- Test that node ID doesn't collide with existing nodes

### Step 6: API Changes
- Modify `POST /v1/plan/workflow` with `provenance` query param
- Add `GET /v1/history/{prompt_id}/provenance` endpoint
- Update inline UI with provenance panel
- Add `output_dir` to `ComfyUIConfig`

### Step 7: Sidecar Writer
- Add `_write_sidecar()` to `GenerationEngine`
- Call it in `_run_generation()` after outputs are extracted
- Test sidecar JSON schema and content

### Step 8: Run Full Test Suite
- `pytest tests/ -v`
- All tests green before considering this complete

---

## 9. Test Requirements

### 9.1 `tests/test_provenance.py` (NEW)

```python
def test_decision_to_dict():
    """Decision serializes all fields including alternatives."""

def test_provenance_record_to_dict():
    """Full record with multiple decisions serializes correctly."""

def test_provenance_to_human_readable():
    """Human-readable output includes all decisions and alternatives."""

def test_safe_value_float():
    """1.5 → '1.5', 1.0 → '1', 7.0 → '7'."""

def test_safe_value_int():
    """42 → '42'."""

def test_provenance_add():
    """ProvenanceRecord.add() appends to decisions list."""

def test_empty_provenance():
    """Empty record serializes without error."""
```

### 9.2 Planner Integration Tests

```python
def test_plan_includes_provenance():
    """GenerationPlan.provenance is populated after plan_generation()."""

def test_provenance_has_checkpoint_decision():
    """Provenance includes a 'checkpoint' decision with source and reason."""

def test_provenance_has_recipe_decision():
    """Provenance includes a 'recipe' decision."""

def test_provenance_has_param_decisions():
    """Provenance includes decisions for steps, cfg, sampler, resolution, seed."""

def test_provenance_checkpoint_alternatives():
    """When multiple models are considered, alternatives list is populated."""

def test_provenance_recipe_fallback():
    """When a recipe is filtered by capability, the fallback is recorded."""

def test_provenance_prompt_adaptation():
    """When prompt is modified (Pony tags, Flux stripping), decision is recorded."""

def test_provenance_no_prompt_decision_when_unchanged():
    """When prompt passes through unchanged, no prompt_adaptation decision."""

def test_provenance_vram_context():
    """Provenance records GPU name, VRAM available, and VRAM estimated."""
```

### 9.3 Composer Tests

```python
def test_compose_annotated_workflow_injects_note():
    """Annotated workflow has a Note node with provenance text."""

def test_note_node_id_no_collision():
    """Note node ID is max(existing) + 1."""

def test_note_node_class_type():
    """Note node uses class_type='Note'."""
```

### 9.4 Updated Existing Tests

- `test_prompt_adapter.py`: All existing tests unpack 3-tuple return
- `test_recipes.py`: All existing tests unpack 2-tuple return
- No test should break — only return types change, existing values stay same

---

## 10. Backward Compatibility

- `GenerationPlan.param_sources` (the existing flat dict) stays unchanged.
  It's a quick-access field for the UI. `provenance` is the rich version.
- `ResolvedParams.sources` stays unchanged. Provenance reads from it but
  doesn't replace it.
- The `POST /v1/plan/workflow` endpoint default behavior changes (now
  includes a Note node). Users who want the old behavior pass
  `?provenance=none`.
- No existing test should break. Return types gain additional elements but
  existing tuple unpacking is updated as part of each step.

---

## 11. What This Does NOT Cover

- **LLM-based provenance explanations.** All reasons are template strings,
  not LLM-generated. This keeps provenance deterministic and fast.
- **Provenance diff between runs.** Comparing two provenance records is out
  of scope for v1. The sidecar files make this possible later.
- **Provenance in image metadata (EXIF/PNG chunks).** Possible future
  extension. Sidecar JSON is sufficient for v1.
- **UI for browsing generation history.** The `GET /v1/history/{id}/provenance`
  endpoint exists but a full history browser UI is out of scope.
