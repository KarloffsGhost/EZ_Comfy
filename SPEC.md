# EZ_Comfy — Spec-Driven ComfyUI Orchestrator

## 1. Vision

EZ_Comfy is a standalone application that takes a natural-language prompt from a user and produces the optimal ComfyUI workflow to fulfill it — automatically selecting the right pipeline, model, sampler, resolution, and parameters based on what the user's hardware can realistically run.

It solves three problems most ComfyUI users face:
1. **Which model?** — recommends the best checkpoint for the task, shows download instructions if missing
2. **Which nodes and how to wire them?** — selects from a library of proven workflow recipes
3. **Which settings?** — applies model-specific, recipe-specific optimal parameters

The user says: *"A cyberpunk city at sunset, cinematic lighting"*
EZ_Comfy figures out: *"You have an RTX 5070 Ti with 16GB VRAM. For cinematic photorealism, Juggernaut XL is ideal — you don't have it installed, but RealVisXL Lightning is installed and fits. Using txt2img with hi-res fix recipe, 1024x1024, euler, 6 steps, CFG 1.5. Auto-adding negative prompt for SDXL photorealism."*

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      FastAPI Server                       │
│  GET /  /v1/generate  /v1/health  /v1/inventory           │
│  /v1/recommendations  /v1/queue  /v1/compare              │
└────────────┬─────────────────────────────────────────────┘
             │
     ┌───────▼────────┐
     │  Prompt Adapter │  ← Rewrites prompt for model-specific syntax
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │   LLM Planner  │  ← Analyzes prompt, selects pipeline + style hints
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │ Hardware Probe  │  ← VRAM, RAM, installed models, available nodes
     └───────┬────────┘
             │
     ┌───────▼─────────┐
     │  Model Catalog   │  ← Curated knowledge base + installed inventory
     └───────┬─────────┘
             │
     ┌───────▼────────┐
     │ Recipe Selector │  ← Matches intent to workflow recipe
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │ Param Resolver  │  ← Model settings > family defaults, resolution bucketing
     └───────┬────────┘
             │
     ┌───────▼─────────┐
     │ Workflow Composer│  ← Builds ComfyUI node graph dict from recipe
     └───────┬─────────┘
             │
     ┌───────▼────────┐
     │  ComfyUI Client │  ← Submit, WebSocket progress, download
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │  Output Handler │  ← Save to disk, sidecar metadata, thumbnails
     └────────────────┘
```

---

## 3. Core Components

### 3.1 Hardware Probe (`ez_comfy/hardware/`)

**Purpose:** Detect available GPU, VRAM, RAM, and query ComfyUI for installed models, LoRAs, VAEs, custom nodes.

**Files:**
- `probe.py` — `HardwareProfile` dataclass + `probe_hardware()` function
- `comfyui_inventory.py` — `ComfyUIInventory` class that queries ComfyUI API

**HardwareProfile:**
```python
@dataclass
class HardwareProfile:
    gpu_name: str           # "NVIDIA GeForce RTX 5070 Ti"
    gpu_vram_gb: float      # 15.9
    system_ram_gb: float    # 127.0
    cuda_version: str       # "12.4"
    platform: str           # "win32"
```

**ComfyUIInventory:**
```python
@dataclass
class ModelInfo:
    filename: str           # "realvisxlV50_lightning.safetensors"
    size_bytes: int
    family: str             # classified family string
    variant: str | None     # "lightning", "turbo", etc

@dataclass
class LoRAInfo:
    filename: str
    size_bytes: int
    compatible_families: list[str]   # inferred from file size + name patterns

@dataclass
class ComfyUIInventory:
    checkpoints: list[ModelInfo]
    loras: list[LoRAInfo]
    vaes: list[str]
    upscale_models: list[str]        # ESRGAN etc
    clip_models: list[str]           # T5, CLIP-L, CLIP-G etc
    controlnet_models: list[str]
    discovered_class_types: set[str]  # all class_types from GET /object_info
    samplers: list[str]              # available sampler names
    schedulers: list[str]            # available scheduler names
```

**How it works:**
- GPU: `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
- RAM: `psutil.virtual_memory().total`
- ComfyUI models: `GET /object_info/{node_class}` for each relevant node type
- Checkpoints: `GET /object_info/CheckpointLoaderSimple` → `.input.required.ckpt_name[0]`
- LoRAs: `GET /object_info/LoraLoader` → `.input.required.lora_name[0]`
- VAEs: `GET /object_info/VAELoader` → `.input.required.vae_name[0]`
- Upscale: `GET /object_info/UpscaleModelLoader` → `.input.required.model_name[0]`
- ControlNets: `GET /object_info/ControlNetLoader` → `.input.required.control_net_name[0]`
- Samplers: `GET /object_info/KSampler` → `.input.required.sampler_name[0]`
- Schedulers: `GET /object_info/KSampler` → `.input.required.scheduler[0]`
- Custom nodes: `GET /object_info` (full) — returns every registered node class_type

**Node Capability Detection:**

Recipes require specific *capabilities*, not package names (package names aren't exposed by ComfyUI's API). The inventory stores discovered node class_types, and a capability map translates recipe requirements to class_type checks:

```python
# Maps capability names (used by recipes) → ComfyUI class_types that prove the capability exists
NODE_CAPABILITY_MAP: dict[str, list[str]] = {
    "adetailer":          ["ADetailer", "ADetailerPipe"],
    "ipadapter":          ["IPAdapterApply", "IPAdapterModelLoader", "IPAdapter"],
    "controlnet":         ["ControlNetLoader", "ControlNetApply"],
    "regional_prompting": ["RegionalPrompt", "RegionalSampler"],
    "animatediff":        ["AnimateDiffLoader", "AnimateDiffSampler", "ADE_AnimateDiffLoaderWithContext"],
    "face_restore":       ["FaceRestoreWithModel", "FaceRestoreCFWithModel"],
    "upscale_model":      ["UpscaleModelLoader", "ImageUpscaleWithModel"],
    "svd":                ["ImageOnlyCheckpointLoader", "SVD_img2vid_Conditioning"],
    "stable_audio":       ["EmptyLatentAudio", "VAEDecodeAudio"],
}

def has_capability(capability: str, discovered_class_types: set[str]) -> bool:
    """Returns True if ANY of the required class_types for this capability are present."""
    required = NODE_CAPABILITY_MAP.get(capability, [])
    return any(ct in discovered_class_types for ct in required)
```

Recipes declare `required_capabilities: list[str]` (e.g., `["adetailer"]`), NOT package names. The `ComfyUIInventory` stores `discovered_class_types: set[str]` from the full `/object_info` dump. Recipe selection filters by `has_capability()`.

If a new custom node package uses different class_type names, only `NODE_CAPABILITY_MAP` needs updating — no recipe changes.

**LoRA family inference (best-effort with warning):**
LoRAs don't declare their base model. Inference is heuristic — the UI should display a warning icon on LoRAs marked ambiguous, with tooltip: *"Compatibility is estimated from filename/size. This LoRA may not work with the selected model."*
- File size > 300MB → likely SD 1.5 full-rank
- File size 50-300MB → check filename for `xl`, `sdxl`, `pony` → SDXL family; else SD 1.5
- File size < 50MB → LoRA/LoKr/LoHa — check filename patterns
- Filenames containing `xl`, `sdxl`, `pony` → `["sdxl"]`
- Filenames containing `flux` → `["flux"]`
- No clear signal → `["sd15", "sdxl"]` (mark as `ambiguous=True`, show warning in UI)

---

### 3.2 Model Catalog (`ez_comfy/models/`)

**Purpose:** Curated knowledge base of ~50 recommended models with task-matching intelligence, plus runtime classification of any installed checkpoint.

**Files:**
- `catalog.py` — `ModelCatalogEntry` dataclass + `MODEL_CATALOG` list + `recommend_models()` function
- `profiles.py` — `ModelProfile` dataclass (per-family defaults) + `PROFILES` registry
- `classifier.py` — `classify_checkpoint()` function for unknown/installed models

#### 3.2.1 Curated Model Catalog (`catalog.py`)

Each entry encodes what community experience has established about a model:

```python
@dataclass
class ModelCatalogEntry:
    id: str                             # unique slug: "juggernaut_xl_v9"
    name: str                           # display name: "Juggernaut XL v9"
    family: str                         # sd15, sdxl, flux, sd3, cascade, svd, stable_audio
    variant: str | None                 # lightning, turbo, lcm, None
    source: str                         # HuggingFace repo or Civitai URL
    filename: str                       # expected checkpoint filename
    size_gb: float                      # download size
    vram_min_gb: float                  # minimum VRAM to run at native res
    vram_recommended_gb: float          # comfortable VRAM (no offloading)

    # Task matching
    strengths: list[str]                # ["photorealism", "portraits", "cinematic"]
    weaknesses: list[str]               # ["anime", "text_rendering"]
    best_for: list[str]                 # natural language: ["product photos", "editorial"]
    style_tags: list[str]               # ["realistic", "cinematic", "editorial", "dark"]

    # Model-specific optimal settings (override family defaults)
    settings: ModelSettings

    # Prompt syntax requirements
    prompt_syntax: PromptSyntax

    # Recommended companions
    recommended_loras: list[LoRARecommendation]
    recommended_vae: str | None         # None = use built-in

    # Dependencies
    requires_clip: str | None           # e.g. "t5xxl_fp16.safetensors" for Flux
    required_capabilities: list[str]    # capability keys from NODE_CAPABILITY_MAP

    # Download info
    download_command: str               # "huggingface-cli download ..."
    license: str                        # "apache-2.0", "creativeml-openrail-m", etc
    gated: bool                         # requires HF login
```

```python
@dataclass
class ModelSettings:
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    clip_skip: int                      # 1 = default, 2 = anime models, etc
    denoise_default: float              # for img2img (0.0-1.0)

@dataclass
class PromptSyntax:
    emphasis_format: str                # "weighted" = (word:1.3), "none" = Flux
    quality_prefix: str | None          # "score_9, score_8_up, " for Pony
    quality_suffix: str | None          # ", masterpiece, best quality" for some anime models
    negative_required: bool             # False for Flux (ignores negatives)
    default_negative: str               # model-family-specific negative prompt
    supports_break: bool                # True if BREAK keyword works (SDXL)

@dataclass
class LoRARecommendation:
    name: str                           # "Detail Tweaker XL"
    filename: str                       # "detail_tweaker_xl.safetensors"
    source: str                         # download URL
    strength: float                     # recommended strength
    reason: str                         # "Sharpens fine detail without artifacts"
```

**v1 Catalog (curated core — ~15 models, expand after usage telemetry):**

Ship with one strong pick per category. The catalog is a Python data structure — adding entries later is a one-line change + test.

**Photorealism (3):**
- RealVisXL V5.0 Lightning — fast photorealism (SDXL Lightning) ← owner has this installed
- Juggernaut XL v9 — editorial/cinematic photorealism (SDXL)
- Realistic Vision V5.1 — classic photorealism (SD 1.5)

**Stylized/Artistic (2):**
- DreamShaper XL Lightning — versatile stylized (SDXL Lightning)
- DreamShaper 8 — versatile all-rounder (SD 1.5) ← owner has this installed

**Anime/Illustration (2):**
- Pony Diffusion V6 XL — stylized characters, needs score tags (SDXL)
- Anything V5 — classic anime (SD 1.5)

**Prompt-Adherent/General (3):**
- Flux.1 Dev — best prompt adherence, text rendering (Flux)
- Flux.1 Schnell — fast Flux variant (Flux)
- SDXL Base 1.0 — reference SDXL (SDXL) ← owner has this installed

**Video (1):**
- Stable Video Diffusion XT — img2vid (SVD)

**Audio (1):**
- Stable Audio Open 1.0 — sound effects, ambient (Stable Audio) ← owner has this installed

**Upscaling (2):**
- RealESRGAN x4plus — general purpose 4x
- 4x-UltraSharp — sharp detail upscaler

*Post-v1 expansion targets: architecture models, product photography, more anime options, AnimateDiff models. Add based on what users actually request.*

#### 3.2.2 Model Recommendation Logic (`recommend_models()`)

```python
def recommend_models(
    prompt: str,
    intent: PipelineIntent,
    hardware: HardwareProfile,
    inventory: ComfyUIInventory,
    catalog: list[ModelCatalogEntry],
    prefer_installed: bool = True,
) -> list[ModelRecommendation]:
```

```python
@dataclass
class ModelRecommendation:
    entry: ModelCatalogEntry
    score: float                        # 0-100 composite score
    installed: bool                     # already on disk?
    fits_vram: bool                     # fits in available VRAM?
    match_reasons: list[str]            # ["photorealism matches prompt", "installed"]
    warnings: list[str]                 # ["requires 12GB VRAM, you have 15.9GB"]
```

**Scoring algorithm:**
1. **Filter** by intent compatibility (TXT2IMG → image models, VIDEO → SVD, etc.)
2. **Filter** by VRAM fit (`vram_min_gb <= hardware.gpu_vram_gb`)
3. **Score** each candidate:
   - +30 points: strength match (prompt keywords vs model strengths)
   - +25 points: installed on disk (skip download wait)
   - +20 points: VRAM headroom (more headroom = can do larger batches/resolutions)
   - +15 points: speed variant bonus (if `preferences.prefer_speed` and model is Lightning/Turbo/Schnell)
   - +10 points: style tag match (prompt style keywords vs model style_tags)
4. **Sort** by score descending
5. **Return** top N (default 3)

**Prompt analysis for strength matching:**
Simple keyword extraction — no LLM needed:
- "photo", "realistic", "portrait" → match `photorealism`, `portraits`
- "anime", "manga", "2D" → match `anime`, `illustration`
- "cinematic", "movie", "film" → match `cinematic`, `editorial`
- "fantasy", "dragon", "magic" → match `fantasy`, `concept_art`
- "architecture", "building", "interior" → match `architectural`
- "product", "packaging", "commercial" → match `product_photography`
- "abstract", "surreal", "artistic" → match `artistic`, `painterly`
- "text", "logo", "typography" → match `text_rendering` (→ strongly prefer Flux)

#### 3.2.3 Family Profiles (`profiles.py`)

Fallback defaults when a checkpoint is installed but not in the curated catalog:

```python
@dataclass
class ModelProfile:
    family: str
    native_resolution: tuple[int, int]
    resolution_buckets: list[tuple[int, int]]   # valid aspect ratio pairs
    vram_requirement_gb: float
    default_settings: ModelSettings
    prompt_syntax: PromptSyntax
    supports_img2img: bool
    supports_inpaint: bool
    supports_controlnet: bool
    clip_type: str
    vae_type: str
```

**Resolution buckets (SDXL example):**
```python
SDXL_BUCKETS = [
    (1024, 1024),  # 1:1
    (1152, 896),   # 9:7
    (896, 1152),   # 7:9
    (1216, 832),   # 19:13
    (832, 1216),   # 13:19
    (1344, 768),   # 7:4
    (768, 1344),   # 4:7
    (1536, 640),   # 12:5 (panoramic)
    (640, 1536),   # 5:12 (tall)
]

SD15_BUCKETS = [
    (512, 512),    # 1:1
    (768, 512),    # 3:2
    (512, 768),    # 2:3
    (640, 448),    # ~3:2
    (448, 640),    # ~2:3
    (768, 432),    # 16:9
    (432, 768),    # 9:16
]
```

**`snap_to_bucket(width, height, buckets)`**: Given a desired resolution, returns the nearest valid bucket. This prevents the silent quality degradation that comes from generating at non-trained resolutions.

#### 3.2.4 Classifier (`classifier.py`)

For checkpoints that aren't in the curated catalog (user downloaded something custom):

```python
def classify_checkpoint(filename: str, size_bytes: int | None = None) -> tuple[str, str | None]:
    """Returns (family, variant) based on filename patterns and file size."""
```

Same pattern matching as before, but enhanced with variant detection:
- `lightning` in name → variant = `"lightning"`
- `turbo` in name → variant = `"turbo"`
- `lcm` in name → variant = `"lcm"`
- `hyper` in name → variant = `"hyper"`
- `schnell` in name → variant = `"schnell"`

---

### 3.3 Prompt Adapter (`ez_comfy/planner/prompt_adapter.py`)

**Purpose:** Rewrite the user's prompt to match the selected model's syntax requirements. This is the component that prevents the #1 user mistake: writing the same prompt for every model.

```python
def adapt_prompt(
    user_prompt: str,
    negative_prompt: str,
    syntax: PromptSyntax,
    style_preset: StylePreset | None = None,
) -> tuple[str, str]:
    """Returns (adapted_positive, adapted_negative)."""
```

**What it does:**

1. **Quality prefix/suffix injection:**
   - Pony models: prepend `"score_9, score_8_up, score_7_up, "` to positive prompt
   - Anime SD1.5 models: append `", masterpiece, best quality"` to positive
   - Flux: no modification (Flux is trained on natural language)

2. **Negative prompt generation:**
   - If user left negative blank and `auto_negative_prompt` is enabled:
     - SDXL photorealism: `"worst quality, low quality, normal quality, lowres, watermark, text, signature, blurry, deformed"`
     - SD 1.5 photorealism: `"worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, extra digits, fewer digits, watermark, signature"`
     - Anime models: `"worst quality, low quality, lowres, bad anatomy, bad hands, text, error, ugly, duplicate, morbid"`
     - Pony: `"score_4, score_3, score_2, score_1, worst quality, low quality"`
     - Flux: `""` (Flux ignores negative prompts entirely — don't waste tokens)

3. **Style preset expansion:**
   - If user selects a style preset (e.g., "cinematic"), append model-appropriate tokens:
     - SDXL cinematic: `", cinematic lighting, film grain, color grading, bokeh, shallow depth of field, anamorphic"`
     - SD 1.5 cinematic: `", cinematic, dramatic lighting, film grain, 35mm photograph"`
     - Flux cinematic: `" in cinematic style with dramatic lighting and shallow depth of field"` (natural language)

4. **Emphasis normalization:**
   - If user wrote `(word:1.3)` but model doesn't support emphasis (Flux) → strip to just `word`
   - If user wrote `word+++` (NAI format) → convert to `(word:1.3)` for SDXL/SD1.5

**Style Presets:**

```python
@dataclass
class StylePreset:
    id: str                     # "cinematic", "anime", "photographic", etc
    name: str                   # display name
    positive_tokens: dict[str, str]  # family → tokens to append
    negative_tokens: dict[str, str]  # family → tokens to append to negative
```

**Built-in style presets:**
- `photographic` — natural photography look
- `cinematic` — film/movie aesthetic
- `anime` — Japanese animation style
- `digital_art` — digital illustration
- `fantasy` — fantasy/concept art
- `pixel_art` — retro pixel art style
- `watercolor` — watercolor painting
- `oil_painting` — classical oil painting
- `3d_render` — 3D rendered look
- `comic` — comic book / graphic novel
- `minimalist` — clean, minimal design
- `noir` — dark, moody, black & white
- `cyberpunk` — neon, futuristic, tech
- `portrait` — optimized for face/portrait photos
- `landscape` — optimized for landscapes/scenery
- `product` — clean product photography

---

### 3.4 Workflow Recipes (`ez_comfy/workflows/`)

**Purpose:** Library of proven workflow patterns, each a Python function that builds a ComfyUI node graph. The recipe selector matches user intent + available nodes to the best recipe.

**Files:**
- `recipes.py` — `Recipe` dataclass + `RECIPES` registry + `select_recipe()` function
- `composer.py` — `compose_workflow(plan, recipe)` dispatcher
- `txt2img.py` — txt2img recipe builders
- `img2img.py` — img2img recipe builders
- `inpaint.py` — inpaint recipe builders
- `upscale.py` — upscale recipe builders
- `video.py` — video recipe builders
- `audio.py` — audio recipe builder

#### Recipe Registry

```python
@dataclass
class Recipe:
    id: str                              # "txt2img_basic"
    name: str                            # "Text to Image (Standard)"
    description: str                     # what it does
    intent: PipelineIntent               # which pipeline this serves
    priority: int                        # higher = preferred when multiple match
    when: str                            # human-readable condition
    required_capabilities: list[str]     # capability keys from NODE_CAPABILITY_MAP (empty = built-in only)
    requires_reference_image: bool       # needs uploaded image?
    requires_mask: bool                  # needs mask?
    supports_lora: bool                  # can inject LoRA nodes?
    supports_controlnet: bool
    builder: str                         # function name: "build_txt2img_basic"
    settings_overrides: dict | None      # recipe-specific param overrides
```

**v1 Recipes (10 core recipes — expand after usage telemetry):**

Ship with the recipes that cover built-in nodes only + the most common custom node (ControlNet). Recipes requiring rarer custom nodes (IPAdapter, ADetailer, AnimateDiff, RegionalPrompting) are post-v1 additions.

**txt2img recipes:**
| ID | Name | Priority | Requires | When |
|----|------|----------|----------|------|
| `txt2img_basic` | Standard txt2img | 10 | — | Default text-to-image |
| `txt2img_hires_fix` | Hi-Res Fix | 20 | — | User requests high resolution or detail |

**img2img recipes:**
| ID | Name | Priority | Requires | When |
|----|------|----------|----------|------|
| `img2img_basic` | Standard img2img | 10 | reference image | Modify existing image |
| `img2img_controlnet_canny` | ControlNet Canny | 15 | capability: `controlnet` + ref image | Keep structure, change style |

**inpaint recipes:**
| ID | Name | Priority | Requires | When |
|----|------|----------|----------|------|
| `inpaint_basic` | Standard Inpaint | 10 | image + mask | Replace masked area |

**upscale recipes:**
| ID | Name | Priority | Requires | When |
|----|------|----------|----------|------|
| `upscale_simple` | Model Upscale | 10 | image + capability: `upscale_model` | Fast upscale (no refinement) |
| `upscale_refine` | Upscale + Refine | 20 | image + capability: `upscale_model` + checkpoint | Upscale then img2img refine at low denoise |

**video recipes:**
| ID | Name | Priority | Requires | When |
|----|------|----------|----------|------|
| `video_svd` | SVD img2vid | 10 | capability: `svd` + reference image | Animate a still image |

**audio recipes:**
| ID | Name | Priority | Requires | When |
|----|------|----------|----------|------|
| `audio_stable` | Stable Audio Open | 10 | capability: `stable_audio` + T5 clip | Sound effects, ambient audio |

*Post-v1 recipe expansion targets (add when custom node ecosystem is better understood):*
- `txt2img_adetailer` — auto face fix (capability: `adetailer`)
- `txt2img_regional` — regional prompting (capability: `regional_prompting`)
- `img2img_ipadapter` — IPAdapter style transfer (capability: `ipadapter`)
- `img2img_controlnet_depth` — ControlNet depth (capability: `controlnet`)
- `inpaint_outpaint` — outpainting
- `video_animatediff` — AnimateDiff text-to-video (capability: `animatediff`)

#### Recipe Selection Logic

```python
def select_recipe(
    intent: PipelineIntent,
    prompt: str,
    has_reference_image: bool,
    has_mask: bool,
    inventory: ComfyUIInventory,
    user_recipe_override: str | None = None,
) -> Recipe:
```

1. Filter recipes by `intent`
2. Filter by capability availability (check each `required_capabilities` entry via `has_capability(cap, inventory.discovered_class_types)`)
3. Filter by image/mask requirements
4. Among remaining, pick highest `priority`
5. If user specified `recipe_override`, use that (error if requirements not met)

#### Workflow Node Graphs

Each recipe builder returns a `dict` in ComfyUI's format. Key patterns:

**txt2img_basic:**
```
Node 1: CheckpointLoaderSimple(ckpt_name)
Node 2: CLIPTextEncode(positive) ← clip[1,1]
Node 3: CLIPTextEncode(negative) ← clip[1,1]
Node 4: EmptyLatentImage(width, height, batch_size)
Node 5: KSampler(seed, steps, cfg, sampler, scheduler, denoise=1.0)
         ← model[1,0], positive[2,0], negative[3,0], latent_image[4,0]
Node 6: VAEDecode ← samples[5,0], vae[1,2]
Node 7: SaveImage(filename_prefix="ezcomfy") ← images[6,0]
```

**txt2img_basic with LoRA (injected when plan.loras is non-empty):**
```
Node 1: CheckpointLoaderSimple(ckpt_name)
Node 8: LoraLoader(lora_name, strength_model, strength_clip)
         ← model[1,0], clip[1,1]
Node 2: CLIPTextEncode(positive) ← clip[8,1]    ← NOTE: clip from LoRA output
Node 3: CLIPTextEncode(negative) ← clip[8,1]
Node 4: EmptyLatentImage(width, height, batch_size)
Node 5: KSampler ← model[8,0], ...              ← NOTE: model from LoRA output
Node 6: VAEDecode ← samples[5,0], vae[1,2]
Node 7: SaveImage ← images[6,0]
```

**Multiple LoRAs: chain them.** Node 8 → Node 9 → ... each LoraLoader takes model/clip from previous.

**txt2img_basic with VAE override:**
```
Node 9: VAELoader(vae_name)
Node 6: VAEDecode ← samples[5,0], vae[9,0]      ← VAE from loader, not checkpoint
```

**txt2img_basic with ControlNet:**
```
Node 10: ControlNetLoader(control_net_name)
Node 11: LoadImage(image_path)
Node 12: ControlNetApply(strength)
          ← conditioning[2,0], control_net[10,0], image[11,0]
Node 5: KSampler ← positive[12,0] instead of [2,0]
```

**txt2img_hires_fix (two-pass):**
```
Pass 1 (same as basic but at reduced resolution):
  Nodes 1-7 as above but width/height = native * 0.5-0.7

Pass 2 (upscale + refine):
  Node 10: LatentUpscale(upscale_method="nearest-exact", width=target, height=target)
           ← samples[5,0]
  Node 11: KSampler(steps=10, denoise=0.45)
           ← model[1,0], positive[2,0], negative[3,0], latent_image[10,0]
  Node 12: VAEDecode ← samples[11,0], vae[1,2]
  Node 13: SaveImage ← images[12,0]
```

**img2img_basic:**
```
Node 1: CheckpointLoaderSimple
Node 2: CLIPTextEncode(positive) ← clip[1,1]
Node 3: CLIPTextEncode(negative) ← clip[1,1]
Node 8: LoadImage(reference_image_filename)
Node 9: VAEEncode ← pixels[8,0], vae[1,2]
Node 5: KSampler(denoise=plan.denoise_strength)  ← latent_image[9,0]
Node 6: VAEDecode ← samples[5,0], vae[1,2]
Node 7: SaveImage ← images[6,0]
```

**inpaint_basic:**
```
Node 8: LoadImage(reference_image)
Node 9: LoadImage(mask_image)
Node 10: VAEEncode ← pixels[8,0], vae[1,2]
Node 11: SetLatentNoiseMask ← samples[10,0], mask[9,0]
Node 5: KSampler(denoise=1.0) ← latent_image[11,0]
...
```

**upscale_simple:**
```
Node 1: LoadImage(input_image)
Node 2: UpscaleModelLoader(model_name)
Node 3: ImageUpscaleWithModel ← upscale_model[2,0], image[1,0]
Node 4: SaveImage ← images[3,0]
```

**upscale_refine:**
```
Nodes 1-3: same as upscale_simple
Node 5: ImageScale(width=target, height=target) ← image[3,0]  (resize to exact target)
Node 6: CheckpointLoaderSimple(ckpt_name)
Node 7: VAEEncode ← pixels[5,0], vae[6,2]
Node 8: CLIPTextEncode(positive) ← clip[6,1]
Node 9: CLIPTextEncode(negative) ← clip[6,1]
Node 10: KSampler(steps=12, denoise=0.3) ← model[6,0], latent[7,0]
Node 11: VAEDecode ← samples[10,0], vae[6,2]
Node 12: SaveImage ← images[11,0]
```

**video_svd:**
```
Node 1: ImageOnlyCheckpointLoader(ckpt_name)
Node 2: LoadImage(reference_image)
Node 3: CLIPVisionEncode ← clip_vision[1,1], image[2,0]
Node 4: SVD_img2vid_Conditioning ← clip_vision[3,0], init_image[2,0]
         (width, height, video_frames, fps, motion_bucket_id=127, augmentation_level=0)
Node 5: KSampler ← model[1,0], positive[4,0], negative[4,1], latent[4,2]
Node 6: VAEDecode ← vae[1,2], samples[5,0]
Node 7: SaveAnimatedWEBP(fps) ← images[6,0]
```

**audio_stable:**
Same as MultiModel's proven `_build_audio_workflow()`:
```
Node 1: CheckpointLoaderSimple(stable_audio_checkpoint) → model, CLIP=None, VAE
Node 9: CLIPLoader(type="stable_audio", clip_name=t5_encoder) → CLIP
Node 2: CLIPTextEncode(positive) ← clip[9,0]
Node 3: CLIPTextEncode(negative) ← clip[9,0]
Node 4: ConditioningStableAudio ← pos[2,0], neg[3,0], seconds_start=0, seconds_total=duration
Node 5: EmptyLatentAudio ← seconds=duration
Node 6: KSampler(steps=100, cfg=7.0, sampler=dpmpp_3m_sde, scheduler=exponential)
Node 7: VAEDecodeAudio ← vae[1,2], samples[6,0]
Node 8: SaveAudioMP3(quality="128k") ← audio[7,0]
```

---

### 3.5 Parameter Resolver (`ez_comfy/planner/param_resolver.py`)

**Purpose:** Resolve final generation parameters using a priority chain.

**Priority chain (highest wins):**
1. User explicit overrides (from request)
2. Recipe-specific overrides
3. Model catalog entry settings (if model is in catalog)
4. Family profile defaults
5. Global defaults

```python
def resolve_params(
    request: GenerationRequest,
    recipe: Recipe,
    catalog_entry: ModelCatalogEntry | None,
    family_profile: ModelProfile,
) -> ResolvedParams:
```

```python
@dataclass
class ResolvedParams:
    steps: int
    cfg_scale: float
    sampler: str
    scheduler: str
    width: int                   # snapped to resolution bucket
    height: int                  # snapped to resolution bucket
    clip_skip: int
    denoise_strength: float
    seed: int                    # -1 resolved to random
    batch_size: int
    # Source tracking (for UI transparency)
    sources: dict[str, str]      # {"steps": "model_catalog", "width": "resolution_bucket", ...}
```

**Resolution resolution (pun intended):**
1. If user specified both width and height → snap to nearest bucket
2. If user specified aspect ratio hint (e.g., "landscape", "portrait", "square", "panoramic", "16:9") → pick matching bucket
3. If neither → use native resolution for the model family
4. Always snap to valid bucket — never generate at arbitrary resolution

---

### 3.6 ComfyUI Client (`ez_comfy/comfyui/`)

**Purpose:** HTTP + WebSocket client for ComfyUI API operations.

**Files:**
- `client.py` — `ComfyUIClient` class
- `vram.py` — VRAM handoff helpers

**ComfyUIClient:**
```python
class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=30)
        self._ws_url = base_url.replace("http", "ws") + "/ws"

    # HTTP methods
    async def health_check(self) -> bool
    async def system_stats(self) -> dict
    async def get_object_info(self, node_class: str | None = None) -> dict
    async def queue_prompt(self, workflow: dict) -> str   # returns prompt_id
    async def get_history(self, prompt_id: str) -> dict
    async def download_output(self, filename: str, subfolder: str, output_type: str) -> bytes
    async def upload_image(self, image_bytes: bytes, filename: str) -> dict
    async def get_queue(self) -> dict
    async def cancel_prompt(self, prompt_id: str) -> None
    async def free_vram(self) -> None
    async def close(self) -> None

    # WebSocket methods
    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 300,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> dict:
        """Connect via WebSocket, stream progress events, return on completion."""
```

**WebSocket Progress Streaming:**

ComfyUI sends progress events via WebSocket at `ws://{base_url}/ws?clientId={client_id}`:
```python
@dataclass
class ProgressEvent:
    event_type: str          # "progress", "executing", "executed", "execution_error"
    node_id: str | None      # which node is running
    step: int | None         # current step (e.g., 3)
    total_steps: int | None  # total steps (e.g., 6)
    preview_b64: str | None  # latent preview as base64 data URI (small JPEG), or None
```

The WebSocket approach replaces polling `/history` and enables:
- Real-time progress bar (KSampler reports step N of M)
- Latent preview images mid-generation
- Immediate error detection (don't wait for timeout)

**Fallback:** If WebSocket connection fails, fall back to polling `/history/{prompt_id}` every 1 second (same as MultiModel's proven pattern).

**VRAM Manager (`vram.py`):**
```python
async def unload_ollama_models(ollama_url: str) -> None
async def free_comfyui_vram(client: ComfyUIClient) -> None

@asynccontextmanager
async def vram_guard(client: ComfyUIClient, ollama_url: str | None):
    """Context manager: unloads Ollama before, frees ComfyUI after."""
    if ollama_url:
        await unload_ollama_models(ollama_url)
    try:
        yield
    finally:
        await free_comfyui_vram(client)
```

---

### 3.7 Generation Engine (`ez_comfy/engine.py`)

**Purpose:** Top-level orchestrator that ties everything together.

```python
class GenerationEngine:
    def __init__(self, settings: Settings):
        self.comfyui = ComfyUIClient(settings.comfyui.base_url)
        self.ollama_url = settings.ollama.base_url if settings.ollama.enabled else None
        self.hardware: HardwareProfile | None = None
        self.inventory: ComfyUIInventory | None = None
        self.catalog = load_catalog()
        self.llm: LLMClient | None = None     # optional
        self.settings = settings

    async def startup(self) -> None:
        """Probe hardware, scan inventory, init LLM client if enabled."""

    async def generate(
        self,
        request: GenerationRequest,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> GenerationResult:
        """Main entry point: prompt → plan → workflow → submit → output."""

    async def plan_only(self, request: GenerationRequest) -> GenerationPlan:
        """Return the plan without executing (for preview)."""

    async def refresh_inventory(self) -> ComfyUIInventory:
        """Re-scan ComfyUI models."""

    async def get_recommendations(self, prompt: str, intent: str | None = None) -> list[ModelRecommendation]:
        """Get model recommendations for a prompt without generating."""
```

**GenerationRequest:**
```python
@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    reference_image: bytes | None = None
    mask_image: bytes | None = None
    intent_override: PipelineIntent | None = None
    checkpoint_override: str | None = None
    recipe_override: str | None = None
    style_preset: str | None = None           # "cinematic", "anime", etc
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None           # "16:9", "square", "portrait", etc
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    seed: int = -1
    loras: list[tuple[str, float]] | None = None
    batch_size: int = 1
    denoise_strength: float = 0.7
    upscale_factor: int = 4
    video_frames: int = 25
    video_fps: int = 8
    audio_duration: float = 5.0
    output_dir: str = "output"
    enhance_prompt: bool = False
```

**GenerationPlan (enriched):**
```python
@dataclass
class GenerationPlan:
    intent: PipelineIntent
    recipe: Recipe
    prompt: str                          # after prompt adaptation
    original_prompt: str                 # what user typed
    negative_prompt: str                 # after auto-generation
    checkpoint: str
    checkpoint_family: str
    catalog_entry: ModelCatalogEntry | None
    params: ResolvedParams
    loras: list[tuple[str, float]]
    vae_override: str | None
    controlnet: str | None
    controlnet_strength: float
    reference_image_path: str | None
    mask_image_path: str | None
    style_preset: StylePreset | None
    estimated_vram_gb: float
    estimated_time_seconds: float
    warnings: list[str]
    recommendations: list[ModelRecommendation]  # shown in UI
    missing_capabilities: list[str]             # capabilities needed but not detected
    param_sources: dict[str, str]               # transparency: where each param came from
```

**GenerationResult:**
```python
@dataclass
class GenerationResult:
    success: bool
    output_paths: list[str]
    plan: GenerationPlan
    generation_time_seconds: float
    seed_used: int                      # actual seed (resolved from -1)
    error: str | None = None
    comfyui_prompt_id: str | None = None
    workflow_json: dict | None = None   # the exact workflow sent (for "Copy Workflow" feature)
```

**`generate()` flow:**
```python
async def generate(self, request, on_progress=None):
    # 1. Ensure startup completed
    if not self.hardware:
        await self.startup()

    # 2. Detect intent
    intent = request.intent_override or detect_intent(request)

    # 3. Get model recommendations
    recommendations = recommend_models(request.prompt, intent, self.hardware, self.inventory, self.catalog)

    # 4. Select checkpoint (first installed recommendation, or override)
    checkpoint, catalog_entry = select_checkpoint(request, recommendations, self.inventory)

    # 5. Select recipe
    recipe = select_recipe(intent, request, self.inventory)

    # 6. Adapt prompt for model syntax
    prompt, negative = adapt_prompt(request.prompt, request.negative_prompt,
                                     catalog_entry.prompt_syntax if catalog_entry else family_syntax,
                                     style_preset)

    # 7. Resolve parameters
    params = resolve_params(request, recipe, catalog_entry, family_profile)

    # 8. Build plan
    plan = GenerationPlan(...)

    # 9. Compose workflow
    workflow = compose_workflow(plan)

    # 10. Upload reference images if needed
    if request.reference_image:
        upload_result = await self.comfyui.upload_image(request.reference_image, "input_ref.png")

    # 11. Execute with VRAM guard
    async with vram_guard(self.comfyui, self.ollama_url):
        prompt_id = await self.comfyui.queue_prompt(workflow)
        result = await self.comfyui.wait_for_completion(prompt_id, on_progress=on_progress)

    # 12. Download and save outputs + sidecar metadata
    output_paths = await self._save_outputs(result, plan, request.output_dir)

    return GenerationResult(
        success=True, output_paths=output_paths, plan=plan,
        workflow_json=workflow, seed_used=params.seed, ...
    )
```

---

### 3.8 Generation Queue + Single-Runner Lock (`ez_comfy/engine.py`)

**Purpose:** Accept multiple generation requests and process them sequentially. The GPU can only run one workflow at a time — all generation entry points go through the queue.

**Single-runner contract:** There is ONE `asyncio.Lock` (`_gpu_lock`) in `GenerationEngine`. Every code path that calls ComfyUI (`generate()`, `compare()`, queue processing) acquires this lock before submitting a workflow. This prevents concurrent jobs from colliding regardless of which endpoint initiated them.

```python
class GenerationEngine:
    def __init__(self, settings: Settings):
        ...
        self._gpu_lock = asyncio.Lock()  # single-runner: one ComfyUI job at a time
        self._queue = GenerationQueue(self)

    async def generate(self, request, on_progress=None):
        async with self._gpu_lock:
            return await self._generate_impl(request, on_progress)

    async def compare(self, request):
        async with self._gpu_lock:
            # Both A and B run inside the same lock acquisition
            result_a = await self._generate_impl(req_a)
            result_b = await self._generate_impl(req_b)
            return ComparisonResult(result_a, result_b, seed)
```

The queue's `_process_loop()` also calls `generate()`, which acquires the lock. If a direct `/v1/generate` call is running, queued items wait. If the queue is running, a direct `/v1/generate` call blocks until the current item finishes. This is intentional — the GPU is the bottleneck.

```python
@dataclass
class QueueEntry:
    id: str                              # uuid
    request: GenerationRequest
    status: str                          # "pending", "running", "completed", "failed", "cancelled"
    result: GenerationResult | None
    progress: float                      # 0.0-1.0
    created_at: float
    started_at: float | None
    completed_at: float | None

class GenerationQueue:
    def __init__(self, engine: GenerationEngine):
        self._entries: list[QueueEntry] = []
        self._engine = engine
        self._processing = False

    async def enqueue(self, request: GenerationRequest) -> str:
        """Add to queue, return entry ID. Starts processing if idle."""

    async def cancel(self, entry_id: str) -> bool:
        """Cancel a pending entry (can't cancel running)."""

    def get_status(self, entry_id: str) -> QueueEntry | None

    def list_entries(self) -> list[QueueEntry]

    async def _process_loop(self) -> None:
        """Process queue entries FIFO. Each calls engine.generate() which acquires _gpu_lock."""
```

The queue enables:
- Submit multiple prompts from UI, go do something else
- Each runs sequentially (GPU lock enforces single-runner)
- UI shows queue status with progress for current item
- Direct `/v1/generate` and `/v1/compare` calls also respect the lock — no collisions

---

### 3.9 A/B Comparison (`ez_comfy/engine.py`)

**Purpose:** Generate the same prompt with two different configurations side by side.

```python
@dataclass
class ComparisonRequest:
    prompt: str
    negative_prompt: str = ""
    config_a: dict                       # overrides for side A (e.g. {"checkpoint": "realvis..."})
    config_b: dict                       # overrides for side B (e.g. {"checkpoint": "juggernaut..."})
    shared_seed: bool = True             # use same seed for fair comparison

@dataclass
class ComparisonResult:
    result_a: GenerationResult
    result_b: GenerationResult
    shared_seed: int | None
```

In the engine:
```python
async def compare(self, request: ComparisonRequest) -> ComparisonResult:
    seed = random_seed() if request.shared_seed else None
    req_a = GenerationRequest(prompt=request.prompt, seed=seed, **request.config_a)
    req_b = GenerationRequest(prompt=request.prompt, seed=seed, **request.config_b)
    # Run sequentially (GPU shared)
    result_a = await self.generate(req_a)
    result_b = await self.generate(req_b)
    return ComparisonResult(result_a, result_b, seed)
```

---

### 3.10 LLM Client (`ez_comfy/planner/llm_client.py`)

**Purpose:** Optional LLM for prompt enhancement and ambiguous intent resolution.

Uses Ollama (local) or OpenAI-compatible API. Single-shot completions, not a multi-turn agent.

```python
class LLMClient:
    def __init__(self, provider: str, model: str, base_url: str, api_key: str | None):
        ...

    async def enhance_prompt(self, user_prompt: str, model_family: str) -> str:
        """Add detail/quality tags appropriate for the target model family."""

    async def detect_intent(self, user_prompt: str, has_image: bool) -> dict:
        """Structured output: {intent, style_hints, negative_suggestion, aspect_ratio}."""

    async def suggest_negative(self, prompt: str, family: str) -> str:
        """Generate optimal negative prompt for the given model family."""
```

Note: `enhance_prompt` now takes `model_family` so it can tailor enhancement to the target model's training style (natural language for Flux, tag-based for SDXL, etc.).

---

### 3.11 FastAPI Server (`ez_comfy/api/`)

**Files:**
- `server.py` — App factory + lifespan
- `routes.py` — All endpoints + inline UI
- `models.py` — Pydantic request/response models

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET /` | GET | Serves the web UI (inline HTML/JS) |
| `GET /v1/health` | GET | ComfyUI reachability + GPU status |
| `GET /v1/inventory` | GET | Return installed models/LoRAs/VAEs |
| `POST /v1/inventory/refresh` | POST | Re-scan ComfyUI models |
| `GET /v1/recommendations` | GET | Get model recommendations for a prompt |
| `GET /v1/recipes` | GET | List available workflow recipes |
| `POST /v1/generate` | POST | Synchronous generation (JSON body) |
| `POST /v1/generate/form` | POST | Synchronous generation (multipart, supports image upload) |
| `POST /v1/plan` | POST | Preview generation plan without executing |
| `POST /v1/plan/workflow` | POST | Export ComfyUI workflow JSON (downloadable) |
| `POST /v1/compare` | POST | A/B comparison generation |
| `POST /v1/queue` | POST | Enqueue generation for background processing |
| `GET /v1/queue` | GET | List all queued/running/completed jobs |
| `GET /v1/queue/{job_id}` | GET | Get job status and result |
| `DELETE /v1/queue/{job_id}` | DELETE | Cancel a queued job |
| `GET /v1/install/plan` | GET | What to install for a given prompt |
| `GET /v1/history/{prompt_id}/provenance` | GET | Provenance record for a completed job |

**WebSocket endpoint (`/v1/ws/{job_id}`):**
Proxies ComfyUI's WebSocket progress events to the UI:
```json
{"type": "progress", "step": 3, "total_steps": 6, "preview_b64": "data:image/jpeg;base64,..."}
{"type": "executing", "node": "KSampler"}
{"type": "complete", "output_urls": ["/v1/output/ezcomfy_00001.png"]}
{"type": "error", "message": "CUDA out of memory"}
```
The `preview_b64` field contains the latent preview as an inline base64 data URI (small JPEG, ~5-15KB). This avoids the need for a separate preview endpoint — the image is embedded directly in the WebSocket message. The field is `null` if ComfyUI doesn't send a preview for that step.

---

### 3.12 Web UI (`GET /`)

**Inline HTML/JS single-page app** (same pattern as MultiModel's Visual Runner).

**Layout:**

```
┌──────────────────────────────────────────────────────────────┐
│  EZ_Comfy                    [Queue: 0]  [GPU: RTX 5070 Ti] │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ Prompt ────────────────────────────────────────────────┐ │
│  │ [textarea: describe what you want to create]           │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ Negative Prompt ──────────────────────────────────────┐ │
│  │ [auto-filled based on model — editable]                │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ Reference Image (drag & drop) ────────────────────────┐ │
│  │ [drop zone / file picker]     [+ Add Mask]             │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ Style ─────┐  ┌─ Pipeline ────┐  ┌─ Aspect ─────────┐ │
│  │ [cinematic▼] │  │ [auto-detect] │  │ [square / 16:9▼] │ │
│  └─────────────┘  └───────────────┘  └───────────────────┘ │
│                                                              │
│  ┌─ Model ─────────────────────────────────────────────────┐ │
│  │ ★ RealVisXL Lightning (installed, SDXL) [auto-selected]│ │
│  │   Juggernaut XL v9 (not installed — 6.5GB) [How to Get]│ │
│  │   Flux.1 Dev (not installed — 23.8GB) [How to Get]    │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ▶ Advanced Settings (collapsed)                             │
│    Resolution: [auto] W:[____] H:[____]                      │
│    Steps: [auto]  CFG: [auto]  CLIP Skip: [auto]  Seed:[-1] │
│    Sampler: [auto]  Scheduler: [auto]                        │
│    Denoise: [0.7]  Batch: [1]                                │
│    Recipe: [auto / dropdown]                                 │
│    LoRAs: [+ Add LoRA] (shows compatible only)               │
│    ☐ Enhance prompt with AI                                  │
│                                                              │
│  [Preview Plan]  [Generate]  [A/B Compare]  [Add to Queue]  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  ┌─ Plan Preview ──────────────────────────────────────────┐ │
│  │ Pipeline: txt2img | Recipe: Hi-Res Fix                  │ │
│  │ Model: realvisxlV50_lightning (SDXL Lightning)          │ │
│  │ Resolution: 1024x1024 (snapped from 1000x1000)         │ │
│  │ Steps: 6 | CFG: 1.5 | Sampler: euler | Sched: sgm_uni │ │
│  │ Negative: (auto-generated for SDXL photorealism)       │ │
│  │ Prompt adapted: added SDXL quality tokens               │ │
│  │ Est. VRAM: 8.2GB / 15.9GB | Est. Time: ~4s            │ │
│  │ ⚠ Missing capabilities: none                            │ │
│  │ Source: steps=model_catalog, width=resolution_bucket    │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ Progress ──────────────────────────────────────────────┐ │
│  │ ████████░░░░░░░░ Step 3/6 (KSampler) — 1.8s elapsed    │ │
│  │ [latent preview image]                                  │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ Output ────────────────────────────────────────────────┐ │
│  │ [generated image(s)]                                    │ │
│  │ Generation time: 3.8s | Seed: 1234567890                │ │
│  │ [Save] [Open Folder] [Copy Workflow JSON] [Reuse Seed]  │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ A/B Comparison (when active) ─────────────────────────┐ │
│  │ ┌─ A ──────────────┐  ┌─ B ──────────────┐            │ │
│  │ │ [image A]         │  │ [image B]         │            │ │
│  │ │ RealVisXL Light.  │  │ Juggernaut XL     │            │ │
│  │ │ 3.8s              │  │ 12.1s             │            │ │
│  │ └──────────────────┘  └──────────────────┘            │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ Queue ─────────────────────────────────────────────────┐ │
│  │ #1 "sunset over mountains" ████████████ Done ✓          │ │
│  │ #2 "portrait of an astronaut" ███░░░░░░ Running (3/20)  │ │
│  │ #3 "abstract fluid art" ░░░░░░░░░░░░ Pending [Cancel]   │ │
│  └────────────────────────────────────────────────────────-┘ │
│                                                              │
│  ┌─ History ───────────────────────────────────────────────┐ │
│  │ [thumb] [thumb] [thumb] [thumb] [thumb] [thumb]         │ │
│  │ click to reuse prompt + settings                        │ │
│  └────────────────────────────────────────────────────────-┘ │
└──────────────────────────────────────────────────────────────┘
```

**Key UI behaviors:**
- **Model recommendation panel:** Shows top 3 models ranked for the current prompt. Installed models marked with ★. Uninstalled show size + [How to Get] button (opens modal with HuggingFace CLI command / Civitai link + file placement instructions).
- **Auto negative prompt:** Negative prompt field auto-fills when model is selected. Greyed out with "(not used)" for Flux.
- **Style preset selector:** Dropdown with visual preview. Changes affect prompt adaptation preview.
- **Pipeline auto-detect:** Updates automatically when reference image added/removed.
- **Aspect ratio selector:** Presets (square, landscape 16:9, portrait 9:16, panoramic 21:9) that snap to model's resolution buckets.
- **LoRA dropdown:** Only shows compatible LoRAs for the selected model family.
- **Plan preview:** Shows EVERYTHING that will happen — adapted prompt, auto-negative, param sources, VRAM estimate. Full transparency.
- **WebSocket progress:** Real-time progress bar with step count + latent preview image.
- **Copy Workflow JSON:** Copies the exact ComfyUI workflow dict — user can paste into ComfyUI's web UI to tweak further.
- **Reuse Seed:** Locks the seed from current output for iterating on prompt/settings.
- **A/B Compare:** Opens side-by-side view, lets user pick two configs (different models, different settings).
- **Queue panel:** Shows all pending/running/completed jobs. Expandable.
- **History:** Thumbnails of recent generations. Click to reload prompt + all settings.

---

## 4. Configuration

### 4.1 `config/settings.yaml`

```yaml
comfyui:
  base_url: "http://127.0.0.1:8000"
  model_base_path: "G:/Documents/ComfyUI/models"
  default_output_dir: "output"
  timeout_seconds: 300

ollama:
  base_url: "http://localhost:11434"
  enabled: true

llm:
  enabled: false
  provider: "ollama"
  model: "llama3.2:3b"
  base_url: "http://localhost:11434"
  api_key_env: "OPENAI_API_KEY"

preferences:
  prefer_speed: true
  max_resolution_multiplier: 1.5
  default_batch_size: 1
  auto_negative_prompt: true
  default_style: null                            # null = no style, or "cinematic" etc

history:
  max_entries: 100
  save_metadata: true
  save_thumbnails: true
```

### 4.2 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EZCOMFY_COMFYUI_URL` | `http://127.0.0.1:8000` | ComfyUI server |
| `EZCOMFY_OLLAMA_URL` | `http://localhost:11434` | Ollama (for VRAM handoff) |
| `EZCOMFY_OUTPUT_DIR` | `./output` | Default output directory |
| `EZCOMFY_PORT` | `8088` | Server port |
| `EZCOMFY_CONFIG` | `config/settings.yaml` | Config file path |
| `EZCOMFY_MODEL_BASE` | `G:/Documents/ComfyUI/models` | ComfyUI models path |

---

## 5. Project Structure

```
I:/EZ_Comfy/
├── pyproject.toml
├── SPEC.md                          # this file
├── CONTEXT.md                       # build instructions
├── CLAUDE.md                        # project instructions for Claude Code
├── config/
│   └── settings.yaml
├── ez_comfy/
│   ├── __init__.py
│   ├── __main__.py                  # CLI entry: serve, generate, check, plan
│   ├── engine.py                    # GenerationEngine + GenerationQueue
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py                # FastAPI app factory + lifespan
│   │   ├── routes.py                # All endpoints + inline UI HTML
│   │   └── models.py                # Pydantic request/response models
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── probe.py                 # HardwareProfile + probe_hardware()
│   │   └── comfyui_inventory.py     # ComfyUIInventory + scan functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── catalog.py               # ModelCatalogEntry + MODEL_CATALOG + recommend_models()
│   │   ├── profiles.py              # ModelProfile family defaults + resolution buckets
│   │   └── classifier.py            # classify_checkpoint() for unknown models
│   ├── planner/
│   │   ├── __init__.py
│   │   ├── intent.py                # PipelineIntent enum + detect_intent()
│   │   ├── planner.py               # plan_generation() → GenerationPlan
│   │   ├── prompt_adapter.py        # adapt_prompt() + StylePreset + PromptSyntax
│   │   ├── param_resolver.py        # resolve_params() with priority chain
│   │   └── llm_client.py            # Optional LLM for prompt enhancement
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── recipes.py               # Recipe dataclass + RECIPES + select_recipe()
│   │   ├── composer.py              # compose_workflow() dispatcher
│   │   ├── txt2img.py               # txt2img recipe builders
│   │   ├── img2img.py               # img2img recipe builders
│   │   ├── inpaint.py               # inpaint recipe builders
│   │   ├── upscale.py               # upscale recipe builders
│   │   ├── video.py                 # video recipe builders
│   │   └── audio.py                 # audio recipe builder
│   ├── comfyui/
│   │   ├── __init__.py
│   │   ├── client.py                # ComfyUIClient — HTTP + WebSocket
│   │   └── vram.py                  # VRAM handoff helpers
│   └── config/
│       ├── __init__.py
│       └── schema.py                # Settings Pydantic model
├── tests/
│   ├── unit/
│   │   ├── test_classifier.py
│   │   ├── test_catalog.py
│   │   ├── test_profiles.py
│   │   ├── test_intent.py
│   │   ├── test_prompt_adapter.py
│   │   ├── test_param_resolver.py
│   │   ├── test_planner.py
│   │   ├── test_recipes.py
│   │   ├── test_workflows.py
│   │   ├── test_engine.py
│   │   └── test_api.py
│   └── integration/
│       ├── test_comfyui_client.py
│       └── test_end_to_end.py
└── docs/
    └── (empty — generated later if needed)
```

---

## 6. Build Phases

### Phase 1: Foundation
**Files:** `pyproject.toml`, `__init__.py`, `__main__.py`, `config/schema.py`, `config/settings.yaml`, `hardware/probe.py`
**Tests:** `test_probe.py` (mock nvidia-smi + psutil)
**Milestone:** `python -m ez_comfy check` prints GPU name + VRAM + platform info

### Phase 2: ComfyUI Client + Inventory
**Files:** `comfyui/client.py`, `comfyui/vram.py`, `hardware/comfyui_inventory.py`
**Tests:** `test_comfyui_client.py` (mock httpx), `test_inventory.py`
**Milestone:** `python -m ez_comfy check` also prints installed checkpoints/LoRAs/custom nodes

### Phase 3: Model Catalog + Profiles + Classifier
**Files:** `models/catalog.py`, `models/profiles.py`, `models/classifier.py`
**Tests:** `test_catalog.py`, `test_classifier.py`, `test_profiles.py`
**Milestone:** Given a checkpoint filename, returns family + optimal defaults. `recommend_models()` returns ranked suggestions for a prompt.

### Phase 4: Prompt Adapter + Style Presets
**Files:** `planner/prompt_adapter.py`
**Tests:** `test_prompt_adapter.py`
**Milestone:** Prompt correctly adapted for each model family. Pony gets score tags, Flux strips emphasis, auto-negatives generated.

### Phase 5: Intent Detection + Parameter Resolver + Planner
**Files:** `planner/intent.py`, `planner/param_resolver.py`, `planner/planner.py`
**Tests:** `test_intent.py`, `test_param_resolver.py`, `test_planner.py`
**Milestone:** Given prompt + hardware + inventory, returns a complete `GenerationPlan` with adapted prompt, resolved params, selected recipe.

### Phase 6: Workflow Recipes + Composers
**Files:** `workflows/recipes.py`, `workflows/composer.py`, all `workflows/*.py` builders
**Tests:** `test_recipes.py`, `test_workflows.py` — validate node graph structure for every recipe
**Milestone:** Every recipe produces a valid ComfyUI node graph dict. LoRA injection, VAE override, ControlNet all tested.

### Phase 7: Generation Engine + Queue
**Files:** `engine.py`
**Tests:** `test_engine.py` (mock ComfyUI client)
**Milestone:** `python -m ez_comfy generate "a cat in space"` produces an image. Queue processes entries sequentially.

### Phase 8: FastAPI Server + Endpoints
**Files:** `api/server.py`, `api/routes.py`, `api/models.py`
**Tests:** `test_api.py` (TestClient with mocked engine)
**Milestone:** All REST + WebSocket endpoints functional. `/v1/recommendations` returns model suggestions.

### Phase 9: Web UI
**Files:** Inline HTML/JS in `routes.py` (GET /)
**Tests:** Manual testing
**Milestone:** Full UI with prompt input, model recommendations, style presets, plan preview, progress bar, output display, A/B comparison, queue panel, history.

### Phase 10: LLM Prompt Enhancement (optional)
**Files:** `planner/llm_client.py`
**Tests:** `test_llm_client.py`
**Milestone:** "Enhance prompt" uses local LLM to improve terse prompts, tailored to target model family.

---

## 7. CLI Commands

```bash
# Health check — GPU, VRAM, ComfyUI status, installed models, missing recommendations
python -m ez_comfy check

# Generate from command line
python -m ez_comfy generate "a cyberpunk city at sunset" --output ./output

# Generate with overrides
python -m ez_comfy generate "a cat" --checkpoint realvisxlV50_lightning --steps 6 --style cinematic

# Show plan without generating
python -m ez_comfy plan "a landscape painting"

# Get model recommendations for a prompt
python -m ez_comfy recommend "anime girl in cherry blossom garden"

# Start web server
python -m ez_comfy serve --port 8088
```

---

## 8. Key Design Decisions

1. **Curated model catalog over blind auto-detection** — Community knowledge about which model excels at what is baked into a maintained catalog (~15 entries v1, expand post-telemetry). Auto-detection is the fallback for unknown models.

2. **Recommendation-only, no auto-download** — The app recommends models and shows download instructions (HuggingFace CLI commands, Civitai links) but does NOT download models itself. Downloads are large (2-24GB), may require authentication (gated repos), and users should control what lands on their disk. The UI shows "[Download Instructions]" modals, not "[Install Now]" buttons.

3. **Prompt adaptation is automatic** — The #1 cause of bad ComfyUI results is wrong prompt syntax for the model. EZ_Comfy silently fixes this (Pony score tags, Flux natural language, family-specific negatives).

4. **Resolution bucketing prevents silent quality loss** — Users pick "1000x1000" and wonder why output looks bad. EZ_Comfy snaps to the nearest trained resolution bucket automatically.

5. **Capability-based node detection, not package names** — Recipes declare required capabilities (e.g., `"adetailer"`), mapped to ComfyUI class_types via `NODE_CAPABILITY_MAP`. This is resilient to package renames, forks, and alternative implementations. Package names are never exposed to ComfyUI's API.

6. **Recipes over fixed pipelines** — The workflow library is extensible (10 recipes v1, expand post-telemetry). Adding a new ComfyUI workflow is: add a recipe entry + a builder function. No architecture changes.

7. **Hardware-first, then quality** — VRAM filter comes before quality ranking. An OOM crash is worse than a slightly lower quality model.

8. **Single-runner GPU lock** — One `asyncio.Lock` guards all ComfyUI submissions. Every code path (`/v1/generate`, `/v1/compare`, queue processing) acquires the lock. No concurrent GPU jobs, no collisions, no matter which endpoint initiates the work.

9. **Full transparency** — The plan preview shows WHERE every parameter came from (model catalog, family default, user override, resolution bucket snap). Users learn what matters.

10. **A/B comparison for learning** — The fastest way to understand which model suits your needs is to see them side by side with the same prompt and seed.

11. **Queue for batch workflows** — Submit 10 prompts, go make coffee. Essential for real creative work.

12. **WebSocket progress with inline preview** — Latent previews are sent as base64 data URIs in WebSocket messages (~5-15KB JPEG). No separate preview endpoint needed. Falls back to HTTP polling if WebSocket connection fails.

13. **LoRA compatibility is best-effort** — Family inference from filename/size is heuristic. Ambiguous LoRAs are marked in the UI with a warning. Users are not blocked from using them.

14. **Copy Workflow JSON for escape hatch** — When EZ_Comfy's automation isn't enough, copy the workflow and paste into ComfyUI's native UI for manual tweaking. The app is a starting point, not a cage.

---

## 9. Dependencies

```toml
[project]
dependencies = [
    "httpx>=0.27",           # ComfyUI + Ollama HTTP
    "websockets>=12.0",      # ComfyUI WebSocket progress
    "pydantic>=2.0",         # config + request validation
    "pyyaml>=6.0",           # settings.yaml
    "fastapi>=0.110",        # web server
    "uvicorn>=0.29",         # ASGI server
    "psutil>=5.9",           # RAM detection
    "pillow>=10.0",          # image handling (thumbnails, metadata)
]

[project.optional-dependencies]
llm = [
    "openai>=1.0",           # LLM prompt enhancement (Ollama compatible)
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.0",
]
```
