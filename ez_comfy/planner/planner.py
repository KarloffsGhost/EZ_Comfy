from __future__ import annotations

from dataclasses import dataclass, field

from ez_comfy.hardware.probe import HardwareProfile
from ez_comfy.hardware.comfyui_inventory import CHECKPOINT_LOADER, DIFFUSION_MODEL_LOADER, ComfyUIInventory
from ez_comfy.models.catalog import (
    MODEL_CATALOG, ModelCatalogEntry, ModelRecommendation,
    _intent_compatible_families, find_catalog_entry, recommend_models, resolve_installed_filename,
)
from ez_comfy.models.profiles import ModelProfile, PromptSyntax, get_profile, snap_to_bucket
from ez_comfy.planner.intent import PipelineIntent, detect_intent
from ez_comfy.planner.param_resolver import ResolvedParams, resolve_params
from ez_comfy.planner.prompt_adapter import StylePreset, adapt_prompt, get_style_preset
from ez_comfy.planner.provenance import Alternative, Decision, ProvenanceRecord
from ez_comfy.workflows.recipes import Recipe, select_recipe


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    reference_image: bytes | None = None
    mask_image: bytes | None = None
    intent_override: str | None = None
    checkpoint_override: str | None = None
    recipe_override: str | None = None
    style_preset: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    seed: int = -1
    loras: list[tuple[str, float]] | None = None
    batch_size: int = 1
    denoise_strength: float = 0.7
    upscale_factor: int = 4
    video_frames: int | None = None   # None = the model family's default
    video_fps: int | None = None
    audio_duration: float = 5.0
    output_dir: str = "output"
    enhance_prompt: bool = False


@dataclass
class GenerationPlan:
    intent: PipelineIntent
    recipe: Recipe
    prompt: str                          # after adaptation
    original_prompt: str
    negative_prompt: str
    checkpoint: str
    checkpoint_family: str
    catalog_entry: ModelCatalogEntry | None
    profile: ModelProfile
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
    recommendations: list[ModelRecommendation]
    missing_capabilities: list[str]
    param_sources: dict[str, str]
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    # Role -> file for multi-file models (diffusion_high, diffusion_low, text_encoder, vae).
    # `checkpoint` keeps its meaning for single-checkpoint recipes.
    models: dict[str, str] = field(default_factory=dict)
    video_frames: int | None = None
    video_fps: int | None = None

    def summary(self) -> dict:
        return {
            "intent": self.intent.value,
            "recipe": self.recipe.id,
            "checkpoint": self.checkpoint,
            "family": self.checkpoint_family,
            "width": self.params.width,
            "height": self.params.height,
            "steps": self.params.steps,
            "cfg": self.params.cfg_scale,
            "sampler": self.params.sampler,
            "scheduler": self.params.scheduler,
            "seed": self.params.seed,
            "estimated_vram_gb": self.estimated_vram_gb,
            "estimated_time_s": self.estimated_time_seconds,
            "warnings": self.warnings,
            "missing_capabilities": self.missing_capabilities,
            "param_sources": self.param_sources,
            "provenance": self.provenance.to_dict(),
            "models": dict(self.models),
        }


def plan_generation(
    request: GenerationRequest,
    hardware: HardwareProfile,
    inventory: ComfyUIInventory,
    prefer_speed: bool = True,
    auto_negative: bool = True,
) -> GenerationPlan:
    """Main planner: turns a GenerationRequest into a fully-resolved GenerationPlan."""
    warnings: list[str] = []
    provenance = ProvenanceRecord()

    # 1. Detect intent
    intent_str = request.intent_override or detect_intent(
        request.prompt,
        has_reference_image=request.reference_image is not None,
        has_mask=request.mask_image is not None,
    ).value
    intent = PipelineIntent(intent_str)

    provenance.add(Decision(
        parameter="intent",
        chosen_value=intent.value,
        source="user" if request.intent_override else "prompt_keyword",
        reason=(
            f"User override: {request.intent_override!r}"
            if request.intent_override
            else "Detected from prompt keywords and input context"
        ),
    ))

    # 2-4. Pick the model(s). Text-to-video with the Wan 2.2 pair takes a multi-file path; every
    # other plan picks one checkpoint as before.
    wan_models = None
    if intent == PipelineIntent.VIDEO and request.reference_image is None and not request.checkpoint_override:
        wan_models = _select_wan_t2v_models(inventory, warnings)

    models: dict[str, str] = {}
    chosen_loader = CHECKPOINT_LOADER
    if wan_models:
        recommendations = []
        models = wan_models
        chosen_loader = DIFFUSION_MODEL_LOADER
        checkpoint = wan_models["diffusion_high"]
        catalog_entry = None
        family = "wan_video"
        profile = get_profile(family)
        warnings.append(
            "Wan 2.2 14B offloads to system RAM on a 16 GB card, so renders are slow. Measured on an RTX 5070 Ti: "
            "512x512, 17 frames, 10 steps took 128 s including the model load; the default size (640x640, 81 frames, "
            "20 steps) takes far longer. Raise --timeout (the default is 300 s)."
        )
        provenance.add(Decision(
            parameter="models",
            chosen_value=", ".join(f"{role}={name}" for role, name in models.items()),
            source="inventory",
            reason="Wan 2.2 text-to-video needs the high-noise and low-noise pair plus the umt5 text encoder and Wan VAE; all are installed",
        ))
    else:
        recommendations = recommend_models(
            prompt=request.prompt,
            intent=intent.value,
            hardware=hardware,
            inventory=inventory,
            prefer_speed=prefer_speed,
            top_n=8,
            runnable_only=True,
        )

        checkpoint, catalog_entry, checkpoint_decisions = _select_checkpoint(
            request, intent, recommendations, inventory, warnings, hardware
        )
        for d in checkpoint_decisions:
            provenance.add(d)

        if catalog_entry:
            family = catalog_entry.effective_family
            profile = get_profile(catalog_entry.family, catalog_entry.variant)
        else:
            from ez_comfy.models.classifier import classify_checkpoint
            family, variant = classify_checkpoint(checkpoint)
            profile = get_profile(family, variant)
            if variant:
                family = f"{family}_{variant}" if f"{family}_{variant}" != family else family

    # 5. Select workflow recipe
    recipe, missing_caps, recipe_decision = _select_recipe(request, intent, inventory, warnings, family)
    provenance.add(recipe_decision)
    if recipe.model_loader != chosen_loader:
        warnings.append(
            f"Recipe '{recipe.name}' needs a {recipe.model_loader!r} model but the planner selected a "
            f"{chosen_loader!r} model; generation may fail."
        )

    # 6. Adapt prompt
    style_preset_obj = get_style_preset(request.style_preset)
    syntax = catalog_entry.prompt_syntax if catalog_entry else profile.prompt_syntax
    adapted_prompt, adapted_negative, prompt_changes = adapt_prompt(
        user_prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        syntax=syntax,
        style_preset=style_preset_obj,
        family=family,
        auto_negative=auto_negative,
    )

    if prompt_changes:
        provenance.add(Decision(
            parameter="prompt_adaptation",
            chosen_value="; ".join(prompt_changes),
            source="family_syntax",
            reason=f"Applied {family}-specific prompt syntax: {'; '.join(prompt_changes)}",
        ))

    # 7. Resolve parameters
    user_overrides = {
        "steps":           request.steps,
        "cfg":             request.cfg_scale,
        "sampler":         request.sampler,
        "scheduler":       request.scheduler,
        "seed":            request.seed,
        "width":           request.width,
        "height":          request.height,
        "batch_size":      request.batch_size,
        "denoise_default": request.denoise_strength,
    }
    recipe_overrides = recipe.settings_overrides or {}
    cat_settings = catalog_entry.settings if catalog_entry else None

    params = resolve_params(
        profile=profile,
        catalog_settings=cat_settings,
        recipe_overrides=recipe_overrides,
        user_overrides=user_overrides,
        aspect_ratio=request.aspect_ratio,
    )

    video_frames, video_fps = _resolve_video_dims(request, family, warnings)

    # Emit parameter decisions
    model_name = catalog_entry.name if catalog_entry else checkpoint
    _emit_param_decisions(provenance, params, recipe, model_name, family)

    # 8. Estimate VRAM and time
    vram_est = _estimate_vram(catalog_entry, profile, params)
    time_est = _estimate_time(params, intent, hardware)

    provenance.vram_available_gb = hardware.gpu_vram_gb
    provenance.vram_estimated_gb = vram_est
    provenance.gpu_name = hardware.gpu_name
    provenance.model_id = catalog_entry.id if catalog_entry else checkpoint
    provenance.model_family = family
    provenance.recipe_id = recipe.id

    if vram_est > hardware.gpu_vram_gb:
        warnings.append(
            f"Estimated VRAM ({vram_est:.1f}GB) exceeds available ({hardware.gpu_vram_gb}GB). "
            "Generation may fail or be slow. Try smaller resolution or fewer batch items."
        )

    # 9. LoRA resolution
    loras = request.loras or []
    recommended_vae = catalog_entry.recommended_vae if catalog_entry else None

    return GenerationPlan(
        intent=intent,
        recipe=recipe,
        prompt=adapted_prompt,
        original_prompt=request.prompt,
        negative_prompt=adapted_negative,
        checkpoint=checkpoint,
        checkpoint_family=family,
        catalog_entry=catalog_entry,
        profile=profile,
        params=params,
        loras=loras,
        vae_override=recommended_vae,
        controlnet=None,
        controlnet_strength=1.0,
        reference_image_path=None,
        mask_image_path=None,
        style_preset=style_preset_obj,
        estimated_vram_gb=vram_est,
        estimated_time_seconds=time_est,
        warnings=warnings,
        recommendations=recommendations,
        missing_capabilities=missing_caps,
        param_sources=params.sources,
        provenance=provenance,
        models=models,
        video_frames=video_frames,
        video_fps=video_fps,
    )


def _select_checkpoint(
    request: GenerationRequest,
    intent: PipelineIntent,
    recommendations: list[ModelRecommendation],
    inventory: ComfyUIInventory,
    warnings: list[str],
    hardware: HardwareProfile,
) -> tuple[str, ModelCatalogEntry | None, list[Decision]]:
    """Pick the best installed checkpoint, or fall back to any available.

    Returns (checkpoint_filename, catalog_entry, decisions).
    """
    decisions: list[Decision] = []
    # Every current builder feeds CheckpointLoaderSimple, so only files that loader can load
    # are candidates; diffusion-model (UNETLoader) files are listed but never selected here.
    loadable = inventory.models_for_loader(CHECKPOINT_LOADER)

    if request.checkpoint_override:
        override = request.checkpoint_override
        entry = find_catalog_entry(override)

        # If user supplied a catalog token/alias, resolve to the exact installed filename.
        if entry:
            installed = resolve_installed_filename(entry, inventory)
            if installed:
                decisions.append(Decision(
                    parameter="checkpoint",
                    chosen_value=installed,
                    source="user",
                    reason=f"User explicitly specified checkpoint: {override!r}",
                ))
                return installed, entry, decisions

        # If user supplied an exact installed filename, pass it through.
        for ck in loadable:
            if ck.filename == override:
                decisions.append(Decision(
                    parameter="checkpoint",
                    chosen_value=ck.filename,
                    source="user",
                    reason=f"User explicitly specified checkpoint: {override!r}",
                ))
                return ck.filename, entry, decisions
        decisions.append(Decision(
            parameter="checkpoint",
            chosen_value=override,
            source="user",
            reason=f"User explicitly specified checkpoint: {override!r} (not found in catalog)",
        ))
        return override, entry, decisions

    # Build alternatives list from all recommendations
    alternatives: list[Alternative] = []
    chosen_rec: ModelRecommendation | None = None

    for rec in recommendations:
        if rec.installed:
            installed = resolve_installed_filename(rec.entry, inventory)
            if installed:
                if chosen_rec is None:
                    chosen_rec = rec
                    chosen_filename = installed
                else:
                    alternatives.append(Alternative(
                        value=rec.entry.name,
                        rejected_reason=f"lower relevance score ({rec.score:.0f})",
                    ))
            else:
                alternatives.append(Alternative(
                    value=rec.entry.name,
                    rejected_reason="not installed (filename not found in ComfyUI inventory)",
                ))
        else:
            if not rec.fits_vram:
                reason = f"requires {rec.entry.vram_min_gb}GB VRAM, only {hardware.gpu_vram_gb}GB available"
            else:
                reason = "not installed"
            alternatives.append(Alternative(value=rec.entry.name, rejected_reason=reason))

    if chosen_rec is not None:
        decisions.append(Decision(
            parameter="checkpoint",
            chosen_value=chosen_filename,
            source="recommendation",
            reason=(
                f"Top-scored installed model for {intent.value} intent "
                f"(score: {chosen_rec.score:.0f}; reasons: {', '.join(chosen_rec.match_reasons)})"
            ),
            alternatives=alternatives,
        ))
        return chosen_filename, chosen_rec.entry, decisions

    # No recommended model installed — use first available checkpoint
    if loadable:
        # Prefer a model whose family suits this intent, then an unclassified one (it may suit),
        # and only then a model known to be the wrong kind (a stable_audio or video checkpoint
        # for an image prompt).
        suitable = _intent_compatible_families(intent.value)
        ck = (
            next((m for m in loadable if m.family in suitable), None)
            or next((m for m in loadable if m.family == "unknown"), None)
            or loadable[0]
        )
        warnings.append(
            f"No recommended model installed. Using {ck.filename!r}. "
            "Consider installing a model from the catalog."
        )
        entry = find_catalog_entry(ck.filename)
        # All recommendations become alternatives
        for rec in recommendations:
            if not any(a.value == rec.entry.name for a in alternatives):
                alternatives.append(Alternative(
                    value=rec.entry.name,
                    rejected_reason="not installed",
                ))
        decisions.append(Decision(
            parameter="checkpoint",
            chosen_value=ck.filename,
            source="fallback",
            reason="No catalog-recommended model installed; using first available checkpoint",
            alternatives=alternatives,
        ))
        return ck.filename, entry, decisions

    raise RuntimeError(
        "No checkpoints found in ComfyUI. Please install a model first."
    )


def _select_recipe(
    request: GenerationRequest,
    intent: PipelineIntent,
    inventory: ComfyUIInventory,
    warnings: list[str],
    family: str | None = None,
) -> tuple[Recipe, list[str], Decision]:
    """Select recipe, tracking missing capabilities. Returns (recipe, missing_caps, decision)."""
    from ez_comfy.hardware.comfyui_inventory import has_capability

    recipe, rejected_recipes = select_recipe(
        intent=intent,
        prompt=request.prompt,
        has_reference_image=request.reference_image is not None,
        has_mask=request.mask_image is not None,
        discovered_class_types=inventory.discovered_class_types,
        recipe_override=request.recipe_override,
        family=family,
    )

    # Check which capabilities of chosen recipe are missing
    missing = [
        cap for cap in recipe.required_capabilities
        if not has_capability(cap, inventory.discovered_class_types)
    ]
    if missing:
        warnings.append(
            f"Recipe '{recipe.name}' requires capabilities not detected in ComfyUI: {missing}. "
            "Generation may fail. Install the required custom nodes or choose a different recipe."
        )

    # Determine source of recipe selection
    if request.recipe_override:
        source = "user"
        reason = f"User explicitly specified recipe: {request.recipe_override!r}"
    elif rejected_recipes and any(
        "capability" in reason.lower() for _, reason in rejected_recipes
    ):
        source = "capability_fallback"
        reason = f"Selected as best available recipe given installed ComfyUI capabilities"
    elif rejected_recipes and any(
        "photorealism" in reason.lower() or "prompt_keyword" in reason.lower()
        or "lower priority" in reason.lower()
        for _, reason in rejected_recipes
    ):
        source = "prompt_keyword"
        reason = f"Selected based on prompt keyword matching"
    else:
        source = "default"
        reason = f"Highest-priority recipe for {intent.value} intent"

    alternatives = [
        Alternative(value=r.id, rejected_reason=rej_reason)
        for r, rej_reason in rejected_recipes
    ]

    decision = Decision(
        parameter="recipe",
        chosen_value=recipe.id,
        source=source,
        reason=reason,
        alternatives=alternatives,
    )

    return recipe, missing, decision


def _emit_param_decisions(
    provenance: ProvenanceRecord,
    params: ResolvedParams,
    recipe: Recipe,
    model_name: str,
    family: str,
) -> None:
    """Add parameter decisions to the provenance record."""
    source_labels = {
        "user": lambda param, val: f"User explicitly set {param}={val}",
        "recipe": lambda param, val: f"Recipe '{recipe.id}' overrides {param} to {val}",
        "model_catalog": lambda param, val: f"Model catalog entry for {model_name!r} specifies {param}={val}",
        "family_profile": lambda param, val: f"Default for {family} family",
        "random": lambda param, val: "Random seed (no user seed specified)",
        "resolution_bucket": lambda param, val: f"Snapped to nearest {family} resolution bucket",
    }

    def _emit(param_key: str, display_key: str, value):
        source = params.sources.get(param_key, "family_profile")
        label_fn = source_labels.get(source, lambda p, v: f"Source: {source}")
        provenance.add(Decision(
            parameter=display_key,
            chosen_value=value,
            source=source,
            reason=label_fn(display_key, value),
        ))

    _emit("steps", "steps", params.steps)
    _emit("cfg", "cfg", params.cfg_scale)
    _emit("sampler", "sampler", params.sampler)
    _emit("scheduler", "scheduler", params.scheduler)

    # Resolution as a single combined decision
    res_source = params.sources.get("width", "family_profile")
    res_label_fn = source_labels.get(res_source, lambda p, v: f"Source: {res_source}")
    provenance.add(Decision(
        parameter="resolution",
        chosen_value=f"{params.width}x{params.height}",
        source=res_source,
        reason=res_label_fn("resolution", f"{params.width}x{params.height}"),
    ))

    _emit("seed", "seed", params.seed)


# Wan 2.2 14B text-to-video file roles. Names follow the template's files
# (wan2.2_t2v_{high,low}_noise_14B_*.safetensors, umt5_xxl_*, wan_2.1_vae.safetensors).
_WAN_FRAME_STEP = 4          # EmptyHunyuanLatentVideo.length: min 1, step 4 (live /object_info, ComfyUI 0.42.0)
_WAN_DEFAULT_FRAMES = 81     # template: floor(5 s * 16 fps) + 1
_WAN_DEFAULT_FPS = 16


def _select_wan_t2v_models(inventory: ComfyUIInventory, warnings: list[str]) -> dict[str, str] | None:
    """Return the Wan 2.2 t2v file roles when the complete set is installed, else None.

    When some Wan t2v files exist but the set is incomplete, say what is missing so the user can
    fix it instead of silently getting another model.
    """
    wan_files = [m.filename for m in inventory.models_for_loader(DIFFUSION_MODEL_LOADER) if m.family == "wan_video"]

    def pick(files: list[str], *tokens: str) -> str | None:
        matches = sorted(f for f in files if all(t in f.lower() for t in tokens))
        return matches[0] if matches else None

    high = pick(wan_files, "t2v", "high_noise", "14b")
    low = pick(wan_files, "t2v", "low_noise", "14b")
    encoder = pick(inventory.clip_models, "umt5")
    vae = pick(inventory.vaes, "wan_2.1_vae") or pick(inventory.vaes, "wan", "vae")

    roles = {"diffusion_high": high, "diffusion_low": low, "text_encoder": encoder, "vae": vae}
    missing = [role for role, name in roles.items() if not name]
    if not missing:
        return {role: name for role, name in roles.items() if name}
    if high or low:
        warnings.append(
            f"Wan 2.2 text-to-video is incomplete (missing: {', '.join(missing)}); not using Wan."
        )
    return None


def _resolve_video_dims(
    request: GenerationRequest, family: str, warnings: list[str]
) -> tuple[int | None, int | None]:
    """Frame count and fps for families that carry them; (None, None) for the rest."""
    if family != "wan_video":
        return None, None
    fps = request.video_fps or _WAN_DEFAULT_FPS
    frames = request.video_frames or _WAN_DEFAULT_FRAMES
    snapped = 1 + _WAN_FRAME_STEP * max(0, int((frames - 1) / _WAN_FRAME_STEP + 0.5))   # nearest, ties up
    if snapped != frames:
        warnings.append(
            f"Wan video length must be 1 plus a multiple of {_WAN_FRAME_STEP} frames; using {snapped} instead of {frames}."
        )
    return snapped, fps


def _estimate_vram(
    catalog_entry: ModelCatalogEntry | None,
    profile: ModelProfile,
    params: ResolvedParams,
) -> float:
    base = (catalog_entry.vram_min_gb if catalog_entry else profile.vram_requirement_gb)
    # Resolution factor: 1024x1024 = baseline; higher res costs more
    res_factor = (params.width * params.height) / (1024 * 1024)
    return round(base * (0.8 + 0.2 * res_factor), 1)


def _estimate_time(params: ResolvedParams, intent: PipelineIntent, hardware: HardwareProfile) -> float:
    """Rough time estimate in seconds."""
    if intent == PipelineIntent.AUDIO:
        return 60.0  # audio is slow
    if intent == PipelineIntent.VIDEO:
        return 120.0

    # Image: base ~0.5s/step for RTX 4090-class; scale by VRAM
    vram_factor = max(0.5, 16.0 / max(hardware.gpu_vram_gb, 1))
    base = params.steps * 0.5 * vram_factor
    res_factor = (params.width * params.height) / (1024 * 1024)
    return round(base * res_factor, 1)
