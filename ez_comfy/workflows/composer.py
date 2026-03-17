from __future__ import annotations

from ez_comfy.planner.planner import GenerationPlan
from ez_comfy.workflows.audio import build_audio_stable
from ez_comfy.workflows.img2img import (
    build_img2img_basic,
    build_img2img_controlnet_canny,
)
from ez_comfy.workflows.inpaint import build_inpaint_basic
from ez_comfy.workflows.txt2img import build_photo_realism_v1, build_txt2img_basic, build_txt2img_hires_fix
from ez_comfy.workflows.upscale import build_upscale_refine, build_upscale_simple
from ez_comfy.workflows.video import build_video_svd

_BUILDERS = {
    "build_photo_realism_v1":           build_photo_realism_v1,
    "build_txt2img_basic":              build_txt2img_basic,
    "build_txt2img_hires_fix":          build_txt2img_hires_fix,
    "build_img2img_basic":              build_img2img_basic,
    "build_img2img_controlnet_canny":   build_img2img_controlnet_canny,
    "build_inpaint_basic":              build_inpaint_basic,
    "build_upscale_simple":             build_upscale_simple,
    "build_upscale_refine":             build_upscale_refine,
    "build_video_svd":                  build_video_svd,
    "build_audio_stable":               build_audio_stable,
}


def compose_workflow(plan: GenerationPlan) -> dict:
    """
    Dispatch to the correct workflow builder based on the plan's recipe.
    Returns a ComfyUI-compatible node graph dict ready for POST /prompt.
    """
    builder_name = plan.recipe.builder
    builder = _BUILDERS.get(builder_name)
    if builder is None:
        raise ValueError(
            f"No workflow builder registered for {builder_name!r}. "
            f"Available: {sorted(_BUILDERS)}"
        )
    return builder(plan)


def compose_annotated_workflow(plan: GenerationPlan) -> dict:
    """Compose workflow and inject a Note node carrying provenance information.

    The Note node renders as a sticky note in ComfyUI's canvas, so provenance
    travels with the workflow when exported and re-imported by power users.
    """
    workflow = compose_workflow(plan)

    # Find a free node ID (max existing numeric ID + 1)
    max_id = max((int(k) for k in workflow if k.isdigit()), default=0)
    note_id = str(max_id + 1)

    workflow[note_id] = {
        "class_type": "Note",
        "inputs": {
            "text": plan.provenance.to_human_readable(),
        },
        "_meta": {"title": "EZ Comfy Provenance"},
    }
    return workflow


def list_builders() -> list[str]:
    return sorted(_BUILDERS)
