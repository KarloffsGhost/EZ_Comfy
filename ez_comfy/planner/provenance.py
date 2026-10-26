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
                lines.append(f"  rejected: {alt.value} \u2014 {alt.rejected_reason}")
        return "\n".join(lines)


def _safe_value(v: Any) -> str:
    """Convert any value to a display-safe string."""
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:g}"
    return str(v)
