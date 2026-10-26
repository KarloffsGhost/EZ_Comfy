from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ComfyUIConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8188"
    model_base_path: str = ""
    default_output_dir: str = "output"
    timeout_seconds: int = 300


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    enabled: bool = True


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    api_key_env: str = "OPENAI_API_KEY"


class PreferencesConfig(BaseModel):
    prefer_speed: bool = True
    max_resolution_multiplier: float = 1.5
    default_batch_size: int = 1
    auto_negative_prompt: bool = True
    default_style: str | None = None


class HistoryConfig(BaseModel):
    max_entries: int = 100
    save_metadata: bool = True
    save_thumbnails: bool = True


class Settings(BaseModel):
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    preferences: PreferencesConfig = Field(default_factory=PreferencesConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    env_path = os.environ.get("EZCOMFY_CONFIG")
    if env_path:
        config_path = Path(env_path)

    if not config_path.exists():
        return _apply_env_overrides(Settings())

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    settings = Settings.model_validate(data)
    return _apply_env_overrides(settings)


def _apply_env_overrides(settings: Settings) -> Settings:
    if url := os.environ.get("EZCOMFY_COMFYUI_URL"):
        settings.comfyui.base_url = url
    if url := os.environ.get("EZCOMFY_OLLAMA_URL"):
        settings.ollama.base_url = url
    if out := os.environ.get("EZCOMFY_OUTPUT_DIR"):
        settings.comfyui.default_output_dir = out
    if base := os.environ.get("EZCOMFY_MODEL_BASE"):
        settings.comfyui.model_base_path = base
    return settings
