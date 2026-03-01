"""Server configuration."""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaMode(str, Enum):
    """Ollama connection mode."""
    local = "local"   # Local Ollama instance (localhost:11434)
    cloud = "cloud"   # Ollama Cloud (ollama.com/api)
    off = "off"       # Disable Ollama entirely


class Settings(BaseSettings):
    """Application settings.

    All settings can be overridden via environment variables with QGC_ prefix.
    Example: QGC_OLLAMA_MODE=cloud QGC_OLLAMA_API_KEY=sk-... uv run qgc-mcp
    """

    model_config = SettingsConfigDict(env_prefix="QGC_")

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Data paths
    gadgets_dir: Path = Path("builtin/gadgets")

    # Registry info
    registry_url: str = "http://localhost:8080/gadgets"
    compiler_version: str = "0.1.0"

    # Ollama
    ollama_mode: OllamaMode = OllamaMode.local
    ollama_local_url: str = "http://localhost:11434/api/chat"
    ollama_cloud_url: str = "https://ollama.com/api/chat"
    ollama_api_key: Optional[str] = None
    ollama_model: str = "qwen3-coder-next"


settings = Settings()
