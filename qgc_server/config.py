"""Server configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="QGC_")

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Data paths
    gadgets_dir: Path = Path("builtin/gadgets")

    # Registry info
    registry_url: str = "http://localhost:8080/gadgets"
    compiler_version: str = "0.1.0"


settings = Settings()
