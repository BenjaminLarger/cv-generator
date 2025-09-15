"""Configuration module for CV Generator."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Default OpenAI model to use")
    temperature: float = Field(default=0.7, description="Temperature for text generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens for generation")


class AppConfig(BaseModel):
    """Main application configuration."""

    # OpenAI settings
    openai: OpenAIConfig

    # File paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    templates_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "templates")
    applications_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "applications")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="cv_generator.log", description="Log file name")

    # Application settings
    debug: bool = Field(default=False, description="Debug mode")

    class Config:
        """Pydantic configuration."""
        env_prefix = "CV_GEN_"


def get_config() -> AppConfig:
    """Get application configuration from environment variables."""

    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your environment or in a .env file."
        )

    # Create OpenAI config
    openai_config = OpenAIConfig(
        api_key=openai_api_key,
        model=os.getenv("CV_GEN_OPENAI_MODEL", "gpt-4"),
        temperature=float(os.getenv("CV_GEN_OPENAI_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("CV_GEN_OPENAI_MAX_TOKENS", "2000"))
    )

    # Create main config
    config = AppConfig(
        openai=openai_config,
        log_level=os.getenv("CV_GEN_LOG_LEVEL", "INFO"),
        log_file=os.getenv("CV_GEN_LOG_FILE", "cv_generator.log"),
        debug=os.getenv("CV_GEN_DEBUG", "false").lower() == "true"
    )

    return config


# Global config instance
config = get_config()