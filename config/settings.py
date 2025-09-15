"""
Configuration settings for CV/Cover Letter Generator
"""
import os
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field


class OpenAISettings(BaseModel):
    """OpenAI API configuration"""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Default model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=2000, description="Maximum tokens to generate")


class LoggingSettings(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_handler: bool = Field(default=True, description="Enable file logging")
    console_handler: bool = Field(default=True, description="Enable console logging")


class ApplicationSettings(BaseModel):
    """Application-specific settings"""
    output_directory: str = Field(default="applications", description="Directory for generated applications")
    template_directory: str = Field(default="templates", description="Directory for CV/cover letter templates")
    cache_directory: str = Field(default="cache", description="Directory for caching data")
    max_concurrent_jobs: int = Field(default=3, description="Maximum concurrent job applications to process")


class Settings(BaseModel):
    """Main application settings"""
    openai: OpenAISettings
    logging: LoggingSettings
    application: ApplicationSettings

    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> "Settings":
        """Load settings from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        else:
            # Return default settings if config file doesn't exist
            return cls(
                openai=OpenAISettings(api_key=os.getenv("OPENAI_API_KEY", "")),
                logging=LoggingSettings(),
                application=ApplicationSettings()
            )

    @classmethod
    def load_from_env(cls) -> "Settings":
        """Load settings from environment variables"""
        return cls(
            openai=OpenAISettings(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")) if os.getenv("OPENAI_MAX_TOKENS") else None
            ),
            logging=LoggingSettings(
                level=os.getenv("LOG_LEVEL", "INFO"),
                file_handler=os.getenv("LOG_FILE", "true").lower() == "true",
                console_handler=os.getenv("LOG_CONSOLE", "true").lower() == "true"
            ),
            application=ApplicationSettings(
                output_directory=os.getenv("OUTPUT_DIR", "applications"),
                template_directory=os.getenv("TEMPLATE_DIR", "templates"),
                cache_directory=os.getenv("CACHE_DIR", "cache"),
                max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
            )
        )


def get_settings() -> Settings:
    """Get application settings, preferring config file over environment variables"""
    try:
        return Settings.load_from_file()
    except Exception:
        return Settings.load_from_env()


# Global settings instance
settings = get_settings()