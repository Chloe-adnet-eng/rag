"""Module for managing environment variables configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Variables(BaseSettings):
    """Configuration variables loaded from .env file."""

    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str


def get_variables() -> Variables:
    """Load and return environment variables from .env file.

    Returns:
        Variables: Configuration variables instance.
    """
    return Variables()
