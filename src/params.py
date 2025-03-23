from pydantic_settings import BaseSettings, SettingsConfigDict


class Variables(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    
def get_variables() -> Variables:
    return Variables()


