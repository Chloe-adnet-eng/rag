#%%
from pydantic_settings import BaseSettings
#%%
class DatabaseSettings(BaseSettings):
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str
    COLLECTION_NAME: str
    OPEN_AI_EMBEDDINGS_SIZE: int
    TOP_VECTOR_SEARCH: int

    class Config:
        env_file = ".env"

database_settings = DatabaseSettings()
OPENAI_KEY = database_settings.OPENAI_API_KEY
EMBEDDING_MODEL = database_settings.EMBEDDING_MODEL
COLLECTION_NAME = database_settings.COLLECTION_NAME
OPEN_AI_EMBEDDINGS_SIZE = database_settings.OPEN_AI_EMBEDDINGS_SIZE
TOP_VECTOR_SEARCH = database_settings.TOP_VECTOR_SEARCH

# %%
