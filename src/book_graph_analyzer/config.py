"""Configuration management for Book Graph Analyzer."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BGA_",
    )

    # Neo4j connection
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="bookgraph123")

    # Ollama (local LLM)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")
    
    # Hugging Face Inference API
    hf_api_key: str = Field(default="")
    hf_model: str = Field(default="meta-llama/Llama-3.1-70B-Instruct")
    llm_provider: str = Field(default="ollama", description="ollama or huggingface")

    # Paths
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))

    # Processing settings
    batch_size: int = Field(default=50, description="Sentences to process per batch")
    max_sentence_length: int = Field(default=1000, description="Skip sentences longer than this")

    @property
    def texts_dir(self) -> Path:
        return self.data_dir / "texts"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def seeds_dir(self) -> Path:
        return self.data_dir / "seeds"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
