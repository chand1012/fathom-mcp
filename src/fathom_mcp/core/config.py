"""Configuration settings for the Fathom MCP server."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

cores = os.cpu_count() or 2
default_threads = cores // 2
PLACEHOLDER_API_KEYS = {
    "your-service-api-key-here",
    "your_service_api_key",
    "changeme",
    "replace-me",
}


class Settings(BaseSettings):
    """Application settings."""

    DEFAULT_EMBEDDING_MODEL_FILENAME: str = "bge-small-en-v1.5-q8_0.gguf"
    DEFAULT_EMBEDDING_MODEL_URL: str = (
        "https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/"
        "bge-small-en-v1.5-q8_0.gguf"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Fathom API settings
    fathom_api_url: str = Field(..., description="Fathom API base URL")
    fathom_api_key: str = Field(...,
                                description="Fathom API key for authentication")
    fathom_webhook_secret: str = Field(
        ..., description="Webhook secret for validation (whsec_... format)"
    )

    # Service API key — secures the /sync endpoint and MCP server
    service_api_key: str = Field(
        ..., description="API key for the sync endpoint and MCP server"
    )

    # Vector store settings
    vector_db_path: str = Field(
        default="./data/vectors.db", description="Path to SQLite database"
    )
    embedding_dimension: int = Field(
        default=384, description="Dimension of embedding vectors"
    )

    # Local llama.cpp embedding settings
    embedding_model_path: Optional[str] = Field(
        default=None, description="Path to a local GGUF embedding model"
    )
    embedding_model_url: str = Field(
        default=DEFAULT_EMBEDDING_MODEL_URL,
        description="Download URL for the local GGUF embedding model",
    )
    embedding_model: str = Field(
        default="bge-small-en-v1.5",
        description="Default embedding model identifier",
    )
    embedding_n_ctx: int = Field(
        default=0,
        description="Context size for the embedding model; 0 uses the model default",
    )
    embedding_n_batch: int = Field(
        default=512, description="Logical maximum batch size for embedding evaluation"
    )
    embedding_n_threads: Optional[int] = Field(
        default=default_threads, description="Threads for llama.cpp evaluation"
    )
    embedding_n_gpu_layers: int = Field(
        default=0, description="Number of model layers to offload to GPU"
    )
    embedding_use_mmap: bool = Field(
        default=True, description="Use mmap when loading the GGUF model"
    )
    embedding_use_mlock: bool = Field(
        default=False, description="Keep the GGUF model resident in RAM"
    )
    embedding_verbose: bool = Field(
        default=False, description="Enable verbose llama.cpp logging"
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_nested_delimiter": "__",
    }

    @model_validator(mode="after")
    def set_default_embedding_model_path(self) -> "Settings":
        """Store the embedding model beside the vector database by default."""
        service_api_key = self.service_api_key.strip()
        if not service_api_key or service_api_key.lower() in PLACEHOLDER_API_KEYS:
            raise ValueError(
                "SERVICE_API_KEY must be set to a non-placeholder value."
            )

        self.service_api_key = service_api_key

        if not self.embedding_model_path:
            db_path = Path(self.vector_db_path).expanduser()
            self.embedding_model_path = str(
                db_path.parent / self.DEFAULT_EMBEDDING_MODEL_FILENAME
            )
        return self


# Global settings instance
settings = Settings()  # type: ignore
