"""Embedding generator backed by a local llama.cpp model."""

import asyncio
import logging
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from typing import Any, ClassVar, List, Optional, Union

import httpx

from fathom_mcp.core.config import settings

logger = logging.getLogger(__name__)


def get_default_embedding_model_path() -> Path:
    """Return the configured local GGUF path."""
    if not settings.embedding_model_path:
        raise ValueError("Embedding model path must be configured")
    return Path(settings.embedding_model_path).expanduser()


def ensure_embedding_model(download_if_missing: bool = True) -> Path:
    """Ensure the configured GGUF embedding model exists before startup."""
    model_path = get_default_embedding_model_path()
    if model_path.exists():
        return model_path

    if not download_if_missing:
        raise FileNotFoundError(
            f"Embedding model not found at {model_path}. "
            f"Download it from {settings.embedding_model_url} before startup."
        )

    download_url = settings.embedding_model_url
    parsed_url = urlparse(download_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported embedding model URL: {download_url}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading embedding model from %s to %s",
                download_url, model_path)
    with httpx.stream("GET", download_url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        with model_path.open("wb") as model_file:
            for chunk in response.iter_bytes():
                if chunk:
                    model_file.write(chunk)

    logger.info("Downloaded embedding model to %s", model_path)
    return model_path


class Embedder:
    """Generate embeddings using a local llama.cpp model kept in memory."""

    _model_cache: ClassVar[dict[tuple[object, ...], Any]] = {}
    _model_locks: ClassVar[dict[tuple[object, ...], Lock]] = {}
    _cache_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize and retain a local llama.cpp embedding model."""
        resolved_model_path = model_path or str(ensure_embedding_model())
        self.model_path = str(Path(resolved_model_path).expanduser())
        self.model = model or settings.embedding_model
        self._cache_key = (
            self.model_path,
            self.model,
            settings.embedding_n_ctx,
            settings.embedding_n_batch,
            settings.embedding_n_threads,
            settings.embedding_n_gpu_layers,
            settings.embedding_use_mmap,
            settings.embedding_use_mlock,
            settings.embedding_verbose,
        )
        self._llama = self._get_or_create_model()
        self._model_lock = self._get_model_lock()

        logger.info(
            "Initialized llama.cpp embedder with model_path=%s model=%s",
            self.model_path,
            self.model,
        )

    def _get_or_create_model(self) -> Any:
        """Load the llama.cpp model once and reuse it across embedder instances."""
        with self._cache_lock:
            cached_model = self._model_cache.get(self._cache_key)
            if cached_model is not None:
                return cached_model

            from llama_cpp import Llama

            logger.info("Loading llama.cpp embedding model from %s",
                        self.model_path)
            model = Llama(
                model_path=self.model_path,
                embedding=True,
                n_ctx=settings.embedding_n_ctx,
                n_batch=settings.embedding_n_batch,
                n_threads=settings.embedding_n_threads,
                n_gpu_layers=settings.embedding_n_gpu_layers,
                use_mmap=settings.embedding_use_mmap,
                use_mlock=settings.embedding_use_mlock,
                verbose=settings.embedding_verbose,
            )
            self._model_cache[self._cache_key] = model
            self._model_locks[self._cache_key] = Lock()
            return model

    def _get_model_lock(self) -> Lock:
        """Return the lock guarding access to the shared llama.cpp model."""
        with self._cache_lock:
            model_lock = self._model_locks.get(self._cache_key)
            if model_lock is None:
                model_lock = Lock()
                self._model_locks[self._cache_key] = model_lock
            return model_lock

    def _create_embedding(self, texts: List[str]) -> dict[str, Any]:
        """Run the blocking llama.cpp embedding call behind a model lock.

        Embeds each text individually to avoid exceeding the model's context
        window (512 tokens for bge-small), which causes llama_decode to return -1.
        """
        with self._model_lock:
            data = []
            for i, text in enumerate(texts):
                result = self._llama.create_embedding(text, model=self.model)
                embedding = result["data"][0]["embedding"]
                data.append({"index": i, "embedding": embedding})
            return {"data": data}

    async def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings from the in-process llama.cpp model."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        try:
            result = await asyncio.to_thread(self._create_embedding, texts)
            data = result.get("data", [])
            embeddings = sorted(data, key=lambda item: item["index"])
            embedding_vectors = [item["embedding"] for item in embeddings]

            if len(embedding_vectors) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, got {len(embedding_vectors)}"
                )

            logger.debug("Generated %s embeddings", len(embedding_vectors))
            return embedding_vectors
        except Exception as exc:
            logger.error("Unexpected error in get_embeddings: %s", exc)
            raise


async def get_embeddings(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Convenience function to get embeddings using the default embedder instance.

    Args:
        texts: A single string or a list of strings to embed.

    Returns:
        A list of embeddings (each embedding is a list of floats).
    """
    embedder = Embedder()
    return await embedder.get_embeddings(texts)
