from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = int(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = float(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    gemini_embedding_model: str
    gemini_embedding_dimension: int
    local_embedding_model: str
    local_embedding_dimension: int

    llm_provider: str
    embedding_provider: str

    rag_samples_dir: Path
    qdrant_path: Path
    rag_collection_name: str
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_embedding_batch_size: int
    rag_embedding_max_retries: int
    rag_embedding_retry_seconds: float
    rag_min_retrieval_score: float
    rag_top_k: int
    rag_max_context_chars: int

    database_url: str
    max_upload_mb: int
    max_tool_rounds: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        chunk_size = _env_int("RAG_CHUNK_SIZE", 1200, 100)
        chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", 200, 0)

        if chunk_overlap >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            gemini_embedding_model=os.getenv(
                "GEMINI_EMBEDDING_MODEL",
                "gemini-embedding-2",
            ),
            gemini_embedding_dimension=_env_int(
                "GEMINI_EMBEDDING_DIMENSION",
                768,
                128,
            ),
            llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
            rag_samples_dir=Path(os.getenv("RAG_SAMPLES_DIR", "data/samples")),
            qdrant_path=Path(os.getenv("QDRANT_PATH", "qdrant_storage")),
            rag_collection_name=os.getenv(
                "RAG_COLLECTION_NAME",
                "research_docs",
            ),
            rag_chunk_size=chunk_size,
            rag_chunk_overlap=chunk_overlap,
            rag_embedding_batch_size=_env_int(
                "RAG_EMBEDDING_BATCH_SIZE",
                16,
                1,
            ),
            rag_embedding_max_retries=_env_int(
                "RAG_EMBEDDING_MAX_RETRIES",
                5,
                0,
            ),
            rag_embedding_retry_seconds=_env_float(
                "RAG_EMBEDDING_RETRY_SECONDS",
                2.0,
                0.1,
            ),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            local_embedding_dimension=_env_int(
                "LOCAL_EMBEDDING_DIMENSION", 384, 128
            ),
            rag_min_retrieval_score=_env_float(
                "RAG_MIN_RETRIEVAL_SCORE",
                0.20,
                -1.0,
            ),
            rag_top_k=_env_int("RAG_TOP_K", 5, 1),
            rag_max_context_chars=_env_int(
                "RAG_MAX_CONTEXT_CHARS",
                12000,
                1000,
            ),
            database_url=os.getenv(
                "DATABASE_URL",
                "sqlite:///./runtime/research_assistant.db",
            ),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 25, 1),
            max_tool_rounds=_env_int("MAX_TOOL_ROUNDS", 3, 1),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


settings = Settings.from_env()
