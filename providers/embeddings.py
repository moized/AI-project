from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from google import genai
from google.genai import types

from core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    dimension: int
    model_name: str

    def embed_documents(
        self,
        documents: Sequence[tuple[str, str]],
    ) -> list[list[float]]:
        ...

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class EmbeddingRequest:
    text: str
    mode: str


class LocalSentenceTransformerEmbeddingProvider:
    """Free local embedding provider; no API quota or network call at runtime."""
    model_name = "BAAI/bge-small-en-v1.5"
    dimension = 384

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.local_embedding_model
        self.dimension = settings.local_embedding_dimension
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, documents: Sequence[tuple[str, str]]) -> list[list[float]]:
        texts = [f"title: {title} | text: {text}" for title, text in documents]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            [f"Represent this sentence for searching relevant passages: {query}" for query in queries],
            normalize_embeddings=True,
        )
        return vectors.tolist()


class GeminiEmbeddingProvider:
    """Gemini Embedding 2 adapter.

    Embeddings 2 does not accept the old task_type parameter. For retrieval,
    task instructions are included in the text prompt instead.
    """

    def __init__(
        self,
        api_key: str | None = settings.gemini_api_key,
        model_name: str = settings.gemini_embedding_model,
        dimension: int = settings.gemini_embedding_dimension,
        batch_size: int = settings.rag_embedding_batch_size,
        max_retries: int = settings.rag_embedding_max_retries,
        retry_seconds: float = settings.rag_embedding_retry_seconds,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_seconds = retry_seconds
        self.client = genai.Client(api_key=api_key) if api_key else None

    @staticmethod
    def _document_text(title: str, text: str) -> str:
        safe_title = title.strip() or "none"
        return f"title: {safe_title} | text: {text.strip()}"

    @staticmethod
    def _query_text(query: str) -> str:
        return f"task: question answering | query: {query.strip()}"

    def _embed_batch(self, contents: Sequence[types.Content]) -> list[list[float]]:
        if not self.client:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured; embedding generation is unavailable."
            )

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=list(contents),
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension,
                    ),
                )
                vectors = [list(item.values) for item in response.embeddings]

                if len(vectors) != len(contents):
                    raise RuntimeError(
                        "Embedding provider returned a different number of vectors "
                        f"({len(vectors)}) than inputs ({len(contents)})."
                    )

                for vector in vectors:
                    if len(vector) != self.dimension:
                        raise RuntimeError(
                            f"Expected embedding dimension {self.dimension}, "
                            f"received {len(vector)}."
                        )

                return vectors

            except Exception as exc:
                message = str(exc).lower()
                transient = any(
                    marker in message
                    for marker in (
                        "429",
                        "resource_exhausted",
                        "quota",
                        "503",
                        "unavailable",
                        "deadline",
                    )
                )

                if not transient or attempt >= self.max_retries:
                    logger.exception("Embedding request failed.")
                    raise

                delay = min(
                    self.retry_seconds * (2**attempt) + (time.time() % 1),
                    60.0,
                )
                logger.warning(
                    "Transient embedding failure; retry %d/%d in %.2fs.",
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Embedding generation failed after retries.")

    def embed_documents(
        self,
        documents: Sequence[tuple[str, str]],
    ) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=self._document_text(title, text)
                        )
                    ],
                )
                for title, text in batch
            ]
            all_vectors.extend(self._embed_batch(contents))

        return all_vectors

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=self._query_text(query))],
                )
                for query in batch
            ]
            all_vectors.extend(self._embed_batch(contents))

        return all_vectors


class DeterministicEmbeddingProvider:
    """Small offline provider used only for tests and local contract checks."""

    dimension = 64
    model_name = "deterministic-test-v1"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        normalized = text.lower()
        for token in normalized.replace("\n", " ").split():
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension
            vector[index] += 1.0

        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_documents(
        self,
        documents: Sequence[tuple[str, str]],
    ) -> list[list[float]]:
        return [self._vector(f"{title} {text}") for title, text in documents]

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        return [self._vector(query) for query in queries]
