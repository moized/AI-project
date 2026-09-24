from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.config import settings
from providers.embeddings import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    LocalSentenceTransformerEmbeddingProvider,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    page: int | None
    text: str
    chunk_index: int


@dataclass(frozen=True)
class SearchResult:
    id: str
    source: str
    page: int | None
    text: str
    score: float


class SimpleRAGPipeline:
    """Document ingestion and retrieval with replaceable embeddings and local Qdrant."""

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

    def __init__(
        self,
        samples_dir: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.samples_dir = Path(samples_dir or settings.rag_samples_dir)
        self.qdrant_path = settings.qdrant_path
        self.collection_name = settings.rag_collection_name
        self.chunk_size = settings.rag_chunk_size
        self.chunk_overlap = settings.rag_chunk_overlap
        self.embedding_model = settings.gemini_embedding_model
        self.embedding_dimension = settings.gemini_embedding_dimension

        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        elif settings.embedding_provider == "local":
            self.embedding_provider = LocalSentenceTransformerEmbeddingProvider()
        elif settings.embedding_provider == "gemini":
            self.embedding_provider = GeminiEmbeddingProvider()
        else:
            raise ValueError(
                f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}"
            )

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.qdrant_path / "index_manifest.json"
        self.qdrant = QdrantClient(path=str(self.qdrant_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        names = {c.name for c in self.qdrant.get_collections().collections}
        dimension = self.embedding_provider.dimension

        if self.collection_name not in names:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            return

        info = self.qdrant.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        size = next(iter(vectors.values())).size if isinstance(vectors, dict) else vectors.size

        if size != dimension:
            raise RuntimeError(
                f"Qdrant collection dimension {size} does not match "
                f"embedding provider dimension {dimension}. "
                "Use a fresh QDRANT_PATH or collection when changing embeddings."
            )

    def _read_document(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".pdf":
            records = []
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    records.append({"text": text, "page": page_number})
            return records

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [{"text": text, "page": None}] if text else []

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", " ", ""]
        pieces = [text]

        for separator in separators:
            if separator and separator not in text:
                continue

            new_pieces = []
            for piece in pieces:
                if len(piece) <= self.chunk_size:
                    new_pieces.append(piece)
                else:
                    new_pieces.extend(
                        piece.split(separator) if separator else list(piece)
                    )
            pieces = new_pieces

            if all(len(piece) <= self.chunk_size for piece in pieces):
                break

        chunks: list[str] = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            candidate = f"{current} {piece}".strip()

            if current and len(candidate) > self.chunk_size:
                chunks.append(current)
                overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = f"{overlap} {piece}".strip()
            else:
                current = candidate

        if current:
            chunks.append(current)

        return chunks

    def _build_chunks(self, path: Path) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        index = 0

        for record in self._read_document(path):
            for text in self._split_text(record["text"]):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{path.name}:{record['page']}:{index}:{digest}",
                    )
                )
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        source=path.name,
                        page=record["page"],
                        text=text,
                        chunk_index=index,
                    )
                )
                index += 1

        return chunks

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}

        try:
            return json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            logger.warning("Invalid manifest; rebuilding index state.")
            return {}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        temp = self.manifest_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.manifest_path)

    def index_documents(self) -> dict[str, int]:
        files = sorted(
            [
                p
                for p in self.samples_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ],
            key=lambda p: p.name.lower(),
        )

        manifest = self._load_manifest()
        current = {p.name for p in files}
        metrics = {
            "indexed": 0,
            "unchanged": 0,
            "failed": 0,
            "chunks": 0,
        }

        for removed in set(manifest) - current:
            try:
                old_ids = manifest[removed].get("point_ids", [])
                if old_ids:
                    self.qdrant.delete(
                        collection_name=self.collection_name,
                        points_selector=old_ids,
                    )
                manifest.pop(removed, None)
            except Exception:
                logger.exception("Failed to remove deleted document %s.", removed)
                metrics["failed"] += 1

        for path in files:
            file_hash = self._file_hash(path)
            old = manifest.get(path.name)

            signature = {
                "hash": file_hash,
                "model": self.embedding_provider.model_name,
                "dimension": self.embedding_provider.dimension,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

            if old and all(old.get(key) == value for key, value in signature.items()):
                metrics["unchanged"] += 1
                continue

            try:
                chunks = self._build_chunks(path)
                if not chunks:
                    logger.warning("Empty or unextractable document: %s", path.name)
                    metrics["failed"] += 1
                    continue

                vectors = self.embedding_provider.embed_documents(
                    [(chunk.source, chunk.text) for chunk in chunks]
                )

                if len(vectors) != len(chunks):
                    raise RuntimeError(
                        f"Embedding provider returned {len(vectors)} vectors "
                        f"for {len(chunks)} chunks."
                    )

                points = [
                    PointStruct(
                        id=chunk.id,
                        vector=vector,
                        payload={
                            "source": chunk.source,
                            "page": chunk.page,
                            "text": chunk.text,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors)
                ]

                new_ids = {chunk.id for chunk in chunks}
                old_ids = set(old.get("point_ids", [])) if old else set()
                stale_ids = list(old_ids - new_ids)

                # Upsert the new version first. Delete only points that do not
                # belong to the new version, so shared unchanged chunks survive.
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )

                if stale_ids:
                    self.qdrant.delete(
                        collection_name=self.collection_name,
                        points_selector=stale_ids,
                    )

                manifest[path.name] = {
                    **signature,
                    "point_ids": [chunk.id for chunk in chunks],
                }

                metrics["indexed"] += 1
                metrics["chunks"] += len(chunks)

            except Exception:
                logger.exception("Failed to index %s", path.name)
                metrics["failed"] += 1

        self._save_manifest(manifest)
        return metrics

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        vector = self.embedding_provider.embed_queries([query])[0]

        result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k or settings.rag_top_k,
            with_payload=True,
        )

        output = []
        for point in result.points:
            score = float(point.score)
            if score < settings.rag_min_retrieval_score:
                continue

            payload = point.payload or {}
            output.append(
                {
                    "id": str(point.id),
                    "source": str(payload.get("source", "unknown")),
                    "page": payload.get("page"),
                    "text": str(payload.get("text", "")),
                    "score": score,
                }
            )

        return output

    def _load_and_chunk_documents(self) -> list[dict[str, Any]]:
        chunks = []
        for path in sorted(self.samples_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                chunks.extend(self._build_chunks(path))

        return [
            {
                "id": chunk.id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]


ProductionRAGPipeline = SimpleRAGPipeline
