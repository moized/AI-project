from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("RAGPipeline")


# =====================================================================
# DATA CONTRACTS
# =====================================================================

@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    page: Optional[int]
    text: str
    chunk_index: int


@dataclass(frozen=True)
class SearchResult:
    id: str
    source: str
    page: Optional[int]
    text: str
    score: float


# =====================================================================
# SQLITE EMBEDDING CACHE
# =====================================================================

class SQLiteEmbeddingCache:
    """Thread-safe, single-file embedding cache replacing per-file JSON storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    hash_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get(self, hash_key: str) -> Optional[List[float]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT vector FROM embedding_cache WHERE hash_key = ?",
                (hash_key,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set_many(self, items: List[tuple[str, str, int, List[float]]]):
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (hash_key, model, dimension, vector)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (key, model, dim, json.dumps(vector))
                    for key, model, dim, vector in items
                ],
            )
            conn.commit()


# =====================================================================
# MAIN PIPELINE
# =====================================================================

class ProductionRAGPipeline:
    """
    Production-hardened RAG indexing and retrieval engine using:
    - Qdrant Vector DB
    - Gemini text-embedding-004 with asymmetric task types
    - Recursive boundary-preserving chunking
    - SQLite persistence cache
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

    def __init__(self):
        # Paths
        self.samples_dir = Path(os.getenv("RAG_SAMPLES_DIR", "data/samples"))
        self.qdrant_path = Path(os.getenv("QDRANT_PATH", "qdrant_storage"))
        self.manifest_path = self.qdrant_path / "index_manifest.json"
        
        # Vector & Embedding Config
        self.collection_name = os.getenv("RAG_COLLECTION_NAME", "research_docs")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self.embedding_dimension = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))
        
        # Chunking & Retrieval Parameters
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
        self.embedding_batch_size = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))
        self.max_retries = int(os.getenv("RAG_EMBEDDING_MAX_RETRIES", "5"))
        self.base_retry_seconds = float(os.getenv("RAG_EMBEDDING_RETRY_SECONDS", "2.0"))

        # Initialization
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)

        self.cache = SQLiteEmbeddingCache(self.qdrant_path / "embedding_cache.db")

        # Gemini Client Initialization
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("[RAG] GEMINI_API_KEY is not set. Inference will fail.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

        # Local Persistent Qdrant
        self.qdrant = QdrantClient(path=str(self.qdrant_path))
        self._ensure_collection()

    # -----------------------------------------------------------------
    # COLLECTION MANAGEMENT
    # -----------------------------------------------------------------

    def _ensure_collection(self):
        collections = [c.name for c in self.qdrant.get_collections().collections]

        if self.collection_name not in collections:
            logger.info("[RAG] Creating Qdrant collection: %s", self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            return

        info = self.qdrant.get_collection(self.collection_name)
        vectors_config = info.config.params.vectors
        size = (
            next(iter(vectors_config.values())).size
            if isinstance(vectors_config, dict)
            else vectors_config.size
        )

        if size != self.embedding_dimension:
            logger.warning(
                "[RAG] Dimension mismatch detected (%d != %d). Rebuilding collection.",
                size,
                self.embedding_dimension,
            )
            self.qdrant.delete_collection(self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            self._save_manifest({})

    # -----------------------------------------------------------------
    # CHUNKING & EXTRACTION
    # -----------------------------------------------------------------

    def _read_document(self, filepath: Path) -> List[Dict[str, Any]]:
        suffix = filepath.suffix.lower()

        if suffix == ".pdf":
            records = []
            try:
                reader = PdfReader(str(filepath))
                for page_num, page in enumerate(reader.pages, start=1):
                    # Using layout mode keeps columnar text and tables intact
                    text = (page.extract_text(extraction_mode="layout") or "").strip()
                    if text:
                        records.append({"text": text, "page": page_num})
                return records
            except Exception as exc:
                logger.error("[RAG] PDF Extraction failed for %s: %s", filepath.name, exc)
                return []

        # Plain text / Markdown
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace").strip()
            return [{"text": content, "page": None}] if content else []
        except Exception as exc:
            logger.error("[RAG] File read error for %s: %s", filepath.name, exc)
            return []

    def _recursive_split(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """Splits text preserving logical boundaries (paragraphs, sentences, words)."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)
        chunks = []
        current_chunk: List[str] = []
        current_len = 0

        for segment in splits:
            seg_len = len(segment) + len(separator)
            if current_len + seg_len > self.chunk_size and current_chunk:
                joined = separator.join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
                
                # Apply overlap
                while current_len > self.chunk_overlap and current_chunk:
                    removed = current_chunk.pop(0)
                    current_len -= len(removed) + len(separator)

            current_chunk.append(segment)
            current_len += seg_len

        if current_chunk:
            final_text = separator.join(current_chunk).strip()
            if final_text:
                chunks.append(final_text)

        # Process any oversized segment that didn't fit
        result = []
        for c in chunks:
            if len(c) > self.chunk_size and new_separators:
                result.extend(self._recursive_split(c, new_separators))
            else:
                result.append(c)

        return result

    def _build_chunks(self, filepath: Path) -> List[DocumentChunk]:
        records = self._read_document(filepath)
        chunks: List[DocumentChunk] = []
        total_idx = 0

        for record in records:
            text_segments = self._recursive_split(record["text"])
            for segment in text_segments:
                content_hash = hashlib.sha256(segment.encode("utf-8")).hexdigest()
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{filepath.name}:{record.get('page')}:{total_idx}:{content_hash}",
                    )
                )
                chunks.append(
                    DocumentChunk(
                        id=point_id,
                        source=filepath.name,
                        page=record.get("page"),
                        text=segment,
                        chunk_index=total_idx,
                    )
                )
                total_idx += 1

        return chunks

    # -----------------------------------------------------------------
    # EMBEDDING GENERATION WITH RETRY & CACHING
    # -----------------------------------------------------------------

    def _generate_cache_key(self, text: str, task_type: str) -> str:
        payload = f"{self.embedding_model}|{self.embedding_dimension}|{task_type}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _embed_with_retry(
        self, texts: List[str], task_type: types.TaskType
    ) -> List[List[float]]:
        """Handles rate limits (429) using exponential backoff with jitter."""
        if not self.client:
            raise RuntimeError("Gemini Client not initialized. Check GEMINI_API_KEY.")

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.embedding_dimension,
                    ),
                )
                return [list(item.values) for item in response.embeddings]
            except Exception as exc:
                err_str = str(exc).lower()
                is_transient = any(
                    code in err_str
                    for code in ["429", "resource_exhausted", "quota", "503", "unavailable"]
                )
                if attempt >= self.max_retries or not is_transient:
                    logger.error("[RAG] Unrecoverable embedding error: %s", exc)
                    raise

                delay = min(self.base_retry_seconds * (2 ** attempt) + (time.time() % 1), 60.0)
                logger.warning(
                    "[RAG] Rate limited. Retrying attempt %d/%d in %.2fs...",
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Embedding generation exceeded maximum retries.")

    def _get_or_create_embeddings(
        self, texts: List[str], task_type: types.TaskType
    ) -> List[List[float]]:
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        cache_keys: List[str] = []

        # Check Cache First
        for i, text in enumerate(texts):
            key = self._generate_cache_key(text, task_type)
            cache_keys.append(key)
            cached_vector = self.cache.get(key)
            if cached_vector:
                embeddings[i] = cached_vector
            else:
                missing_indices.append(i)

        # Batch Fetch Missing from API
        if missing_indices:
            for start in range(0, len(missing_indices), self.embedding_batch_size):
                batch_idxs = missing_indices[start : start + self.embedding_batch_size]
                batch_texts = [texts[idx] for idx in batch_idxs]
                
                computed = self._embed_with_retry(batch_texts, task_type)
                
                cache_updates = []
                for idx, vector in zip(batch_idxs, computed):
                    embeddings[idx] = vector
                    cache_updates.append(
                        (cache_keys[idx], self.embedding_model, self.embedding_dimension, vector)
                    )

                self.cache.set_many(cache_updates)

        return [e for e in embeddings if e is not None]

    # -----------------------------------------------------------------
    # MANIFEST & INDEXING
    # -----------------------------------------------------------------

    def _file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, manifest: Dict[str, Any]):
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def index_documents(self) -> Dict[str, Any]:
        """Incrementally syncs local files with the Qdrant vector index."""
        files = sorted(
            [p for p in self.samples_dir.iterdir() if p.suffix.lower() in self.SUPPORTED_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        manifest = self._load_manifest()
        current_filenames = {p.name for p in files}
        old_filenames = set(manifest.keys())

        # Cleanup deleted documents
        for removed in (old_filenames - current_filenames):
            point_ids = manifest.get(removed, {}).get("point_ids", [])
            if point_ids:
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids,
                )
            manifest.pop(removed, None)

        metrics = {"indexed": 0, "unchanged": 0, "failed": 0, "chunks": 0}

        for file in files:
            source = file.name
            current_hash = self._file_hash(file)
            entry = manifest.get(source)

            # Check if file has changed
            if (
                entry
                and entry.get("hash") == current_hash
                and entry.get("chunk_size") == self.chunk_size
                and entry.get("chunk_overlap") == self.chunk_overlap
                and entry.get("model") == self.embedding_model
            ):
                metrics["unchanged"] += 1
                continue

            chunks = self._build_chunks(file)
            if not chunks:
                logger.warning("[RAG] Empty or unextractable document: %s", source)
                metrics["failed"] += 1
                continue

            # Embed with RETRIEVAL_DOCUMENT task type
            vectors = self._get_or_create_embeddings(
                texts=[c.text for c in chunks],
                task_type=types.TaskType.RETRIEVAL_DOCUMENT,
            )

            points = [
                PointStruct(
                    id=c.id,
                    vector=v,
                    payload={"source": c.source, "page": c.page, "text": c.text},
                )
                for c, v in zip(chunks, vectors)
            ]

            # Atomic swap: remove stale points for this doc, then upsert new points
            if entry and entry.get("point_ids"):
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=entry["point_ids"],
                )

            self.qdrant.upsert(collection_name=self.collection_name, points=points)

            manifest[source] = {
                "hash": current_hash,
                "model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "point_ids": [c.id for c in chunks],
            }

            metrics["indexed"] += 1
            metrics["chunks"] += len(chunks)

        self._save_manifest(manifest)
        return metrics

    # -----------------------------------------------------------------
    # SEARCH / RETRIEVAL
    # -----------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query = query.strip()
        if not query:
            return []

        # Queries must use RETRIEVAL_QUERY task type
        query_vectors = self._get_or_create_embeddings(
            texts=[query],
            task_type=types.TaskType.RETRIEVAL_QUERY,
        )
        query_vector = query_vectors[0]

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            SearchResult(
                id=str(pt.id),
                source=pt.payload.get("source", "unknown"),
                page=pt.payload.get("page"),
                text=pt.payload.get("text", ""),
                score=pt.score,
            )
            for pt in results.points
        ]


# =====================================================================
# VERIFICATION RUNNER
# =====================================================================

if __name__ == "__main__":
    pipeline = ProductionRAGPipeline()
    print("\n--- Indexing Documents ---")
    stats = pipeline.index_documents()
    print("Indexing Complete:", stats)

    test_query = "What are the core findings of the report?"
    print(f"\n--- Searching: '{test_query}' ---")
    hits = pipeline.retrieve(test_query, top_k=3)
    for hit in hits:
        print(f"[{hit.score:.4f}] {hit.source} (Page {hit.page}): {hit.text[:80]}...")