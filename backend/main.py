from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from agent.agent_core import ResearchAgent
from backend.database import ChatMessage, SessionLocal, init_db
from core.config import settings
from core.schemas import (
    DocumentListResponse,
    HealthResponse,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from rag.rag_pipeline import SimpleRAGPipeline

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AI Research Assistant API",
    version="2.1.0",
    description="Modular monolith API for document-grounded research assistance.",
    lifespan=lifespan,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache(maxsize=1)
def get_rag() -> SimpleRAGPipeline:
    return SimpleRAGPipeline()


@lru_cache(maxsize=1)
def get_agent() -> ResearchAgent:
    return ResearchAgent(rag=get_rag())


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Liveness probe: the web process is responding."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse, tags=["system"])
def readiness_check() -> HealthResponse:
    """Readiness probe for the database and local vector store."""
    try:
        init_db()
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        qdrant = QdrantClient(path=str(settings.qdrant_path))
        qdrant.get_collections()
        qdrant.close()
        return HealthResponse(status="ok")
    except Exception as exc:
        logger.exception("Readiness check failed.")
        raise HTTPException(status_code=503, detail="Service is not ready.") from exc


@app.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    tags=["documents"],
)
def list_documents() -> DocumentListResponse:
    directory = settings.rag_samples_dir
    directory.mkdir(parents=True, exist_ok=True)

    documents = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt"}
    )
    return DocumentListResponse(documents=documents)


@app.post(
    "/api/v1/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
) -> UploadResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".md", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail="Only PDF, Markdown and TXT files are supported.",
        )

    destination_dir = settings.rag_samples_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    max_bytes = settings.max_upload_mb * 1024 * 1024

    temp_path: Path | None = None
    total = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_dir,
            prefix=".upload-",
            suffix=extension,
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)

            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds MAX_UPLOAD_MB="
                            f"{settings.max_upload_mb}."
                        ),
                    )
                temp.write(chunk)

        temp_path.replace(destination)
        return UploadResponse(status="ok", filename=filename)

    except HTTPException:
        raise
    except OSError as exc:
        logger.exception("Document upload failed.")
        raise HTTPException(
            status_code=500,
            detail="Document upload failed.",
        ) from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning(
                    "Could not clean temporary upload file %s.",
                    temp_path,
                )


@app.post(
    "/api/v1/documents/index",
    response_model=IndexResponse,
    tags=["documents"],
)
def index_documents(
    rag: SimpleRAGPipeline = Depends(get_rag),  # noqa: B008
) -> IndexResponse:
    try:
        result = rag.index_documents()
        outcome = "ok" if result["failed"] == 0 else "partial"
        return IndexResponse(
            status=outcome,
            message="Document indexing completed.",
            result=result,
        )
    except Exception as exc:
        logger.exception("Document indexing failed.")
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed.",
        ) from exc


@app.post(
    "/api/v1/chat",
    response_model=QueryResponse,
    tags=["chat"],
)
def chat_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),  # noqa: B008
    agent: ResearchAgent = Depends(get_agent),  # noqa: B008
) -> QueryResponse:
    try:
        result = agent.run(request.query)
        db.add(
            ChatMessage(
                query=request.query,
                answer=result["answer"],
            )
        )
        db.commit()
        return QueryResponse.model_validate(result)
    except Exception as exc:
        db.rollback()
        logger.exception("Chat request failed.")
        raise HTTPException(
            status_code=500,
            detail="Chat request failed.",
        ) from exc
