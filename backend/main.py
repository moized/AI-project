from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from agent.agent_core import ResearchAgent
from backend.database import ChatMessage, SessionLocal, init_db
from core.config import settings
from core.schemas import HealthResponse, IndexResponse, QueryRequest, QueryResponse
from rag.rag_pipeline import SimpleRAGPipeline

logging.basicConfig(level=settings.log_level)

rag = SimpleRAGPipeline()
agent = ResearchAgent(rag=rag)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AI Research Assistant API",
    version="2.0.0",
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@app.post("/documents/index", response_model=IndexResponse)
def index_documents():
    try:
        result = rag.index_documents()
        return {
            "status": "ok",
            "message": "Document indexing completed.",
            "result": result,
        }
    except Exception as exc:
        logging.exception("Document indexing failed.")
        raise HTTPException(status_code=500, detail="Document indexing failed.") from exc


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest, db: Session = Depends(get_db)):
    try:
        result = agent.run(request.query)
        db.add(ChatMessage(query=request.query, answer=result["answer"]))
        db.commit()
        return result
    except Exception as exc:
        logging.exception("Chat request failed.")
        db.rollback()
        raise HTTPException(status_code=500, detail="Chat request failed.") from exc
