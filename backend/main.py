from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Any, Dict

from backend.database import SessionLocal, init_db, ChatMessage
from agent.agent_core import ResearchAgent


app = FastAPI(
    title="AI Research Assistant API",
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    init_db()


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------

agent = ResearchAgent()


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    tool_output: Any = None


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


# ---------------------------------------------------------------------
# Document indexing
# ---------------------------------------------------------------------

@app.post("/documents/index")
def index_documents():
    """
    Index new or changed documents from RAG_SAMPLES_DIR.

    The RAG pipeline uses its manifest and embedding cache,
    so unchanged documents should not be embedded again.
    """

    try:
        result = agent.rag.index_documents()

        return {
            "status": "ok",
            "message": "Doküman indeksleme tamamlandı.",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Doküman indeksleme hatası: {exc}",
        )


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=QueryResponse,
)
def chat_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    try:
        result = agent.run(
            request.query
        )

        db_message = ChatMessage(
            query=request.query,
            answer=result.get(
                "answer",
                "",
            ),
        )

        db.add(db_message)
        db.commit()

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )