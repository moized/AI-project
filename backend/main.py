from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Any, Dict

from backend.database import SessionLocal, init_db, ChatMessage
from agent.agent_core import ResearchAgent

app = FastAPI(title="AI Research Assistant API", version="1.0.0")

@app.on_event("startup")
def startup_event():
    init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

agent = ResearchAgent()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    tool_output: Any = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest, db: Session = Depends(get_db)):
    try:
        result = agent.run(request.query)
        
        db_message = ChatMessage(
            query=request.query,
            answer=result.get("answer", "")
        )
        db.add(db_message)
        db.commit()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
