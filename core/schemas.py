from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    source: str
    page: int | None = None
    score: float


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8000)


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class IndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    message: str
    result: dict[str, int]


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    filename: str


class DocumentListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
