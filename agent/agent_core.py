from __future__ import annotations

import logging
from typing import Any

from core.config import settings
from core.schemas import Source, ToolCallRecord
from providers.llm import GeminiLLMProvider, LLMProvider
from rag.rag_pipeline import SimpleRAGPipeline
from tools.tool_implementation import TOOL_DECLARATIONS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(
        self,
        rag: SimpleRAGPipeline | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.rag = rag or SimpleRAGPipeline()
        self.llm = llm or GeminiLLMProvider()
        self.tools = TOOL_FUNCTIONS

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        return tool(**arguments)

    def run(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {
                "query": query,
                "answer": "Please enter a question or research topic.",
                "sources": [],
                "tool_calls": [],
            }

        retrieved = self.rag.retrieve(query, top_k=settings.rag_top_k)
        sources = [
            Source(
                id=item["id"],
                source=item["source"],
                page=item.get("page"),
                score=item["score"],
            )
            for item in retrieved
        ]

        context = []
        total_chars = 0
        for item in retrieved:
            block = f"[{item['source']}"
            if item.get("page"):
                block += f", page {item['page']}"
            block += f"]\n{item['text']}"
            if total_chars + len(block) > settings.rag_max_context_chars:
                break
            context.append(block)
            total_chars += len(block)

        prompt = f"""You are an AI research assistant.
Answer the user's question using the retrieved context when it is relevant.
Do not invent document evidence. If the context is insufficient, say so.
Cite document claims with the source labels shown in the context.
Use the same language as the user.
You may use the provided tools when the user's request requires them.

Retrieved context:
{chr(10).join(context) if context else "(no relevant context)"}

User question:
{query}
"""

        if not self.llm:
            raise RuntimeError("LLM provider is not configured.")

        result = self.llm.answer(
            prompt=prompt,
            tools=TOOL_DECLARATIONS,
            tool_executor=self._execute_tool,
        )

        return {
            "query": query,
            "answer": result.text,
            "sources": [source.model_dump() for source in sources],
            "tool_calls": [
                ToolCallRecord(
                    name=call.name,
                    arguments=call.arguments,
                    result=call.result,
                ).model_dump()
                for call in result.tool_calls
            ],
        }
