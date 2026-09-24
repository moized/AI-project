from __future__ import annotations

import logging
import re
from typing import Any

from core.config import settings
from core.schemas import Source, ToolCallRecord
from providers.llm import GeminiLLMProvider, LLMProvider
from rag.rag_pipeline import SimpleRAGPipeline
from tools.tool_implementation import TOOL_DECLARATIONS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "based",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "give",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "please",
        "that",
        "the",
        "this",
        "to",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "you",
        "according",
        "provided",
        "context",
        "document",
        "documents",
        "knowledge",
        "base",
        "main",
        "topics",
        "discussed",
        "hakkında",
        "hakkındaki",
        "göre",
        "nedir",
        "nedeniyle",
        "nasıl",
        "hangi",
        "ile",
        "ve",
        "bir",
        "bu",
        "şu",
        "için",
        "olan",
        "olarak",
        "kısaca",
        "açıkla",
        "açıklayın",
        "açıklaması",
    }
)


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

    @staticmethod
    def _terms(text: str) -> set[str]:
        tokens = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
        return {
            token
            for token in tokens
            if len(token) >= 3 and not token.isdigit() and token not in _STOPWORDS
        }

    @classmethod
    def _is_document_request(cls, query: str) -> bool:
        lowered = query.casefold()
        markers = (
            "according to the document",
            "according to the documents",
            "provided document",
            "provided documents",
            "knowledge base",
            "in the documents",
            "in the document",
            "dokümanlara göre",
            "dokümanlarda",
            "belgelere göre",
            "belgelerde",
            "kaynaklara göre",
        )
        return any(marker in lowered for marker in markers)

    @classmethod
    def _context_is_relevant(
        cls,
        query: str,
        item: dict[str, Any],
    ) -> bool:
        score = float(item.get("score", 0.0))
        if score >= 0.68:
            return True

        query_terms = cls._terms(query)
        if not query_terms:
            return False

        context_terms = cls._terms(str(item.get("text", "")))
        overlap = query_terms & context_terms

        if len(overlap) >= 2:
            return True

        return any(len(term) >= 9 for term in overlap)

    @classmethod
    def _filter_retrieved(
        cls,
        query: str,
        retrieved: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if cls._is_document_request(query):
            return retrieved

        return [
            item
            for item in retrieved
            if cls._context_is_relevant(query, item)
        ]

    @staticmethod
    def _unique_sources(retrieved: list[dict[str, Any]]) -> list[Source]:
        seen: set[tuple[str, int | None]] = set()
        sources: list[Source] = []

        for item in retrieved:
            key = (str(item["source"]), item.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                Source(
                    id=item["id"],
                    source=item["source"],
                    page=item.get("page"),
                    score=item["score"],
                )
            )

        return sources

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
        relevant = self._filter_retrieved(query, retrieved)
        sources = self._unique_sources(relevant)

        context = []
        total_chars = 0
        for item in relevant:
            block = f"[{item['source']}"
            if item.get("page"):
                block += f", page {item['page']}"
            block += f"]\n{item['text']}"
            if total_chars + len(block) > settings.rag_max_context_chars:
                break
            context.append(block)
            total_chars += len(block)

        document_request = self._is_document_request(query)
        prompt = f"""You are an AI research assistant.
Use retrieved document evidence only when it directly helps answer the user's question.
Retrieved context is optional evidence, not mandatory content.

Important rules:
- Never treat an unrelated retrieved document as evidence for the answer.
- Never cite a document merely because it was retrieved.
- For general questions, answer from your general knowledge when the retrieved context is absent or irrelevant.
- When the user explicitly asks about the provided documents, only make document-grounded claims supported by the retrieved context. If the evidence is insufficient, say so clearly.
- Cite document claims with the exact source labels shown in the context.
- Use the same language as the user.
- You may use the provided tools when the user's request requires them.

User explicitly requested document-grounded evidence: {"yes" if document_request else "no"}

Retrieved context:
{chr(10).join(context) if context else "(no relevant context)"}

User question:
{query}
"""

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
