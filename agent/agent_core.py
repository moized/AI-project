from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai

from rag.rag_pipeline import SimpleRAGPipeline
from tools.tool_implementation import TOOL_FUNCTIONS

load_dotenv()


class ResearchAgent:
    def __init__(self):
        self.rag = SimpleRAGPipeline()
        self.tools = TOOL_FUNCTIONS

        api_key = os.getenv("GEMINI_API_KEY")

        self.model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite",
        )

        self.client = genai.Client(api_key=api_key) if api_key else None

    def run(self, query: str) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "query": query,
            "answer": "",
            "sources": [],
            "tool_output": None,
        }

        query = query.strip()

        if not query:
            response["answer"] = "Lütfen bir soru veya araştırma konusu girin."
            return response

        query_lower = query.lower()

        # --------------------------------------------------------------
        # Simple deterministic tools
        # --------------------------------------------------------------

        greetings = {
            "merhaba",
            "selam",
            "hello",
            "hi",
            "hey",
            "nasılsın",
            "günaydın",
            "iyi günler",
        }

        if query_lower in greetings:
            response["answer"] = (
                "Merhaba! AI Research Assistant olarak "
                "dokümanlarınızı araştırmanıza yardımcı olabilirim."
            )
            return response

        if "tarih" in query_lower:
            date_func = self.tools.get("get_current_date")

            if date_func:
                result = date_func()
                response["tool_output"] = result
                response["answer"] = f"Mevcut tarih: {result}"
                return response

        if any(op in query for op in ["+", "-", "*", "/"]):
            calc_func = self.tools.get("calculator")

            if calc_func:
                for word in query.split():
                    if (
                        any(character.isdigit() for character in word)
                        and any(op in word for op in ["+", "-", "*", "/"])
                    ):
                        try:
                            result = calc_func(word)

                            response["tool_output"] = result
                            response["answer"] = (
                                f"Hesaplama sonucu: {result}"
                            )
                            return response
                        except Exception:
                            pass

        # --------------------------------------------------------------
        # RAG
        # --------------------------------------------------------------

        retrieved_chunks = self.rag.retrieve(
            query,
            top_k=5,
        )

        sources: List[Dict[str, Any]] = []
        context_parts: List[str] = []

        for chunk in retrieved_chunks:
            source = chunk.get("source", "unknown")
            page = chunk.get("page")

            citation = (
                f"[{source}, sayfa {page}]"
                if page
                else f"[{source}]"
            )

            sources.append(
                {
                    "id": chunk.get("id"),
                    "source": source,
                    "page": page,
                    "score": chunk.get("score"),
                }
            )

            context_parts.append(
                f"{citation}\n{chunk.get('text', '')}"
            )

        response["sources"] = sources

        context_text = "\n\n".join(context_parts)

        # --------------------------------------------------------------
        # Gemini
        # --------------------------------------------------------------

        if not self.client:
            response["answer"] = (
                "Gemini API anahtarı yapılandırılmamış. "
                "Projenin .env dosyasında GEMINI_API_KEY tanımlayın."
            )
            return response

        if context_text:
            prompt = f"""
You are an AI Research Assistant.

Answer the user's question using the retrieved document context below.

Rules:
- Prefer the provided documents over unsupported assumptions.
- If the documents do not contain enough information, say so clearly.
- Do not invent sources or facts.
- When making a claim based on a document, refer to its source label.
- Give a concise but useful answer.
- Answer in the same language as the user.

Retrieved context:

{context_text}

User question:

{query}
"""
        else:
            prompt = f"""
You are an AI Research Assistant.

No relevant document passages were retrieved for this question.

Tell the user clearly that the available documents do not contain enough
relevant information.

Do not invent document sources.

User question:

{query}
"""

        try:
            gemini_response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            response["answer"] = (
                gemini_response.text
                if gemini_response.text
                else "Model boş bir yanıt döndürdü."
            )

        except Exception as exc:
            response["answer"] = (
                "Gemini API çağrısı sırasında bir hata oluştu. "
                f"Hata: {exc}"
            )

        return response