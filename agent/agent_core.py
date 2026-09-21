import os
from typing import Dict, Any, List
from rag.rag_pipeline import SimpleRAGPipeline
from tools.tool_implementation import TOOL_FUNCTIONS

class ResearchAgent:
    def __init__(self):
        self.rag = SimpleRAGPipeline()
        self.tools = TOOL_FUNCTIONS

    def run(self, query: str) -> Dict[str, Any]:
        """Ajanın sorguyu işleme, araçları kullanma ve RAG'den bilgi getirme döngüsü."""
        response = {
            "query": query,
            "answer": "",
            "sources": [],
            "tool_output": None
        }

        query_lower = query.lower().strip()

        # Selamlama kontrolü
        if query_lower in ["merhaba", "selam", "hello", "hi", "hey"]:
            response["answer"] = "Merhaba! Ben AI Araştırma Asistanıyım. Size teknik dokümanlar hakkında yardımcı olabilir, hesaplama yapabilir veya güncel tarihi söyleyebilirim."
            return response

        # 1. Basit Araç Tetikleme Kontrolü (örn: tarih veya hesaplama)
        if "tarih" in query_lower:
            date_func = self.tools.get("get_current_date")
            if date_func:
                res = date_func()
                response["tool_output"] = f"Mevcut tarih: {res}"
                response["answer"] = f"Tarih bilgisi alındı: {res}"
                return response

        if any(op in query for op in ["+", "-", "*", "/"]):
            calc_func = self.tools.get("calculator")
            if calc_func:
                words = query.split()
                for word in words:
                    if any(c.isdigit() for c in word) and any(op in word for op in ["+", "-", "*", "/"]):
                        res = calc_func(word)
                        response["tool_output"] = res
                        response["answer"] = f"Hesaplama sonucu: {res}"
                        return response

        # 2. RAG ile Bilgi Getirme (Retrieval)
        retrieved_chunks = self.rag.retrieve(query, top_k=2)
        sources = []
        context_texts = []

        for chunk in retrieved_chunks:
            sources.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "score": chunk["score"]
            })
            context_texts.append(chunk["text"])

        response["sources"] = sources

        if context_texts:
            combined_context = "\n".join(context_texts)
            response["answer"] = f"Bulunan dokümanlara göre yanıt: {combined_context[:300]}..."
        else:
            response["answer"] = f"'{query}' ile ilgili eşleşen bir doküman bulamadım. Lütfen 'architecture', 'gemini' veya 'react' gibi anahtar kelimeler deneyin."

        return response
