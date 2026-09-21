import os
from typing import Dict, Any, List
from google import genai
from rag.rag_pipeline import SimpleRAGPipeline
from tools.tool_implementation import TOOL_FUNCTIONS

class ResearchAgent:
    def __init__(self):
        self.rag = SimpleRAGPipeline()
        self.tools = TOOL_FUNCTIONS
        
        # Google GenAI API yapılandırması (google-genai paketi ve güncel rehberlik)
        api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def run(self, query: str) -> Dict[str, Any]:
        """Ajanın sorguyu işleme, araçları kullanma, RAG'den bilgi getirme ve Google GenAI ile yanıt üretme döngüsü."""
        response = {
            "query": query,
            "answer": "",
            "sources": [],
            "tool_output": None
        }

        query_lower = query.lower().strip()

        # Esnek selamlama ve yardım talebi kontrolü
        greetings = ["merhaba", "selam", "hello", "hi", "hey", "nasılsın", "günaydın", "iyi günler"]
        if any(g in query_lower for g in greetings) or "nasıl yardımcı" in query_lower:
            response["answer"] = "Merhaba! Ben AI Araştırma Asistanıyım. Size yüklediğiniz dokümanlar hakkında yardımcı olabilir, hesaplama yapabilir veya güncel tarihi söyleyebilirim. Nasıl yardımcı olabilirim?"
            return response

        # 1. Basit Araç Tetikleme Kontrolü (tarih veya hesaplama)
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

        # 2. RAG ile Bilgi Getirme (Retrieval) - Kara Kutu Olarak Kullanım
        retrieved_chunks = self.rag.retrieve(query, top_k=3)
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

        # 3. Google GenAI Entegrasyonu ile Yanıt Üretimi
        context_str = "\n\n".join(context_texts) if context_texts else "İlgili doküman bulunamadı."
        
        prompt = f"""Sen profesyonel bir AI Araştırma Asistanısın. Kullanıcının sorusunu aşağıda sağlanan doküman bağlamını kullanarak yanıtla. Eğer bağlamda yeterli bilgi yoksa, genel bilginle ancak dürüstçe yardımcı ol.

Bağlam:
{context_str}

Kullanıcı Sorusu: {query}
Yanıt:"""

        if self.client:
            try:
                gemini_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                response["answer"] = gemini_response.text
            except Exception as e:
                if context_texts:
                    response["answer"] = f"[LLM Hatası: {str(e)}] Bulunan doküman özeti:\n{context_str[:500]}"
                else:
                    response["answer"] = f"Google GenAI API hatası oluştu ve eşleşen doküman bulunamadı: {str(e)}"
        else:
            if context_texts:
                response["answer"] = f"[GEMINI_API_KEY tanımlı değil, RAG Özeti] {context_str[:600]}..."
            else:
                response["answer"] = f"'{query}' ile ilgili eşleşen bir doküman bulamadım. Lütfen geçerli bir anahtar kelime girin veya doküman yükleyin."

        return response
