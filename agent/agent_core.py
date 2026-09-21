import os
from typing import Dict, Any, List
from google import genai
from rag.rag_pipeline import SimpleRAGPipeline
from tools.tool_implementation import TOOL_FUNCTIONS

class ResearchAgent:
    def __init__(self):
        self.rag = SimpleRAGPipeline()
        self.tools = TOOL_FUNCTIONS
        
        # Google GenAI API yapılandırması (google-genai paketi)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_name = 'gemini-2.5-flash' # veya uygun güncel model
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
                response["answer"] = f"Google GenAI API hatası oluştu: {str(e)}. Bulunan doküman özeti: {context_str[:300]}"
        else:
            # API anahtarı yoksa fallback olarak RAG bağlamını döndür
            if context_texts:
                response["answer"] = f"[GEMINI_API_KEY bulunamadı, RAG Özeti] {context_str[:400]}..."
            else:
                response["answer"] = f"'{query}' ile ilgili eşleşen bir doküman bulamadım."

        return response
