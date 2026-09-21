import os
import glob
from typing import List, Dict, Any

class SimpleRAGPipeline:
    def __init__(self, samples_dir: str = "data/samples"):
        self.samples_dir = samples_dir
        self.chunks = self._load_and_chunk_documents()

    def _load_and_chunk_documents(self) -> List[Dict[str, Any]]:
        chunks = []
        if not os.path.exists(self.samples_dir):
            return chunks
        
        pattern = os.path.join(self.samples_dir, "*.md")
        for filepath in glob.glob(pattern):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                filename = os.path.basename(filepath)
                # Basit paragraf bazlı chunking
                paragraphs = content.split("\n\n")
                for i, para in enumerate(paragraphs):
                    if para.strip():
                        chunks.append({
                            "id": f"{filename}_chunk_{i}",
                            "text": para.strip(),
                            "source": filename,
                            "score": 1.0
                        })
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Sorgu ile kelime eşleşmesine dayalı basit ve hafif retrieval (RAG)."""
        query_terms = set(query.lower().split())
        scored_chunks = []

        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            match_count = sum(1 for term in query_terms if term in text_lower)
            if match_count > 0:
                scored_chunks.append({
                    **chunk,
                    "score": float(match_count)
                })

        # Skora göre sırala
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

rag_pipeline = SimpleRAGPipeline()
