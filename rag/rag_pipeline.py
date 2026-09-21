import os
import glob
from typing import List, Dict, Any
from pypdf import PdfReader

class SimpleRAGPipeline:
    def __init__(self, samples_dir: str = "data/samples"):
        self.samples_dir = samples_dir

    def _load_and_chunk_documents(self) -> List[Dict[str, Any]]:
        chunks = []
        if not os.path.exists(self.samples_dir):
            return chunks
        
        patterns = [
            os.path.join(self.samples_dir, "*.md"),
            os.path.join(self.samples_dir, "*.txt"),
            os.path.join(self.samples_dir, "*.pdf")
        ]
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    filename = os.path.basename(filepath)
                    content = ""
                    if filepath.endswith(".pdf"):
                        reader = PdfReader(filepath)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                content += text + "\n\n"
                    else:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()

                    paragraphs = content.split("\n\n")
                    for i, para in enumerate(paragraphs):
                        if para.strip():
                            chunks.append({
                                "id": f"{filename}_chunk_{i}",
                                "text": para.strip(),
                                "source": filename,
                                "score": 1.0
                            })
                except Exception:
                    pass
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Dinamik olarak dokümanları (PDF, MD, TXT) yükleyip kelime eşleşmesine göre RAG retrieval yapar."""
        chunks = self._load_and_chunk_documents()
        query_terms = set(query.lower().split())
        scored_chunks = []

        for chunk in chunks:
            text_lower = chunk["text"].lower()
            match_count = sum(1 for term in query_terms if term in text_lower)
            if match_count > 0:
                scored_chunks.append({
                    **chunk,
                    "score": float(match_count)
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

rag_pipeline = SimpleRAGPipeline()
