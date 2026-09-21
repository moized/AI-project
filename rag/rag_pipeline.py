import os
import glob
import logging
from typing import List, Dict, Any
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class SimpleRAGPipeline:
    def __init__(self, samples_dir: str = "data/samples", collection_name: str = "research_docs", vector_size: int = 768):
        self.samples_dir = samples_dir
        self.collection_name = collection_name
        self.vector_size = vector_size # gemini-embedding-2 MRL destekli boyut (önerilen: 768, 1536 veya 3072)
        
        # Qdrant yerel kalıcı vektör veritabanı (disk persistence)
        self.qdrant_client = QdrantClient(path="./qdrant_storage")
        
        # Google GenAI SDK istemcisi
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                self.genai_client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.error(f"GenAI Client başlatılamadı: {e}")
                self.genai_client = None
        else:
            self.genai_client = None

        self._ensure_collection()
        self._index_documents()

    def _ensure_collection(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
        except Exception as e:
            logger.error(f"Koleksiyon oluşturulurken hata: {e}")

    def _get_embedding(self, text: str, task_prefix: str = "task: search result | query: ") -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.vector_size

        if self.genai_client:
            try:
                # Resmi belgelere göre gemini-embedding-2 modeli ve output_dimensionality kullanımı
                formatted_text = f"{task_prefix}{text}" if not text.startswith("task:") else text
                response = self.genai_client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=formatted_text,
                    config=types.EmbedContentConfig(output_dimensionality=self.vector_size)
                )
                if response and response.embeddings:
                    return response.embeddings[0].values
            except Exception as e:
                logger.warning(f"Gemini Embedding API hatası, sıfır vektör döndürülüyor: {e}")

        # Fallback olarak sıfır vektör veya dummy
        return [0.0] * self.vector_size

    def _load_and_chunk_documents(self) -> List[Dict[str, Any]]:
        chunks = []
        if not os.path.exists(self.samples_dir):
            return chunks
        
        patterns = [
            os.path.join(self.samples_dir, "*.md"),
            os.path.join(self.samples_dir, "*.txt"),
            os.path.join(self.samples_dir, "*.pdf")
        ]
        
        chunk_id_counter = 0
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    filename = os.path.basename(filepath)
                    content = ""
                    if filepath.endswith(".pdf"):
                        reader = PdfReader(filepath)
                        for page_num, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                paragraphs = text.split("\n\n")
                                for para in paragraphs:
                                    if para.strip():
                                        chunks.append({
                                            "id": f"{filename}_page_{page_num}_chunk_{chunk_id_counter}",
                                            "text": para.strip(),
                                            "source": filename,
                                            "page": page_num + 1,
                                            "score": 0.0
                                        })
                                        chunk_id_counter += 1
                    else:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        paragraphs = content.split("\n\n")
                        for para in paragraphs:
                            if para.strip():
                                chunks.append({
                                    "id": f"{filename}_chunk_{chunk_id_counter}",
                                    "text": para.strip(),
                                    "source": filename,
                                    "page": 1,
                                    "score": 0.0
                                })
                                chunk_id_counter += 1
                except Exception as e:
                    logger.error(f"Dosya okunurken hata ({filepath}): {e}")
        return chunks

    def _index_documents(self):
        try:
            chunks = self._load_and_chunk_documents()
            if not chunks:
                return

            # Mevcut indekslenmiş nokta sayısını kontrol et
            count_result = self.qdrant_client.count(collection_name=self.collection_name, exact=True)
            
            # Eğer doküman sayısı değiştiyse veya veritabanı boşsa yeniden indeksle (gereksiz tekrarı önleme)
            if count_result.count != len(chunks):
                points = []
                for idx, chunk in enumerate(chunks):
                    # Resmi belgelere uygun olarak dokümanlar için document yapısı
                    doc_prefix = f"title: {chunk['source']} | text: "
                    vector = self._get_embedding(chunk["text"], task_prefix=doc_prefix)
                    
                    points.append(
                        PointStruct(
                            id=idx,
                            vector=vector,
                            payload={
                                "chunk_id": chunk["id"],
                                "text": chunk["text"],
                                "source": chunk["source"],
                                "page": chunk.get("page", 1)
                            }
                        )
                    )
                if points:
                    self.qdrant_client.recreate_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                    )
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
        except Exception as e:
            logger.error(f"İndeksleme sırasında hata: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Qdrant vektör veritabanı üzerinden anlamsal (semantic) arama yapar."""
        if not query or not query.strip():
            return []

        try:
            self._index_documents()
            query_vector = self._get_embedding(query, task_prefix="task: search result | query: ")

            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points

            results = []
            for hit in search_result:
                if hit.score > 0.01:
                    payload = hit.payload
                    results.append({
                        "id": payload.get("chunk_id"),
                        "text": payload.get("text"),
                        "source": payload.get("source"),
                        "page": payload.get("page", 1),
                        "score": round(float(hit.score), 4)
                    })
            return results
        except Exception as e:
            logger.error(f"Retrieval sırasında hata: {e}")
            return []

rag_pipeline = SimpleRAGPipeline()
