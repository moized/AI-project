import os
import glob
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai

class SimpleRAGPipeline:
    def __init__(self, samples_dir: str = "data/samples", collection_name: str = "research_docs"):
        self.samples_dir = samples_dir
        self.collection_name = collection_name
        
        # Qdrant yerel kalıcı vektör veritabanı (sıfır maliyetli ve yerel)
        self.qdrant_client = QdrantClient(path="./qdrant_storage")
        
        # Google GenAI SDK (gemini-embedding-2 veya yedek olarak yerel model)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                self.genai_client = genai.Client(api_key=api_key)
            except Exception:
                self.genai_client = None
        else:
            self.genai_client = None

        # Yerel yedek embedding modeli (Sentence Transformers - all-MiniLM-L6-v2)
        self.local_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_size = 384

        self._ensure_collection()
        self._index_documents_if_needed()

    def _ensure_collection(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
        except Exception:
            pass

    def _get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.vector_size

        # Öncelikli olarak Gemini Embedding API kullanılabilir, ancak kota/hata durumunda yerel fallback devreye girer.
        if self.genai_client:
            try:
                # gemini-embedding-2 veya text-embedding-004
                response = self.genai_client.models.embed_content(
                    model="text-embedding-004",
                    contents=text
                )
                if response and response.embedding:
                    return response.embedding.values
            except Exception:
                pass

        # Yerel fallback (Sentence Transformers)
        embedding = self.local_encoder.encode(text)
        return embedding.tolist()

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
                except Exception:
                    pass
        return chunks

    def _index_documents_if_needed(self):
        try:
            chunks = self._load_and_chunk_documents()
            if not chunks:
                return

            count_result = self.qdrant_client.count(collection_name=self.collection_name, exact=True)
            if count_result.count == 0:
                points = []
                for idx, chunk in enumerate(chunks):
                    vector = self._get_embedding(chunk["text"])
                    # Boyut uyumsuzluğunu önlemek için kontrol
                    if len(vector) != self.vector_size:
                        vector = self.local_encoder.encode(chunk["text"]).tolist()
                    
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
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Qdrant vektör veritabanı üzerinden anlamsal (semantic) arama yapar."""
        if not query or not query.strip():
            return []

        try:
            self._index_documents_if_needed()
            query_vector = self._get_embedding(query)
            if len(query_vector) != self.vector_size:
                query_vector = self.local_encoder.encode(query).tolist()

            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points

            results = []
            for hit in search_result:
                if hit.score > 0.05:
                    payload = hit.payload
                    results.append({
                        "id": payload.get("chunk_id"),
                        "text": payload.get("text"),
                        "source": payload.get("source"),
                        "page": payload.get("page", 1),
                        "score": round(float(hit.score), 4)
                    })
            return results
        except Exception:
            return []

rag_pipeline = SimpleRAGPipeline()
