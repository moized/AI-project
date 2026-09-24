from providers.embeddings import DeterministicEmbeddingProvider
from rag.rag_pipeline import SimpleRAGPipeline


def test_chunking_and_metadata(tmp_path):
    rag = SimpleRAGPipeline(
        samples_dir="data/samples",
        embedding_provider=DeterministicEmbeddingProvider(),
        qdrant_path=tmp_path / "qdrant",
    )
    chunks = rag._load_and_chunk_documents()
    assert isinstance(chunks, list)
    assert chunks
    assert {"id", "source", "text"}.issubset(chunks[0])


def test_semantic_retrieve(tmp_path):
    rag = SimpleRAGPipeline(
        samples_dir="data/samples",
        embedding_provider=DeterministicEmbeddingProvider(),
        qdrant_path=tmp_path / "qdrant",
    )
    rag.index_documents()
    results = rag.retrieve("architecture", top_k=2)
    assert isinstance(results, list)
    for result in results:
        assert {"id", "text", "source", "score"}.issubset(result)
        assert isinstance(result["score"], float)


def test_empty_query(tmp_path):
    rag = SimpleRAGPipeline(
        samples_dir="data/samples",
        embedding_provider=DeterministicEmbeddingProvider(),
        qdrant_path=tmp_path / "qdrant",
    )
    assert rag.retrieve("", top_k=2) == []
