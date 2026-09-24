from rag.rag_pipeline import SimpleRAGPipeline
from providers.embeddings import DeterministicEmbeddingProvider


def test_chunking_and_metadata():
    rag = SimpleRAGPipeline(
        samples_dir="data/samples",
        embedding_provider=DeterministicEmbeddingProvider(),
    )
    chunks = rag._load_and_chunk_documents()
    assert isinstance(chunks, list)
    assert chunks
    assert {"id", "source", "text"}.issubset(chunks[0])


def test_semantic_retrieve():
    rag = SimpleRAGPipeline(
        samples_dir="data/samples",
        embedding_provider=DeterministicEmbeddingProvider(),
    )
    rag.index_documents()
    results = rag.retrieve("architecture", top_k=2)
    assert isinstance(results, list)
    for result in results:
        assert {"id", "text", "source", "score"}.issubset(result)
        assert isinstance(result["score"], float)


def test_empty_query():
    rag = SimpleRAGPipeline(
        samples_dir="data/samples",
        embedding_provider=DeterministicEmbeddingProvider(),
    )
    assert rag.retrieve("", top_k=2) == []
