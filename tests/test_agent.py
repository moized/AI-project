from agent.agent_core import ResearchAgent
from providers.embeddings import DeterministicEmbeddingProvider
from providers.llm import FakeLLMProvider
from rag.rag_pipeline import SimpleRAGPipeline


def build_agent(qdrant_path):
    rag = SimpleRAGPipeline(
        samples_dir="data/samples",
        embedding_provider=DeterministicEmbeddingProvider(),
        qdrant_path=qdrant_path,
    )
    rag.index_documents()
    return ResearchAgent(
        rag=rag,
        llm=FakeLLMProvider("Test response."),
    )


def test_agent_returns_contract(tmp_path):
    result = build_agent(tmp_path / "qdrant").run("What is architecture?")
    assert result["answer"] == "Test response."
    assert isinstance(result["sources"], list)
    assert isinstance(result["tool_calls"], list)


def test_agent_empty_query(tmp_path):
    result = build_agent(tmp_path / "qdrant").run("")
    assert result["answer"]


def test_context_relevance_rejects_unrelated_retrieval():
    item = {
        "score": 0.559,
        "text": "Metamaterials and metasurface electromagnetic properties.",
        "source": "holloway2012.pdf",
        "page": 5,
        "id": "1",
    }
    assert not ResearchAgent._context_is_relevant(
        "What is supervised learning?",
        item,
    )


def test_context_relevance_accepts_matching_term():
    item = {
        "score": 0.55,
        "text": "RAG architecture uses retrieval and generation.",
        "source": "rag_architecture.md",
        "page": None,
        "id": "2",
    }
    assert ResearchAgent._context_is_relevant("What is RAG architecture?", item)


def test_document_request_keeps_retrieved_context():
    item = {
        "score": 0.53,
        "text": "Metamaterials are artificial structures.",
        "source": "holloway2012.pdf",
        "page": 5,
        "id": "3",
    }
    assert ResearchAgent._filter_retrieved(
        "According to the documents, what are the main topics?",
        [item],
    ) == [item]
