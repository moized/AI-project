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
