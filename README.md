# AI Research Assistant

A modular-monolith research assistant for technical documents, with local vector retrieval, Gemini generation, controlled function calling, a FastAPI backend, a Streamlit UI, SQLite persistence, and Docker deployment.

## Architecture

    Streamlit
       |
    FastAPI /api/v1
       |
    ResearchAgent
       |---- controlled tools
       |
       RAG
       |---- extraction/chunking
       |---- replaceable embedding provider
       |---- local Qdrant
       |
       Gemini LLM provider

The system is intentionally designed for low or near-zero cost:

- Gemini 3.5 Flash-Lite is the default generation model, with model failover configured for Gemini 3.1 Flash-Lite and Gemini 3.8 Flash.
- The default embedding provider is local FastEmbed with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, so document indexing does not consume Gemini embedding quota.
- Qdrant runs locally.
- SQLite runs locally.
- Docker and GitHub Actions are used for reproducible development and CI.

Gemini generation is configurable through `GEMINI_MODEL` and `GEMINI_FALLBACK_MODELS`. The provider treats daily quota exhaustion as model-specific failover, while transient 429/5xx errors use bounded exponential retry delays of 2, 4, 8, 16, and 32 seconds; server-provided retry guidance takes precedence. Gemini Embedding 2 remains an optional provider.

## Local setup

    python -m venv .venv

Windows:

    .venv\Scripts\activate

Linux/macOS:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Copy .env.example to .env and add your Gemini API key.

Run tests:

    pytest -q

Run the backend:

    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Run the frontend in another terminal:

    streamlit run frontend/app.py

Open:

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Docker

Create .env, then:

    docker compose up --build

The backend is available on port 8000 and Streamlit on port 8501.

Persistent runtime state is intentionally not committed to Git. Docker stores Qdrant, SQLite, and Hugging Face model cache in named volumes.

## API

The frontend uses these versioned endpoints:

- GET /api/v1/documents
- POST /api/v1/documents/upload
- POST /api/v1/documents/index
- POST /api/v1/chat

The black-box API contract is in contracts/backend_api_contract.json.

## Testing strategy

The automated suite avoids paid or external model calls:

- deterministic embedding provider for RAG tests
- fake LLM provider for agent tests
- FastAPI contract/health tests
- security tests for the calculator tool

CI runs linting, Python compilation, and pytest on every relevant push and pull request.

## Production direction

This repository is currently a production-oriented learning system, not a claim of unlimited-scale production readiness. The current design emphasizes:

- explicit contracts
- replaceable providers
- bounded tool execution
- input validation
- atomic file replacement
- incremental indexing
- runtime-state isolation
- automated CI
- local/free development

At larger scale, the natural next steps would include authentication/authorization, request IDs and structured logging, tracing/metrics, rate limiting, background indexing, object storage, PostgreSQL where needed, and a managed vector database only when justified.