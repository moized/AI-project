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
- The default embedding provider is local FastEmbed with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, while the Render free deployment overrides this to Gemini Embedding 2 at 768 dimensions because Render's free instance has 512 MB RAM.
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

Persistent runtime state is intentionally not committed to Git. Docker stores Qdrant, SQLite, and FastEmbed model cache in named volumes.

Docker is optional for the free cloud deployment below.

## Free cloud deployment

The repository includes a native Render Blueprint at render.yaml for the FastAPI backend. It uses Render's free web-service plan, the Python 3.11 runtime, the /ready health check, Gemini Embedding 2 at 768 dimensions to avoid the memory-heavy local ONNX embedding runtime, and INDEX_ON_STARTUP=true so bundled sample documents can rebuild the local demo index after a restart. Render free services have ephemeral filesystems and can spin down after 15 minutes of inactivity, so uploaded documents, SQLite data, and the local Qdrant index are not intended to be permanent on the free demo deployment.

The Streamlit frontend automatically retries the Render `/health` endpoint during cold starts for about 2–3 minutes, so a sleeping free backend does not normally require a manual page refresh. Normal API calls use a longer connection/read timeout. The agent also filters weak, unrelated retrieval results before adding document evidence or citations to an answer.

The Streamlit frontend can be deployed separately on Streamlit Community Cloud. Set these Streamlit secrets:

    BACKEND_URL = "https://YOUR-RENDER-SERVICE.onrender.com"
    BACKEND_ACCESS_TOKEN = "same random token configured on Render"

Set these Render environment variables/secrets:

    GEMINI_API_KEY = <secret>
    BACKEND_ACCESS_TOKEN = <same random token>
    GEMINI_MODEL = gemini-3.5-flash-lite
    GEMINI_FALLBACK_MODELS = gemini-3.1-flash-lite,gemini-3.8-flash
    GEMINI_EMBEDDING_MODEL = gemini-embedding-2
    GEMINI_EMBEDDING_DIMENSION = 768

Do not commit .env, Streamlit secrets.toml, API keys, or access tokens.

Render free services are suitable for a hobby/demo deployment, not a durable production database. Streamlit Community Cloud is free for sharing Streamlit apps.

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