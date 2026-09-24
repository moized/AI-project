# AI Research Assistant — Current Implementation Plan

## Principle
Do Pass 2 first: make the system work and harden it; then perform Pass 1 as a learning/explanation pass over the finished architecture.

## Current target
A low-cost production-oriented modular monolith:

`Streamlit -> FastAPI -> ResearchAgent -> RAG / Tools -> Gemini`

Qdrant and SQLite remain local runtime components.

## Provider policy
- LLM: Gemini 3.8 Flash through `google-genai`
- Embeddings: FastEmbed local by default
- Gemini Embedding 2: optional provider
- Test doubles: deterministic embeddings + fake LLM

## Work order
1. Infrastructure/configuration/contracts
2. Tools and deterministic data
3. RAG ingestion/retrieval
4. Agent + function calling
5. Backend/database/API
6. Frontend/deployment
7. Integration/evaluation/observability

## Production checks
- explicit contracts
- typed inputs/outputs
- safe tool allow-list
- bounded tool loops
- bounded context
- incremental indexing
- runtime-state isolation
- API versioning
- safe error responses
- automated lint/compile/test CI
- Docker reproducibility

## Current known next engineering layers
- retrieval evaluation (Hit@K/MRR)
- reranking/hybrid retrieval if evaluation justifies it
- structured logging/request IDs
- readiness endpoint
- authentication/rate limiting when exposed beyond local use
- background indexing if ingestion becomes slow

## Learning rule
Do not add a technology just because it is fashionable. Every added component must solve a measurable project requirement.
