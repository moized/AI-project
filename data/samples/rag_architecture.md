# RAG (Retrieval-Augmented Generation) Architecture Overview

## Introduction
Retrieval-Augmented Generation (RAG) is a technique used to improve the accuracy and reliability of generative AI models by fetching facts from an external knowledge base before generating a response.

## Core Components
1. **Document Ingestion & Chunking**: Raw documents (PDF, Markdown, TXT) are split into smaller, manageable chunks (e.g., 500-1000 tokens) with appropriate overlap.
2. **Embeddings**: Chunks are converted into dense vector representations using embedding models (e.g., local Sentence Transformers or Gemini embedding models).
3. **Vector Database**: Vectors are stored in a local vector database (such as Chroma or FAISS) for fast similarity search.
4. **Retrieval**: Given a user query, the system retrieves the top-k most relevant chunks based on vector cosine similarity.
5. **Generation**: The retrieved context is combined with the user prompt and passed to the LLM (e.g., Gemini free tier) to generate a grounded response with citations.
