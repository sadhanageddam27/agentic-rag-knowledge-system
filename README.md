# Agentic RAG Knowledge System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Ollama-purple)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow)

Production-grade Agentic RAG system — agent-driven retrieval pipeline
that grounds LLM responses in real documents, with multi-step reasoning
for complex queries. Built from scratch in Python with FastAPI, ChromaDB,
and an OpenAI-compatible LLM backend.

## Live Project Page
**https://sadhanageddam27.github.io/agentic-rag-knowledge-system/**

---

## What Makes This "Agentic"

Standard RAG does one retrieval and generates an answer.
**Agentic RAG** decides *how* to answer:

    Simple question → single retrieval → generate answer

    Complex question → classify → decompose into sub-questions
                    → retrieve evidence per sub-question
                    → synthesize final answer from all evidence

This gives dramatically better results on multi-hop questions like:
*"How does the refund policy differ for digital vs physical products
and what are the processing times for each?"*

---

## Project Structure

    agentic-rag-knowledge-system/
    ├── src/
    │   ├── rag_pipeline.py    # Core RAG — loader, chunker, embedder, vector store, pipeline
    │   └── agentic_rag.py     # Agentic layer — classify, decompose, retrieve, synthesize
    ├── tests/
    │   └── test_rag.py        # Unit tests for all components (mocked external deps)
    ├── api.py                 # FastAPI REST API — ingest and query endpoints
    ├── demo.py                # CLI demo with sample documents
    ├── requirements.txt
    └── .env.example

---

## Architecture

    Documents
        │
        ▼
    DocumentLoader ── txt, md, pdf support
        │
        ▼
    TextChunker ── overlapping chunks (512 tokens, 64 overlap)
        │
        ▼
    Embedder ── local (sentence-transformers) OR openai
        │
        ▼
    VectorStore (ChromaDB) ── persistent, cosine similarity
        │
        ▼
    ┌─────────────────────────────────────────┐
    │           AgenticRAG                    │
    │                                         │
    │  query → classify → simple/complex?     │
    │                                         │
    │  simple:  retrieve → generate           │
    │                                         │
    │  complex: decompose → retrieve×N        │
    │           → synthesize final answer     │
    └─────────────────────────────────────────┘

---

## Quick Start

**1. Clone and install**

    git clone https://github.com/sadhanageddam27/agentic-rag-knowledge-system.git
    cd agentic-rag-knowledge-system
    pip install -r requirements.txt

**2. Configure**

    cp .env.example .env
    # Add your OPENAI_API_KEY, or use Ollama (free, local)

**3. Run the demo**

    python demo.py

**4. Or start the API**

    python api.py
    # API available at http://localhost:8000
    # Docs at http://localhost:8000/docs

---

## Using Ollama (Free, No API Key)

Run completely offline with a local LLM:

    # Install Ollama: https://ollama.com
    ollama pull llama3

    # Set in .env:
    LLM_BACKEND=ollama
    LLM_MODEL=llama3
    EMBEDDER_BACKEND=local   # sentence-transformers, no key needed

---

## API Endpoints

    POST /ingest/text    Add raw text to knowledge base
    POST /ingest/file    Upload .txt, .md, or .pdf file
    POST /query          Standard RAG query
    POST /agent/query    Agentic multi-step query
    GET  /health         Health check
    GET  /stats          Vector store statistics
    DELETE /knowledge    Clear all documents

**Example:**

    # Ingest a document
    curl -X POST http://localhost:8000/ingest/text \
      -H "Content-Type: application/json" \
      -d '{"text": "Our return policy allows 30-day returns...", "source": "policy"}'

    # Query it
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question": "What is the return window?"}'

    # Agentic query (multi-step)
    curl -X POST http://localhost:8000/agent/query \
      -H "Content-Type: application/json" \
      -d '{"question": "How does the return policy differ for digital vs physical?"}'

---

## Running Tests

    pip install pytest
    pytest tests/ -v

Tests mock all external dependencies (LLM, ChromaDB, embedder) —
no API keys or running services required.

---

## Key Design Decisions

**Why ChromaDB?**
Persistent, embedded, no separate server needed. Easy to swap for
Pinecone or Weaviate in production by replacing the VectorStore class.

**Why local embeddings by default?**
sentence-transformers/all-MiniLM-L6-v2 is free, fast, and good enough
for most use cases. Swap to OpenAI embeddings for higher accuracy.

**Why overlapping chunks?**
64-token overlap between chunks prevents information loss at boundaries.
Critical for questions that span paragraph breaks.

**Why classify before decomposing?**
Decomposition adds latency (extra LLM calls). Classifying first means
simple questions get fast single-retrieval responses.

---

## Tech Stack
Python · FastAPI · ChromaDB · sentence-transformers · OpenAI API · pytest

## Topics
`rag` `llm` `agentic-ai` `vector-search` `chromadb` `fastapi`
`python` `nlp` `prompt-engineering` `openai`
