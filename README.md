
# Agentic RAG Knowledge System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Ollama-purple)
![Tests](https://img.shields.io/badge/Tests-18%20passing-brightgreen)

Production-grade Agentic RAG system — agent-driven retrieval pipeline
that grounds LLM responses in real documents, with multi-step reasoning
for complex queries. Built from scratch in Python with FastAPI, ChromaDB,
and an OpenAI-compatible backend that works with any local or cloud LLM.

## Live Project Page
**https://sadhanageddam27.github.io/agentic-rag-knowledge-system/**

---

## What Makes This "Agentic"

Standard RAG does one retrieval and generates an answer.
This agent decides HOW to answer first:

    Simple question → single retrieval → generate answer

    Complex question → classify → decompose into sub-questions
                    → retrieve evidence per sub-question
                    → synthesize final grounded answer

Example of a question that benefits from agentic reasoning:
"How does the refund policy differ for digital vs physical products
and what are the processing times for each?"

---

## Project Structure

    agentic-rag-knowledge-system/
    ├── src/
    │   ├── rag_pipeline.py    # Document, Loader, Chunker, Embedder, VectorStore, RAGPipeline
    │   └── agentic_rag.py     # AgentStep, AgentResponse, AgenticRAG
    ├── tests/
    │   └── test_rag.py        # 18 unit tests — all mocked, no API key needed
    ├── api.py                 # FastAPI REST API — 7 endpoints
    ├── demo.py                # CLI demo with 3 sample docs + example queries
    ├── requirements.txt
    └── .env.example

---

## Architecture

    Documents (.txt / .md / .pdf)
          │
          ▼
    DocumentLoader
          │
          ▼
    TextChunker ── 512 tokens, 64 overlap
          │
          ▼
    Embedder ── local (sentence-transformers) OR openai
          │
          ▼
    VectorStore (ChromaDB) ── cosine similarity, persistent
          │
          ▼
    AgenticRAG
          │
    ┌─────┴──────┐
    │            │
  simple      complex
    │            │
  retrieve    decompose → retrieve×N → synthesize
    │            │
    └─────┬──────┘
          │
      final answer

---

## Quick Start

**1. Clone and install**

    git clone https://github.com/sadhanageddam27/agentic-rag-knowledge-system.git
    cd agentic-rag-knowledge-system
    pip install -r requirements.txt

**2. Configure**

    cp .env.example .env
    # Add OPENAI_API_KEY, or use Ollama (free, local — see below)

**3. Run the demo**

    python demo.py

**4. Start the API**

    python api.py
    # Docs at http://localhost:8000/docs

**5. Run tests (no API key needed)**

    pytest tests/ -v

---

## API Endpoints

    POST   /ingest/text     Add raw text to knowledge base
    POST   /ingest/file     Upload .txt, .md, or .pdf file
    POST   /query           Standard RAG query
    POST   /agent/query     Agentic multi-step query with reasoning trace
    GET    /health          Health check
    GET    /stats           Vector store stats
    DELETE /knowledge       Clear all documents

**Example — agentic query:**

    curl -X POST http://localhost:8000/agent/query \
      -H "Content-Type: application/json" \
      -d '{"question": "How does the refund policy differ for digital vs physical?"}'

**Response includes:**

    {
      "answer": "...",
      "is_complex": true,
      "reasoning_steps": [
        "Classified as: complex",
        "Decomposed into 2 sub-questions",
        "Retrieved 3 chunks for sub-question 1 (score: 0.91)",
        "Retrieved 3 chunks for sub-question 2 (score: 0.88)",
        "Synthesized final answer from all evidence"
      ],
      "latency_ms": 842.3
    }

---

## Using Ollama (Free, No API Key)

Run completely offline:

    # Install Ollama: https://ollama.com
    ollama pull llama3

    # Set in .env:
    LLM_BACKEND=ollama
    LLM_MODEL=llama3
    EMBEDDER_BACKEND=local

---

## Key Design Decisions

**Overlapping chunks** — 64-token overlap between chunks prevents
information loss at paragraph boundaries.

**Classify before decomposing** — avoids extra LLM calls on simple
questions that only need one retrieval.

**Modular backends** — swap ChromaDB for Pinecone, or sentence-transformers
for OpenAI embeddings, by changing one class. Core logic unchanged.

**All tests mocked** — unit tests mock LLM, ChromaDB, and embedder,
so pytest runs without any API keys or running services.

---

## Tech Stack
Python · FastAPI · ChromaDB · sentence-transformers · OpenAI API · pytest · uvicorn

## Topics
`rag` `llm` `agentic-ai` `vector-search` `chromadb`
`fastapi` `python` `nlp` `prompt-engineering` `openai`
