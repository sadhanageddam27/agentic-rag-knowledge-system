"""
api.py — FastAPI REST API for the Agentic RAG System

Endpoints:
    POST /ingest/text    — Add raw text to knowledge base
    POST /ingest/file    — Upload a file to knowledge base
    POST /query          — Query the RAG pipeline
    POST /agent/query    — Query the Agentic RAG (multi-step)
    GET  /health         — Health check
    GET  /stats          — Vector store stats
    DELETE /knowledge    — Clear knowledge base
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os
import logging

from src.rag_pipeline import RAGPipeline, Embedder, VectorStore, LLMClient
from src.agentic_rag import AgenticRAG

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG Knowledge System",
    description="Document-grounded Q&A with multi-step agentic reasoning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Initialize pipeline (singleton) ────────────────────────────────────────────

embedder = Embedder(
    backend=os.environ.get("EMBEDDER_BACKEND", "local")
)
vector_store = VectorStore(
    collection_name="rag_knowledge",
    persist_dir=os.environ.get("CHROMA_DIR", "./chroma_db"),
    embedder=embedder
)
llm = LLMClient(
    backend=os.environ.get("LLM_BACKEND", "openai"),
    model=os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
    base_url=os.environ.get("LLM_BASE_URL", None)
)

pipeline = RAGPipeline(
    embedder=embedder,
    vector_store=vector_store,
    llm=llm,
    top_k=int(os.environ.get("TOP_K", "5")),
    score_threshold=float(os.environ.get("SCORE_THRESHOLD", "0.3"))
)

agent = AgenticRAG(rag_pipeline=pipeline)


# ── Request / Response Models ─────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str
    source: Optional[str] = "api_input"

class IngestTextResponse(BaseModel):
    chunks_added: int
    source: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class SourceInfo(BaseModel):
    content: str
    score: float
    source: str
    chunk_index: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: List[SourceInfo]
    latency_ms: float

class AgentQueryResponse(BaseModel):
    answer: str
    query: str
    is_complex: bool
    reasoning_steps: List[str]
    sources: List[SourceInfo]
    latency_ms: float

class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    embedder_backend: str
    llm_model: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "agentic-rag"}


@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse(
        total_chunks=vector_store.count(),
        collection_name=vector_store.collection_name,
        embedder_backend=embedder.backend,
        llm_model=llm.model
    )


@app.post("/ingest/text", response_model=IngestTextResponse)
def ingest_text(req: IngestTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    chunks_added = pipeline.ingest_text(req.text, source=req.source)
    return IngestTextResponse(chunks_added=chunks_added, source=req.source)


@app.post("/ingest/file", response_model=IngestTextResponse)
async def ingest_file(file: UploadFile = File(...)):
    allowed = {".txt", ".md", ".pdf"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks_added = pipeline.ingest_file(tmp_path)
        return IngestTextResponse(chunks_added=chunks_added, source=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = pipeline.query(req.question)
        return QueryResponse(
            answer=response.answer,
            query=response.query,
            sources=[
                SourceInfo(
                    content=s.content[:300] + "..." if len(s.content) > 300 else s.content,
                    score=s.score,
                    source=s.metadata.get("filename", s.metadata.get("source", "unknown")),
                    chunk_index=s.metadata.get("chunk_index")
                )
                for s in response.sources
            ],
            latency_ms=response.latency_ms
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = agent.query(req.question)
        return AgentQueryResponse(
            answer=response.answer,
            query=response.query,
            is_complex=response.is_complex,
            reasoning_steps=[s.output for s in response.steps],
            sources=[
                SourceInfo(
                    content=s.content[:300] + "..." if len(s.content) > 300 else s.content,
                    score=s.score,
                    source=s.metadata.get("filename", s.metadata.get("source", "unknown")),
                    chunk_index=s.metadata.get("chunk_index")
                )
                for s in response.all_sources
            ],
            latency_ms=response.latency_ms
        )
    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/knowledge")
def clear_knowledge():
    vector_store.clear()
    return {"status": "cleared", "message": "All documents removed from knowledge base"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
