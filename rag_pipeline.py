"""
agentic_rag.py — Agentic RAG with Multi-Step Reasoning

The agent can:
    1. Decompose complex questions into sub-questions
    2. Retrieve evidence for each sub-question independently
    3. Synthesize a final answer from all retrieved evidence
    4. Decide when it has enough information to answer

This is the key difference from basic RAG — the agent REASONS
about what it needs to retrieve, rather than doing a single lookup.
"""

import logging
import json
from typing import List, Optional
from dataclasses import dataclass, field

from src.rag_pipeline import RAGPipeline, RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)


# ── Agent State ────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """One reasoning step taken by the agent."""
    step_type: str        # "decompose" | "retrieve" | "synthesize" | "answer"
    input: str
    output: str
    chunks_used: List[RetrievedChunk] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Full agentic response with reasoning trace."""
    answer: str
    query: str
    steps: List[AgentStep]
    all_sources: List[RetrievedChunk]
    is_complex: bool
    latency_ms: float


# ── Agentic RAG ────────────────────────────────────────────────────────────────

class AgenticRAG:
    """
    Agent-driven RAG that decides HOW to answer a question.

    For simple questions: single retrieval + generation (standard RAG)
    For complex questions: decompose → retrieve per sub-question → synthesize

    Interview talking point:
        "The agent uses an LLM to first classify whether a question needs
         multi-step reasoning. If it does, it breaks it into sub-questions,
         retrieves evidence for each independently, then synthesizes a final
         answer — giving much better results on multi-hop questions."
    """

    DECOMPOSE_PROMPT = """You are a question analysis assistant.

Given a complex question, decompose it into 2-4 simpler sub-questions
that together answer the original question.

Return ONLY a JSON array of strings. No explanation.

Example:
Input: "How does the refund policy differ for digital vs physical products?"
Output: ["What is the refund policy for digital products?", "What is the refund policy for physical products?"]

Input: {question}
Output:"""

    CLASSIFY_PROMPT = """Classify whether this question requires multi-step reasoning
to answer accurately, or if it can be answered with a single search.

Return ONLY "complex" or "simple". Nothing else.

Question: {question}"""

    SYNTHESIZE_PROMPT = """You are synthesizing information from multiple sources to answer a question.

Original question: {question}

Evidence gathered from sub-questions:
{evidence}

Provide a comprehensive, accurate answer to the original question using the evidence above.
If any part cannot be answered from the evidence, say so clearly."""

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None, max_steps: int = 5):
        self.rag = rag_pipeline or RAGPipeline()
        self.max_steps = max_steps

    def query(self, question: str) -> AgentResponse:
        """
        Main agent entry point.
        Automatically chooses simple or complex reasoning path.
        """
        import time
        start = time.time()
        steps = []
        all_sources = []

        # Step 1: Classify complexity
        complexity = self._classify(question)
        is_complex = complexity == "complex"

        steps.append(AgentStep(
            step_type="classify",
            input=question,
            output=f"Classified as: {complexity}"
        ))

        if not is_complex:
            # Simple path — standard RAG
            rag_response = self.rag.query(question)
            steps.append(AgentStep(
                step_type="retrieve",
                input=question,
                output=f"Retrieved {len(rag_response.sources)} chunks",
                chunks_used=rag_response.sources
            ))
            steps.append(AgentStep(
                step_type="answer",
                input=question,
                output=rag_response.answer
            ))
            return AgentResponse(
                answer=rag_response.answer,
                query=question,
                steps=steps,
                all_sources=rag_response.sources,
                is_complex=False,
                latency_ms=round((time.time() - start) * 1000, 1)
            )

        # Complex path — decompose → retrieve → synthesize
        # Step 2: Decompose
        sub_questions = self._decompose(question)
        steps.append(AgentStep(
            step_type="decompose",
            input=question,
            output=f"Decomposed into {len(sub_questions)} sub-questions: {sub_questions}"
        ))

        # Step 3: Retrieve for each sub-question
        evidence_parts = []
        for i, sub_q in enumerate(sub_questions[:self.max_steps]):
            chunks = self.rag.vector_store.search(sub_q, top_k=3)
            all_sources.extend(chunks)

            context = "\n\n".join(c.content for c in chunks) if chunks else "No relevant documents found."
            evidence_parts.append(f"Sub-question {i+1}: {sub_q}\nEvidence: {context}")

            steps.append(AgentStep(
                step_type="retrieve",
                input=sub_q,
                output=f"Retrieved {len(chunks)} chunks (top score: {chunks[0].score if chunks else 0})",
                chunks_used=chunks
            ))

        # Step 4: Synthesize
        evidence_text = "\n\n---\n\n".join(evidence_parts)
        answer = self.rag.llm.complete(
            system="You are a precise research assistant. Synthesize evidence to answer questions accurately.",
            user=self.SYNTHESIZE_PROMPT.format(
                question=question,
                evidence=evidence_text
            )
        )

        steps.append(AgentStep(
            step_type="synthesize",
            input=f"{len(sub_questions)} sub-questions answered",
            output=answer
        ))

        return AgentResponse(
            answer=answer,
            query=question,
            steps=steps,
            all_sources=list({c.doc_id: c for c in all_sources}.values()),
            is_complex=True,
            latency_ms=round((time.time() - start) * 1000, 1)
        )

    def _classify(self, question: str) -> str:
        """Ask LLM if question needs multi-step reasoning."""
        try:
            result = self.rag.llm.complete(
                system="You classify questions as complex or simple.",
                user=self.CLASSIFY_PROMPT.format(question=question)
            )
            return "complex" if "complex" in result.lower() else "simple"
        except Exception:
            return "simple"  # fallback to simple on error

    def _decompose(self, question: str) -> List[str]:
        """Break complex question into sub-questions."""
        try:
            raw = self.rag.llm.complete(
                system="You decompose questions into sub-questions. Return only JSON.",
                user=self.DECOMPOSE_PROMPT.format(question=question)
            )
            # Parse JSON array
            raw = raw.strip()
            if raw.startswith("["):
                sub_questions = json.loads(raw)
                return [q for q in sub_questions if isinstance(q, str)]
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}, using original question")

        return [question]  # fallback
