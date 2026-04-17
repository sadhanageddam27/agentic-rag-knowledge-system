"""
demo.py — Interactive CLI demo for the Agentic RAG system

Run:
    python demo.py

This will:
    1. Load sample documents into the knowledge base
    2. Run example queries through both simple RAG and agentic RAG
    3. Show the full reasoning trace
"""

from src.rag_pipeline import RAGPipeline, Embedder, VectorStore, LLMClient
from src.agentic_rag import AgenticRAG
import os

# ── Sample Documents ──────────────────────────────────────────────────────────

SAMPLE_DOCS = {
    "refund_policy.txt": """
Refund Policy — TechStore Inc.

Digital Products:
All digital product purchases are non-refundable once the download has been accessed.
If the product has not been downloaded within 7 days, a full refund can be requested.
Defective digital products are eligible for a free replacement or store credit.

Physical Products:
Physical products can be returned within 30 days of purchase for a full refund.
Items must be in original, unopened condition to qualify for a refund.
Shipping costs for returns are covered by the customer unless the item is defective.
Defective physical products receive free return shipping and a full refund.

Processing Time:
Refunds are processed within 5-7 business days after the return is received.
Credit card refunds may take an additional 3-5 days to appear on your statement.

Exceptions:
Customized or personalized items cannot be returned unless they are defective.
Clearance items are final sale and not eligible for returns or refunds.
""",

    "shipping_policy.txt": """
Shipping Policy — TechStore Inc.

Standard Shipping:
Standard shipping takes 5-7 business days and costs $4.99.
Free standard shipping is available on orders over $50.

Express Shipping:
Express shipping delivers within 2-3 business days and costs $12.99.
Express shipping is available for orders placed before 2pm EST.

Overnight Shipping:
Overnight shipping is available for $24.99 on eligible items.
Orders must be placed before 12pm EST for same-day dispatch.

International Shipping:
International shipping is available to 45 countries.
Delivery times range from 7-21 business days depending on destination.
Customers are responsible for any customs duties or import taxes.

Order Tracking:
All orders receive a tracking number via email within 24 hours of dispatch.
Real-time tracking is available through our website or the carrier's portal.
""",

    "product_warranty.txt": """
Product Warranty — TechStore Inc.

Standard Warranty:
All TechStore products come with a 1-year limited warranty against manufacturing defects.
The warranty covers hardware failures and defects under normal use conditions.

Extended Warranty:
Customers can purchase a 2-year or 3-year extended warranty at checkout.
Extended warranty adds accidental damage protection not covered in standard warranty.

What Is Covered:
- Manufacturing defects and hardware failures
- Battery issues (capacity drops below 80% within warranty period)
- Display defects (dead pixels, backlight issues)

What Is Not Covered:
- Physical damage from drops, spills, or misuse
- Damage from unauthorized repairs or modifications
- Normal wear and tear
- Software issues or data loss

Warranty Claims:
To file a warranty claim, contact support@techstore.com with your order number.
Approved claims receive a replacement unit within 5-7 business days.
A prepaid shipping label is provided for returning the defective unit.
"""
}


def print_separator(title: str = ""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def setup_pipeline():
    """Initialize the RAG pipeline with local embedder."""
    print("\n🔧 Initializing RAG pipeline...")

    embedder = Embedder(
        backend=os.environ.get("EMBEDDER_BACKEND", "local")
    )
    vector_store = VectorStore(
        collection_name="demo_knowledge",
        persist_dir="./demo_chroma_db",
        embedder=embedder
    )

    # Use OpenAI if key available, otherwise need Ollama
    llm_backend = "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"
    llm = LLMClient(
        backend=llm_backend,
        model="gpt-3.5-turbo" if llm_backend == "openai" else "llama3",
        temperature=0.1
    )

    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        top_k=4,
        score_threshold=0.2
    )

    agent = AgenticRAG(rag_pipeline=pipeline)
    return pipeline, agent


def ingest_sample_docs(pipeline: RAGPipeline):
    """Load sample documents into the knowledge base."""
    print_separator("INGESTING DOCUMENTS")

    # Write sample docs to temp files
    import tempfile
    import os
    tmpdir = tempfile.mkdtemp()

    total = 0
    for filename, content in SAMPLE_DOCS.items():
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        chunks = pipeline.ingest_file(filepath)
        total += chunks
        print(f"  ✓ {filename} → {chunks} chunks")

    print(f"\n  Total: {total} chunks indexed in vector store")
    return tmpdir


def run_simple_queries(pipeline: RAGPipeline):
    """Demonstrate standard RAG queries."""
    print_separator("SIMPLE RAG QUERIES")

    queries = [
        "What is the return window for physical products?",
        "How long does standard shipping take?",
        "What does the warranty cover?",
    ]

    for q in queries:
        print(f"\n❓ Question: {q}")
        response = pipeline.query(q)
        print(f"💬 Answer: {response.answer}")
        print(f"📎 Sources: {len(response.sources)} chunks retrieved")
        print(f"⚡ Latency: {response.latency_ms}ms")

        if response.sources:
            top = response.sources[0]
            src = top.metadata.get("filename", "unknown")
            print(f"🔝 Top source: {src} (score: {top.score})")


def run_agentic_queries(agent: AgenticRAG):
    """Demonstrate multi-step agentic reasoning."""
    print_separator("AGENTIC RAG — MULTI-STEP REASONING")

    complex_queries = [
        "How does the refund policy differ between digital and physical products, and what are the processing times?",
        "If I order a product with express shipping and it arrives defective, what are my options for a refund and how long will it take?",
    ]

    for q in complex_queries:
        print(f"\n❓ Complex question:\n   {q}")
        response = agent.query(q)

        print(f"\n🤖 Is complex: {response.is_complex}")
        print(f"\n🧠 Reasoning steps:")
        for i, step in enumerate(response.steps, 1):
            print(f"   {i}. {step.output[:100]}{'...' if len(step.output) > 100 else ''}")

        print(f"\n💬 Final answer:\n   {response.answer}")
        print(f"\n📎 Total sources used: {len(response.all_sources)}")
        print(f"⚡ Total latency: {response.latency_ms}ms")


def main():
    print("\n" + "🚀 " * 20)
    print("   AGENTIC RAG KNOWLEDGE SYSTEM — DEMO")
    print("🚀 " * 20)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Trying Ollama (http://localhost:11434)")
        print("   Set OPENAI_API_KEY in .env or install Ollama: https://ollama.com")

    try:
        pipeline, agent = setup_pipeline()
        tmpdir = ingest_sample_docs(pipeline)
        run_simple_queries(pipeline)
        run_agentic_queries(agent)

        print_separator("DEMO COMPLETE")
        print("  ✅ RAG pipeline working correctly")
        print("  ✅ Agentic multi-step reasoning demonstrated")
        print(f"  ✅ {pipeline.vector_store.count()} chunks in knowledge base")
        print("\n  Next steps:")
        print("  - Run the API:  python api.py")
        print("  - Add your own docs to ./data/ and call ingest_directory()")
        print("  - Swap embedder backend: EMBEDDER_BACKEND=openai")
        print("  - Swap LLM: LLM_BACKEND=ollama LLM_MODEL=llama3")

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
