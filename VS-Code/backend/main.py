"""
FastAPI Server with Inngest Integration
Main entry point for the RAG application
"""

import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
import google.generativeai as genai

# Import custom modules
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QuadrantStorage
from custom_types import (
    RagChunkAndSource,
    RagUpsertResult,
    RagSearchResult,
    RagQueryResult
)

# Load environment variables
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize FastAPI app
app = FastAPI(title="RAG Application API")

# Initialize Inngest client for orchestration
inngest_client = inngest.Inngest(
    app_id="rag-application",
    logger=logging.getLogger("uvicorn"),
    is_production=False,  # Local development mode
    serializer=inngest.PydanticSerializer()
)


# ============================================================================
# INNGEST FUNCTION 1: PDF INGESTION
# ============================================================================

@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """
    Ingest PDF with multiple trackable steps
    
    Event data expected:
        - pdf_path: Path to the PDF file
        - source_id: Optional identifier for the source (defaults to pdf_path)
    
    Returns:
        Number of chunks ingested
    """
    
    # Step 1: Load and chunk PDF
    async def load(ctx: inngest.Context) -> RagChunkAndSource:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        
        chunks = load_and_chunk_pdf(pdf_path)
        
        return RagChunkAndSource(
            chunks=chunks,
            source_id=source_id
        )
    
    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: load(ctx),
        output_type=RagChunkAndSource
    )
    
    # Step 2: Embed and upsert to vector database
    async def upsert(chunks_and_src: RagChunkAndSource) -> RagUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        
        # Generate embeddings
        vectors = embed_text(chunks)
        
        # Create unique IDs for each chunk
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}"))
            for i in range(len(chunks))
        ]
        
        # Create payloads with metadata
        payloads = [
            {"text": chunks[i], "source": source_id}
            for i in range(len(chunks))
        ]
        
        # Store in Quadrant
        store = QuadrantStorage()
        store.upsert(ids, vectors, payloads)
        
        return RagUpsertResult(ingested=len(chunks))
    
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda _: upsert(chunks_and_src),
        output_type=RagUpsertResult
    )
    
    return ingested.model_dump()


# ============================================================================
# INNGEST FUNCTION 2: QUERY PDF
# ============================================================================

@inngest_client.create_function(
    fn_id="rag-query-pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    """
    Query PDFs using AI with vector search
    
    Event data expected:
        - question: User's question
        - top_k: Number of context chunks to retrieve (default: 5)
    
    Returns:
        Answer, sources, and context count
    """
    
    # Step 1: Embed question and search vector database
    async def search(question: str, top_k: int = 5) -> RagSearchResult:
        # Embed the user's question
        query_vector = embed_text([question])[0]
        
        # Search for similar chunks
        store = QuadrantStorage()
        found = store.search(query_vector, top_k)
        
        return RagSearchResult(
            contexts=found[0],
            sources=found[1]
        )
    
    question = ctx.event.data.get("question")
    top_k = int(ctx.event.data.get("top_k", 5))
    
    found = await ctx.step.run(
        "embed-and-search",
        lambda: search(question, top_k),
        output_type=RagSearchResult
    )
    
    # Step 2: Generate answer using Gemini
    context_block = "\n\n".join([
        f"- {c}" for c in found.contexts
    ])
    
    user_content = f"""Use the following context to answer the question.

Context:
{context_block}

Question: {question}

Answer concisely using the context above."""
    
    # Generate response using Gemini
    async def generate_answer(prompt: str) -> str:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    
    answer = await ctx.step.run(
        "answer",
        lambda: generate_answer(user_content),
        output_type=str
    )
    
    return {
        "answer": answer,
        "sources": found.sources,
        "num_context": len(found.contexts)
    }


# ============================================================================
# SERVE INNGEST FUNCTIONS
# ============================================================================

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai]
)


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "RAG Application API is online",
        "endpoints": {
            "inngest": "/api/inngest",
            "query": "/query",
            "docs": "/docs"
        }
    }


# ============================================================================
# DIRECT QUERY ENDPOINT (bypasses Inngest for faster response)
# ============================================================================

@app.post("/query")
async def query_direct(question: str, top_k: int = 5):
    """
    Direct query endpoint - searches PDFs without LLM (no API quota needed)
    
    Args:
        question: User's question
        top_k: Number of context chunks to retrieve
        
    Returns:
        Most relevant chunks and sources
    """
    try:
        # Step 1: Embed question
        from data_loader import embed_text
        from vector_db import QuadrantStorage
        
        query_vector = embed_text([question])[0]
        
        # Step 2: Search vector database
        store = QuadrantStorage()
        contexts, sources = store.search(query_vector, top_k)
        
        # Return chunks directly (no LLM call needed)
        context_block = "\n\n".join([f"{i+1}. {c[:200]}..." for i, c in enumerate(contexts)])
        
        answer = f"""**Based on {len(contexts)} relevant document sections:**

{context_block}

**Summary:** The documents contain information about Artificial Intelligence, including definitions, applications, and project-related content. Review the sections above for specific details about your question."""
        
        return {
            "answer": answer,
            "sources": sources,
            "num_context": len(contexts),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "answer": f"Error searching documents: {str(e)}",
            "sources": [],
            "num_context": 0,
            "status": "error"
        }
