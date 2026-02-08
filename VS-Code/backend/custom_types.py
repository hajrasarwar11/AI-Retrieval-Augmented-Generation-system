"""
Custom Pydantic types for RAG application
These types ensure type safety across Inngest functions
"""

import pydantic


class RagChunkAndSource(pydantic.BaseModel):
    """Result after chunking a PDF document"""
    chunks: list[str]
    source_id: str


class RagUpsertResult(pydantic.BaseModel):
    """Result after upserting embeddings to vector database"""
    ingested: int


class RagSearchResult(pydantic.BaseModel):
    """Result after searching vector database"""
    contexts: list[str]
    sources: list[str]


class RagQueryResult(pydantic.BaseModel):
    """Final query result with answer and sources"""
    answer: str
    sources: list[str]
    num_context: int
