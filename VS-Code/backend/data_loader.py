"""
Data Loading and Embedding Module
Handles PDF loading, text chunking, and embedding generation
"""

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize local embedding model (no API quota limits)
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EMBED_DIMENSION = 384


def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load PDF and split into text chunks
    
    Args:
        path: Path to PDF file
        
    Returns:
        List of text chunks
    """
    # Load PDF using LlamaIndex
    reader = PDFReader()
    documents = reader.load_data(file=path)
    
    # Extract text from all pages
    text = [doc.text for doc in documents if getattr(doc, 'text', None)]
    
    # Split text into chunks with overlap
    splitter = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = []
    for t in text:
        chunks.extend(splitter.split_text(t))
    
    return chunks


def embed_text(text: list[str]) -> list[list[float]]:
    """
    Convert text chunks to embeddings using local sentence-transformers model
    
    Args:
        text: List of text strings to embed
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    embeddings = EMBED_MODEL.encode(text, convert_to_tensor=False)
    return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
