"""
Quadrant Vector Database Client
Handles all interactions with the Quadrant vector database
"""
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

class QuadrantStorage:
    """
    Client for managing vector storage and retrieval in Quadrant
    """
    def __init__(
        self, 
        url: str = "http://localhost:6333",
        collection: str = "documents",
        dim: int = 384
    ):
        """
        Initialize Quadrant client and create collection if it doesn't exist
        
        Args:
            url: Quadrant server URL
            collection: Collection name for storing vectors
            dim: Dimension of vectors (must match embedding model)
        """
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        
        # Create collection if it doesn't exist
        if not self.client.collection_exists(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created Quadrant collection: {self.collection}")
    
    def upsert(self, ids: list, vectors: list, payloads: list):
        """
        Insert or update vectors in the database
        
        Args:
            ids: List of unique identifiers
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries (text, source, etc.)
        """
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]
        
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )
    
    def search(self, query_vector: list[float], top_k: int = 5):
        """
        Search for similar vectors in the database
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (contexts, sources) where contexts are text chunks
            and sources are unique document names
        """
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        contexts = []
        sources = set()
        
        for result in results:
            payload = getattr(result, 'payload', {})
            text = payload.get('text', '')
            source = payload.get('source', '')
            
            if text:
                contexts.append(text)
                sources.add(source)
        
        return contexts, list(sources)
