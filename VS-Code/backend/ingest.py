from PyPDF2 import PdfReader
import os
from pathlib import Path
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# Simple text splitter function
def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

base_dir = Path(__file__).resolve().parent
pdf_folder = base_dir.parent / "data"
found_pdf = False

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        found_pdf = True
        path = os.path.join(pdf_folder, file)
        print(f"{'='*70}")
        print(f"Processing: {file}")
        print(f"{'='*70}\n")
        
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        print(f"✓ Loaded {file}")
        print(f"  Text length: {len(text)} characters\n")

        # Split text into chunks
        chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        print(f"✓ Created {len(chunks)} chunks\n")

        # Show preview of first 3 chunks
        print(f"First 3 chunks preview:")
        for idx, chunk in enumerate(chunks[:3], start=1):
            print(f"\nChunk {idx} (first 150 chars):")
            print(f"  {chunk[:150]}...\n")

        # Generate TF-IDF embeddings (faster than transformers, no model download needed)
        print(f"Generating embeddings for {len(chunks)} chunks using TF-IDF...")
        try:
            vectorizer = TfidfVectorizer(max_features=384, lowercase=True, stop_words='english')
            embeddings = vectorizer.fit_transform(chunks).toarray()
            print(f"\n✓ Generated embeddings for all {len(chunks)} chunks")
            print(f"✓ Embedding dimensions: {embeddings.shape[1]}\n")
            
            # Create vector list with metadata
            vector_list = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_list.append({
                    "chunk_id": i,
                    "text": chunk,
                    "embedding_size": len(embedding),
                    "embedding_sample": embedding[:5].tolist()
                })
            
            # Display embedding information
            print(f"Embedding Details:")
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Embedding dimensions: {len(embeddings[0])}")
            print(f"  First embedding sample (first 5 values):")
            print(f"    {embeddings[0][:5]}\n")
            
            # Save metadata to JSON
            output_file = base_dir / f"{Path(file).stem}_embeddings_metadata.json"
            with open(output_file, 'w') as f:
                json.dump(vector_list, f, indent=2)
            print(f"✓ Embeddings metadata saved to {output_file}\n")
            
            # Save full embeddings as numpy array
            embeddings_file = base_dir / f"{Path(file).stem}_embeddings.npy"
            np.save(embeddings_file, embeddings)
            print(f"✓ Full embeddings array saved to {embeddings_file}")
            print(f"  Shape: {embeddings.shape}\n")

            # Push vectors to Qdrant
            try:
                host = os.getenv("QDRANT_HOST", "localhost")
                port = int(os.getenv("QDRANT_PORT", "6333"))
                collection_name = os.getenv("QDRANT_COLLECTION", "rag_vectors")

                client = QdrantClient(host=host, port=port, timeout=5.0)
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
                )
                points = [
                    PointStruct(
                        id=i,
                        vector=embeddings[i].tolist(),
                        payload={
                            "text": chunks[i],
                            "chunk_id": i,
                            "source_file": file,
                        },
                    )
                    for i in range(len(chunks))
                ]
                client.upsert(collection_name=collection_name, points=points)
                print(f"✓ Upserted {len(points)} vectors into Qdrant collection '{collection_name}' at {host}:{port}\n")
            except Exception as e:
                print(f"✗ Error pushing vectors to Qdrant: {e}")
            
            print(f"✓✓ Embedding generation complete!\n")
            
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")

if not found_pdf:
    print("✗ No PDF files found in the data folder.")

