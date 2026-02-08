"""
Direct PDF ingestion without Streamlit
Bypasses UI and directly calls the ingestion pipeline
"""
import sys
sys.path.insert(0, r"d:\FJWU\Semester 5\Artificial Intelligence-Dr. Irum Matloob\Assignment3\AI_RAG_Project\backend")

from pathlib import Path
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QuadrantStorage
import uuid

data_dir = Path(r"d:\FJWU\Semester 5\Artificial Intelligence-Dr. Irum Matloob\Assignment3\AI_RAG_Project\data")
pdf_files = list(data_dir.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF(s) to ingest")

for pdf_path in pdf_files:
    print(f"\n📄 Processing: {pdf_path.name}")
    
    try:
        # Step 1: Load and chunk
        print("  1️⃣ Loading and chunking PDF...")
        chunks = load_and_chunk_pdf(str(pdf_path))
        print(f"     → {len(chunks)} chunks created")
        
        # Step 2: Embed
        print("  2️⃣ Generating embeddings...")
        vectors = embed_text(chunks)
        print(f"     → {len(vectors)} embeddings generated")
        
        # Step 3: Create IDs
        print("  3️⃣ Creating unique IDs...")
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_path.name}_{i}"))
            for i in range(len(chunks))
        ]
        
        # Step 4: Create payloads
        payloads = [
            {"text": chunks[i], "source": pdf_path.name}
            for i in range(len(chunks))
        ]
        
        # Step 5: Upsert to Qdrant
        print("  4️⃣ Upserting to Qdrant...")
        store = QuadrantStorage()
        store.upsert(ids, vectors, payloads)
        print(f"     → Successfully ingested {len(chunks)} chunks")
        
        print(f"✅ {pdf_path.name} complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n✨ Re-ingestion complete!")
