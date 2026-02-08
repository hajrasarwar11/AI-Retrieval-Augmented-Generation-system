"""
Re-ingest PDFs with new embedding model
"""
import requests
import time
import json
from pathlib import Path

# Find PDF files
data_dir = Path(__file__).parent / "data"
pdf_files = list(data_dir.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF(s) to ingest")

# Inngest API endpoint
INNGEST_API = "http://localhost:8288/v1"

for pdf_path in pdf_files:
    print(f"\n📄 Ingesting: {pdf_path.name}")
    
    # Send ingest event via HTTP
    event_data = {
        "name": "rag/ingest_pdf",
        "data": {
            "pdf_path": str(pdf_path),
            "source_id": pdf_path.name
        }
    }
    
    try:
        response = requests.post(
            f"{INNGEST_API}/events",
            json=[event_data],
            timeout=10
        )
        
        result = response.json() if response.status_code == 200 else {}
    except Exception as e:
        print(f"❌ Error sending event: {e}")
        continue
    
    if response.status_code == 200 and result.get("ids"):
        event_id = result["ids"][0]
        print(f"✓ Event triggered (ID: {event_id})")
        
        # Poll for result
        start_time = time.time()
        timeout = 180  # 3 minutes
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{INNGEST_API}/events/{event_id}/runs",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("data") and len(data["data"]) > 0:
                        run = data["data"][0]
                        status = run.get("status")
                        
                        if status in ["Completed", "Succeeded", "Success"]:
                            output = run.get("output")
                            ingested = output.get("ingested", 0) if output else 0
                            print(f"✅ Success! Ingested {ingested} chunks")
                            break
                        elif status in ["Failed", "Cancelled"]:
                            print(f"❌ Failed with status: {status}")
                            break
                
                time.sleep(2)
                
            except Exception as e:
                print(f"⚠️  Error checking status: {e}")
                time.sleep(2)
        else:
            print(f"⏱️  Timeout waiting for completion")
    else:
        print(f"❌ Failed to trigger event")

print("\n✨ Re-ingestion complete!")
