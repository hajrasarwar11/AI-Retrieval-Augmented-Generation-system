"""
Streamlit Frontend for RAG Application
User interface for PDF upload and question answering
"""

import streamlit as st
import requests
import time
import os
import json
from pathlib import Path

# Configuration
INNGEST_API_BASE = os.getenv("INNGEST_API_BASE", "http://localhost:8288/v1")
BACKEND_API = os.getenv("BACKEND_API", "http://127.0.0.1:8000/api/inngest")
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_event_result(event_id: str, timeout: int = 60):
    """
    Poll Inngest API for event result
    
    Args:
        event_id: The event ID to check
        timeout: Maximum time to wait in seconds
    
    Returns:
        Event result data or None
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Fetch event runs
            response = requests.get(
                f"{INNGEST_API_BASE}/events/{event_id}/runs",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("data") and len(data["data"]) > 0:
                    # Get the most recent run
                    run = data["data"][0]
                    status = run.get("status")
                    
                    if status in ["Completed", "Succeeded", "Success"]:
                        return run.get("output")
                    elif status in ["Failed", "Cancelled"]:
                        st.error(f"Run failed with status: {status}")
                        return None
            
            time.sleep(2)  # Poll every 2 seconds
            
        except Exception as e:
            st.error(f"Error fetching result: {str(e)}")
            return None
    
    st.warning("Query timeout - please try again")
    return None


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI RAG Chatbot")
st.markdown("Upload PDF documents and ask questions about their content")

# ============================================================================
# SIDEBAR: PDF UPLOAD
# ============================================================================

with st.sidebar:
    st.header("📄 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to add to the knowledge base"
    )
    
    if uploaded_file and st.button("📥 Ingest PDF", use_container_width=True):
        try:
            # Save PDF to data directory
            pdf_path = DATA_DIR / uploaded_file.name
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Trigger ingestion via HTTP API
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                event_payload = {
                    "name": "rag/ingest_pdf",
                    "data": {
                        "pdf_path": str(pdf_path),
                        "source_id": uploaded_file.name
                    }
                }
                
                response = requests.post(
                    f"{INNGEST_API_BASE}/events",
                    json=[event_payload],
                    timeout=10
                )
                
                result = response.json() if response.status_code == 200 else {}
                
                if result.get("ids"):
                    event_id = result["ids"][0]
                    
                    # Wait for ingestion to complete
                    ingestion_result = fetch_event_result(event_id, timeout=120)
                    
                    if ingestion_result:
                        st.success(f"✅ Successfully ingested {uploaded_file.name}!")
                        st.json(ingestion_result)
                    else:
                        st.warning("Ingestion triggered but result not confirmed. Check Inngest dashboard.")
                else:
                    st.error("Failed to trigger ingestion")
                    
        except Exception as e:
            st.error(f"Error during ingestion: {str(e)}")
    
    st.divider()
    
    st.markdown("### 📊 Settings")
    top_k = st.slider(
        "Number of context chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant chunks to retrieve"
    )
    
    st.divider()
    
    st.markdown("### ℹ️ Instructions")
    st.markdown("""
    1. Upload a PDF document
    2. Click 'Ingest PDF' to process it
    3. Ask questions in the main area
    4. View AI-generated answers with sources
    """)

# ============================================================================
# MAIN AREA: CHAT INTERFACE
# ============================================================================

st.header("💬 Ask Questions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                # Query backend directly (faster than Inngest)
                response = requests.post(
                    f"http://127.0.0.1:8000/query",
                    params={
                        "question": question,
                        "top_k": top_k
                    },
                    timeout=30
                )
                
                answer_data = response.json() if response.status_code == 200 else {}
                
                if answer_data.get("status") == "success":
                    answer = answer_data.get("answer", "No answer generated")
                    sources = answer_data.get("sources", [])
                    num_context = answer_data.get("num_context", 0)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display metadata
                    st.caption(f"Used {num_context} context chunks")
                    
                    # Display sources
                    if sources:
                        with st.expander("📎 Sources"):
                            for source in sources:
                                st.markdown(f"- {source}")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {answer_data.get('answer', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error during query: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>AI RAG System • Powered by OpenAI, Quadrant & Inngest</small>
</div>
""", unsafe_allow_html=True)
