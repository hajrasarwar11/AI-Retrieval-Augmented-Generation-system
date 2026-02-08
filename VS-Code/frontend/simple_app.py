"""
Simplified Streamlit Frontend for RAG Application
"""
import streamlit as st
import requests

st.set_page_config(page_title="AI RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 AI RAG Chatbot")
st.markdown("Upload PDF documents and ask questions about their content")

st.header("💬 Ask Questions")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    st.write("**You:** " + question)
    
    with st.spinner("Searching and generating answer..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/query",
                params={"question": question, "top_k": 5},
                timeout=30
            )
            
            data = response.json()
            
            if response.status_code == 200 and data.get("status") == "success":
                st.write("**Answer:**")
                st.write(data.get("answer", "No answer"))
                
                if data.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in data.get("sources", []):
                            st.write(f"- {source}")
            else:
                st.error(f"Error: {data.get('answer', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

st.divider()
st.markdown("<div style='text-align: center;'><small>AI RAG System</small></div>", unsafe_allow_html=True)
