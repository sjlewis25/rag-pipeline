import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

st.title("📚 RAG Document Q&A System")
st.markdown("Upload documents and ask questions using AI")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ {data['message']}")
                    st.info(f"Created {data['chunks']} chunks")
                else:
                    st.error(f"❌ Error: {response.text}")

with col2:
    st.header("❓ Ask Questions")
    
    question = st.text_input("Enter your question:", placeholder="What is this document about?")
    
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching and generating answer..."):
                payload = {"question": question, "top_k": 3}
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.subheader("💡 Answer")
                    st.write(data["answer"])
                    
                    st.subheader("📄 Sources")
                    for i, source in enumerate(data["sources"], 1):
                        with st.expander(f"Source {i}: {source['source']} (score: {source['score']:.3f})"):
                            st.write(source["text"])
                else:
                    st.error(f"❌ Error: {response.text}")

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This RAG system:
- Uses Ollama for local LLM
- Qdrant for vector storage
- Supports PDF and TXT files
- Provides source citations
""")

st.sidebar.header("🔧 System Status")
try:
    health = requests.get(f"{API_URL}/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("✅ API: Online")
    else:
        st.sidebar.error("❌ API: Error")
except:
    st.sidebar.error("❌ API: Offline")