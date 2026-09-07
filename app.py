import os
import tempfile
import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag import RAG

st.set_page_config(
    page_title="DocuMind AI — Neural Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global style refinements */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 1rem;
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .card-box {
        padding: 1.25rem;
        border-radius: 0.75rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    .doc-pill {
        padding: 0.85rem;
        border-radius: 0.6rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.75rem;
    }
    .chat-bubble-user {
        background-color: #f1f5f9;
        border-radius: 0.75rem 0.75rem 0.1rem 0.75rem;
        padding: 0.9rem 1.2rem;
        margin: 0.75rem 0;
        border-left: 4px solid #64748b;
    }
    .chat-bubble-ai {
        background-color: #f0fdf4;
        border-radius: 0.75rem 0.75rem 0.75rem 0.1rem;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0 1.25rem 0;
        border-left: 4px solid #10b981;
        border: 1px solid #dcfce7;
    }
    .info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
    
if "rag" not in st.session_state:
    st.session_state.rag = RAG(st.session_state.vector_store)
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_document(uploaded_file):
    """Process an uploaded document."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        chunks = st.session_state.processor.process_document(tmp_path)
        st.session_state.vector_store.add_documents(chunks)
        
        if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "chunks": len(chunks)
            })
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def clear_documents():
    """Clear all uploaded documents and reset the vector store."""
    st.session_state.vector_store = VectorStore()
    st.session_state.uploaded_files = []
    st.session_state.rag = RAG(st.session_state.vector_store)

# Hero Header
st.markdown("""
<div class="main-header">
    <h1>🧠 DocuMind AI</h1>
    <p>Enterprise-grade Retrieval-Augmented Generation (RAG) system for interactive document intelligence.</p>
    <div class="badge-container">
        <span class="tech-badge">⚡ FAISS Indexing</span>
        <span class="tech-badge">🧬 Sentence Transformers</span>
        <span class="tech-badge">🦜️ LangChain</span>
        <span class="tech-badge">🤖 OpenAI GPT</span>
        <span class="tech-badge">🔒 In-Memory Privacy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📂 Ingest Documents")
    uploaded_file = st.file_uploader("Upload PDF document", type="pdf", help="Select any text-based PDF to index into the vector store")
    
    if uploaded_file is not None:
        if st.button("🚀 Process & Embed", type="primary", use_container_width=True):
            with st.spinner("Extracting text, chunking, and computing embeddings..."):
                process_document(uploaded_file)
                st.success(f"Indexed {uploaded_file.name} successfully!")
    
    st.divider()
    st.header("⚙️ Indexing Parameters")
    with st.expander("Chunking Settings", expanded=False):
        chunk_size = st.slider("Chunk Size (characters)", min_value=100, max_value=2000, value=1000, step=100, help="Target size for text segmentation")
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50, help="Overlap ensures contextual continuity between chunks")
        
        if chunk_size != st.session_state.processor.chunk_size or chunk_overlap != st.session_state.processor.chunk_overlap:
            st.session_state.processor = DocumentProcessor(chunk_size, chunk_overlap)
    
    if st.button("🗑️ Reset Knowledge Base", type="secondary", use_container_width=True):
        clear_documents()
        st.info("Vector store reset.")
    
    st.divider()
    st.header("📑 Active Documents")
    if st.session_state.uploaded_files:
        for doc in st.session_state.uploaded_files:
            st.markdown(f"""
            <div class="doc-pill">
                <strong>📄 {doc['name']}</strong><br>
                <small style="color: #64748b;">Chunks: <b>{doc['chunks']}</b> | Size: <b>{doc['size']/1024:.1f} KB</b></small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No documents loaded into active memory.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Query Knowledge Base")
    question = st.text_input("Ask a question about your documents:", placeholder="e.g., Summarize the core methodology and key findings...", label_visibility="collapsed")
    
    if st.button("Send Query", type="primary"):
        if not st.session_state.uploaded_files:
            st.warning("⚠️ Please upload and process at least one document first.")
        elif not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Retrieving relevant context and generating response..."):
                answer = st.session_state.rag.generate_answer(question)
                st.session_state.chat_history.append({"question": question, "answer": answer})
    
    # Display conversation history
    if st.session_state.chat_history:
        st.subheader("💭 Conversation History")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-bubble-user">
                <strong>👤 Query:</strong><br>{chat['question']}
            </div>
            <div class="chat-bubble-ai">
                <strong>🤖 DocuMind Assistant:</strong><br>{chat['answer']}
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.subheader("📊 Session Analytics")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        total_docs = len(st.session_state.uploaded_files)
        st.metric("Documents", total_docs)
    with col_stats2:
        total_chunks = sum(doc["chunks"] for doc in st.session_state.uploaded_files) if st.session_state.uploaded_files else 0
        st.metric("Total Chunks", total_chunks)
    
    st.subheader("🔍 Architecture Pipeline")
    with st.expander("How RAG Operates", expanded=True):
        st.markdown("""
        1. **Ingestion & Segmentation**: PDFs are extracted and parsed into overlapping semantic chunks.
        2. **Dense Vector Embeddings**: Each segment is mapped to high-dimensional space via `all-MiniLM-L6-v2`.
        3. **FAISS Fast Search**: Sub-millisecond similarity index finds the most relevant passages.
        4. **Grounded Generation**: OpenAI synthesizes verifiable responses exclusively from retrieved evidence.
        """)

    st.markdown("""
    <div class="info-card">
        <h4 style="margin: 0 0 0.5rem 0; color: #0f172a;">🔒 Privacy & Isolation</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #475569; line-height: 1.4;">
            All indexed vector representations remain strictly within local memory for this session. Raw PDF files are deleted immediately after parsing.
        </p>
    </div>
    """, unsafe_allow_html=True)