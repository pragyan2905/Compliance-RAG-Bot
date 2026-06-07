import os
import streamlit as st
import tempfile
import requests
import json

# Adjust Python path if running from ui/
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_parser import DocumentParser
from core.embedding_manager import EmbeddingManager
from core.vector_store import QdrantManager
from core.bm25_retriever import BM25Manager
from core.retriever import HybridRetriever
from core.llm_generator import ComplianceGenerator
from core.llm_critic import ComplianceCritic
from schemas.retrieval import VectorRecord

st.set_page_config(page_title="Compliance RAG Bot", layout="wide")

# Custom CSS for Premium Chat Design with Vibrant Colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #09090b 0%, #18181b 100%);
        color: #f8fafc;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    h2, h3 {
        color: #cbd5e1;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.4);
    }
    
    .risk-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .risk-high { border-left: 6px solid #f43f5e; }
    .risk-medium { border-left: 6px solid #fbbf24; }
    .risk-low { border-left: 6px solid #34d399; }
    
    .critic-pass { border-left: 6px solid #10b981; }
    .critic-fail { border-left: 6px solid #f43f5e; }
</style>
""", unsafe_allow_html=True)

from qdrant_client import QdrantClient

@st.cache_resource
def load_backend_components():
    """Initializes and caches the backend singletons."""
    parser = DocumentParser(chunk_size=1000, chunk_overlap=200)
    embedder = EmbeddingManager()
    
    # Initialize a SINGLE shared Qdrant client to prevent SQLite lock errors
    os.makedirs("data/qdrant", exist_ok=True)
    shared_qdrant_client = QdrantClient(path="data/qdrant")
    
    # Fast Local (Single Document) Instances
    qdrant_local = QdrantManager(collection_name="compliance_ui", vector_size=embedder.embedding_dimension, client=shared_qdrant_client)
    bm25_local = BM25Manager(persist_dir="data/bm25_ui")
    retriever_local = HybridRetriever(qdrant_local, bm25_local, embedder)
    
    # Global Precedent (CUAD) Instances
    qdrant_global = QdrantManager(collection_name="compliance_global", vector_size=embedder.embedding_dimension, client=shared_qdrant_client)
    bm25_global = BM25Manager(persist_dir="data/bm25_cuad")
    retriever_global = HybridRetriever(qdrant_global, bm25_global, embedder)
    
    generator = ComplianceGenerator()
    critic = ComplianceCritic()
    
    return parser, embedder, qdrant_local, bm25_local, retriever_local, retriever_global, generator, critic

def process_document(uploaded_file, parser, embedder, qdrant, bm25):
    """Processes the uploaded PDF through the pipeline for Fast Local Mode."""
    with st.spinner("Analyzing Document Hierarchy..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        parsed_doc = parser.parse(tmp_path)
        os.remove(tmp_path)
        
    with st.spinner(f"Generating Embeddings for {len(parsed_doc.chunks)} chunks..."):
        texts = [chunk.text for chunk in parsed_doc.chunks]
        embeddings = embedder.embed_batch(texts)
        
    with st.spinner("Indexing into Vector Database..."):
        records = []
        for chunk, vector in zip(parsed_doc.chunks, embeddings):
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=vector,
                    text=chunk.text,
                    document_id=parsed_doc.document_id,
                    page=chunk.page,
                    section=chunk.section,
                    subsection=chunk.subsection,
                    clause_id=chunk.clause_id
                )
            )
        qdrant.upsert_records(records)
        bm25.index_chunks(parsed_doc.chunks)
        
    return parsed_doc.document_id

def general_chat(query: str) -> str:
    """Makes a standard non-RAG call to Groq for general compliance queries."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is missing."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful, expert compliance and legal AI assistant."},
            {"role": "user", "content": query}
        ],
        "temperature": 0.5
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error communicating with Groq: {e}"

def render_about_project():
    st.header("Project Architecture")
    st.markdown("""
    **Compliance RAG Bot** is a modular, deterministic enterprise AI retrieval and compliance auditing system. 
    It features a dual-mode architecture to handle both fast, local document audits and deep precedent searches.
    
    🔗 **Open Source Code**: [View the GitHub Repository here](https://github.com/pragyan2905/Compliance-RAG-Bot/tree/main)
    
    ### Technologies Used
    - **Document Parsing**: `PyMuPDF` with heuristic font analysis to preserve hierarchical metadata (Sections, Clauses).
    - **Embeddings**: `BAAI/bge-small-en-v1.5` via `SentenceTransformers` for rapid dense vectorization.
    - **Global Knowledge Base**: The **CUAD** (Contract Understanding Atticus Dataset) containing over 500 expert-annotated commercial contracts.
    - **Vector Storage**: `Qdrant` running on local-disk to store embeddings without external SaaS dependencies.
    - **Sparse Retrieval**: `rank-bm25` for in-memory keyword matching.
    - **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` to mathematically score the top hybrid candidates against the user's query context.
    - **Text Generation**: The `llama-3.3-70b-versatile` model via the **Groq API** to force structured JSON outputs.
    - **Data Validation**: `Pydantic v2` for strongly typed pipeline schemas.
    """)

def main():
    st.title("Compliance RAG Bot")
    st.caption("Enterprise Contract Intelligence & Global Precedent Search")
    
    tab_chat, tab_about = st.tabs(["Auditor Interface", "About Project"])
    
    with tab_about:
        render_about_project()
        
    with tab_chat:
        # Init backend
        try:
            parser, embedder, qdrant_local, bm25_local, retriever_local, retriever_global, generator, critic = load_backend_components()
        except Exception as e:
            st.error(f"Failed to initialize backend. Check your GROQ_API_KEY. Error: {e}")
            return
    
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your Compliance AI. Choose an Audit Mode in the sidebar to begin!", "type": "text"}
            ]
    
        # Sidebar for Dual-Mode Configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            search_mode = st.radio(
                "Select Audit Mode:",
                ["Fast Audit (Upload PDF)", "Deep Precedent Search (CUAD DB)"],
                index=0
            )
            st.divider()
            
            if search_mode == "Fast Audit (Upload PDF)":
                st.subheader("Upload Contract")
                uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
                
                if uploaded_file is not None:
                    if st.button("Process Document for Chat"):
                        doc_id = process_document(uploaded_file, parser, embedder, qdrant_local, bm25_local)
                        st.session_state["doc_indexed"] = True
                        st.success(f"Document '{uploaded_file.name}' indexed successfully.")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"I have processed '{uploaded_file.name}'. You can now ask me to audit specific clauses, liabilities, or risks in this document.", "type": "text"}
                        )
            else:
                st.success("🌐 Connected to Global CUAD Knowledge Base (500+ Contracts).")
                st.info("You don't need to upload anything. Ask me general questions to find legal precedents!")
    
        # Render Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["type"] == "text":
                    st.markdown(message["content"])
                elif message["type"] == "rag_result":
                    analysis = message["analysis"]
                    validation = message["validation"]
                    chunks = message["chunks"]
                    
                    st.markdown(f"**Executive Summary:**\n{analysis.summary}")
                    
                    # Risks
                    if analysis.risks:
                        for risk in analysis.risks:
                            severity_class = f"risk-{risk.severity.lower()}"
                            st.markdown(f"""
                            <div class="risk-card {severity_class}">
                                <h4 style="margin-top: 0; color: #f8fafc; font-weight: 600;">{risk.severity} Severity: {risk.risk_type}</h4>
                                <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 8px;"><strong>Relevant Clause:</strong> {risk.relevant_clause_id or 'N/A'}</p>
                                <p style="color: #e2e8f0; line-height: 1.5;">{risk.description}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No compliance risks identified for this query.")
                        
                    # Critic
                    st.markdown("<br>", unsafe_allow_html=True)
                    critic_class = "critic-fail" if validation.is_hallucinated else "critic-pass"
                    st.markdown(f"""
                    <div class="risk-card {critic_class}" style="padding: 15px;">
                        <h5 style="margin-top: 0;">Critic Validation (Confidence: {validation.confidence_score}/10)</h5>
                        <p style="margin-bottom: 0; color: #cbd5e1;">{validation.feedback}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Context Expander
                    with st.expander("View Retrieved Legal Evidence"):
                        for i, chunk in enumerate(chunks):
                            doc_id_label = chunk.get('document_id', 'N/A')
                            st.markdown(f"**Rank {i+1}** | Source: `{doc_id_label}`")
                            st.caption(chunk.get("text"))
                            st.markdown("---")
    
        # Chat Input
        if prompt := st.chat_input("Ask a compliance question..."):
            st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
            
            with st.chat_message("user"):
                st.markdown(prompt)
    
            with st.chat_message("assistant"):
                # Determine which retriever to use based on mode
                if search_mode == "Deep Precedent Search (CUAD DB)":
                    with st.spinner("Searching 500+ Global Contracts..."):
                        retrieved_chunks = retriever_global.retrieve(prompt, top_k=5)
                        if not retrieved_chunks:
                            response_msg = "I couldn't find any relevant precedents in the global database."
                            st.markdown(response_msg)
                            st.session_state.messages.append({"role": "assistant", "content": response_msg, "type": "text"})
                        else:
                            analysis = generator.generate(retrieved_chunks, prompt)
                            validation = critic.validate(retrieved_chunks, analysis)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "rag_result",
                                "analysis": analysis,
                                "validation": validation,
                                "chunks": retrieved_chunks
                            })
                            st.rerun()
                
                elif search_mode == "Fast Audit (Upload PDF)" and st.session_state.get("doc_indexed"):
                    with st.spinner("Retrieving evidence and running audit..."):
                        retrieved_chunks = retriever_local.retrieve(prompt, top_k=5)
                        if not retrieved_chunks:
                            response_msg = "I couldn't find any relevant clauses in the uploaded document."
                            st.markdown(response_msg)
                            st.session_state.messages.append({"role": "assistant", "content": response_msg, "type": "text"})
                        else:
                            analysis = generator.generate(retrieved_chunks, prompt)
                            validation = critic.validate(retrieved_chunks, analysis)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "rag_result",
                                "analysis": analysis,
                                "validation": validation,
                                "chunks": retrieved_chunks
                            })
                            st.rerun()
                else:
                    with st.spinner("Analyzing..."):
                        response_text = general_chat(prompt)
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text, "type": "text"})

if __name__ == "__main__":
    main()
