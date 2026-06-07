# ⚖️ ComplianceRAG Bot

**Enterprise Contract Intelligence & Global Precedent Search**

ComplianceRAG is a modular, deterministic, local-first retrieval-augmented generation (RAG) system built for legal and compliance auditing. It allows users to instantly audit individual contracts against compliance rules, or perform deep semantic precedent searches across a massive global database of real-world commercial contracts.

---

## 🏗 Architecture

This project was built from the ground up to be lightweight, inspectable, and free-tier friendly. It strictly avoids heavy orchestration frameworks (like LangChain) in favor of pure, deterministic Python code.

- **Dual-Mode System**:
  1. **Fast Audit Mode**: Upload a single PDF. The system parses it, embeds it on the fly, and uses the Groq LLM to flag risks, liabilities, and standard compliance checks.
  2. **Global Precedent Search**: Ask a general question (e.g., "What is a standard limitation of liability?"). The system searches a pre-indexed database of over 500 expert-annotated commercial contracts (the CUAD dataset) to find and summarize real market precedents.
- **Hierarchical Document Parsing**: Uses `PyMuPDF` with heuristic font analysis to preserve the structural integrity of legal documents (Sections, Subsections, Clauses).
- **Hybrid Retrieval**: Fuses dense semantic search (`BAAI/bge-small-en-v1.5`) with sparse keyword matching (`rank-bm25`) using Reciprocal Rank Fusion (RRF).
- **Local Vector Storage**: Uses `Qdrant` running on the local disk (SQLite), requiring no external database servers or Docker containers.
- **Strict Data Validation**: Uses `Pydantic v2` to enforce strict JSON schemas on the LLM output, guaranteeing deterministic Risk Cards and Critic Validations without hallucinated formatting.
- **LLM Engine**: Uses the blazing-fast `llama-3.3-70b-versatile` model via the **Groq API**.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

```bash
git clone <your-repository-url>
cd ComplianceRAG
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your Groq API Key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. (Optional but Recommended) Ingest the Global CUAD Database
To enable the "Deep Precedent Search" mode, you need the CUAD dataset database. Generating embeddings for 500+ long contracts takes time, so we provided a script optimized for Google Colab/Kaggle free GPUs.
1. Upload `scripts/colab_ingest_cuad.py` to Google Colab or Kaggle.
2. Run the script on a GPU instance (T4 or P100). It will process the dataset in ~5 minutes.
3. Download the resulting `cuad_data.zip` file.
4. Extract the zip file into the `data/` folder of this project so it looks like this:
   ```text
   ComplianceRAG/
   ├── data/
   │   ├── qdrant/
   │   └── bm25_cuad/
   ```

### 4. Run the Application
Launch the Streamlit UI:
```bash
PYTHONPATH=. streamlit run ui/app.py
```
The application will be available at `http://localhost:8501`.

---

## 🌍 Deployment

This application is extremely portable and production-ready for small to medium enterprise deployments. 

### Streamlit Community Cloud (Free)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set the Main file path to `ui/app.py`.
4. Add your `GROQ_API_KEY` in the Streamlit Advanced Settings -> Secrets.
5. Deploy! *(Note: If you want to include the CUAD database, you must push the `data/` folder to GitHub using Git LFS due to file size limits).*

### VPS / Docker (Enterprise Security)
Because Qdrant and the Embedding Models run entirely locally, you can deploy this on any standard Linux VPS (like Render, AWS EC2, or DigitalOcean) without needing complex cloud architecture. Just copy the repository, `pip install`, and run the Streamlit command.

---

## 📁 Directory Structure
```text
ComplianceRAG/
├── core/                   # The backend engine
│   ├── document_parser.py  # PyMuPDF hierarchical parsing
│   ├── embedding_manager.py# SentenceTransformers local embeddings
│   ├── retriever.py        # Hybrid RRF Search + Cross-Encoder
│   ├── vector_store.py     # Qdrant local disk management
│   ├── llm_generator.py    # Groq structured generation
│   └── llm_critic.py       # Hallucination validation
├── schemas/                # Pydantic data models
├── scripts/                # Colab ingestion scripts
├── ui/                     # Frontend
│   └── app.py              # Streamlit dual-mode interface
└── data/                   # Local databases (Qdrant & BM25)
```
