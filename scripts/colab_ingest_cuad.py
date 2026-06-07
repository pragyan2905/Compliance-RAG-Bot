import os
import shutil
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
import pickle
import uuid

def chunk_text(text, chunk_size=1000, overlap=200):
    """Simple text chunker since raw CUAD text doesn't have PDF hierarchy."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def main():
    print("1. Loading CUAD dataset from Hugging Face...")
    # Using the modernized Parquet repository to bypass the trust_remote_code error
    ds = load_dataset("umarbutler/better-cuad", split="train")
    
    # Identify the correct text column dynamically (typically 'text' or 'Text')
    text_col = next((col for col in ["text", "Text", "document", "context"] if col in ds.column_names), ds.column_names[0])
    
    # The better-cuad dataset has exactly 1 row per contract, so we just extract the text directly
    unique_contexts = [doc for doc in ds[text_col] if doc and len(str(doc).strip()) > 50]
    print(f"   Found {len(unique_contexts)} unique legal contracts.")

    print("\n2. Initializing Embedding Model (GPU)...")
    # EXPLICITLY set device="cuda" to prevent it from running on CPU
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    vector_size = embedder.get_sentence_embedding_dimension()

    print("\n3. Initializing Qdrant...")
    # Wipe the directory if it exists to prevent Qdrant file lock errors across multiple notebook runs
    shutil.rmtree("data", ignore_errors=True)
    os.makedirs("data/qdrant", exist_ok=True)
    os.makedirs("data/bm25_cuad", exist_ok=True)

    qdrant = QdrantClient(path="data/qdrant")
    collection_name = "compliance_global"

    # Create collection
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print("\n4. Chunking, Embedding, and Indexing Contracts...")
    all_chunks = []
    chunk_metadata = []

    for doc_idx, context in enumerate(tqdm(unique_contexts, desc="Processing Contracts")):
        chunks = chunk_text(str(context))
        for chunk_idx, text in enumerate(chunks):
            chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
            all_chunks.append(text)
            chunk_metadata.append({
                "chunk_id": chunk_id,
                "document_id": f"cuad_{doc_idx}",
                "text": text,
                "section": "General", # Fallback since raw text has no headers
                "clause_id": f"Paragraph {chunk_idx + 1}"
            })

    print(f"   Generated {len(all_chunks)} total chunks.")

    print("\n5. Embedding all chunks (This will be FAST on Colab/Kaggle GPU!)...")
    # Reduced batch size to 256 to prevent CUDA Out-of-Memory / Device-side assertion errors
    embeddings = embedder.encode(all_chunks, batch_size=256, show_progress_bar=True)

    print("\n6. Upserting to Qdrant...")
    points = []
    for i, (meta, vector) in enumerate(zip(chunk_metadata, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=meta
            )
        )

        # Upsert in batches of 1000
        if len(points) >= 1000:
            qdrant.upsert(collection_name=collection_name, points=points)
            points = []

    # Upsert remaining
    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    print("\n7. Building BM25 Keyword Index...")
    tokenized_corpus = [doc.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 and metadata
    with open("data/bm25_cuad/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open("data/bm25_cuad/chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunk_metadata, f)

    # CRITICAL: Close the Qdrant connection to release file locks!
    qdrant.close()

    print("\n8. Zipping the 'data' folder.")
    shutil.make_archive("cuad_data", "zip", "data")

    print("\n DONE! Please download the 'cuad_data.zip' file from the Colab file browser (left sidebar)!")

if __name__ == "__main__":
    main()
