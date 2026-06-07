import os
from test_parser import create_test_pdf
from core.document_parser import DocumentParser
from core.embedding_manager import EmbeddingManager
from core.vector_store import QdrantManager
from core.bm25_retriever import BM25Manager
from core.retriever import HybridRetriever
from schemas.retrieval import VectorRecord

def main():
    test_pdf = "test_contract_hybrid.pdf"
    
    print("1. Creating and parsing test PDF...")
    create_test_pdf(test_pdf)
    parser = DocumentParser(chunk_size=200, chunk_overlap=50)
    parsed_doc = parser.parse(test_pdf)

    print("\n2. Initializing Embedding Manager...")
    embedding_manager = EmbeddingManager()
    texts = [chunk.text for chunk in parsed_doc.chunks]
    embeddings = embedding_manager.embed_batch(texts)
    
    print("\n3. Initializing Qdrant & Upserting...")
    qdrant_manager = QdrantManager(collection_name="hybrid_test", vector_size=embedding_manager.embedding_dimension)
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
    qdrant_manager.upsert_records(records)
    
    print("\n4. Initializing BM25 & Indexing...")
    bm25_manager = BM25Manager(persist_dir="data/bm25_test")
    bm25_manager.index_chunks(parsed_doc.chunks)

    print("\n5. Initializing Hybrid Retriever (with Cross-Encoder)...")
    hybrid_retriever = HybridRetriever(
        qdrant_manager=qdrant_manager,
        bm25_manager=bm25_manager,
        embedding_manager=embedding_manager
    )
    
    print("\n6. Performing Hybrid Search & Re-ranking...")
    query = "What is the definition of confidential information?"
    print(f"   Query: '{query}'")
    
    results = hybrid_retriever.retrieve(query, top_k=2)
    
    for i, res in enumerate(results):
        print(f"\n   --- Final Rank {i+1} ---")
        print(f"   Section: {res.get('section')}")
        print(f"   RRF Score: {res.get('rrf_score'):.4f} | Rerank Score: {res.get('rerank_score'):.4f}")
        print(f"   Text: {res.get('text')[:100]}...")
        
    # Cleanup
    if os.path.exists(test_pdf):
        os.remove(test_pdf)
        
if __name__ == "__main__":
    main()
