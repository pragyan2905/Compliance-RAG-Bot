import os
from test_parser import create_test_pdf
from core.document_parser import DocumentParser
from core.embedding_manager import EmbeddingManager
from core.vector_store import QdrantManager
from schemas.retrieval import VectorRecord

def main():
    test_pdf = "test_contract_vector.pdf"
    
    print("1. Creating and parsing test PDF...")
    create_test_pdf(test_pdf)
    parser = DocumentParser(chunk_size=200, chunk_overlap=50)
    parsed_doc = parser.parse(test_pdf)
    print(f"   Parsed {len(parsed_doc.chunks)} chunks.")

    print("\n2. Initializing Embedding Manager (BAAI/bge-small-en-v1.5)...")
    embedding_manager = EmbeddingManager()
    
    print("\n3. Generating embeddings for chunks...")
    texts = [chunk.text for chunk in parsed_doc.chunks]
    embeddings = embedding_manager.embed_batch(texts)
    
    print("\n4. Initializing Qdrant Vector Store...")
    qdrant_manager = QdrantManager(collection_name="test_contracts", vector_size=embedding_manager.embedding_dimension)
    
    print("\n5. Upserting records into Qdrant...")
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
    print("   Successfully upserted records.")
    
    print("\n6. Performing Semantic Search...")
    query = "What happens if we terminate the agreement?"
    print(f"   Query: '{query}'")
    query_vector = embedding_manager.embed_text(query)
    
    results = qdrant_manager.search(query_vector, limit=2)
    
    for i, res in enumerate(results):
        print(f"\n   --- Result {i+1} (Score: {res['score']:.4f}) ---")
        print(f"   Section: {res.get('section')}")
        print(f"   Clause ID: {res.get('clause_id')}")
        print(f"   Text: {res.get('text')[:100]}...")
        
    # Cleanup
    if os.path.exists(test_pdf):
        os.remove(test_pdf)
        
if __name__ == "__main__":
    main()
