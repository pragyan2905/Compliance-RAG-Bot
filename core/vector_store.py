import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from schemas.retrieval import VectorRecord
from schemas.document import ParsedDocument

class QdrantManager:
    """
    Manages vector storage and retrieval using a local Qdrant instance.
    """
    
    def __init__(self, collection_name: str = "contracts", db_path: str = "data/qdrant", vector_size: int = 384, client: Optional[QdrantClient] = None):
        """
        Initializes the local Qdrant client or uses an existing shared client.
        
        Args:
            collection_name: Name of the collection to store embeddings.
            db_path: Path to store local Qdrant database files.
            vector_size: Dimensionality of the embeddings (384 for bge-small).
            client: Optional shared QdrantClient instance.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        if client is not None:
            self.client = client
        else:
            # Ensure the data directory exists
            os.makedirs(db_path, exist_ok=True)
            # Initialize local Qdrant client
            self.client = QdrantClient(path=db_path)
        
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Creates the collection if it doesn't already exist.
        """
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=rest.Distance.COSINE
                )
            )

    def upsert_records(self, records: List[VectorRecord]):
        """
        Upserts a batch of VectorRecords into the Qdrant collection.
        """
        if not records:
            return
            
        points = []
        for record in records:
            payload = {
                "text": record.text,
                "document_id": record.document_id,
                "page": record.page,
                "section": record.section,
                "subsection": record.subsection,
                "clause_id": record.clause_id
            }
            
            points.append(
                rest.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a dense semantic search.
        
        Returns a list of dictionaries containing the payload and the similarity score.
        """
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points
        
        results = []
        for scored_point in search_result:
            result_item = scored_point.payload.copy() if scored_point.payload else {}
            result_item["score"] = scored_point.score
            result_item["chunk_id"] = scored_point.id
            results.append(result_item)
            
        return results
