from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from core.vector_store import QdrantManager
from core.bm25_retriever import BM25Manager
from core.embedding_manager import EmbeddingManager

class HybridRetriever:
    """
    Orchestrates Hybrid Search (Dense + Sparse) followed by RRF and Cross-Encoder Re-ranking.
    """
    
    def __init__(
        self,
        qdrant_manager: QdrantManager,
        bm25_manager: BM25Manager,
        embedding_manager: EmbeddingManager,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        self.qdrant_manager = qdrant_manager
        self.bm25_manager = bm25_manager
        self.embedding_manager = embedding_manager
        
        try:
            self.reranker = CrossEncoder(reranker_model_name, device=device)
        except Exception as e:
            print(f"Warning: Failed to load reranker on {device}. Falling back to CPU. Error: {e}")
            self.reranker = CrossEncoder(reranker_model_name, device="cpu")

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Fuses ranked lists using Reciprocal Rank Fusion (RRF).
        RRF score = 1 / (k + rank)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Process dense results
        for rank, res in enumerate(dense_results):
            chunk_id = res['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
            if chunk_id not in chunk_metadata:
                chunk_metadata[chunk_id] = res
                
        # Process sparse results
        for rank, res in enumerate(sparse_results):
            chunk_id = res['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
            if chunk_id not in chunk_metadata:
                chunk_metadata[chunk_id] = res
                
        # Sort by RRF score descending
        fused_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for chunk_id, score in fused_items:
            result_dict = chunk_metadata[chunk_id].copy()
            result_dict["rrf_score"] = score
            fused_results.append(result_dict)
            
        return fused_results

    def retrieve(self, query: str, top_k: int = 5, retrieve_k: int = 20) -> List[Dict[str, Any]]:
        """
        Executes the full hybrid retrieval and re-ranking pipeline.
        
        1. Query Qdrant for top `retrieve_k` chunks.
        2. Query BM25 for top `retrieve_k` chunks.
        3. Fuse with RRF.
        4. Re-rank top fused results using Cross-Encoder.
        5. Return top `top_k` results.
        """
        # 1. Dense Retrieval
        query_vector = self.embedding_manager.embed_text(query)
        dense_results = self.qdrant_manager.search(query_vector, limit=retrieve_k)
        
        # 2. Sparse Retrieval
        sparse_results = self.bm25_manager.search(query, limit=retrieve_k)
        
        # 3. RRF Fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # If no results, return empty
        if not fused_results:
            return []
            
        # 4. Cross-Encoder Re-ranking
        # Take the top fused results (e.g., up to retrieve_k) to re-rank
        candidates = fused_results[:retrieve_k]
        
        # Prepare pairs for the Cross-Encoder: (query, document_text)
        cross_encoder_pairs = [(query, cand["text"]) for cand in candidates]
        
        # Predict relevance scores
        re_rank_scores = self.reranker.predict(cross_encoder_pairs)
        
        # Attach new scores and sort
        for idx, score in enumerate(re_rank_scores):
            candidates[idx]["rerank_score"] = float(score)
            
        # Sort by the new rerank score descending
        final_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        # 5. Return Top K
        return final_results[:top_k]
