import re
import pickle
import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from schemas.document import DocumentChunk

class BM25Manager:
    """
    Manages sparse retrieval using BM25.
    Maintains an in-memory BM25 index and a mapping back to chunks.
    """
    
    def __init__(self, persist_dir: str = "data/bm25"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "bm25_index.pkl")
        self.corpus_path = os.path.join(self.persist_dir, "bm25_corpus.pkl")
        
        self.bm25: BM25Okapi = None
        self.chunks: List[DocumentChunk] = []
        self._load_if_exists()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        # Convert to lowercase and split by non-alphanumeric
        return [word for word in re.split(r'\W+', text.lower()) if word]

    def _load_if_exists(self):
        """Loads index from disk if it exists."""
        if os.path.exists(self.index_path) and os.path.exists(self.corpus_path):
            with open(self.index_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(self.corpus_path, 'rb') as f:
                self.chunks = pickle.load(f)

    def _save(self):
        """Saves index to disk."""
        if self.bm25 is not None:
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.bm25, f)
            with open(self.corpus_path, 'wb') as f:
                pickle.dump(self.chunks, f)

    def index_chunks(self, chunks: List[DocumentChunk]):
        """
        Builds the BM25 index from a list of DocumentChunks.
        Currently, this recreates the index from scratch for simplicity, 
        but in a production environment with incremental updates, you might
        maintain a persistent document store and rebuild periodically.
        """
        if not chunks:
            return
            
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save()

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index for the top `limit` results.
        Returns a list of dictionaries simulating the Qdrant search output structure.
        """
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue # Skip zero score results
                
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_id": "unknown", # BM25 doesn't currently store doc id directly, just chunks
                "page": chunk.page,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "clause_id": chunk.clause_id,
                "score": scores[idx]
            })
            
        return results
