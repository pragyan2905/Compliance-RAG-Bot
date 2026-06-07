import os
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """
    Manages the generation of dense vector embeddings using SentenceTransformers.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        """
        Initializes the EmbeddingManager.
        
        Args:
            model_name: The HuggingFace model string or path.
            device: 'cpu' or 'cuda' or 'mps' for Mac M-series.
        """
        self.model_name = model_name
        self.device = device
        # Ensure we set device properly for Mac (mps) if available, otherwise fallback
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"Warning: Failed to load model on {device}. Falling back to CPU. Error: {e}")
            self.model = SentenceTransformer(model_name, device="cpu")
            
        # Get the dimension of the embeddings
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a dense embedding for a single string.
        """
        # Prefixing with "Represent this sentence for searching relevant passages:" 
        # is recommended for BAAI/bge models for asymmetric retrieval queries, 
        # but for document encoding, we typically just encode the raw text.
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates dense embeddings for a batch of strings.
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return embeddings.tolist()
