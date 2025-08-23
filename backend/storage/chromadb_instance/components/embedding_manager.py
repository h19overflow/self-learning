"""
Embedding Manager Component

This module manages sentence transformer models for query and document encoding,
providing a clean interface for embedding operations with fallback mechanisms.

Dependencies:
- sentence-transformers for embedding models
- Error handling and logging
"""

import logging
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer

from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig


class EmbeddingManager:
    """Manager for sentence transformer embedding models."""

    def __init__(self, config: ChromaConfig):
        """Initialize the embedding manager.
        
        Args:
            config: ChromaDB configuration containing embedding settings
        """
        self.config = config
        self.embedding_model = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize embedding model
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize the sentence transformer embedding model with fallback."""
        try:
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.embedding_device,
                trust_remote_code=True
            )
            self.logger.info(f"Embedding model '{self.config.embedding_model}' ready")
            
        except Exception as e:
            self.logger.warning(f"Failed to load {self.config.embedding_model}, trying fallback")
            self._load_fallback_model()

    def _load_fallback_model(self):
        """Load fallback embedding model if primary model fails."""
        fallback_models = [
            'BAAI/bge-large-en-v1.5',
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2'
        ]
        
        for fallback_model in fallback_models:
            try:
                self.embedding_model = SentenceTransformer(
                    fallback_model,
                    device=self.config.embedding_device
                )
                self.logger.info(f"Loaded fallback embedding model: {fallback_model}")
                return
            except Exception as e:
                self.logger.warning(f"Fallback model {fallback_model} failed: {e}")
                continue
        
        # If all fallbacks fail, raise error
        raise RuntimeError("Failed to load any embedding model (including fallbacks)")

    def encode_query(self, query: str) -> List[float]:
        """Encode a single query text to embedding vector.
        
        Args:
            query: Query text to encode
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            RuntimeError: If encoding fails
        """
        try:
            if not self.embedding_model:
                raise RuntimeError("Embedding model not initialized")
            
            embedding = self.embedding_model.encode([query])
            return embedding[0].tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to encode query: {e}")
            raise RuntimeError(f"Query encoding failed: {e}")

    def encode_documents(self, documents: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Encode multiple documents to embedding vectors.
        
        Args:
            documents: List of document texts to encode
            batch_size: Optional batch size for processing (uses config default if None)
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If encoding fails
        """
        try:
            if not self.embedding_model:
                raise RuntimeError("Embedding model not initialized")
            
            if not documents:
                return []
            
            # Use config batch size if not specified
            if batch_size is None:
                batch_size = self.config.batch_size
            
            # Process in batches if needed
            all_embeddings = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch)
                all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode documents: {e}")
            raise RuntimeError(f"Document encoding failed: {e}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension size
            
        Raises:
            RuntimeError: If model not available
        """
        try:
            if not self.embedding_model:
                raise RuntimeError("Embedding model not initialized")
            
            return self.embedding_model.get_sentence_embedding_dimension()
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding dimension: {e}")
            raise RuntimeError(f"Cannot determine embedding dimension: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        if not self.embedding_model:
            return {"status": "not_initialized"}
        
        try:
            return {
                "model_name": getattr(self.embedding_model, '_model_name', self.config.embedding_model),
                "device": str(self.embedding_model.device),
                "max_seq_length": getattr(self.embedding_model, '_max_seq_length', 'unknown'),
                "embedding_dimension": self.get_embedding_dimension(),
                "status": "ready"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def is_available(self) -> bool:
        """Check if embedding model is available and ready.
        
        Returns:
            True if model is ready, False otherwise
        """
        return self.embedding_model is not None

    def similarity_search_embeddings(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """Perform similarity search using embeddings directly.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            import numpy as np
            from numpy.linalg import norm
            
            query_vec = np.array(query_embedding)
            similarities = []
            
            for i, candidate_emb in enumerate(candidate_embeddings):
                candidate_vec = np.array(candidate_emb)
                
                # Calculate cosine similarity
                similarity = np.dot(query_vec, candidate_vec) / (norm(query_vec) * norm(candidate_vec))
                similarities.append((i, float(similarity)))
            
            # Sort by similarity (descending) and take top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []

    def reload_model(self, new_model_name: Optional[str] = None):
        """Reload the embedding model, optionally with a different model.
        
        Args:
            new_model_name: Optional new model name to load
        """
        try:
            if new_model_name:
                self.config.embedding_model = new_model_name
            
            self.embedding_model = None  # Clear current model
            self._initialize_embedding_model()
            self.logger.info("Embedding model reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reload embedding model: {e}")
            raise RuntimeError(f"Model reload failed: {e}")

    async def finalize(self):
        """Clean up resources."""
        self.embedding_model = None
        self.logger.info("Embedding manager resources cleaned up")