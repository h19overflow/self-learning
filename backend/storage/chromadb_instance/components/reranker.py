"""
Cross-Encoder Reranker Component

This module provides cross-encoder reranking functionality to improve retrieval precision
by re-scoring initial candidates based on query-document relevance.

Dependencies:
- sentence-transformers for cross-encoder models
- numpy for array operations and similarity calculations
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer

from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig, RetrievalConfig


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval precision."""

    def __init__(self, config: ChromaConfig):
        """Initialize the cross-encoder reranker.
        
        Args:
            config: ChromaDB configuration containing reranking settings
        """
        self.config = config
        self.reranker = None
        self.embedding_model = None  # For diversity filtering
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize reranker if enabled
        if config.enable_reranking_by_default:
            self._initialize_reranker()

    def _initialize_reranker(self):
        """Initialize the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            # Set cache directory if specified
            if self.config.rerank_model_cache_dir:
                import os
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = self.config.rerank_model_cache_dir
            
            # Load cross-encoder model
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.reranker = CrossEncoder(model_name, device=self.config.embedding_device)
            self.logger.info(f"Cross-encoder reranker '{model_name}' ready")
            
        except ImportError:
            self.logger.warning("sentence-transformers CrossEncoder not available. Reranking disabled.")
        except Exception as e:
            self.logger.warning(f"Failed to initialize reranker: {e}. Reranking disabled.")

    def is_available(self) -> bool:
        """Check if reranker is available and ready."""
        return self.reranker is not None

    def rerank_results(
        self, 
        query: str, 
        raw_results: Dict[str, Any], 
        retrieval_config: RetrievalConfig
    ) -> Dict[str, Any]:
        """Apply cross-encoder reranking to search results.
        
        Args:
            query: Original search query
            raw_results: Raw results from ChromaDB
            retrieval_config: Configuration with reranking settings
            
        Returns:
            Reranked results in same format as ChromaDB output
        """
        if not self.is_available():
            self.logger.warning("Reranker not available, returning original results")
            return raw_results

        try:
            # Extract data from raw results
            documents = raw_results.get('documents', [[]])[0]
            metadatas = raw_results.get('metadatas', [[]])[0]
            ids = raw_results.get('ids', [[]])[0]
            
            if not documents:
                return raw_results
            
            # Create query-document pairs for cross-encoder
            pairs = [(query, doc) for doc in documents]
            
            # Get reranking scores in batches
            rerank_scores = self._batch_predict(pairs, retrieval_config.rerank_batch_size)
            
            # Normalize cross-encoder scores to 0-1 range using sigmoid
            import numpy as np
            normalized_scores = [1 / (1 + np.exp(-score)) for score in rerank_scores]
            
            # Create scored results for sorting
            scored_results = [
                (score, doc, meta, doc_id) 
                for score, doc, meta, doc_id in zip(normalized_scores, documents, metadatas, ids)
            ]
            
            # Sort by rerank score (descending)
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Take top-k after reranking
            top_k = retrieval_config.top_k
            scored_results = scored_results[:top_k]
            
            # Apply diversity filtering if enabled
            if retrieval_config.enable_diversity:
                scored_results = self._apply_diversity_filtering(scored_results, retrieval_config)
            
            # Reconstruct results in ChromaDB format
            reranked_results = {
                'documents': [[result[1] for result in scored_results]],
                'metadatas': [[result[2] for result in scored_results]],
                'ids': [[result[3] for result in scored_results]],
                'distances': [[1.0 - result[0] for result in scored_results]]  # Convert score back to distance
            }
            
            self.logger.info(f"Reranked {len(documents)} candidates -> {len(scored_results)} results")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}. Returning original results.")
            return raw_results

    def _batch_predict(self, pairs: List[Tuple[str, str]], batch_size: int) -> List[float]:
        """Predict reranking scores in batches.
        
        Args:
            pairs: List of (query, document) pairs
            batch_size: Batch size for processing
            
        Returns:
            List of reranking scores
        """
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            try:
                batch_scores = self.reranker.predict(batch_pairs)
                all_scores.extend(batch_scores)
            except Exception as e:
                self.logger.error(f"Batch prediction failed: {e}. Using fallback scores.")
                # Fallback to neutral scores for this batch
                all_scores.extend([0.5] * len(batch_pairs))
        
        return all_scores

    def _apply_diversity_filtering(
        self, 
        scored_results: List[Tuple[float, str, Dict, str]], 
        retrieval_config: RetrievalConfig
    ) -> List[Tuple[float, str, Dict, str]]:
        """Apply diversity filtering to reduce redundant results.
        
        Args:
            scored_results: List of (score, document, metadata, id) tuples
            retrieval_config: Configuration with diversity settings
            
        Returns:
            Filtered results with reduced redundancy
        """
        if not scored_results or len(scored_results) <= 1:
            return scored_results

        try:
            # Initialize embedding model for diversity if needed
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(
                    self.config.embedding_model,
                    device=self.config.embedding_device,
                    trust_remote_code=True
                )
            
            # Extract documents for similarity comparison
            documents = [result[1] for result in scored_results]
            
            # Encode documents for diversity comparison
            doc_embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
            
            # Start with the highest-scored result
            diverse_results = [scored_results[0]]
            used_indices = {0}
            
            for i, candidate in enumerate(scored_results[1:], 1):
                candidate_embedding = doc_embeddings[i]
                
                # Check similarity with already selected results
                is_diverse = True
                for selected_idx in [scored_results.index(res) for res in diverse_results]:
                    if selected_idx in used_indices:
                        selected_embedding = doc_embeddings[selected_idx]
                        
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(candidate_embedding, selected_embedding)
                        
                        if similarity > retrieval_config.diversity_threshold:
                            is_diverse = False
                            break
                
                if is_diverse:
                    diverse_results.append(candidate)
                    used_indices.add(i)
            
            self.logger.info(f"Diversity filtering: {len(scored_results)} -> {len(diverse_results)} results")
            return diverse_results
            
        except Exception as e:
            self.logger.error(f"Diversity filtering failed: {e}. Returning original results.")
            return scored_results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            return 0.0

    def calculate_retrieval_k(self, retrieval_config: RetrievalConfig) -> int:
        """Calculate how many candidates to retrieve for reranking.
        
        Args:
            retrieval_config: Configuration with reranking settings
            
        Returns:
            Number of candidates to retrieve
        """
        if retrieval_config.enable_reranking and self.is_available():
            # Retrieve more candidates for reranking, but cap at reasonable limit
            return min(
                retrieval_config.top_k * retrieval_config.rerank_top_k_multiplier, 
                100  # Maximum candidates to avoid performance issues
            )
        return retrieval_config.top_k

    async def finalize(self):
        """Clean up resources."""
        self.reranker = None
        self.embedding_model = None
        self.logger.info("Reranker resources cleaned up")