"""
Search Result Formatter Component

This module handles the formatting and processing of raw ChromaDB search results
into structured SearchResult objects with proper scoring and metadata handling.

Dependencies:
- SearchResult and RetrievalResults models
- Score calculation utilities
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.storage.chromadb_instance.models.chroma_config import RetrievalConfig
from backend.storage.chromadb_instance.models.search_result import SearchResult, RetrievalResults


class SearchResultFormatter:
    """Formatter for converting raw ChromaDB results to structured objects."""

    def __init__(self):
        """Initialize the search result formatter."""
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def format_search_results(
        self, 
        raw_results: Dict[str, Any], 
        config: RetrievalConfig,
        limit_to_top_k: bool = True
    ) -> List[SearchResult]:
        """Format raw ChromaDB results into SearchResult objects.
        
        Args:
            raw_results: Raw results from ChromaDB query
            config: Retrieval configuration for formatting options
            limit_to_top_k: Whether to limit results to configured top_k
            
        Returns:
            List of formatted SearchResult objects
        """
        try:
            search_results = []
            
            # Extract results (ChromaDB returns nested lists)
            documents = raw_results.get('documents', [[]])[0]
            metadatas = raw_results.get('metadatas', [[]])[0]
            distances = raw_results.get('distances', [[]])[0]
            ids = raw_results.get('ids', [[]])[0]
            
            if not documents:
                self.logger.info("No documents found in raw results")
                return search_results
            
            # Process each result
            for i, (doc, metadata, distance, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                # Convert distance to similarity score
                score = self._calculate_score_from_distance(distance)
                
                # Apply score threshold if configured
                if config.score_threshold is not None and score < config.score_threshold:
                    continue
                
                # Create SearchResult
                result = self._create_search_result(
                    document=doc,
                    document_id=doc_id,
                    score=score,
                    metadata=metadata,
                    chunk_index=i,
                    config=config
                )
                
                search_results.append(result)
                
                # Limit to top_k if requested
                if limit_to_top_k and len(search_results) >= config.top_k:
                    break
            
            self.logger.info(f"Formatted {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to format search results: {e}")
            return []

    def create_retrieval_results(
        self,
        query: str,
        search_results: List[SearchResult],
        retrieval_time_ms: float,
        collection_name: str,
        config: RetrievalConfig,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResults:
        """Create a complete RetrievalResults object.
        
        Args:
            query: Original search query
            search_results: Formatted search results
            retrieval_time_ms: Time taken for retrieval
            collection_name: Name of the collection searched
            config: Retrieval configuration used
            additional_metadata: Optional additional metadata
            
        Returns:
            Complete RetrievalResults object
        """
        try:
            metadata = {
                "collection_name": collection_name,
                "top_k_requested": config.top_k,
                "score_threshold_used": config.score_threshold,
                "reranking_enabled": config.enable_reranking,
                "diversity_enabled": config.enable_diversity,
                "search_timestamp": datetime.now().isoformat()
            }
            
            # Add additional metadata if provided
            if additional_metadata:
                metadata.update(additional_metadata)
            
            return RetrievalResults(
                query=query,
                results=search_results,
                total_results=len(search_results),
                retrieval_time_ms=retrieval_time_ms,
                collection_name=collection_name,
                top_k_requested=config.top_k,
                score_threshold_used=config.score_threshold
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create RetrievalResults: {e}")
            # Return minimal results object on error
            return RetrievalResults(
                query=query,
                results=[],
                total_results=0,
                retrieval_time_ms=retrieval_time_ms,
                collection_name=collection_name,
                top_k_requested=config.top_k,
                score_threshold_used=config.score_threshold
            )

    def _create_search_result(
        self,
        document: str,
        document_id: str,
        score: float,
        metadata: Dict[str, Any],
        chunk_index: int,
        config: RetrievalConfig
    ) -> SearchResult:
        """Create a single SearchResult object.
        
        Args:
            document: Document text content
            document_id: Unique document identifier
            score: Relevance score
            metadata: Document metadata
            chunk_index: Index of this chunk in results
            config: Retrieval configuration
            
        Returns:
            SearchResult object
        """
        try:
            return SearchResult(
                content=document,
                document_id=document_id,
                score=score,
                source_file=metadata.get('source_file', 'unknown'),
                chunk_index=metadata.get('chunk_index', chunk_index),
                page_index=metadata.get('page_index'),
                word_count=metadata.get('word_count'),
                char_count=metadata.get('char_count'),
                metadata=metadata if config.include_metadata else None,
                embedding=None  # Embeddings not included by default for performance
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create SearchResult: {e}")
            # Return minimal result on error
            return SearchResult(
                content=document,
                document_id=document_id,
                score=score,
                source_file="unknown",
                chunk_index=chunk_index
            )

    def _calculate_score_from_distance(self, distance: float, distance_function: str = "cosine") -> float:
        """Calculate relevance score from distance metric.
        
        Args:
            distance: Distance value from ChromaDB
            distance_function: Type of distance function used
            
        Returns:
            Relevance score (0.0 to 1.0, higher is better)
        """
        try:
            if distance is None:
                return 0.0
            
            if distance_function == "cosine":
                # Cosine distance: 0 = identical, 2 = opposite
                return max(0.0, 1.0 - distance)
            elif distance_function == "l2":
                # L2 distance: 0 = identical, higher = more different
                return 1.0 / (1.0 + distance)
            elif distance_function == "ip":
                # Inner product: higher = more similar (already similarity)
                return max(0.0, min(1.0, distance))
            else:
                # Default to cosine-like conversion
                return max(0.0, 1.0 - distance)
                
        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}. Using default score.")
            return 0.0

    def filter_results_by_score(
        self, 
        results: List[SearchResult], 
        min_score: float
    ) -> List[SearchResult]:
        """Filter search results by minimum score threshold.
        
        Args:
            results: List of search results to filter
            min_score: Minimum score threshold
            
        Returns:
            Filtered list of results
        """
        try:
            filtered = [result for result in results if result.score >= min_score]
            self.logger.info(f"Score filtering: {len(results)} -> {len(filtered)} results (min_score={min_score})")
            return filtered
            
        except Exception as e:
            self.logger.error(f"Score filtering failed: {e}")
            return results

    def deduplicate_results(
        self, 
        results: List[SearchResult],
        similarity_threshold: float = 0.95
    ) -> List[SearchResult]:
        """Remove duplicate or highly similar results.
        
        Args:
            results: List of search results
            similarity_threshold: Threshold for considering results duplicates
            
        Returns:
            Deduplicated list of results
        """
        try:
            if len(results) <= 1:
                return results
            
            deduplicated = []
            seen_content = set()
            
            for result in results:
                # Simple content-based deduplication
                content_hash = hash(result.content.strip().lower())
                
                if content_hash not in seen_content:
                    deduplicated.append(result)
                    seen_content.add(content_hash)
            
            self.logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)} results")
            return deduplicated
            
        except Exception as e:
            self.logger.error(f"Deduplication failed: {e}")
            return results

    def group_results_by_source(self, results: List[SearchResult]) -> Dict[str, List[SearchResult]]:
        """Group search results by source file.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary mapping source files to their results
        """
        try:
            grouped = {}
            
            for result in results:
                source = result.source_file
                if source not in grouped:
                    grouped[source] = []
                grouped[source].append(result)
            
            # Sort results within each group by score
            for source in grouped:
                grouped[source].sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"Grouped {len(results)} results into {len(grouped)} sources")
            return grouped
            
        except Exception as e:
            self.logger.error(f"Grouping by source failed: {e}")
            return {"unknown": results}