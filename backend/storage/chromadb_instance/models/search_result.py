"""
Search result models for ChromaDB retrieval operations.

This module defines data structures for representing search results 
from ChromaDB queries.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SearchResult:
    """Represents a single search result from ChromaDB."""
    
    # Content and identification
    content: str
    document_id: str
    score: float
    
    # Metadata
    source_file: str
    chunk_index: int
    page_index: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate search result after initialization."""
        if not self.content.strip():
            raise ValueError("Search result content cannot be empty")
        
        if self.score < 0 or self.score > 1:
            raise ValueError("Search result score must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary format."""
        return {
            "content": self.content,
            "document_id": self.document_id,
            "score": self.score,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "page_index": self.page_index,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
            "has_embedding": self.embedding is not None
        }


@dataclass
class RetrievalResults:
    """Container for multiple search results with metadata."""
    
    # Query information
    query: str
    results: List[SearchResult]
    
    # Retrieval metadata
    total_results: int
    retrieval_time_ms: float
    collection_name: str
    
    # Query processing info
    top_k_requested: int
    score_threshold_used: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.results) > 0
    
    @property
    def average_score(self) -> float:
        """Calculate average score of all results."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)
    
    @property
    def unique_sources(self) -> List[str]:
        """Get list of unique source files in results."""
        return list(set(r.source_file for r in self.results))
    
    def filter_by_score(self, min_score: float) -> 'RetrievalResults':
        """Create new RetrievalResults with filtered results by minimum score."""
        filtered_results = [r for r in self.results if r.score >= min_score]
        
        return RetrievalResults(
            query=self.query,
            results=filtered_results,
            total_results=len(filtered_results),
            retrieval_time_ms=self.retrieval_time_ms,
            collection_name=self.collection_name,
            top_k_requested=self.top_k_requested,
            score_threshold_used=min_score,
            timestamp=self.timestamp
        )
    
    def filter_by_source(self, source_file: str) -> 'RetrievalResults':
        """Create new RetrievalResults with results from specific source file."""
        filtered_results = [r for r in self.results if r.source_file == source_file]
        
        return RetrievalResults(
            query=self.query,
            results=filtered_results,
            total_results=len(filtered_results),
            retrieval_time_ms=self.retrieval_time_ms,
            collection_name=self.collection_name,
            top_k_requested=self.top_k_requested,
            score_threshold_used=self.score_threshold_used,
            timestamp=self.timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval results to dictionary format."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "retrieval_time_ms": self.retrieval_time_ms,
            "collection_name": self.collection_name,
            "top_k_requested": self.top_k_requested,
            "score_threshold_used": self.score_threshold_used,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "summary": {
                "has_results": self.has_results,
                "average_score": self.average_score,
                "unique_sources": self.unique_sources
            }
        }