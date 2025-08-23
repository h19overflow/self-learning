"""
Configuration models for ChromaDB operations.

This module defines configuration classes for ChromaDB ingestion and retrieval,
providing a clean interface for managing settings and parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB instance and operations."""
    
    # Database settings
    persist_directory: Path
    collection_name: str = "academic_papers"
    
    # Embedding settings
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # Must match the model used during ingestion
    embedding_device: str = "cuda"
    
    # Collection settings
    distance_function: str = "cosine"  # cosine, l2, ip
    
    # Batch processing
    batch_size: int = 100
    max_concurrent_batches: int = 3
    
    # Error handling
    continue_on_error: bool = True
    max_retries: int = 3
    
    # Reranking settings (global defaults)
    enable_reranking_by_default: bool = True
    rerank_model_cache_dir: Optional[str] = None  # Custom cache directory for models
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure persist directory exists
        self.persist_directory = Path(self.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Validate distance function
        valid_distances = ["cosine", "l2", "ip"]
        if self.distance_function not in valid_distances:
            raise ValueError(f"distance_function must be one of {valid_distances}")


class RetrievalConfig(BaseModel):
    """Configuration for document retrieval operations."""
    
    # Retrieval parameters
    top_k: int = Field(default=5, description="Number of results to return", gt=0)
    score_threshold: Optional[float] = Field(
        default=None, 
        description="Minimum similarity score threshold", 
        ge=0.0, 
        le=1.0
    )
    
    # Query processing
    query_expansion: bool = Field(default=False, description="Enable query expansion")
    include_metadata: bool = Field(default=True, description="Include document metadata in results")
    include_embeddings: bool = Field(default=False, description="Include embeddings in results")
    
    # Filtering
    source_filter: Optional[str] = Field(default=None, description="Filter by specific source file")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata filters")
    
    # Reranking configuration
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking for better precision")
    rerank_top_k_multiplier: int = Field(default=3, description="Retrieve N times more candidates for reranking", gt=1, le=10)
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model for reranking")
    rerank_batch_size: int = Field(default=16, description="Batch size for reranking operations", gt=0, le=64)
    
    # Diversity filtering (optional post-reranking step)
    enable_diversity: bool = Field(default=False, description="Enable diversity filtering to reduce redundancy")
    diversity_threshold: float = Field(default=0.8, description="Similarity threshold for diversity filtering", ge=0.0, le=1.0)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True