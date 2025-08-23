"""
Chunk data models for semantic chunking.

This module defines the data structures used to represent text chunks
and their associated metadata.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    
    source_file: str  # Clean filename without extension
    chunk_index: int  # Position of chunk within document
    chunk_id: str  # Unique identifier for the chunk
    start_position: int  # Character position where chunk starts
    end_position: int  # Character position where chunk ends
    page_index: int  # Page number where chunk starts (from MinerU)
    word_count: int  # Number of words in chunk
    char_count: int  # Number of characters in chunk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "page_index": self.page_index,
            "word_count": self.word_count,
            "char_count": self.char_count
        }


@dataclass
class Chunk:
    """Represents a semantically meaningful text chunk."""
    
    content: str  # The actual text content
    metadata: ChunkMetadata  # Associated metadata
    embedding: Optional[Any] = None  # Optional embedding vector
    
    def __post_init__(self):
        """Validate chunk data after initialization."""
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        
        # Update metadata with actual content statistics
        self.metadata.word_count = len(self.content.split())
        self.metadata.char_count = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "has_embedding": self.embedding is not None
        }
    
    @classmethod
    def create_chunk_id(cls, source_file: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        # Remove file extension and create clean ID
        clean_name = Path(source_file).stem
        return f"{clean_name}_chunk_{chunk_index:04d}"
