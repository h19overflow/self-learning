"""Simplified chunking configuration."""

from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """Simple configuration for text chunking."""
    chunk_size: int = 8192
    overlap: int = 500
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    
    @classmethod
    def create_default(cls) -> "ChunkingConfig":
        return cls()
    
    @classmethod
    def create_large_chunks(cls) -> "ChunkingConfig":
        return cls(chunk_size=8192, overlap=500)
    
    @classmethod
    def create_small_chunks(cls) -> "ChunkingConfig":
        return cls(chunk_size=4096, overlap=250)
    
    # Legacy compatibility properties
    @property
    def min_chunk_size(self): return self.chunk_size // 4
    
    @property 
    def max_chunk_size(self): return self.chunk_size
    
    @property
    def overlap_size(self): return self.overlap
    
    @property
    def similarity_threshold(self): return 0.95
    
    @property
    def preserve_paragraphs(self): return True
    
    @property
    def preserve_code_blocks(self): return True
    
    @property
    def preserve_tables(self): return True
    
    @property
    def generate_embeddings(self): return False
    
    @property
    def clean_text(self): return True
    
    # Legacy methods for backward compatibility
    @classmethod
    def create_high_quality(cls) -> "ChunkingConfig":
        return cls()
    
    @classmethod
    def create_long_context(cls) -> "ChunkingConfig":
        return cls()
    
    def validate(self) -> None:
        """Basic validation."""
        pass