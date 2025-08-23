"""
Simplified semantic chunking for MinerU documents.
"""

from .semantic_chunker import SemanticChunker

# Backward compatibility imports
from .models.chunk import Chunk, ChunkMetadata
from .models.chunking_config import ChunkingConfig
from .utils.file_processing_utils import FileProcessingUtils
from .utils.page_mapping_utils import PageMappingUtils

__all__ = [
    "SemanticChunker",
    "Chunk", 
    "ChunkMetadata",
    "ChunkingConfig",
    "FileProcessingUtils",
    "PageMappingUtils", 
]