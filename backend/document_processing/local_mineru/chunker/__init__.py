"""
Simplified semantic chunking for MinerU documents.
"""

from .semantic_chunker import SemanticChunker

# Backward compatibility imports
from .models.chunk import Chunk, ChunkMetadata
from .utils.file_processing_utils import FileProcessingUtils
from .utils.page_mapping_utils import PageMappingUtils

__all__ = [
    "SemanticChunker",
    "Chunk", 
    "ChunkMetadata",
    "FileProcessingUtils",
    "PageMappingUtils", 
]