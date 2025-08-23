"""Simplified file processing utilities."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from ..models import Chunk, ChunkMetadata
from .page_mapping_utils import PageMappingUtils


class FileProcessingUtils:
    """Minimal file processing for semantic chunking."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def find_markdown_files(self, directory: Path) -> List[Path]:
        """Find all markdown files in directory."""
        if not directory.exists():
            return []
        return [f for f in directory.rglob("*.md") if not f.name.endswith('.backup')]
    
    def read_markdown_file(self, file_path: Path) -> tuple[str, str]:
        """Read markdown file and return content and source name."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, file_path.stem
    
    def create_chunks_with_metadata(
        self,
        text: str,
        source_name: str,
        file_path: Path,
        text_processor,  # Semantic chunker
        page_mapper: PageMappingUtils
    ) -> List[Chunk]:
        """Create chunks with metadata using semantic chunker."""
        page_mapping = page_mapper.load_page_mapping(file_path)
        chunk_texts = text_processor.chunk_text(text, source_name)
        
        chunks = []
        current_position = 0
        
        for i, chunk_text in enumerate(chunk_texts):
            start_pos = text.find(chunk_text[:100], current_position) if len(chunk_text) > 100 else text.find(chunk_text, current_position)
            start_pos = start_pos if start_pos != -1 else current_position
            end_pos = start_pos + len(chunk_text)
            
            page_idx = page_mapper.get_page_for_position(page_mapping, start_pos)
            
            metadata = ChunkMetadata(
                source_file=source_name,
                chunk_index=i,
                chunk_id=Chunk.create_chunk_id(source_name, i),
                start_position=start_pos,
                end_position=end_pos,
                page_index=page_idx,
                word_count=0,
                char_count=0
            )
            
            chunks.append(Chunk(content=chunk_text, metadata=metadata))
            current_position = end_pos
        
        return chunks
    
    def create_file_result(self, chunks: List[Chunk], chunk_size: int = 8192, overlap: int = 500) -> Dict[str, Any]:
        """Create result dictionary for processed file."""
        if not chunks:
            return {"source_file": "unknown", "total_chunks": 0, "chunks": []}
        
        return {
            "source_file": chunks[0].metadata.source_file,
            "total_chunks": len(chunks),
            "chunks": [chunk.to_dict() for chunk in chunks]
        }
    
    def process_single_file(
        self,
        file_path: Path,
        text_processor,  # Semantic chunker
        page_mapper: PageMappingUtils
    ) -> Dict[str, Any]:
        """Process single file with semantic chunker."""
        content, source_name = self.read_markdown_file(file_path)
        chunks = self.create_chunks_with_metadata(content, source_name, file_path, text_processor, page_mapper)
        return self.create_file_result(chunks, text_processor.chunk_size, text_processor.overlap)
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def create_summary(self, file_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary from file results."""
        total_files = len(file_results)
        total_chunks = sum(result.get('total_chunks', 0) for result in file_results.values())
        
        return {
            "summary": {
                "total_files_processed": total_files,
                "total_chunks_generated": total_chunks,
                "average_chunks_per_file": total_chunks / total_files if total_files > 0 else 0
            },
            "files": file_results
        }


# HELPER FUNCTIONS

def create_file_processor(logger: logging.Logger = None) -> FileProcessingUtils:
    """Create file processor instance."""
    return FileProcessingUtils(logger)

def process_directory_simple(directory: Path, chunk_size: int = 8192, overlap: int = 500) -> List[Chunk]:
    """Simple directory processing with semantic chunker."""
    from ..semantic_chunker import SemanticChunker
    
    processor = FileProcessingUtils()
    text_processor = SemanticChunker(chunk_size, overlap)
    page_mapper = PageMappingUtils()
    
    all_chunks = []
    for file_path in processor.find_markdown_files(directory):
        try:
            content, source_name = processor.read_markdown_file(file_path)
            chunks = processor.create_chunks_with_metadata(content, source_name, file_path, text_processor, page_mapper)
            all_chunks.extend(chunks)
        except Exception:
            continue
    
    return all_chunks