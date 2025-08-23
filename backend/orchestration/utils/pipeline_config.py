"""
Pipeline Configuration for Document Processing Orchestration

Centralized configuration management for the entire document processing pipeline.
"""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class PipelineConfiguration:
    """Comprehensive configuration for the document processing pipeline."""
    
    # Input/Output paths
    pdf_input_directory: Path
    output_directory: Path
    chunked_output_file: Path
    
    # Processing settings
    chunk_size: int = 5128
    overlap: int = 500
    max_concurrent_vlm_requests: int = 5
    
    # VLM settings
    gemini_model: str = "gemini-2.5-flash"
    enable_vlm_enhancement: bool = True
    backup_original_files: bool = True
    
    # ChromaDB RAG settings
    enable_rag_ingestion: bool = True
    chromadb_persist_directory: Optional[Path] = None
    chromadb_collection_name: str = "academic_papers"
    chromadb_embedding_device: str = "cuda"
    
    # Video transcription settings
    enable_video_transcription: bool = False
    video_urls: Optional[List[str]] = None
    video_transcript_language: str = "en"
    max_videos_per_playlist: Optional[int] = None
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: int = 30
    continue_on_errors: bool = True
    
    # Output directory cleanup
    auto_clean_old_outputs: bool = True
    dry_run_cleanup: bool = False  # Set to True to see what would be cleaned without actually deleting
    
    # Logging
    log_level: str = "INFO"


def create_default_config(
    pdf_input_directory: str,
    output_directory: str,
    chunked_output_file: str
) -> PipelineConfiguration:
    """
    Create a default pipeline configuration.
    
    Args:
        pdf_input_directory: Directory containing PDF files
        output_directory: Directory for processed output
        chunked_output_file: File path for chunked results
        
    Returns:
        PipelineConfiguration: Default configuration
    """
    return PipelineConfiguration(
        pdf_input_directory=Path(pdf_input_directory),
        output_directory=Path(output_directory),
        chunked_output_file=Path(chunked_output_file)
    )