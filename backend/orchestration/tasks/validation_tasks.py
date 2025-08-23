"""
Validation tasks for the pipeline orchestration system.

This module contains tasks for validating source files and preventing duplicates.
"""

from typing import Dict, Any
from prefect import task
from prefect.logging import get_run_logger

from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig
from backend.storage.chromadb_instance.components.chroma_retriever import ChromaRetriever
from ..utils.pipeline_config import PipelineConfiguration
from ..utils.file_validation_utils import clean_source_filename


@task(
    name="Source File Validation",
    description="Check existing sources in ChromaDB to prevent duplicate processing",
    retries=2,
    retry_delay_seconds=10
)
async def validate_source_files_task(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Validate source files against ChromaDB to prevent duplicate ingestion.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dict[str, Any]: Validation results with files to process
    """
    logger = get_run_logger()
    logger.info(f"Validating source files in: {config.pdf_input_directory}")
    
    try:
        # Initialize ChromaRetriever to check existing sources
        chroma_persist_dir = config.chromadb_persist_directory or config.output_directory.parent / "chromadb_storage"
        chroma_config = ChromaConfig(
            persist_directory=chroma_persist_dir,
            collection_name=config.chromadb_collection_name,
            embedding_device=config.chromadb_embedding_device
        )
        
        retriever = ChromaRetriever(chroma_config)
        
        # Get existing sources from ChromaDB
        existing_sources = retriever.list_source_files()
        logger.info(f"Found {len(existing_sources)} existing sources in ChromaDB")
        
        # Clean existing source names for comparison
        cleaned_existing = set(clean_source_filename(src) for src in existing_sources)
        
        # Get PDF files from input directory
        pdf_files = []
        if config.pdf_input_directory.exists():
            pdf_files = list(config.pdf_input_directory.glob("*.pdf"))
        
        # Clean PDF filenames and check for duplicates
        files_to_process = []
        skipped_files = []
        
        for pdf_file in pdf_files:
            cleaned_name = clean_source_filename(pdf_file.stem)
            
            if cleaned_name in cleaned_existing:
                skipped_files.append({
                    "file": pdf_file.name,
                    "reason": "Already exists in ChromaDB",
                    "existing_source": cleaned_name
                })
                logger.info(f"Skipping {pdf_file.name} - already exists in ChromaDB as '{cleaned_name}'")
            else:
                files_to_process.append(pdf_file)
        
        # Clean up retriever
        await retriever.finalize()
        
        validation_results = {
            "status": "success",
            "total_pdf_files": len(pdf_files),
            "files_to_process": len(files_to_process),
            "files_skipped": len(skipped_files),
            "files_to_process_list": [f.name for f in files_to_process],
            "skipped_files_list": skipped_files,
            "existing_sources_count": len(existing_sources),
            "proceed_with_processing": len(files_to_process) > 0
        }
        
        logger.info(f"Validation completed: {len(files_to_process)} files to process, {len(skipped_files)} skipped")
        return validation_results
        
    except Exception as e:
        logger.error(f"Source file validation failed: {e}")
        # Return safe default to continue processing
        return {
            "status": "failed",
            "error": str(e),
            "proceed_with_processing": True,  # Continue processing on validation failure
            "files_to_process": 0,
            "total_pdf_files": 0
        }