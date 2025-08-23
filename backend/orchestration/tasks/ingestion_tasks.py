"""
Ingestion tasks for the pipeline orchestration system.

This module contains tasks for ingesting processed content into vector databases.
"""

from typing import Dict, Any
from prefect import task
from prefect.logging import get_run_logger

from backend.storage.chromadb_instance.chromadb_manager import ChromaDBManager
from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig
from ..utils.pipeline_config import PipelineConfiguration


@task(
    name="ChromaDB RAG Ingestion",
    description="Ingest semantic chunks into ChromaDB vector database",
    retries=2,
    retry_delay_seconds=20
)
async def chromadb_rag_ingestion_task(config: PipelineConfiguration, chunk_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest semantic chunks into ChromaDB vector database.
    
    Args:
        config: Pipeline configuration
        chunk_results: Results from semantic chunking task
        
    Returns:
        Dict[str, Any]: ChromaDB ingestion results
    """
    logger = get_run_logger()
    
    if not config.enable_rag_ingestion:
        logger.info("RAG ingestion disabled, skipping")
        return {"skipped": True, "reason": "RAG ingestion disabled"}
    
    logger.info(f"Starting ChromaDB ingestion from: {config.chunked_output_file}")
    
    chroma_manager = None
    try:
        # Set up ChromaDB persist directory
        if config.chromadb_persist_directory is None:
            chroma_persist_dir = config.output_directory.parent / "chromadb_storage"
        else:
            chroma_persist_dir = config.chromadb_persist_directory
        
        # Create ChromaDB configuration
        chroma_config = ChromaConfig(
            persist_directory=chroma_persist_dir,
            collection_name=config.chromadb_collection_name,
            embedding_device=config.chromadb_embedding_device,
            batch_size=200,  # Smaller batches for stability
            max_concurrent_batches=4,
            continue_on_error=config.continue_on_errors
        )
        
        # Initialize ChromaDB manager
        chroma_manager = ChromaDBManager(chroma_config)
        
        # Ingest chunks from file
        ingestion_results = await chroma_manager.ingest_chunks_from_file(config.chunked_output_file)
        
        # Get collection information
        collection_info = chroma_manager.get_collection_info()
        
        # Format results to match expected interface
        if ingestion_results.get("success", False):
            stats = ingestion_results.get("statistics", {})
            results = {
                "success": True,
                "total_chunks_processed": stats.get("successful_ingestions", 0),
                "total_files_processed": stats.get("total_documents", 0),
                "failed_ingestions": stats.get("failed_ingestions", 0),
                "processing_time": stats.get("processing_time", 0.0),
                "collection_name": config.chromadb_collection_name,
                "total_documents_in_db": collection_info.get("total_documents", 0),
                "ingestion_completed": True
            }
            
            logger.info(f"ChromaDB ingestion completed: {results['total_chunks_processed']} chunks ingested from {results['total_files_processed']} files")
            logger.info(f"Collection now contains {results['total_documents_in_db']} total documents")
            return results
        else:
            error_msg = ingestion_results.get("error", "Unknown error during ingestion")
            logger.error(f"ChromaDB ingestion failed: {error_msg}")
            return {
                "failed": True, 
                "error": error_msg,
                "statistics": ingestion_results.get("statistics", {})
            }
        
    except Exception as e:
        logger.error(f"ChromaDB ingestion failed: {e}")
        if not config.continue_on_errors:
            raise
        return {"failed": True, "error": str(e)}
    finally:
        # Ensure proper cleanup of ChromaDB resources
        if chroma_manager:
            try:
                await chroma_manager.finalize()
                logger.info("ChromaDB manager finalized successfully")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")