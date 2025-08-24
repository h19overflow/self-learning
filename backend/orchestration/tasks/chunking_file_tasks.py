"""
File-level semantic chunking tasks for Kafka-based pipeline orchestration.

Each task chunks a single markdown file independently, allowing for better
error isolation and parallel processing via Kafka.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from prefect import task
from prefect.logging import get_run_logger

from backend.document_processing.local_mineru.chunker.semantic_chunker import SemanticChunker


@task(
    name="Chunk Single Markdown File",
    description="Create semantic chunks from a single markdown file",
    retries=2,
    retry_delay_seconds=15
)
async def chunk_single_markdown_task(
    markdown_file_path: str,
    chunk_size: int = 8192,
    overlap: int = 500,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create semantic chunks from a single markdown file.
    
    Args:
        markdown_file_path: Path to the markdown file to chunk
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks
        task_id: Optional task identifier for tracking
        
    Returns:
        Dict containing chunking results and chunk data
    """
    logger = get_run_logger()
    md_path = Path(markdown_file_path)
    
    if not md_path.exists():
        error_msg = f"Markdown file not found: {markdown_file_path}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "file_path": markdown_file_path,
            "task_id": task_id
        }
    
    logger.info(f"Chunking markdown file: {md_path.name}")
    
    try:
        chunker = SemanticChunker(
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = chunker.chunk_text(content, str(md_path))
        
        if chunks:
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                chunk_obj = {
                    "chunk_id": f"{md_path.stem}_chunk_{i}",
                    "content": chunk_text,
                    "source_file": str(md_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk_text)
                }
                chunk_objects.append(chunk_obj)
            
            logger.info(f"Chunking completed: {len(chunks)} chunks generated")
            return {
                "status": "success",
                "file_path": markdown_file_path,
                "chunks_generated": len(chunks),
                "chunks": chunk_objects,
                "task_id": task_id,
                "processing_completed_at": asyncio.get_event_loop().time()
            }
        else:
            error_msg = "No chunks generated from file"
            logger.warning(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "file_path": markdown_file_path,
                "task_id": task_id
            }
            
    except Exception as e:
        error_msg = f"Chunking exception: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "file_path": markdown_file_path,
            "task_id": task_id
        }


# HELPER FUNCTIONS

def prepare_chunks_for_kafka(chunks: List[Dict[str, Any]], batch_size: int = 10) -> List[List[Dict[str, Any]]]:
    """
    Split chunks into batches for Kafka message size optimization.
    
    Args:
        chunks: List of chunk objects
        batch_size: Number of chunks per batch
        
    Returns:
        List of chunk batches
    """
    batches = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batches.append(batch)
    return batches