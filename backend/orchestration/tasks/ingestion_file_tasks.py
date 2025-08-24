"""
File-level ChromaDB ingestion tasks for Kafka-based pipeline orchestration.

Each task ingests chunks from a single file independently, allowing for better
error isolation and parallel processing via Kafka.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from prefect import task
from prefect.logging import get_run_logger

from backend.storage.chromadb_instance.chromadb_manager import ChromaDBManager
from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig


@task(
    name="Ingest Single File Chunks",
    description="Ingest chunks from a single file into ChromaDB",
    retries=2,
    retry_delay_seconds=20
)
async def ingest_single_file_chunks_task(
    chunks_data: Dict[str, Any],
    chromadb_persist_directory: str,
    collection_name: str = "self_learning_rag",
    embedding_device: str = "cpu",
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest chunks from a single file into ChromaDB vector database.
    
    Args:
        chunks_data: Chunk data from chunk_single_markdown_task
        chromadb_persist_directory: ChromaDB persistence directory
        collection_name: Name of the ChromaDB collection
        embedding_device: Device for embedding computation
        task_id: Optional task identifier for tracking
        
    Returns:
        Dict containing ingestion results and metadata
    """
    logger = get_run_logger()
    
    if chunks_data.get("status") != "success":
        error_msg = f"Cannot ingest chunks: {chunks_data.get('error', 'Invalid chunk data')}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "task_id": task_id
        }
    
    chunks = chunks_data.get("chunks", [])
    if not chunks:
        error_msg = "No chunks available for ingestion"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "task_id": task_id
        }
    
    logger.info(f"Ingesting {len(chunks)} chunks from: {chunks_data.get('file_path')}")
    
    chroma_manager = None
    try:
        chroma_config = ChromaConfig(
            persist_directory=Path(chromadb_persist_directory),
            collection_name=collection_name,
            embedding_device=embedding_device,
            batch_size=50,
            max_concurrent_batches=2,
            continue_on_error=True
        )
        
        chroma_manager = ChromaDBManager(chroma_config)
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk["content"])
            metadatas.append({
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "chunk_size": chunk["chunk_size"]
            })
            ids.append(chunk["chunk_id"])
        
        ingestion_results = await chroma_manager.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        if ingestion_results.get("success", False):
            logger.info(f"Successfully ingested {len(chunks)} chunks")
            return {
                "status": "success",
                "file_path": chunks_data.get("file_path"),
                "chunks_ingested": len(chunks),
                "collection_name": collection_name,
                "task_id": task_id,
                "processing_completed_at": asyncio.get_event_loop().time()
            }
        else:
            error_msg = f"Ingestion failed: {ingestion_results.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "file_path": chunks_data.get("file_path"),
                "task_id": task_id
            }
            
    except Exception as e:
        error_msg = f"ChromaDB ingestion exception: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "file_path": chunks_data.get("file_path"),
            "task_id": task_id
        }
    finally:
        if chroma_manager:
            try:
                await chroma_manager.finalize()
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")


@task(
    name="Ingest Chunk Batch",
    description="Ingest a batch of chunks into ChromaDB for performance optimization",
    retries=2,
    retry_delay_seconds=20
)
async def ingest_chunk_batch_task(
    chunk_batch: List[Dict[str, Any]],
    chromadb_persist_directory: str,
    collection_name: str = "self_learning_rag",
    embedding_device: str = "cpu",
    batch_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest a batch of chunks into ChromaDB for better performance.
    
    Args:
        chunk_batch: List of chunk objects to ingest
        chromadb_persist_directory: ChromaDB persistence directory
        collection_name: Name of the ChromaDB collection
        embedding_device: Device for embedding computation
        batch_id: Optional batch identifier for tracking
        
    Returns:
        Dict containing batch ingestion results
    """
    logger = get_run_logger()
    
    if not chunk_batch:
        error_msg = "Empty chunk batch provided"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "batch_id": batch_id
        }
    
    logger.info(f"Ingesting batch of {len(chunk_batch)} chunks")
    
    chroma_manager = None
    try:
        chroma_config = ChromaConfig(
            persist_directory=Path(chromadb_persist_directory),
            collection_name=collection_name,
            embedding_device=embedding_device,
            batch_size=100,
            max_concurrent_batches=4,
            continue_on_error=True
        )
        
        chroma_manager = ChromaDBManager(chroma_config)
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunk_batch:
            documents.append(chunk["content"])
            metadatas.append({
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "chunk_size": chunk["chunk_size"]
            })
            ids.append(chunk["chunk_id"])
        
        ingestion_results = await chroma_manager.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        if ingestion_results.get("success", False):
            logger.info(f"Successfully ingested batch of {len(chunk_batch)} chunks")
            return {
                "status": "success",
                "chunks_ingested": len(chunk_batch),
                "collection_name": collection_name,
                "batch_id": batch_id,
                "processing_completed_at": asyncio.get_event_loop().time()
            }
        else:
            error_msg = f"Batch ingestion failed: {ingestion_results.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "batch_id": batch_id
            }
            
    except Exception as e:
        error_msg = f"Batch ingestion exception: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "batch_id": batch_id
        }
    finally:
        if chroma_manager:
            try:
                await chroma_manager.finalize()
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")