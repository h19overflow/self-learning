"""
Clean Prefect Flow Definitions for Document Processing Pipeline

This module contains only the flow orchestration logic, keeping the main pipeline
file clean and focused on workflow coordination.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner
from prefect.logging import get_run_logger

# Import configuration and tasks from organized modules
from .utils.pipeline_config import PipelineConfiguration, create_default_config
from .tasks.validation_tasks import validate_source_files_task
from .tasks.processing_tasks import process_pdfs_task, vlm_enhancement_task, semantic_chunking_task
from .tasks.ingestion_tasks import chromadb_rag_ingestion_task
from .tasks.transcription_tasks import video_transcription_task


@flow(
    name="Document Processing Pipeline",
    description="Complete pipeline for processing academic papers from PDF to RAG",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
async def document_processing_pipeline(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Main document processing pipeline flow.
    
    Args:
        config: Complete pipeline configuration
        
    Returns:
        Dict[str, Any]: Comprehensive pipeline results
    """
    logger = get_run_logger()
    pipeline_start_time = datetime.now()
    
    logger.info("Starting Document Processing Pipeline")
    logger.info(f"Input directory: {config.pdf_input_directory}")
    logger.info(f"Output directory: {config.output_directory}")
    logger.info(f"Configuration: VLM={config.enable_vlm_enhancement}, RAG={config.enable_rag_ingestion}")
    
    # Ensure output directories exist
    config.output_directory.mkdir(parents=True, exist_ok=True)
    config.chunked_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Stage 0: Source File Validation
        logger.info("Stage 0: Source File Validation")
        validation_results = await validate_source_files_task(config)
        
        # Check if we should proceed with processing
        if not validation_results.get("proceed_with_processing", True):
            logger.info("No new files to process - all files already exist in ChromaDB")
            pipeline_duration = datetime.now() - pipeline_start_time
            return {
                "pipeline_status": "SKIPPED",
                "pipeline_duration_seconds": pipeline_duration.total_seconds(),
                "reason": "No new files to process",
                "validation_results": validation_results,
                "stages": {
                    "source_validation": validation_results,
                    "pdf_processing": {"skipped": True, "reason": "No new files"},
                    "video_transcription": {"skipped": True, "reason": "No new files"},
                    "vlm_enhancement": {"skipped": True, "reason": "No new files"},
                    "semantic_chunking": {"skipped": True, "reason": "No new files"},
                    "chromadb_rag_ingestion": {"skipped": True, "reason": "No new files"}
                }
            }
        
        # Stage 1: PDF Processing
        logger.info("Stage 1: PDF Processing")
        pdf_results = await process_pdfs_task(config, validation_results)
        
        # Stage 1.5: Video Transcription (optional)
        logger.info("Stage 1.5: Video Transcription")
        video_results = await video_transcription_task(config)
        
        # Stage 2: VLM Enhancement (optional)
        logger.info("Stage 2: VLM Enhancement")
        vlm_results = await vlm_enhancement_task(config, pdf_results)
        
        # Stage 3: Semantic Chunking
        logger.info("Stage 3: Semantic Chunking")
        chunk_results = await semantic_chunking_task(config, vlm_results)
        
        # Stage 4: ChromaDB RAG Ingestion (optional)
        logger.info("Stage 4: ChromaDB RAG Ingestion")
        rag_results = await chromadb_rag_ingestion_task(config, chunk_results)
        
        # Calculate pipeline metrics
        pipeline_duration = datetime.now() - pipeline_start_time
        
        # Compile comprehensive results
        pipeline_results = {
            "pipeline_status": "SUCCESS",
            "pipeline_duration_seconds": pipeline_duration.total_seconds(),
            "stages": {
                "source_validation": validation_results,
                "pdf_processing": pdf_results,
                "video_transcription": video_results,
                "vlm_enhancement": vlm_results,
                "semantic_chunking": chunk_results,
                "chromadb_rag_ingestion": rag_results
            },
            "summary": {
                "total_files_processed": pdf_results.get("processed_files", 0),
                "total_videos_transcribed": video_results.get("successful_extractions", 0),
                "total_images_enhanced": vlm_results.get("successful_descriptions", 0),
                "total_chunks_generated": chunk_results.get("summary", {}).get("total_chunks_generated", 0),
                "total_chunks_ingested": rag_results.get("total_chunks_processed", 0)
            },
            "configuration": {
                "chunk_size": config.chunk_size,
                "overlap": config.overlap,
                "vlm_enabled": config.enable_vlm_enhancement,
                "video_transcription_enabled": config.enable_video_transcription,
                "rag_enabled": config.enable_rag_ingestion
            }
        }
        
        logger.info("Document Processing Pipeline Completed Successfully")
        logger.info(f"Duration: {pipeline_duration}")
        logger.info(f"Files processed: {pipeline_results['summary']['total_files_processed']}")
        logger.info(f"Videos transcribed: {pipeline_results['summary']['total_videos_transcribed']}")
        logger.info(f"Images enhanced: {pipeline_results['summary']['total_images_enhanced']}")
        logger.info(f"Chunks generated: {pipeline_results['summary']['total_chunks_generated']}")
        logger.info(f"Chunks ingested: {pipeline_results['summary']['total_chunks_ingested']}")
        
        return pipeline_results
        
    except Exception as e:
        pipeline_duration = datetime.now() - pipeline_start_time
        logger.error(f"Pipeline failed after {pipeline_duration}: {e}")
        
        error_results = {
            "pipeline_status": "FAILED",
            "pipeline_duration_seconds": pipeline_duration.total_seconds(),
            "error": str(e),
            "configuration": {
                "chunk_size": config.chunk_size,
                "overlap": config.overlap,
                "vlm_enabled": config.enable_vlm_enhancement,
                "rag_enabled": config.enable_rag_ingestion
            }
        }
        
        if config.continue_on_errors:
            return error_results
        else:
            raise


@flow(
    name="VLM Enhancement Pipeline",
    description="Pipeline starting from VLM enhancement stage",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
async def run_from_vlm_stage(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Run pipeline starting from VLM enhancement stage.
    Assumes PDF processing has already been completed.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    logger = get_run_logger()
    pipeline_start_time = datetime.now()
    
    logger.info("Starting Pipeline from VLM Enhancement Stage")
    logger.info(f"Output directory: {config.output_directory}")
    
    try:
        # Mock PDF results for dependency
        pdf_results = {"status": "completed", "processed_files": "existing", "message": "PDF processing already completed"}
        
        # Stage 2: VLM Enhancement
        logger.info("Stage 2: VLM Enhancement")
        vlm_results = await vlm_enhancement_task(config, pdf_results)
        
        # Stage 3: Semantic Chunking
        logger.info("Stage 3: Semantic Chunking")
        chunk_results = await semantic_chunking_task(config, vlm_results)
        
        # Stage 4: ChromaDB RAG Ingestion (optional)
        logger.info("Stage 4: ChromaDB RAG Ingestion")
        rag_results = await chromadb_rag_ingestion_task(config, chunk_results)
        
        # Calculate pipeline metrics
        pipeline_duration = datetime.now() - pipeline_start_time
        
        # Compile results
        pipeline_results = {
            "pipeline_status": "SUCCESS",
            "pipeline_duration_seconds": pipeline_duration.total_seconds(),
            "starting_stage": "VLM Enhancement",
            "stages": {
                "pdf_processing": {"skipped": True, "reason": "Starting from VLM stage"},
                "video_transcription": {"skipped": True, "reason": "Starting from VLM stage"},
                "vlm_enhancement": vlm_results,
                "semantic_chunking": chunk_results,
                "chromadb_rag_ingestion": rag_results
            },
            "summary": {
                "total_images_enhanced": vlm_results.get("successful_descriptions", 0),
                "total_chunks_generated": chunk_results.get("summary", {}).get("total_chunks_generated", 0),
                "total_chunks_ingested": rag_results.get("total_chunks_processed", 0)
            }
        }
        
        logger.info("Pipeline from VLM stage completed successfully")
        return pipeline_results
        
    except Exception as e:
        pipeline_duration = datetime.now() - pipeline_start_time
        logger.error(f"Pipeline failed after {pipeline_duration}: {e}")
        raise


@flow(
    name="Semantic Chunking Pipeline",
    description="Pipeline starting from semantic chunking stage",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
async def run_from_chunking_stage(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Run pipeline starting from semantic chunking stage.
    Assumes PDF processing and VLM enhancement have already been completed.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    logger = get_run_logger()
    pipeline_start_time = datetime.now()
    
    logger.info("Starting Pipeline from Semantic Chunking Stage")
    logger.info(f"Output directory: {config.output_directory}")
    
    try:
        # Mock VLM results for dependency
        vlm_results = {"status": "completed", "successful_descriptions": "existing", "message": "VLM processing already completed"}
        
        # Stage 3: Semantic Chunking
        logger.info("Stage 3: Semantic Chunking")
        chunk_results = await semantic_chunking_task(config, vlm_results)
        
        # Stage 4: ChromaDB RAG Ingestion (optional)
        logger.info("Stage 4: ChromaDB RAG Ingestion")
        rag_results = await chromadb_rag_ingestion_task(config, chunk_results)
        
        # Calculate pipeline metrics
        pipeline_duration = datetime.now() - pipeline_start_time
        
        # Compile results
        pipeline_results = {
            "pipeline_status": "SUCCESS",
            "pipeline_duration_seconds": pipeline_duration.total_seconds(),
            "starting_stage": "Semantic Chunking",
            "stages": {
                "pdf_processing": {"skipped": True, "reason": "Starting from chunking stage"},
                "video_transcription": {"skipped": True, "reason": "Starting from chunking stage"},
                "vlm_enhancement": {"skipped": True, "reason": "Starting from chunking stage"},
                "semantic_chunking": chunk_results,
                "chromadb_rag_ingestion": rag_results
            },
            "summary": {
                "total_chunks_generated": chunk_results.get("summary", {}).get("total_chunks_generated", 0),
                "total_chunks_ingested": rag_results.get("total_chunks_processed", 0)
            }
        }
        
        logger.info("Pipeline from chunking stage completed successfully")
        return pipeline_results
        
    except Exception as e:
        pipeline_duration = datetime.now() - pipeline_start_time
        logger.error(f"Pipeline failed after {pipeline_duration}: {e}")
        raise


@flow(
    name="RAG Ingestion Pipeline",
    description="Pipeline starting from RAG ingestion stage",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
async def run_from_rag_stage(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Run pipeline starting from RAG ingestion stage.
    Assumes all previous stages have been completed and chunks file exists.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    logger = get_run_logger()
    pipeline_start_time = datetime.now()
    
    logger.info("Starting Pipeline from RAG Ingestion Stage")
    logger.info(f"Chunked file: {config.chunked_output_file}")
    
    try:
        # Verify chunks file exists
        if not config.chunked_output_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {config.chunked_output_file}")
        
        # Mock chunk results for dependency
        chunk_results = {"status": "completed", "summary": {"total_chunks_generated": "existing"}}
        
        # Stage 4: ChromaDB RAG Ingestion
        logger.info("Stage 4: ChromaDB RAG Ingestion")
        rag_results = await chromadb_rag_ingestion_task(config, chunk_results)
        
        # Calculate pipeline metrics
        pipeline_duration = datetime.now() - pipeline_start_time
        
        # Compile results
        pipeline_results = {
            "pipeline_status": "SUCCESS",
            "pipeline_duration_seconds": pipeline_duration.total_seconds(),
            "starting_stage": "RAG Ingestion",
            "stages": {
                "pdf_processing": {"skipped": True, "reason": "Starting from RAG stage"},
                "video_transcription": {"skipped": True, "reason": "Starting from RAG stage"},
                "vlm_enhancement": {"skipped": True, "reason": "Starting from RAG stage"},
                "semantic_chunking": {"skipped": True, "reason": "Starting from RAG stage"},
                "chromadb_rag_ingestion": rag_results
            },
            "summary": {
                "total_chunks_ingested": rag_results.get("total_chunks_processed", 0)
            }
        }
        
        logger.info("Pipeline from RAG stage completed successfully")
        return pipeline_results
        
    except Exception as e:
        pipeline_duration = datetime.now() - pipeline_start_time
        logger.error(f"Pipeline failed after {pipeline_duration}: {e}")
        raise


# High-Level Utility Functions for Easy Pipeline Execution

async def run_pipeline_from_stage(
    stage: str,
    pdf_directory: str,
    output_directory: str,
    chunked_output_file: str = None,
    enable_vlm: bool = True,
    enable_rag: bool = True,
    enable_video_transcription: bool = False
) -> Dict[str, Any]:
    """
    Run pipeline starting from a specific stage.
    
    Args:
        stage: Stage to start from ('pdf', 'vlm', 'chunking', 'rag')
        pdf_directory: Directory containing PDF files
        output_directory: Directory for processed output
        chunked_output_file: File path for chunked results (optional)
        enable_vlm: Whether to enable VLM enhancement
        enable_rag: Whether to enable RAG ingestion
        enable_video_transcription: Whether to enable video transcription
        
    Returns:
        Dict[str, Any]: Pipeline results
        
    Raises:
        ValueError: If invalid stage is specified
    """
    if chunked_output_file is None:
        chunked_output_file = str(Path(output_directory).parent / "chunked_output_books" / "semantic_chunks.json")
    
    config = PipelineConfiguration(
        pdf_input_directory=Path(pdf_directory),
        output_directory=Path(output_directory),
        chunked_output_file=Path(chunked_output_file),
        enable_vlm_enhancement=enable_vlm,
        enable_rag_ingestion=enable_rag,
        enable_video_transcription=enable_video_transcription
    )
    
    stage = stage.lower()
    
    if stage in ['pdf', 'full', 'complete']:
        return await document_processing_pipeline(config)
    elif stage in ['vlm', 'vlm_enhancement', 'enhancement']:
        return await run_from_vlm_stage(config)
    elif stage in ['chunking', 'chunk', 'semantic_chunking']:
        return await run_from_chunking_stage(config)
    elif stage in ['rag', 'ingestion', 'chromadb']:
        return await run_from_rag_stage(config)
    else:
        raise ValueError(f"Invalid stage '{stage}'. Valid stages: 'pdf', 'vlm', 'chunking', 'rag'")


async def run_pipeline_simple(
    pdf_directory: str,
    output_directory: str,
    enable_vlm: bool = True,
    enable_rag: bool = True
) -> Dict[str, Any]:
    """
    Simple function to run the complete pipeline with minimal configuration.
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory for processed output
        enable_vlm: Whether to enable VLM enhancement
        enable_rag: Whether to enable RAG ingestion
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    config = PipelineConfiguration(
        pdf_input_directory=Path(pdf_directory),
        output_directory=Path(output_directory),
        chunked_output_file=Path(output_directory).parent / "chunked_output" / "semantic_chunks.json",
        enable_vlm_enhancement=enable_vlm,
        enable_rag_ingestion=enable_rag
    )
    
    return await document_processing_pipeline(config)