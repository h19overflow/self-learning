"""
Professional Document Processing Pipeline with Prefect Orchestration

A production-ready pipeline that orchestrates PDF processing, VLM enhancement, 
semantic chunking, and RAG ingestion using Prefect workflow management.

Dependencies:
- Prefect for workflow orchestration and monitoring
- All existing pipeline components (PDF pipeline, VLM enhancement, chunking, RAG ingestion)

Architecture:
This orchestrator wraps all refactored components with Prefect tasks and flows,
providing monitoring, error handling, retries, and professional workflow management.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from prefect import task, flow
from prefect.task_runners import ConcurrentTaskRunner
from prefect.logging import get_run_logger

# Import our refactored components
from backend.document_processing.local_mineru.pipelines.pdf_to_enriched_semantic_chunks_pipeline import PDFToEnrichedMarkdownPipeline
from backend.document_processing.local_mineru.vlm_enhancing.vlm_pipeline import VLMPipeline, PipelineConfig
from backend.document_processing.local_mineru.chunker.semantic_chunker import SemanticChunker
from backend.storage.chromadb_instance.chromadb_manager import ChromaDBManager
from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig
from backend.document_processing.video_transcription import VideoTranscriptionManager


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
    
    # Logging
    log_level: str = "INFO"


@task(
    name="PDF Processing",
    description="Convert PDFs to enriched markdown using MinerU",
    retries=3,
    retry_delay_seconds=30
)
async def process_pdfs_task(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Process PDF files to enriched markdown format.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dict[str, Any]: Processing results and statistics
    """
    logger = get_run_logger()
    logger.info(f"Starting PDF processing for directory: {config.pdf_input_directory}")
    
    try:
        # Initialize PDF pipeline with proper arguments
        pdf_pipeline = PDFToEnrichedMarkdownPipeline(
            pdf_directory=config.pdf_input_directory,
            output_directory=config.output_directory,
            enable_vlm=False,  # VLM will be handled separately
            enable_chunking=False  # Chunking will be handled separately
        )
        
        # Process PDFs to markdown
        success = await pdf_pipeline.process()
        
        if success:
            logger.info("PDF processing completed successfully")
            return {
                "status": "success",
                "processed_files": 1,  # The pipeline doesn't return detailed stats
                "message": "PDF processing completed successfully"
            }
        else:
            logger.error("PDF processing failed")
            return {
                "status": "failed", 
                "processed_files": 0,
                "message": "PDF processing failed"
            }
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise


@task(
    name="VLM Enhancement",
    description="Enhance markdown files with AI-generated image descriptions",
    retries=2,
    retry_delay_seconds=45
)
async def vlm_enhancement_task(config: PipelineConfiguration, pdf_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance markdown files with VLM-generated image descriptions.
    
    Args:
        config: Pipeline configuration
        pdf_results: Results from PDF processing task
        
    Returns:
        Dict[str, Any]: VLM enhancement results
    """
    logger = get_run_logger()
    
    if not config.enable_vlm_enhancement:
        logger.info("VLM enhancement disabled, skipping")
        return {"skipped": True, "reason": "VLM enhancement disabled"}
    
    logger.info(f"Starting VLM enhancement for directory: {config.output_directory}")
    
    try:
        # Configure VLM pipeline
        vlm_config = PipelineConfig(
            gemini_model=config.gemini_model,
            backup_original_files=config.backup_original_files,
            max_concurrent_requests=config.max_concurrent_vlm_requests,
            log_level=config.log_level
        )
        
        # Initialize and run VLM pipeline
        vlm_pipeline = VLMPipeline(vlm_config)
        results = await vlm_pipeline.process_directory(config.output_directory)
        
        logger.info(f"VLM enhancement completed: {results['successful_descriptions']} descriptions generated")
        return results
        
    except Exception as e:
        logger.error(f"VLM enhancement failed: {e}")
        if not config.continue_on_errors:
            raise
        return {"failed": True, "error": str(e)}


@task(
    name="Semantic Chunking",
    description="Create semantic chunks from processed markdown files",
    retries=2,
    retry_delay_seconds=15
)
async def semantic_chunking_task(config: PipelineConfiguration, vlm_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create semantic chunks from processed markdown files.
    
    Args:
        config: Pipeline configuration
        vlm_results: Results from VLM enhancement task
        
    Returns:
        Dict[str, Any]: Chunking results
    """
    logger = get_run_logger()
    logger.info(f"Starting semantic chunking with chunk_size={config.chunk_size}, overlap={config.overlap}")
    
    try:
        # Initialize semantic chunker
        chunker = SemanticChunker(
            chunk_size=config.chunk_size,
            overlap=config.overlap
        )
        
        # Process all markdown files in output directory
        results = chunker.process_output_directory(
            output_dir=config.output_directory,
            save_to=config.chunked_output_file
        )
        
        chunks_generated = results.get('summary', {}).get('total_chunks_generated', 0)
        logger.info(f"Semantic chunking completed: {chunks_generated} chunks generated")
        return results
        
    except Exception as e:
        logger.error(f"Semantic chunking failed: {e}")
        raise


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


@task(
    name="Video Transcription",
    description="Extract transcripts from YouTube videos",
    retries=2,
    retry_delay_seconds=15
)
async def video_transcription_task(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Extract transcripts from YouTube videos and save to file.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dict[str, Any]: Video transcription results
    """
    logger = get_run_logger()
    
    if not config.enable_video_transcription:
        logger.info("Video transcription disabled, skipping")
        return {"skipped": True, "reason": "Video transcription disabled"}
    
    logger.info("Starting video transcription from playlist sources")
    
    try:
        # Initialize video transcription manager
        manager = VideoTranscriptionManager()
        
        # Extract transcripts from playlist sources
        logger.info("Extracting transcripts from playlist_sources.json")
        results = await manager.extract_all_playlists()
        
        # Save transcripts to file for semantic chunking
        transcripts_file = config.output_directory / "video_transcripts.json"
        transcripts_data = {
            "transcripts": [result.to_dict() for result in results],
            "summary": {
                "total_videos": len(results),
                "successful_extractions": sum(1 for r in results if r.success),
                "failed_extractions": sum(1 for r in results if not r.success)
            }
        }
        
        # Save to JSON file
        import json
        with open(transcripts_file, 'w', encoding='utf-8') as f:
            json.dump(transcripts_data, f, indent=2, ensure_ascii=False)
        
        # Create enhanced markdown files for semantic chunking with video metadata
        # Save directly to output_directory for semantic chunker processing
        transcript_texts_dir = config.output_directory
        transcript_texts_dir.mkdir(exist_ok=True, parents=True)
        
        for result in results:
            if result.success and result.transcript_text:
                # Create rich markdown file with structured metadata for better chunking
                text_file = transcript_texts_dir / f"{result.video_id}.md"
                with open(text_file, 'w', encoding='utf-8') as f:
                    # Header with video metadata
                    f.write(f"# {result.video_title}\n\n")
                    
                    # Metadata section for credibility and traceability
                    f.write("## Video Information\n\n")
                    f.write(f"- **Video ID**: {result.video_id}\n")
                    f.write(f"- **Source URL**: {result.video_url}\n")
                    f.write(f"- **Language**: {result.language}\n")
                    f.write(f"- **Duration**: {result.duration:.1f} seconds ({result.duration/60:.1f} minutes)\n")
                    f.write(f"- **Transcript Source**: {result.source}\n\n")
                    
                    # Enhanced content section
                    f.write("## Video Content\n\n")
                    
                    # Add structured transcript with better formatting for semantic chunking
                    # Split into logical paragraphs for better semantic understanding
                    transcript_paragraphs = result.transcript_text.split('. ')
                    current_paragraph = ""
                    paragraph_length = 0
                    
                    for sentence in transcript_paragraphs:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        # Add sentence to current paragraph
                        if current_paragraph:
                            current_paragraph += ". " + sentence
                        else:
                            current_paragraph = sentence
                        
                        paragraph_length += len(sentence)
                        
                        # Break into new paragraph after ~300-500 characters for better chunking
                        if paragraph_length > 300 or "okay so" in sentence.lower() or "now" in sentence.lower()[:10]:
                            f.write(current_paragraph)
                            if not current_paragraph.endswith('.'):
                                f.write('.')
                            f.write("\n\n")
                            current_paragraph = ""
                            paragraph_length = 0
                    
                    # Write remaining content
                    if current_paragraph:
                        f.write(current_paragraph)
                        if not current_paragraph.endswith('.'):
                            f.write('.')
                        f.write("\n\n")
                    
                    # Footer with source attribution for credibility
                    f.write("---\n\n")
                    f.write(f"*Transcript extracted from: [{result.video_url}]({result.video_url})*\n")
                    f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Video transcription completed: {transcripts_data['summary']['successful_extractions']} successful")
        return {
            "success": True,
            "total_videos": transcripts_data['summary']['total_videos'],
            "successful_extractions": transcripts_data['summary']['successful_extractions'],
            "failed_extractions": transcripts_data['summary']['failed_extractions'],
            "transcripts_file": str(transcripts_file),
            "transcript_texts_directory": str(transcript_texts_dir)
        }
        
    except Exception as e:
        logger.error(f"Video transcription failed: {e}")
        if not config.continue_on_errors:
            raise
        return {"failed": True, "error": str(e)}


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
        # Stage 1: PDF Processing
        logger.info("Stage 1: PDF Processing")
        pdf_results = await process_pdfs_task(config)
        
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


# Convenience Functions

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


# Stage-Specific Pipeline Functions

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


# High-Level Utility Functions

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


def print_stage_help():
    """Print help information about available pipeline stages."""
    help_text = """
Pipeline Stage Options:
======================

1. 'pdf' or 'full' or 'complete':
   - Runs the complete pipeline from the beginning
   - Includes: PDF processing → Video transcription → VLM enhancement → Chunking → RAG ingestion

2. 'vlm' or 'vlm_enhancement' or 'enhancement':
   - Starts from VLM enhancement stage
   - Assumes PDF processing is already completed
   - Includes: VLM enhancement → Chunking → RAG ingestion

3. 'chunking' or 'chunk' or 'semantic_chunking':
   - Starts from semantic chunking stage
   - Assumes PDF processing and VLM enhancement are completed
   - Includes: Chunking → RAG ingestion

4. 'rag' or 'ingestion' or 'chromadb':
   - Starts from RAG ingestion stage only
   - Assumes all previous stages are completed and chunks file exists
   - Includes: RAG ingestion only

Usage Examples:
==============

# Resume from VLM enhancement after disconnection
await run_pipeline_from_stage(
    stage='vlm',
    pdf_directory='local_mineru/books',
    output_directory='output_books'
)

# Skip to chunking stage
await run_pipeline_from_stage(
    stage='chunking',
    pdf_directory='local_mineru/books',
    output_directory='output_books'
)

# Only run RAG ingestion
await run_pipeline_from_stage(
    stage='rag',
    pdf_directory='local_mineru/books',
    output_directory='output_books',
    chunked_output_file='chunked_output_books/semantic_chunks.json'
)
"""
    print(help_text)


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


# Example Usage
async def main():
    """Example of how to use the pipeline."""
    # Get the script directory and build absolute paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up to project root
    
    # Create configuration with absolute paths
    config = PipelineConfiguration(
        pdf_input_directory=script_dir / "local_mineru" / "books",
        output_directory=script_dir / "output_books", 
        chunked_output_file=script_dir / "chunked_output_books" / "semantic_chunks.json",
        enable_vlm_enhancement=True,
        enable_rag_ingestion=True,
        enable_video_transcription=True,
        max_concurrent_vlm_requests=5
    )
    
    # Run the pipeline
    results = await document_processing_pipeline(config)
    
    # Print results
    print(f"Pipeline Status: {results['pipeline_status']}")
    print(f"Duration: {results['pipeline_duration_seconds']:.2f} seconds")
    print(f"Files Processed: {results['summary']['total_files_processed']}")
    print(f"Chunks Generated: {results['summary']['total_chunks_generated']}")


async def resume_from_vlm_example():
    """Example of resuming pipeline from VLM enhancement stage."""
    script_dir = Path(__file__).parent
    
    # Resume from VLM enhancement stage after disconnection
    results = await run_pipeline_from_stage(
        stage='vlm',
        pdf_directory=str(script_dir / "local_mineru" / "books"),
        output_directory=str(script_dir / "output_books"),
        enable_vlm=True,
        enable_rag=True
    )
    
    print(f"Pipeline Status: {results['pipeline_status']}")
    print(f"Starting Stage: {results['starting_stage']}")
    print(f"Images Enhanced: {results['summary'].get('total_images_enhanced', 0)}")
    print(f"Chunks Generated: {results['summary'].get('total_chunks_generated', 0)}")


async def resume_from_chunking_example():
    """Example of resuming pipeline from semantic chunking stage."""
    script_dir = Path(__file__).parent
    
    # Resume from chunking stage
    results = await run_pipeline_from_stage(
        stage='chunking',
        pdf_directory=str(script_dir / "local_mineru" / "books"),
        output_directory=str(script_dir / "output_books"),
        enable_rag=True
    )
    
    print(f"Pipeline Status: {results['pipeline_status']}")
    print(f"Starting Stage: {results['starting_stage']}")
    print(f"Chunks Generated: {results['summary'].get('total_chunks_generated', 0)}")


async def rag_only_example():
    """Example of running only RAG ingestion stage."""
    script_dir = Path(__file__).parent
    
    # Run only RAG ingestion
    results = await run_pipeline_from_stage(
        stage='rag',
        pdf_directory=str(script_dir / "local_mineru" / "books"),
        output_directory=str(script_dir / "output_books"),
        chunked_output_file=str(script_dir / "chunked_output_books" / "semantic_chunks.json")
    )
    
    print(f"Pipeline Status: {results['pipeline_status']}")
    print(f"Starting Stage: {results['starting_stage']}")
    print(f"Chunks Ingested: {results['summary'].get('total_chunks_ingested', 0)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        stage = sys.argv[1].lower()
        if stage == 'help':
            print_stage_help()
        elif stage == 'vlm':
            asyncio.run(resume_from_vlm_example())
        elif stage == 'chunking':
            asyncio.run(resume_from_chunking_example())
        elif stage == 'rag':
            asyncio.run(rag_only_example())
        else:
            asyncio.run(main())
    else:
        asyncio.run(main())