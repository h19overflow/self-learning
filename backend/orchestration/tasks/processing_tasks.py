"""
Processing tasks for the pipeline orchestration system.

This module contains tasks for PDF processing, VLM enhancement, and semantic chunking.
"""

from typing import Dict, Any
from prefect import task
from prefect.logging import get_run_logger

from backend.document_processing.local_mineru.pipelines.pdf_to_enriched_semantic_chunks_pipeline import PDFToEnrichedMarkdownPipeline
from backend.document_processing.local_mineru.vlm_enhancing.vlm_pipeline import VLMPipeline, PipelineConfig
from backend.document_processing.local_mineru.chunker.semantic_chunker import SemanticChunker
from ..utils.pipeline_config import PipelineConfiguration
from ..utils.file_validation_utils import validate_and_clean_output_directory, clean_old_output_files


@task(
    name="PDF Processing",
    description="Convert PDFs to enriched markdown using MinerU",
    retries=3,
    retry_delay_seconds=30
)
async def process_pdfs_task(config: PipelineConfiguration, validation_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process PDF files to enriched markdown format.
    
    Args:
        config: Pipeline configuration
        validation_results: Optional validation results from source validation task
        
    Returns:
        Dict[str, Any]: Processing results and statistics
    """
    logger = get_run_logger()
    logger.info(f"Starting PDF processing for directory: {config.pdf_input_directory}")
    
    # Check validation results to potentially skip processing
    if validation_results and validation_results.get("files_to_process", 0) == 0:
        logger.info("No new PDF files to process based on validation results")
        return {
            "status": "skipped",
            "processed_files": 0,
            "message": "No new files to process - all already exist in ChromaDB",
            "validation_info": validation_results
        }
    
    # Log validation summary if available
    if validation_results:
        files_to_process = validation_results.get("files_to_process", 0)
        files_skipped = validation_results.get("files_skipped", 0)
        logger.info(f"Processing {files_to_process} new files, skipping {files_skipped} duplicates")
    
    # Validate and potentially clean output directory
    output_validation = None
    cleanup_results = None
    
    if config.auto_clean_old_outputs and validation_results:
        files_to_process_list = validation_results.get("files_to_process_list", [])
        
        if files_to_process_list:
            # Remove .pdf extensions for comparison
            files_without_ext = [f[:-4] if f.endswith('.pdf') else f for f in files_to_process_list]
            
            output_validation = validate_and_clean_output_directory(
                output_dir=config.output_directory,
                files_to_process=files_without_ext,
                stage_name="PDF Processing",
                file_extensions=['.md'],  # PDF processing creates markdown files
                logger=logger
            )
            
            if output_validation["cleanup_recommended"]:
                cleanup_results = clean_old_output_files(
                    output_dir=config.output_directory,
                    files_to_clean=output_validation["old_files_to_clean"],
                    stage_name="PDF Processing",
                    dry_run=config.dry_run_cleanup,
                    logger=logger
                )
    
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
            result = {
                "status": "success",
                "processed_files": 1,  # The pipeline doesn't return detailed stats
                "message": "PDF processing completed successfully"
            }
            
            # Add cleanup information if available
            if output_validation:
                result["output_validation"] = output_validation
            if cleanup_results:
                result["cleanup_results"] = cleanup_results
                
            return result
        else:
            logger.error("PDF processing failed")
            result = {
                "status": "failed", 
                "processed_files": 0,
                "message": "PDF processing failed"
            }
            
            # Add cleanup information even on failure
            if output_validation:
                result["output_validation"] = output_validation
            if cleanup_results:
                result["cleanup_results"] = cleanup_results
                
            return result
        
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
    
    # Validate and potentially clean output directory
    output_validation = None
    cleanup_results = None
    
    if config.auto_clean_old_outputs and pdf_results:
        # Get processed files from PDF results or validation info
        files_to_process_list = []
        
        if pdf_results.get("validation_info") and pdf_results["validation_info"].get("files_to_process_list"):
            # Remove .pdf extensions for comparison
            files_to_process_list = [
                f[:-4] if f.endswith('.pdf') else f 
                for f in pdf_results["validation_info"]["files_to_process_list"]
            ]
        elif config.pdf_input_directory.exists():
            # Fallback: get all PDF files in input directory
            pdf_files = list(config.pdf_input_directory.glob("*.pdf"))
            files_to_process_list = [f.stem for f in pdf_files]
        
        if files_to_process_list:
            output_validation = validate_and_clean_output_directory(
                output_dir=config.output_directory,
                files_to_process=files_to_process_list,
                stage_name="VLM Enhancement",
                file_extensions=['.md'],  # VLM enhancement modifies markdown files
                logger=logger
            )
            
            if output_validation["cleanup_recommended"]:
                cleanup_results = clean_old_output_files(
                    output_dir=config.output_directory,
                    files_to_clean=output_validation["old_files_to_clean"],
                    stage_name="VLM Enhancement",
                    dry_run=config.dry_run_cleanup,
                    logger=logger
                )
    
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
        
        # Add cleanup information if available
        if output_validation:
            results["output_validation"] = output_validation
        if cleanup_results:
            results["cleanup_results"] = cleanup_results
            
        return results
        
    except Exception as e:
        logger.error(f"VLM enhancement failed: {e}")
        if not config.continue_on_errors:
            raise
        
        error_result = {"failed": True, "error": str(e)}
        
        # Add cleanup information even on failure
        if output_validation:
            error_result["output_validation"] = output_validation
        if cleanup_results:
            error_result["cleanup_results"] = cleanup_results
            
        return error_result


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
    
    # Handle chunked output file cleanup
    output_file_cleanup = None
    
    if config.auto_clean_old_outputs:
        # Check if chunked output file already exists
        if config.chunked_output_file.exists():
            logger.info(f"Semantic Chunking: Found existing chunked output file: {config.chunked_output_file.name}")
            
            if not config.dry_run_cleanup:
                # Remove existing file to ensure fresh output
                config.chunked_output_file.unlink()
                logger.info(f"Semantic Chunking: Removed existing chunked output file for fresh processing")
                output_file_cleanup = {
                    "existing_file_removed": True,
                    "file_name": config.chunked_output_file.name,
                    "dry_run": False
                }
            else:
                logger.info(f"Semantic Chunking: [DRY RUN] Would remove existing chunked output file")
                output_file_cleanup = {
                    "existing_file_removed": False,
                    "file_name": config.chunked_output_file.name,
                    "dry_run": True,
                    "would_remove": True
                }
    
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
        
        # Add cleanup information if available
        if output_file_cleanup:
            results["output_file_cleanup"] = output_file_cleanup
            
        return results
        
    except Exception as e:
        logger.error(f"Semantic chunking failed: {e}")
        raise