"""
File-level PDF processing tasks for Kafka-based pipeline orchestration.

Each task processes a single PDF file independently, allowing for better
error isolation and parallel processing via Kafka.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import shutil
import asyncio
from prefect import task
from prefect.logging import get_run_logger

from backend.document_processing.local_mineru.pipelines.pdf_to_enriched_semantic_chunks_pipeline import PDFToEnrichedMarkdownPipeline


@task(
    name="Process Single PDF File",
    description="Convert a single PDF file to enriched markdown",
    retries=2,
    retry_delay_seconds=30
)
async def process_single_pdf_task(
    pdf_file_path: str,
    output_directory: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single PDF file to enriched markdown format.
    
    Args:
        pdf_file_path: Path to the PDF file to process
        output_directory: Directory to save the markdown output
        task_id: Optional task identifier for tracking
        
    Returns:
        Dict containing processing results, output file path, and metadata
    """
    logger = get_run_logger()
    pdf_path = Path(pdf_file_path)
    output_dir = Path(output_directory)
    
    if not pdf_path.exists():
        error_msg = f"PDF file not found: {pdf_file_path}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "file_path": pdf_file_path,
            "task_id": task_id
        }
    
    logger.info(f"Processing PDF file: {pdf_path.name}")
    
    try:
        temp_input_dir = output_dir / "temp_pdf_input"
        temp_input_dir.mkdir(parents=True, exist_ok=True)
        
        temp_pdf_path = temp_input_dir / pdf_path.name
        shutil.copy2(pdf_path, temp_pdf_path)
        
        pdf_pipeline = PDFToEnrichedMarkdownPipeline(
            pdf_directory=temp_input_dir,
            output_directory=output_dir,
            enable_vlm=False,
            enable_chunking=False
        )
        
        success = await pdf_pipeline.process()
        
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        
        if success:
            expected_output_file = output_dir / f"{pdf_path.stem}.md"
            logger.info(f"PDF processing completed: {expected_output_file}")
            return {
                "status": "success",
                "file_path": pdf_file_path,
                "output_file": str(expected_output_file),
                "task_id": task_id,
                "processing_completed_at": asyncio.get_event_loop().time()
            }
        else:
            error_msg = f"PDF processing failed for {pdf_path.name}"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "file_path": pdf_file_path,
                "task_id": task_id
            }
            
    except Exception as e:
        error_msg = f"PDF processing exception: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "file_path": pdf_file_path,
            "task_id": task_id
        }