"""
File-level VLM enhancement tasks for Kafka-based pipeline orchestration.

Each task enhances a single markdown file with AI-generated image descriptions
independently, allowing for better error isolation and parallel processing.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import asyncio
from prefect import task
from prefect.logging import get_run_logger

from backend.document_processing.local_mineru.vlm_enhancing.vlm_pipeline import VLMPipeline, PipelineConfig


@task(
    name="Enhance Single Markdown File with VLM",
    description="Enhance a single markdown file with AI-generated image descriptions",
    retries=2,
    retry_delay_seconds=45
)
async def enhance_single_markdown_task(
    markdown_file_path: str,
    gemini_model: str = "gemini-2.5-flash-lite",
    backup_original: bool = True,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhance a single markdown file with VLM-generated image descriptions.
    
    Args:
        markdown_file_path: Path to the markdown file to enhance
        gemini_model: Gemini model to use for VLM enhancement
        backup_original: Whether to backup the original file
        task_id: Optional task identifier for tracking
        
    Returns:
        Dict containing enhancement results and metadata
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
    
    logger.info(f"Enhancing markdown file: {md_path.name}")
    
    try:
        vlm_config = PipelineConfig(
            gemini_model=gemini_model,
            backup_original_files=backup_original,
            max_concurrent_requests=1,
            log_level="INFO"
        )
        
        vlm_pipeline = VLMPipeline(vlm_config)
        
        results = await vlm_pipeline.process_single_file(md_path)
        
        if results.get("success", False):
            logger.info(f"VLM enhancement completed: {results.get('descriptions_generated', 0)} descriptions")
            return {
                "status": "success",
                "file_path": markdown_file_path,
                "descriptions_generated": results.get("descriptions_generated", 0),
                "images_processed": results.get("images_processed", 0),
                "task_id": task_id,
                "processing_completed_at": asyncio.get_event_loop().time()
            }
        else:
            error_msg = f"VLM enhancement failed: {results.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "file_path": markdown_file_path,
                "task_id": task_id
            }
            
    except Exception as e:
        error_msg = f"VLM enhancement exception: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "file_path": markdown_file_path,
            "task_id": task_id
        }