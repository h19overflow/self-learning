"""
Clean Document Processing Pipeline Orchestrator

This is the main entry point for the document processing pipeline system.
All task definitions and utility functions have been extracted to separate modules
for better organization and maintainability.

Usage Examples:
    # Run complete pipeline
    results = await main()
    
    # Run from specific stage
    results = await resume_from_vlm_example()
    
    # Simple pipeline execution
    results = await run_pipeline_simple(
        pdf_directory="input_pdfs",
        output_directory="output_markdown"
    )
"""

import asyncio
from pathlib import Path

# Import clean flow definitions and utilities
from .prefect_flows import (
    document_processing_pipeline,
    run_from_vlm_stage, 
    run_from_chunking_stage,
    run_from_rag_stage,
    run_pipeline_from_stage,
    run_pipeline_simple
)
from .utils.pipeline_config import PipelineConfiguration, create_default_config


def print_stage_help():
    """Print help information about available pipeline stages."""
  

# Example Usage Functions

async def main():
    """Example of how to use the complete pipeline."""
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