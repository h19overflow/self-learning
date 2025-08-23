"""
Document Processing Pipeline Orchestration

This package provides a clean, modular orchestration system for processing 
documents through PDF extraction, VLM enhancement, semantic chunking, 
and RAG ingestion stages.

Main Components:
- prefect_flows: Clean flow definitions 
- clean_prefect_orchestrator: Main entry point with examples
- tasks/: Individual task definitions organized by functionality
- utils/: Helper functions and configuration management

Usage:
    from backend.orchestration import run_pipeline_simple
    
    results = await run_pipeline_simple(
        pdf_directory="input_pdfs",
        output_directory="output_markdown"
    )
"""

# Import main flow functions for easy access
from .prefect_flows import (
    document_processing_pipeline,
    run_pipeline_from_stage,
    run_pipeline_simple
)

# Import configuration utilities
from .utils import PipelineConfiguration, create_default_config

__all__ = [
    'document_processing_pipeline',
    'run_pipeline_from_stage', 
    'run_pipeline_simple',
    'PipelineConfiguration',
    'create_default_config'
]