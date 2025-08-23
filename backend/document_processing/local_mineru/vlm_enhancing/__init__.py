"""
VLM Enhancement Pipeline for MinerU Output.

This package provides a complete pipeline for enriching MinerU-generated markdown files
with AI-generated image descriptions using Gemini 2.0 Flash VLM.

Key Features:
- Extract image references from markdown files
- Analyze contextual information around images
- Generate contextual descriptions using Gemini 2.0 Flash
- Enrich markdown files with enhanced image tags and descriptions
- Batch processing with error handling and progress tracking
"""

from .vlm_pipeline import VLMPipeline, PipelineConfig
from .components import ImageExtractor, ContextAnalyzer, GeminiDescriber, MarkdownEnricher
from .models import (
    ImageReference, 
    DocumentContext, 
    ImageContext,
    DescriptionResult, 
    EnrichmentResult, 
    ProcessingStatus
)

__version__ = "1.0.0"

__all__ = [
    # Main pipeline
    "VLMPipeline",
    "PipelineConfig",
    
    # Components
    "ImageExtractor",
    "ContextAnalyzer", 
    "GeminiDescriber",
    "MarkdownEnricher",
    
    # Data models
    "ImageReference",
    "DocumentContext", 
    "ImageContext",
    "DescriptionResult",
    "EnrichmentResult",
    "ProcessingStatus",
]