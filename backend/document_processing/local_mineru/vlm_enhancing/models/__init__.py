"""
Data models for the VLM enhancement pipeline.

This package contains all the data structures and models used throughout
the image description and markdown enrichment pipeline.
"""

from .image_context import ImageReference, DocumentContext, ImageContext
from .description_result import DescriptionResult, EnrichmentResult, ProcessingStatus

__all__ = [
    "ImageReference",
    "DocumentContext", 
    "ImageContext",
    "DescriptionResult",
    "EnrichmentResult",
    "ProcessingStatus",
]