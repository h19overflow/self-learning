"""
Data models for VLM description results and processing outcomes.

This module defines data structures for storing and managing the results
of image description generation and markdown enrichment operations.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ProcessingStatus(Enum):
    """Status of image processing operations."""
    
    PENDING = "pending"
    SUCCESS = "success" 
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DescriptionResult:
    """Result of generating a description for an image."""
    
    description: Optional[str]
    status: ProcessingStatus
    error_message: Optional[str] = None
    
    @classmethod
    def success(cls, description: str) -> "DescriptionResult":
        """Create a successful result."""
        return cls(description=description, status=ProcessingStatus.SUCCESS)
    
    @classmethod
    def failed(cls, error_message: str) -> "DescriptionResult":
        """Create a failed result."""
        return cls(description=None, status=ProcessingStatus.FAILED, error_message=error_message)
    
    @classmethod
    def skipped(cls, reason: str) -> "DescriptionResult":
        """Create a skipped result."""
        return cls(description=None, status=ProcessingStatus.SKIPPED, error_message=reason)
    
    def is_successful(self) -> bool:
        """Check if the description generation was successful."""
        return self.status == ProcessingStatus.SUCCESS and self.description is not None


@dataclass
class EnrichmentResult:
    """Result of enriching a markdown file with image descriptions."""
    
    original_image_count: int
    processed_image_count: int
    successful_descriptions: int
    failed_descriptions: int
    skipped_descriptions: int
    
    def success_rate(self) -> float:
        """Calculate the success rate of description generation."""
        if self.processed_image_count == 0:
            return 0.0
        return self.successful_descriptions / self.processed_image_count
    
    def summary(self) -> str:
        """Generate a summary string of the enrichment results."""
        return (
            f"Processed {self.processed_image_count}/{self.original_image_count} images. "
            f"Success: {self.successful_descriptions}, "
            f"Failed: {self.failed_descriptions}, "
            f"Skipped: {self.skipped_descriptions} "
            f"({self.success_rate():.1%} success rate)"
        )