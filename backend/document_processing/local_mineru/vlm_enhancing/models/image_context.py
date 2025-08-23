"""
Data models for image context and processing.

This module defines the core data structures used throughout the VLM enhancement pipeline
for representing images, their context, and processing metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class ImageReference:
    """Represents an image reference found in markdown."""
    
    filename: str
    line_number: int
    markdown_tag: str
    image_path: Path
    
    def exists(self) -> bool:
        """Check if the referenced image file exists."""
        return self.image_path.exists()


@dataclass
class DocumentContext:
    """Context information surrounding an image in a document."""
    
    paragraph_before: Optional[str]
    paragraph_after: Optional[str]
    section_title: Optional[str]
    
    def has_context(self) -> bool:
        """Check if any context information is available."""
        return bool(self.paragraph_before or self.paragraph_after or self.section_title)


@dataclass
class ImageContext:
    """Complete context for an image including reference and surrounding text."""
    
    image_ref: ImageReference
    context: DocumentContext
    markdown_file_path: Path
    
    def create_prompt_context(self) -> str:
        """Create a context string for VLM prompting."""
        context_parts = []
        
        if self.context.section_title:
            context_parts.append(f"Section: {self.context.section_title}")
            
        if self.context.paragraph_before:
            context_parts.append(f"Text before image: {self.context.paragraph_before}")
            
        if self.context.paragraph_after:
            context_parts.append(f"Text after image: {self.context.paragraph_after}")
            
        return "\n\n".join(context_parts) if context_parts else "No context available."