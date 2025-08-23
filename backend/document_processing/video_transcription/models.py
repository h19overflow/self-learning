"""
Simple data models for video transcript extraction.

This module contains basic data structures for storing transcript information.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TranscriptResult:
    """Simple container for video transcript data."""
    
    # Basic video information
    video_id: str
    video_title: str
    video_url: str
    
    # Transcript content
    transcript_text: str
    
    # Metadata
    language: str = "en"
    duration: Optional[float] = None  # Duration in seconds
    
    # Timestamps (list of dicts with 'start', 'duration', 'text')
    timestamps: Optional[List[Dict[str, Any]]] = None
    
    # Processing info
    source: str = "youtube-transcript-api"  # Where we got the transcript from
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "video_url": self.video_url,
            "transcript_text": self.transcript_text,
            "language": self.language,
            "duration": self.duration,
            "timestamps": self.timestamps,
            "source": self.source,
            "success": self.success,
            "error_message": self.error_message
        }
    
    @classmethod
    def create_error_result(cls, video_url: str, error_message: str) -> 'TranscriptResult':
        """Create a result object for failed transcript extraction."""
        return cls(
            video_id="unknown",
            video_title="unknown", 
            video_url=video_url,
            transcript_text="",
            success=False,
            error_message=error_message
        )