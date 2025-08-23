"""
Video Transcription Module

Clean architecture for extracting transcripts from YouTube playlists.
Uses playlist_sources.json for configuration and async task grouping for efficiency.

Architecture:
- VideoTranscriptionManager: Main orchestrator
- components/: Business logic components
- utils/: Utility functions and helpers
- models: Data structures
"""

from .video_transcription_manager import VideoTranscriptionManager
from .models import TranscriptResult
from .components.transcript_extractor import VideoTranscriptExtractor

__all__ = [
    'VideoTranscriptionManager',
    'TranscriptResult', 
    'VideoTranscriptExtractor'
]