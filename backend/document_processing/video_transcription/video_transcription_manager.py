"""
Video Transcription Manager

Main orchestrator for video transcript extraction from playlist sources.
Coordinates components and provides a clean interface for the pipeline.
"""

import asyncio
from pathlib import Path
from typing import List

from .components.transcript_extractor import VideoTranscriptExtractor
from .models import TranscriptResult


class VideoTranscriptionManager:
    """Main manager for video transcription operations."""

    def __init__(self):
        """Initialize the video transcription manager."""
        self.extractor = VideoTranscriptExtractor()
    
    async def extract_all_playlists(self) -> List[TranscriptResult]:
        """Extract transcripts from all playlists in playlist_sources.json.
        
        Returns:
            List[TranscriptResult]: Results for all videos from all playlists
        """
        return await self.extractor.extract_playlists_from_sources()
    
    def get_playlist_sources_path(self) -> Path:
        """Get the path to playlist_sources.json file.
        
        Returns:
            Path: Path to playlist_sources.json
        """
        return Path(__file__).parent / "playlist_sources.json"
    
    def playlist_sources_exists(self) -> bool:
        """Check if playlist_sources.json exists.
        
        Returns:
            bool: True if playlist_sources.json exists
        """
        return self.get_playlist_sources_path().exists()