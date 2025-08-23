"""
Simple Playlist Transcript Extractor

Extracts transcripts from playlists defined in playlist_sources.json.
Uses async task grouping for efficient I/O bound operations.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import List

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ..models import TranscriptResult
from ..utils.playlist_loader import PlaylistLoader


class VideoTranscriptExtractor:
    """Simple extractor for transcripts from playlist sources."""

    def __init__(self):
        """Initialize the transcript extractor."""
        # Setup simple logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Check required libraries
        if not TRANSCRIPT_API_AVAILABLE:
            raise ImportError("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Install with: pip install requests")
        
        # Initialize YouTube transcript API
        self.youtube_api = YouTubeTranscriptApi()
        
        self.logger.info("Playlist transcript extractor ready")

    async def extract_playlists_from_sources(self) -> List[TranscriptResult]:
        """Extract transcripts from all playlists in playlist_sources.json.
        
        Returns:
            List[TranscriptResult]: Results for all videos from all playlists
        """
        # Get playlist_sources.json from parent video_transcription directory
        sources_file = Path(__file__).parent.parent / "playlist_sources.json"
        
        if not sources_file.exists():
            self.logger.error(f"playlist_sources.json not found at: {sources_file}")
            return []
        
        # Load playlists
        playlist_loader = PlaylistLoader(sources_file)
        playlist_urls = playlist_loader.load_playlist_urls()
        settings = playlist_loader.get_playlist_settings()
        
        if not playlist_urls:
            self.logger.warning("No enabled playlists found in playlist_sources.json")
            return []
        
        # Extract settings
        max_videos_per_playlist = settings.get('max_videos_per_playlist')
        language = settings.get('language', 'en')
        
        self.logger.info(f"Processing {len(playlist_urls)} playlists from playlist_sources.json with async task grouping")
        
        # Get all video URLs from playlists
        all_video_urls = []
        for playlist_url in playlist_urls:
            try:
                video_urls = self._extract_playlist_video_urls(playlist_url)
                if max_videos_per_playlist and len(video_urls) > max_videos_per_playlist:
                    video_urls = video_urls[:max_videos_per_playlist]
                    self.logger.info(f"Limited playlist to {max_videos_per_playlist} videos")
                all_video_urls.extend(video_urls)
            except Exception as e:
                self.logger.error(f"Failed to process playlist {playlist_url}: {e}")
        
        self.logger.info(f"Total videos to process: {len(all_video_urls)}")
        
        # Create async tasks for each video
        tasks = []
        for video_url in all_video_urls:
            task = self._extract_transcript_async(video_url, language)
            tasks.append(task)
        
        # Process all videos concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        final_results = []
        successful = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Video {all_video_urls[i]} failed: {result}")
                final_results.append(TranscriptResult.create_error_result(all_video_urls[i], str(result)))
            else:
                final_results.append(result)
                if result.success:
                    successful += 1
        
        self.logger.info(f"Async processing completed: {successful}/{len(all_video_urls)} successful extractions")
        return final_results

    def _extract_transcript(self, video_url: str, language: str = "en") -> TranscriptResult:
        """Extract transcript from a single YouTube video."""
        try:
            video_id = self._extract_video_id(video_url)
            if not video_id:
                return TranscriptResult.create_error_result(video_url, "Invalid YouTube URL")
            
            transcript_data = self.youtube_api.fetch(video_id, [language, 'en'])
            
            if hasattr(transcript_data, 'snippets'):
                transcript_data = [
                    {
                        'text': snippet.text,
                        'start': snippet.start,
                        'duration': snippet.duration
                    }
                    for snippet in transcript_data.snippets
                ]
            
            full_text = ' '.join([segment['text'] for segment in transcript_data])
            video_title = f"Video {video_id}"
            duration = transcript_data[-1].get('start', 0) + transcript_data[-1].get('duration', 0) if transcript_data else 0.0
            
            return TranscriptResult(
                video_id=video_id,
                video_title=video_title,
                video_url=video_url,
                transcript_text=full_text,
                language=language,
                duration=duration,
                timestamps=transcript_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract transcript from {video_url}: {e}")
            return TranscriptResult.create_error_result(video_url, str(e))

    async def _extract_transcript_async(self, video_url: str, language: str = "en") -> TranscriptResult:
        """Async wrapper for transcript extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_transcript, video_url, language)

    # HELPER FUNCTIONS
    
    def _extract_video_id(self, video_url: str) -> str:
        """Extract video ID from YouTube URL."""
        if len(video_url) == 11 and not video_url.startswith('http'):
            return video_url
        
        patterns = [
            r'youtube\.com/watch\?v=([^&]+)',
            r'youtu\.be/([^?]+)',
            r'youtube\.com/embed/([^?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
        
        return None

    def _extract_playlist_video_urls(self, playlist_url: str) -> List[str]:
        """Extract video URLs from YouTube playlist."""
        playlist_id_match = re.search(r'list=([^&]+)', playlist_url)
        if not playlist_id_match:
            self.logger.error("Could not extract playlist ID")
            return []
        
        playlist_id = playlist_id_match.group(1)
        
        try:
            playlist_page_url = f"https://www.youtube.com/playlist?list={playlist_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(playlist_page_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            video_ids = re.findall(r'"videoId":"([^"]+)"', response.text)
            
            # Remove duplicates while preserving order
            unique_video_ids = []
            seen = set()
            for video_id in video_ids:
                if video_id not in seen:
                    unique_video_ids.append(video_id)
                    seen.add(video_id)
            
            video_urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in unique_video_ids]
            
            self.logger.info(f"Extracted {len(video_urls)} videos from playlist")
            return video_urls
            
        except Exception as e:
            self.logger.error(f"Failed to extract playlist videos: {e}")
            return []