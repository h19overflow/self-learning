"""
Simple Playlist Source Loader

Utility for loading playlist URLs from playlist_sources.json file.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any


class PlaylistLoader:
    """Simple loader for playlist sources from configuration file."""

    def __init__(self, config_file: Path):
        """Initialize playlist loader.
        
        Args:
            config_file: Path to playlist_sources.json file
        """
        self.config_file = config_file
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def load_playlist_urls(self) -> List[str]:
        """Load playlist URLs from playlist_sources.json.
        
        Returns:
            List of enabled playlist URLs
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            playlists = config.get('playlists', [])
            
            # Only return enabled playlists
            enabled_playlists = [p for p in playlists if p.get('enabled', True)]
            urls = [playlist['url'] for playlist in enabled_playlists if 'url' in playlist]
            
            self.logger.info(f"Loaded {len(urls)} enabled playlist URLs")
            return urls
            
        except Exception as e:
            self.logger.error(f"Failed to load playlist URLs: {e}")
            return []
    
    def get_playlist_settings(self) -> Dict[str, Any]:
        """Get playlist processing settings from playlist_sources.json.
        
        Returns:
            Dictionary of settings
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return config.get('settings', {})
            
        except Exception as e:
            self.logger.error(f"Failed to load playlist settings: {e}")
            return {}