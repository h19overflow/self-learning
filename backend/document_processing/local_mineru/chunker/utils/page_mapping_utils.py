"""Simplified page mapping utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional


class PageMappingUtils:
    """Simple page mapping for MinerU content."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def load_page_mapping(self, markdown_file_path: Path) -> Dict[int, int]:
        """Load page mapping from MinerU content list JSON."""
        json_file_path = markdown_file_path.parent / f"{markdown_file_path.stem}_content_list.json"
        
        if not json_file_path.exists():
            return {}
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            if not isinstance(content_list, list):
                return {}
            
            return self._build_page_mapping(content_list)
        except Exception:
            return {}
    
    def _build_page_mapping(self, content_list: list) -> Dict[int, int]:
        """Build character position to page mapping."""
        page_mapping = {}
        char_position = 0
        
        for item in content_list:
            if not isinstance(item, dict):
                continue
            
            item_type = item.get('type')
            page_idx = item.get('page_idx', 0)
            
            if item_type == 'text':
                text_content = item.get('text', '')
                text_length = len(text_content)
                
                for i in range(char_position, char_position + text_length):
                    page_mapping[i] = page_idx
                
                char_position += text_length + 2  # Add some buffer
            
            elif item_type in ['image', 'table', 'formula']:
                placeholder_length = {'image': 7, 'table': 7, 'formula': 9}.get(item_type, 10)
                
                for i in range(char_position, char_position + placeholder_length):
                    page_mapping[i] = page_idx
                
                char_position += placeholder_length
        
        return page_mapping
    
    def get_page_for_position(self, page_mapping: Dict[int, int], position: int) -> int:
        """Get page number for character position."""
        if position in page_mapping:
            return page_mapping[position]
        
        # Find closest position
        valid_positions = [pos for pos in page_mapping.keys() if pos <= position]
        if valid_positions:
            closest = max(valid_positions)
            return page_mapping[closest]
        
        return 0
    
    def get_page_range_for_chunk(self, page_mapping: Dict[int, int], start_pos: int, end_pos: int) -> tuple[int, int]:
        """Get page range for chunk."""
        start_page = self.get_page_for_position(page_mapping, start_pos)
        end_page = self.get_page_for_position(page_mapping, end_pos)
        return start_page, end_page
    
    def validate_page_mapping(self, page_mapping: Dict[int, int], text_length: int) -> bool:
        """Basic validation of page mapping."""
        if not page_mapping:
            return False
        
        max_position = max(page_mapping.keys())
        coverage = max_position / text_length if text_length > 0 else 0
        return coverage > 0.3  # At least 30% coverage


# HELPER FUNCTIONS

def create_page_mapper(logger: logging.Logger = None) -> PageMappingUtils:
    """Create page mapping utility."""
    return PageMappingUtils(logger)

def load_page_mapping_for_file(markdown_file_path: Path) -> Dict[int, int]:
    """Load page mapping for single file."""
    mapper = PageMappingUtils()
    return mapper.load_page_mapping(markdown_file_path)