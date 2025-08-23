"""
Context analyzer component.

This module handles extracting contextual information around images in markdown files,
including surrounding paragraphs and section titles to provide context for VLM description generation.
"""

import re
from pathlib import Path
from typing import List, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models import ImageReference, DocumentContext, ImageContext


class ContextAnalyzer:
    """Analyzes markdown content to extract context around image references."""
    
    # Regex patterns for markdown elements
    SECTION_HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    PARAGRAPH_SEPARATOR = re.compile(r'\n\s*\n')
    
    def __init__(self):
        """Initialize the context analyzer."""
        pass
    
    def analyze_image_context(self, image_ref: ImageReference, markdown_path: Path) -> ImageContext:
        """
        Analyze the context around an image reference.
        
        Args:
            image_ref: The image reference to analyze
            markdown_path: Path to the markdown file containing the image
            
        Returns:
            ImageContext object with extracted context information
            
        Raises:
            FileNotFoundError: If the markdown file doesn't exist
        """
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
        
        content = markdown_path.read_text(encoding='utf-8')
        context = self._extract_context(content, image_ref)
        
        return ImageContext(
            image_ref=image_ref,
            context=context,
            markdown_file_path=markdown_path
        )
    
    def _extract_context(self, content: str, image_ref: ImageReference) -> DocumentContext:
        """Extract context information around an image reference."""
        lines = content.splitlines()
        
        # Find the line containing the image
        image_line_index = None
        for i, line in enumerate(lines):
            if image_ref.markdown_tag in line:
                image_line_index = i
                break
        
        if image_line_index is None:
            # Image reference not found, return empty context
            return DocumentContext(None, None, None)
        
        # Extract surrounding paragraphs
        paragraph_before = self._get_paragraph_before(lines, image_line_index)
        paragraph_after = self._get_paragraph_after(lines, image_line_index)
        
        # Extract section title
        section_title = self._get_section_title(lines, image_line_index)
        
        return DocumentContext(
            paragraph_before=paragraph_before,
            paragraph_after=paragraph_after,
            section_title=section_title
        )
    
    def _get_paragraph_before(self, lines: List[str], image_line_index: int) -> Optional[str]:
        """Get the paragraph immediately before the image."""
        if image_line_index == 0:
            return None
        
        # Look backwards for non-empty lines
        paragraph_lines = []
        for i in range(image_line_index - 1, -1, -1):
            line = lines[i].strip()
            
            # Stop at empty line or section header
            if not line or self._is_section_header(line):
                break
            
            # Skip other image references
            if line.startswith('!['):
                continue
                
            paragraph_lines.insert(0, line)
        
        return ' '.join(paragraph_lines) if paragraph_lines else None
    
    def _get_paragraph_after(self, lines: List[str], image_line_index: int) -> Optional[str]:
        """Get the paragraph immediately after the image."""
        if image_line_index >= len(lines) - 1:
            return None
        
        # Look forwards for non-empty lines
        paragraph_lines = []
        for i in range(image_line_index + 1, len(lines)):
            line = lines[i].strip()
            
            # Stop at empty line or section header
            if not line or self._is_section_header(line):
                break
            
            # Skip other image references
            if line.startswith('!['):
                continue
                
            paragraph_lines.append(line)
        
        return ' '.join(paragraph_lines) if paragraph_lines else None
    
    def _get_section_title(self, lines: List[str], image_line_index: int) -> Optional[str]:
        """Get the title of the section containing the image."""
        # Look backwards for the nearest section header
        for i in range(image_line_index, -1, -1):
            line = lines[i].strip()
            if self._is_section_header(line):
                return self._extract_section_title(line)
        
        return None
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a markdown section header."""
        return bool(self.SECTION_HEADER_PATTERN.match(line))
    
    def _extract_section_title(self, header_line: str) -> str:
        """Extract the title from a section header line."""
        match = self.SECTION_HEADER_PATTERN.match(header_line)
        if match:
            return match.group(2).strip()
        return header_line.lstrip('#').strip()
    
    def analyze_multiple_images(self, image_refs: List[ImageReference], markdown_path: Path) -> List[ImageContext]:
        """
        Analyze context for multiple image references in the same file.
        
        Args:
            image_refs: List of image references to analyze
            markdown_path: Path to the markdown file
            
        Returns:
            List of ImageContext objects
        """
        print(f"📝 CONTEXT ANALYZER: Analyzing {len(image_refs)} images in {markdown_path.name}")
        
        contexts = []
        for i, image_ref in enumerate(image_refs, 1):
            try:
                print(f"   📝 {i}/{len(image_refs)}: Analyzing context for {image_ref.filename}")
                context = self.analyze_image_context(image_ref, markdown_path)
                
                # Log extracted context
                doc_context = context.context
                print(f"      📍 Section: {doc_context.section_title or 'None'}")
                print(f"      ⬆️  Before: {doc_context.paragraph_before[:100] + '...' if doc_context.paragraph_before and len(doc_context.paragraph_before) > 100 else doc_context.paragraph_before or 'None'}")
                print(f"      ⬇️  After: {doc_context.paragraph_after[:100] + '...' if doc_context.paragraph_after and len(doc_context.paragraph_after) > 100 else doc_context.paragraph_after or 'None'}")
                
                contexts.append(context)
            except Exception as e:
                print(f"❌ Error analyzing context for {image_ref.filename}: {e}")
                import traceback
                traceback.print_exc()
                # Create a context with no information rather than failing
                empty_context = DocumentContext(None, None, None)
                contexts.append(ImageContext(image_ref, empty_context, markdown_path))
        
        print(f"📝 Context analysis complete: {len(contexts)} contexts generated")
        return contexts