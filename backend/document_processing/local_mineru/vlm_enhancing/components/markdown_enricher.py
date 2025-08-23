"""
Markdown enricher component.

This module handles enriching markdown files by inserting AI-generated image descriptions
while preserving the original document structure and formatting.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models import ImageContext, DescriptionResult, EnrichmentResult, ProcessingStatus


class MarkdownEnricher:
    """Enriches markdown files by adding AI-generated image descriptions."""
    
    def __init__(self, backup_original: bool = True):
        """
        Initialize the markdown enricher.
        
        Args:
            backup_original: Whether to create backup copies of original files
        """
        self.backup_original = backup_original
    
    def enrich_file(
        self, 
        markdown_path: Path, 
        descriptions: List[Tuple[ImageContext, DescriptionResult]]
    ) -> EnrichmentResult:
        """
        Enrich a markdown file with image descriptions.
        
        Args:
            markdown_path: Path to the markdown file to enrich
            descriptions: List of tuples containing ImageContext and DescriptionResult
            
        Returns:
            EnrichmentResult with processing statistics
            
        Raises:
            FileNotFoundError: If the markdown file doesn't exist
        """
        print(f"📝 MARKDOWN ENRICHER: Enriching {markdown_path.name}")
        print(f"🔧 Backup enabled: {self.backup_original}")
        print(f"📊 Processing {len(descriptions)} descriptions")
        
        if not markdown_path.exists():
            error_msg = f"Markdown file not found: {markdown_path}"
            print(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Create backup if requested
        if self.backup_original:
            backup_path = self._create_backup(markdown_path)
            print(f"💾 Created backup: {backup_path}")
        
        # Read the original content
        original_content = markdown_path.read_text(encoding='utf-8')
        print(f"📖 Original content: {len(original_content)} characters")
        
        # Process descriptions and collect statistics
        successful = 0
        failed = 0
        skipped = 0
        
        enriched_content = original_content
        
        # Sort descriptions by line number (descending) to avoid offset issues
        sorted_descriptions = sorted(descriptions, key=lambda x: x[0].image_ref.line_number, reverse=True)
        print(f"🔄 Processing descriptions in reverse line order...")
        
        for i, (image_context, description_result) in enumerate(sorted_descriptions, 1):
            image_name = image_context.image_ref.filename
            print(f"   📝 {i}/{len(sorted_descriptions)}: Processing {image_name}")
            
            # Check if description was successful
            is_successful = hasattr(description_result, 'success') and description_result.success
            if hasattr(description_result, 'is_successful'):
                is_successful = description_result.is_successful()
            
            if is_successful:
                description_text = getattr(description_result, 'description', 'No description available')
                print(f"      ✅ Success: {description_text[:100]}{'...' if len(description_text) > 100 else ''}")
                
                enriched_content = self._insert_description(
                    enriched_content, 
                    image_context, 
                    description_text
                )
                successful += 1
            elif hasattr(description_result, 'status') and description_result.status == ProcessingStatus.FAILED:
                error_msg = getattr(description_result, 'error_message', 'Unknown error')
                print(f"      ❌ Failed: {error_msg}")
                failed += 1
            else:
                print(f"      ⏭️  Skipped")
                skipped += 1
        
        # Check if content was actually modified
        content_changed = enriched_content != original_content
        print(f"📊 Content changed: {content_changed}")
        print(f"📊 New content length: {len(enriched_content)} characters")
        
        # Write the enriched content back to file
        print(f"💾 Writing enriched content to {markdown_path}")
        markdown_path.write_text(enriched_content, encoding='utf-8')
        print(f"✅ File written successfully")
        
        result = EnrichmentResult(
            original_image_count=len(descriptions),
            processed_image_count=len(descriptions),
            successful_descriptions=successful,
            failed_descriptions=failed,
            skipped_descriptions=skipped
        )
        
        print(f"📊 Enrichment result: {successful} successful, {failed} failed, {skipped} skipped")
        return result
    
    def _create_backup(self, markdown_path: Path) -> Path:
        """Create a backup copy of the original markdown file."""
        backup_path = markdown_path.with_suffix(f"{markdown_path.suffix}.backup")
        backup_path.write_text(markdown_path.read_text(encoding='utf-8'), encoding='utf-8')
        return backup_path
    
    def _insert_description(self, content: str, image_context: ImageContext, description: str) -> str:
        """
        Insert a description for an image in the markdown content.
        
        Args:
            content: The markdown content
            image_context: Context information for the image
            description: The generated description
            
        Returns:
            Updated markdown content with the description inserted
        """
        print(f"      🔄 Inserting description for {image_context.image_ref.filename}")
        print(f"      📍 Expected line: {image_context.image_ref.line_number}")
        print(f"      🏷️  Original tag: {image_context.image_ref.markdown_tag}")
        
        lines = content.splitlines()
        print(f"      📄 Content has {len(lines)} lines")
        
        # Find the line containing the image reference
        target_line = image_context.image_ref.line_number - 1  # Convert to 0-based index
        
        if target_line >= len(lines):
            print(f"      ⚠️  Target line {target_line} beyond file length {len(lines)}, using last line")
            target_line = len(lines) - 1
        
        # Find the actual line with the image (in case line numbers shifted)
        actual_line = self._find_image_line(lines, image_context.image_ref.markdown_tag, target_line)
        
        if actual_line is None:
            print(f"      ❌ Could not find image tag in content, returning unchanged")
            return content
        
        print(f"      ✅ Found image at line {actual_line + 1}: {lines[actual_line][:100]}...")
        
        # Create enhanced image markdown
        enhanced_image = self._create_enhanced_image_markdown(
            image_context.image_ref.markdown_tag, 
            description
        )
        print(f"      🎨 Enhanced markdown: {enhanced_image[:150]}...")
        
        # Replace the original image tag with the enhanced version
        original_line = lines[actual_line]
        lines[actual_line] = enhanced_image
        
        print(f"      ✅ Replaced line {actual_line + 1}")
        print(f"         Before: {original_line}")
        print(f"         After:  {enhanced_image}")
        
        return '\n'.join(lines)
    
    def _find_image_line(self, lines: List[str], image_tag: str, start_line: int) -> int:
        """
        Find the line containing the image tag, starting from a given line.
        
        Args:
            lines: List of lines in the content
            image_tag: The markdown image tag to find
            start_line: Starting line index for search
            
        Returns:
            Line index containing the image tag, or None if not found
        """
        # Search in a window around the expected line
        search_window = 10
        start_search = max(0, start_line - search_window)
        end_search = min(len(lines), start_line + search_window + 1)
        
        for i in range(start_search, end_search):
            if image_tag in lines[i]:
                return i
        
        # If not found in window, search the entire file
        for i, line in enumerate(lines):
            if image_tag in line:
                return i
        
        return None
    
    def _create_enhanced_image_markdown(self, original_tag: str, description: str) -> str:
        """
        Create enhanced markdown for an image with description.
        
        Args:
            original_tag: Original image markdown tag
            description: AI-generated description
            
        Returns:
            Enhanced markdown with description
        """
        # Extract the image path from the original tag
        match = re.search(r'\(([^)]+)\)', original_tag)
        image_path = match.group(1) if match else "unknown"
        
        # Create enhanced markdown with description only in alt text
        enhanced = f"![{description}]({image_path})"
        
        return enhanced
    
    def restore_from_backup(self, markdown_path: Path) -> bool:
        """
        Restore a markdown file from its backup.
        
        Args:
            markdown_path: Path to the markdown file to restore
            
        Returns:
            True if successfully restored, False otherwise
        """
        backup_path = markdown_path.with_suffix(f"{markdown_path.suffix}.backup")
        
        if not backup_path.exists():
            return False
        
        try:
            content = backup_path.read_text(encoding='utf-8')
            markdown_path.write_text(content, encoding='utf-8')
            return True
        except Exception:
            return False