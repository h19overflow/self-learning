"""
VLM Enhancement Utilities

Handles VLM image enhancement processing for the PDF pipeline.
This utility creates a temporary VLM script and executes it with proper configuration.
"""

import asyncio
import os
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Any


class VLMEnhancementUtils:
    """Utility class for VLM enhancement processing."""
    
    def __init__(self, output_directory: Path, logger: logging.Logger = None):
        """
        Initialize VLM enhancement utilities.
        
        Args:
            output_directory: Directory containing processed files to enhance
            logger: Optional logger instance for structured logging
        """
        self.output_directory = Path(output_directory)
        self.logger = logger or self._setup_default_logger()
    
    def _setup_default_logger(self) -> logging.Logger:
        """Set up a default logger if none provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _create_vlm_script_content(self) -> str:
        """
        Create the content for the temporary VLM script.
        
        Returns:
            str: Complete VLM script content
        """
        return f'''
import asyncio
import os
import sys
from pathlib import Path

# Add the VLM directories to path - use absolute path resolution
script_dir = Path(__file__).parent
vlm_dir = script_dir / "local_mineru" / "vlm_enhancing"
components_dir = vlm_dir / "components"
models_dir = vlm_dir / "models"

sys.path.insert(0, str(vlm_dir))
sys.path.insert(0, str(components_dir))
sys.path.insert(0, str(models_dir))

# Import components directly without using the main vlm_pipeline file
from image_extractor import ImageExtractor
from context_analyzer import ContextAnalyzer
from gemini_describer import GeminiDescriber
from markdown_enricher import MarkdownEnricher
from description_result import EnrichmentResult, ProcessingStatus

# Import the VLMPipeline class by recreating it here to avoid relative import issues
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class PipelineConfig:
    """Configuration for the VLM enhancement pipeline."""
    
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash-lite"
    backup_original_files: bool = True
    max_concurrent_requests: int = 3
    skip_existing_descriptions: bool = True
    log_level: str = "INFO"

class VLMPipeline:
    """Main pipeline orchestrator for VLM image enhancement."""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.image_extractor = ImageExtractor(base_output_dir=Path())
        self.context_analyzer = ContextAnalyzer()
        self.gemini_describer = GeminiDescriber(
            api_key=config.gemini_api_key,
            model=config.gemini_model
        )
        self.markdown_enricher = MarkdownEnricher(
            backup_original=config.backup_original_files
        )
    
    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def process_directory(self, output_directory):
        """Process all markdown files in the output directory."""
        output_directory = Path(output_directory)
        
        self.logger.info(f"Starting VLM enhancement for: {{output_directory}}")
        
        if not output_directory.exists():
            self.logger.error(f"Output directory does not exist: {{output_directory}}")
            return {{'processed_files': 0, 'total_images': 0, 'successful_descriptions': 0, 'failed_descriptions': 0, 'skipped_descriptions': 0, 'errors': []}}
        
        # Find all markdown files with images
        markdown_files_with_images = []
        for paper_dir in output_directory.iterdir():
            if paper_dir.is_dir():
                auto_dir = paper_dir / "auto"
                if auto_dir.exists():
                    for markdown_file in auto_dir.glob("*.md"):
                        markdown_files_with_images.append(markdown_file)
        
        self.logger.info(f"Found {{len(markdown_files_with_images)}} markdown files to process")
        
        results = {{'processed_files': 0, 'total_images': 0, 'successful_descriptions': 0, 'failed_descriptions': 0, 'skipped_descriptions': 0, 'errors': []}}
        
        for markdown_path in markdown_files_with_images:
            try:
                image_refs = self.image_extractor.extract_from_file(markdown_path)
                
                self.logger.info(f"Processing {{markdown_path}} with {{len(image_refs)}} images")
                
                if image_refs:
                    file_result = await self.process_single_file(markdown_path, image_refs)
                    
                    # Update results
                    results['processed_files'] += 1
                    results['total_images'] += len(image_refs)
                    results['successful_descriptions'] += file_result.successful_descriptions
                    results['failed_descriptions'] += file_result.failed_descriptions
                    results['skipped_descriptions'] += file_result.skipped_descriptions
                    
                    self.logger.info(f"Completed {{markdown_path.name}}: {{file_result.summary()}}")
                
            except Exception as e:
                error_msg = f"Error processing {{markdown_path}}: {{e}}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        # Print summary
        total_processed = results['successful_descriptions'] + results['failed_descriptions']
        success_rate = (results['successful_descriptions'] / total_processed) if total_processed > 0 else 0
        
        self.logger.info(
            f"VLM enhancement complete: {{results['processed_files']}} files, "
            f"{{results['successful_descriptions']}}/{{results['total_images']}} successful descriptions, "
            f"{{success_rate:.1%}} success rate"
        )
        
        return results
    
    async def process_single_file(self, markdown_path, image_refs):
        """Process a single markdown file with its image references."""
        markdown_path = Path(markdown_path)
        
        if not image_refs:
            return EnrichmentResult(markdown_path, [], 0, 0, 0)
        
        # Validate image references
        valid_refs, invalid_refs = self.image_extractor.validate_references(image_refs)
        
        if not valid_refs:
            self.logger.info(f"No valid images found in {{markdown_path}}")
            return EnrichmentResult(markdown_path, [], 0, 0, len(image_refs))
        
        # Generate context for each valid image
        image_contexts = []
        for ref in valid_refs:
            context = self.context_analyzer.analyze_image_context(ref, markdown_path)
            if context:
                image_contexts.append(context)
        
        if not image_contexts:
            self.logger.info(f"All images in {{markdown_path}} already have descriptions, skipping")
            return EnrichmentResult(markdown_path, [], 0, 0, len(valid_refs))
        
        # Generate descriptions using Gemini
        description_results = await self.gemini_describer.describe_multiple_images(image_contexts)
        
        # Count successful and failed descriptions
        successful_descriptions = []
        failed_descriptions = []
        
        for context, result in description_results:
            if hasattr(result, 'success') and result.success and hasattr(result, 'description') and result.description:
                successful_descriptions.append((context, result))
            else:
                failed_descriptions.append((context, result))
        
        # Enrich the markdown file with descriptions
        enrichment_result = self.markdown_enricher.enrich_file(markdown_path, successful_descriptions)
        
        return EnrichmentResult(
            original_image_count=len(image_refs),
            processed_image_count=len(image_contexts),
            successful_descriptions=len(successful_descriptions),
            failed_descriptions=len(failed_descriptions),
            skipped_descriptions=0
        )

async def main():
    config = PipelineConfig(
        gemini_model="gemini-2.5-flash-lite",
        backup_original_files=True,
        max_concurrent_requests=3,
        skip_existing_descriptions=True,
        log_level="INFO"
    )
    
    pipeline = VLMPipeline(config)
    output_dir = Path("{str(self.output_directory).replace(chr(92), chr(92)+chr(92))}")
    
    if not output_dir.exists():
        return False
    
    results = await pipeline.process_directory(output_dir)
    return True

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    async def run_vlm_enhancement(self) -> bool:
        """
        Run VLM enhancement on the pipeline output.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Setting up VLM enhancement for directory: {self.output_directory}")
            
            # Create temporary script content
            script_content = self._create_vlm_script_content()
            
            # Write temporary script
            temp_script_path = Path(__file__).parent.parent / "temp_vlm_runner.py"
            with open(temp_script_path, 'w') as f:
                f.write(script_content)
            
            try:
                # Run the temporary script
                self.logger.info(f"Running VLM enhancement script: {temp_script_path}")
                env = os.environ.copy()
                result = subprocess.run([
                    sys.executable, str(temp_script_path)
                ], cwd=temp_script_path.parent, env=env, capture_output=True, text=True)
                
                self.logger.info(f"VLM return code: {result.returncode}")
                
                if result.stdout:
                    self.logger.debug(f"VLM STDOUT: {result.stdout}")
                if result.stderr:
                    self.logger.warning(f"VLM STDERR: {result.stderr}")
                
                if result.returncode == 0:
                    self.logger.info("VLM enhancement completed successfully")
                    return True
                else:
                    self.logger.error(f"VLM enhancement failed with return code: {result.returncode}")
                    return False
                    
            finally:
                # Clean up temporary script
                try:
                    temp_script_path.unlink()
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp script: {cleanup_error}")
            
        except Exception as e:
            self.logger.error(f"VLM enhancement failed with exception: {e}")
            return False

# HELPER FUNCTIONS

def validate_vlm_dependencies() -> bool:
    """
    Validate that VLM dependencies are available.
    
    Returns:
        bool: True if dependencies are available, False otherwise
    """
    try:
        vlm_dir = Path(__file__).parent.parent / "local_mineru" / "vlm_enhancing"
        required_components = ["image_extractor.py", "context_analyzer.py", "gemini_describer.py", "markdown_enricher.py"]
        
        components_dir = vlm_dir / "components"
        for component in required_components:
            if not (components_dir / component).exists():
                return False
        
        return True
    except Exception:
        return False