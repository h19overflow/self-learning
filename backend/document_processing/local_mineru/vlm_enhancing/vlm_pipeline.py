"""
VLM Enhancement Pipeline Orchestrator

A clean orchestrator that coordinates image extraction, context analysis, description generation,
and markdown enrichment for MinerU-generated files.

Dependencies:
- ImageExtractor for finding image references
- ContextAnalyzer for analyzing image context
- GeminiDescriber for generating AI descriptions
- MarkdownEnricher for updating markdown files

Architecture:
This orchestrator coordinates the VLM pipeline but doesn't contain
business logic. All heavy processing is delegated to specialized component classes.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .components import ImageExtractor, ContextAnalyzer, GeminiDescriber, MarkdownEnricher
from .models import EnrichmentResult, ProcessingStatus


@dataclass
class PipelineConfig:
    """Configuration for the VLM enhancement pipeline."""
    
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash-exp"
    backup_original_files: bool = True
    max_concurrent_requests: int = 5
    skip_existing_descriptions: bool = True
    log_level: str = "INFO"


class VLMPipeline:
    """Clean VLM pipeline orchestrator with structured logging."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the VLM pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging for the VLM pipeline."""
        logger = logging.getLogger("vlm_pipeline")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all VLM pipeline components."""
        self.image_extractor = ImageExtractor(base_output_dir=Path())
        self.context_analyzer = ContextAnalyzer()
        self.gemini_describer = GeminiDescriber(
            api_key=self.config.gemini_api_key,
            model=self.config.gemini_model
        )
        self.markdown_enricher = MarkdownEnricher(
            backup_original=self.config.backup_original_files
        )
        
        self.logger.info("VLM pipeline components initialized successfully")
    
    async def process_directory(self, output_directory: Path) -> Dict[str, Any]:
        """
        Process all markdown files in a directory structure.
        
        Args:
            output_directory: Path to the MinerU output directory
            
        Returns:
            Dict[str, Any]: Processing results and statistics
            
        Raises:
            ValueError: If output directory doesn't exist
        """
        self.logger.info(f"Starting VLM enhancement for directory: {output_directory}")
        
        if not output_directory.exists():
            error_msg = f"Output directory does not exist: {output_directory}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize results tracking
        results = self._initialize_results()
        
        # Process all markdown files with images
        await self._process_all_files(output_directory, results)
        
        # Log and return final summary
        self._log_final_summary(results)
        return results
    
    def _initialize_results(self) -> Dict[str, Any]:
        """Initialize the results tracking dictionary."""
        return {
            "processed_files": 0,
            "total_images": 0,
            "successful_descriptions": 0,
            "failed_descriptions": 0,
            "skipped_descriptions": 0,
            "file_results": {},
            "errors": []
        }
    
    async def _process_all_files(self, output_directory: Path, results: Dict[str, Any]):
        """
        Process all markdown files found in the directory.
        
        Args:
            output_directory: Directory to scan for files
            results: Results dictionary to update
        """
        files_found = False
        
        # Process each markdown file with images
        for markdown_path, image_refs in self.image_extractor.extract_from_directory(output_directory):
            files_found = True
            
            try:
                self.logger.info(f"Processing {markdown_path.name} with {len(image_refs)} image references")
                
                file_result = await self._process_single_file(markdown_path, image_refs)
                
                # Update overall statistics
                self._update_results(results, file_result, markdown_path)
                
                self.logger.info(f"Completed {markdown_path.name}: {file_result.summary()}")
                
            except Exception as e:
                error_msg = f"Failed to process {markdown_path.name}: {e}"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)
        
        if not files_found:
            self._handle_no_files_found(output_directory)
    
    def _update_results(self, results: Dict[str, Any], file_result: EnrichmentResult, markdown_path: Path):
        """Update the overall results with a single file's results."""
        results["processed_files"] += 1
        results["total_images"] += file_result.original_image_count
        results["successful_descriptions"] += file_result.successful_descriptions
        results["failed_descriptions"] += file_result.failed_descriptions
        results["skipped_descriptions"] += file_result.skipped_descriptions
        results["file_results"][str(markdown_path)] = file_result
    
    def _handle_no_files_found(self, output_directory: Path):
        """Handle the case where no markdown files with images are found."""
        self.logger.warning(f"No markdown files with images found in {output_directory}")
        
        # Check for any markdown files at all
        all_md_files = list(output_directory.rglob("*.md"))
        if all_md_files:
            self.logger.info(f"Found {len(all_md_files)} markdown files without processable images")
            # Log first few for debugging
            for md_file in all_md_files[:3]:
                self.logger.debug(f"Found markdown file: {md_file}")
        else:
            self.logger.warning("No markdown files found at all in the directory")
    
    async def _process_single_file(self, markdown_path: Path, image_refs: List) -> EnrichmentResult:
        """
        Process a single markdown file with clean error handling.
        
        Args:
            markdown_path: Path to the markdown file
            image_refs: List of image references found in the file
            
        Returns:
            EnrichmentResult: Processing statistics for this file
        """
        self.logger.debug(f"Processing {len(image_refs)} image references in {markdown_path.name}")
        
        # Validate image references
        valid_refs, invalid_refs = self.image_extractor.validate_references(image_refs)
        
        if invalid_refs:
            self.logger.warning(f"Found {len(invalid_refs)} invalid image references in {markdown_path.name}")
        
        if not valid_refs:
            self.logger.info(f"No valid images found in {markdown_path.name}")
            return EnrichmentResult(
                original_image_count=len(image_refs),
                processed_image_count=0,
                successful_descriptions=0,
                failed_descriptions=0,
                skipped_descriptions=len(image_refs)
            )
        
        # Analyze image contexts
        image_contexts = self.context_analyzer.analyze_multiple_images(valid_refs, markdown_path)
        self.logger.debug(f"Generated {len(image_contexts)} image contexts")
        
        # Filter out existing descriptions if configured
        if self.config.skip_existing_descriptions:
            image_contexts = self._filter_existing_descriptions(image_contexts)
            self.logger.debug(f"After filtering existing descriptions: {len(image_contexts)} contexts remain")
        
        if not image_contexts:
            self.logger.info(f"All images in {markdown_path.name} already have descriptions, skipping")
            return EnrichmentResult(
                original_image_count=len(image_refs),
                processed_image_count=len(image_refs),
                successful_descriptions=0,
                failed_descriptions=0,
                skipped_descriptions=len(image_refs)
            )
        
        # Generate descriptions using Gemini
        self.logger.info(f"Generating descriptions for {len(image_contexts)} images using {self.config.gemini_model}")
        description_results = await self.gemini_describer.describe_multiple_images(image_contexts)
        
        # Count successful and failed descriptions
        successful_count, failed_count = self._count_description_results(description_results)
        self.logger.info(f"Description generation complete: {successful_count} successful, {failed_count} failed")
        
        # Enrich the markdown file
        enrichment_result = self.markdown_enricher.enrich_file(markdown_path, description_results)
        self.logger.info(f"Markdown enrichment complete for {markdown_path.name}")
        
        return enrichment_result
    
    def _count_description_results(self, description_results: List) -> tuple:
        """
        Count successful and failed description results.
        
        Args:
            description_results: List of (context, result) tuples
            
        Returns:
            tuple: (successful_count, failed_count)
        """
        successful_count = 0
        failed_count = 0
        
        for context, result in description_results:
            if hasattr(result, 'success') and result.success:
                successful_count += 1
                self.logger.debug(f"Successfully described: {context.image_ref.filename}")
            else:
                failed_count += 1
                error_msg = getattr(result, 'error_message', 'Unknown error')
                self.logger.debug(f"Failed to describe {context.image_ref.filename}: {error_msg}")
        
        return successful_count, failed_count
    
    def _filter_existing_descriptions(self, image_contexts: List) -> List:
        """
        Filter out images that already have AI-generated descriptions.
        
        Args:
            image_contexts: List of ImageContext objects
            
        Returns:
            List: Filtered list of ImageContext objects
        """
        # For now, return all contexts. In future versions, we could check
        # if the markdown already contains AI-generated description markers
        return image_contexts
    
    def _log_final_summary(self, results: Dict[str, Any]):
        """Log a comprehensive summary of the processing results."""
        total_images = results["total_images"]
        successful = results["successful_descriptions"]
        failed = results["failed_descriptions"]
        skipped = results["skipped_descriptions"]
        success_rate = (successful / total_images) if total_images > 0 else 0.0
        
        self.logger.info("VLM Enhancement Pipeline Complete")
        self.logger.info(f"Files processed: {results['processed_files']}")
        self.logger.info(f"Total images: {total_images}")
        self.logger.info(f"Successful descriptions: {successful}")
        self.logger.info(f"Failed descriptions: {failed}")
        self.logger.info(f"Skipped descriptions: {skipped}")
        self.logger.info(f"Success rate: {success_rate:.1%}")
        
        if results["errors"]:
            self.logger.warning(f"Encountered {len(results['errors'])} errors during processing")
    
    async def process_single_document(self, markdown_path: Path) -> EnrichmentResult:
        """
        Process a single markdown document.
        
        Args:
            markdown_path: Path to the markdown file to process
            
        Returns:
            EnrichmentResult: Processing statistics
            
        Raises:
            FileNotFoundError: If markdown file doesn't exist
        """
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
        
        self.logger.info(f"Processing single document: {markdown_path}")
        
        # Extract image references
        image_refs = self.image_extractor.extract_from_file(markdown_path)
        
        if not image_refs:
            self.logger.info(f"No images found in {markdown_path}")
            return EnrichmentResult(0, 0, 0, 0, 0)
        
        # Process the file
        return await self._process_single_file(markdown_path, image_refs)


# Convenience Functions

def create_vlm_pipeline(
    gemini_model: str = "gemini-2.0-flash-exp",
    backup_files: bool = True,
    log_level: str = "INFO"
) -> VLMPipeline:
    """
    Factory function to create a VLM pipeline with common settings.
    
    Args:
        gemini_model: Gemini model to use for descriptions
        backup_files: Whether to backup original files
        log_level: Logging level
        
    Returns:
        VLMPipeline: Configured pipeline
    """
    config = PipelineConfig(
        gemini_model=gemini_model,
        backup_original_files=backup_files,
        log_level=log_level
    )
    return VLMPipeline(config)

async def enhance_directory_simple(
    output_directory: Path,
    gemini_model: str = "gemini-2.0-flash-exp"
) -> Dict[str, Any]:
    """
    Simple function to enhance all markdown files in a directory.
    
    Args:
        output_directory: Directory containing markdown files
        gemini_model: Gemini model to use
        
    Returns:
        Dict[str, Any]: Processing results
    """
    pipeline = create_vlm_pipeline(gemini_model=gemini_model)
    return await pipeline.process_directory(output_directory)

async def enhance_single_file(
    markdown_file: Path,
    gemini_model: str = "gemini-2.0-flash-exp"
) -> EnrichmentResult:
    """
    Simple function to enhance a single markdown file.
    
    Args:
        markdown_file: Path to markdown file
        gemini_model: Gemini model to use
        
    Returns:
        EnrichmentResult: Processing results
    """
    pipeline = create_vlm_pipeline(gemini_model=gemini_model)
    return await pipeline.process_single_document(markdown_file)


async def main():
    """Example usage of the VLM pipeline."""
    # Configure the pipeline
    config = PipelineConfig(
        gemini_model="gemini-2.0-flash-exp",
        backup_original_files=True,
        log_level="INFO"
    )
    
    # Initialize pipeline
    pipeline = VLMPipeline(config)
    
    # Process the output directory
    output_dir = Path("C:/Users/User/Projects/paper_verifier/backend/ingestion_pipeline/mineru/file_ingesting/output")
    results = await pipeline.process_directory(output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing complete! Processed {results['processed_files']} files")


if __name__ == "__main__":
    asyncio.run(main())