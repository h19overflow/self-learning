"""
PDF to Enriched Markdown Pipeline

Complete orchestrator that chains MinerU PDF processing, VLM image enhancement,
and semantic chunking. This is the complete preprocessing pipeline.

Dependencies:
- MinerU for PDF processing
- VLM enhancement utilities for image description
- Semantic chunker for content chunking
- File management utilities for safe file operations

Architecture:
This orchestrator coordinates the pipeline components but doesn't contain
business logic. All heavy processing is delegated to specialized utility classes.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List

# Add the parent directories to the Python path for imports
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from utils.file_management_utils import FileManagementUtils, find_pdf_files
from utils.mineru_processing_utils import MinerUProcessingUtils, cleanup_intermediate_files
from utils.vlm_enhancement_utils import VLMEnhancementUtils


class PDFToEnrichedMarkdownPipeline:
    """Complete pipeline that chains MinerU + VLM enhancement + semantic chunking."""
    
    def __init__(self, pdf_directory: Path, output_directory: Path, enable_vlm: bool = True, enable_chunking: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            pdf_directory: Directory containing PDF files to process
            output_directory: Directory for final output
            enable_vlm: Whether to run VLM image enhancement
            enable_chunking: Whether to run semantic chunking
        """
        self.pdf_directory = Path(pdf_directory)
        self.output_directory = Path(output_directory)
        self.enable_vlm = enable_vlm
        self.enable_chunking = enable_chunking
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Ensure directories exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize utility classes
        self._initialize_utilities()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging for the pipeline."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_utilities(self):
        """Initialize all utility classes with proper configuration."""
        # File management utilities
        filename_mapping_file = Path(__file__).parent.parent.parent / "filename_mapping.json"
        self.file_manager = FileManagementUtils(filename_mapping_file, self.logger)
        
        # MinerU processing utilities
        mineru_script = Path(__file__).parent.parent / "file_ingesting" / "mineru_ingestor.py"
        self.mineru_processor = MinerUProcessingUtils(mineru_script, self.logger)
        
        # VLM enhancement utilities
        self.vlm_enhancer = VLMEnhancementUtils(self.output_directory, self.logger)
        
        # Chunked output directory
        self.chunked_output_dir = Path(__file__).parent.parent.parent / "chunked_output"
    
    def find_pdf_files(self) -> List[Path]:
        """Find all PDF files in the input directory."""
        return find_pdf_files(self.pdf_directory)
    
    def run_mineru_processing(self, pdf_files: List[Path]) -> bool:
        """
        Run MinerU processing on PDF files.
        
        Args:
            pdf_files: List of PDF files to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directories exist
            research_papers_dir = Path(__file__).parent.parent / "file_ingesting" / "research_papers"
            research_papers_dir.mkdir(parents=True, exist_ok=True)
            
            if not pdf_files:
                self.logger.warning("No PDF files found to process")
                return True
            
            # Copy PDFs to the expected location with safe filenames
            self.file_manager.copy_files_with_safe_names(pdf_files, research_papers_dir)
            
            # Run MinerU processing
            if not self.mineru_processor.run_mineru_processing():
                return False
            
            # Validate MinerU output
            mineru_output_dir = Path(__file__).parent.parent / "file_ingesting" / "output"
            if not self.mineru_processor.validate_mineru_output(mineru_output_dir):
                return False
            
            # Copy results to final output directory with restored filenames
            self.file_manager.copy_output_with_restored_names(mineru_output_dir, self.output_directory)
            
            return True
            
        except Exception as e:
            self.logger.error(f"MinerU processing failed: {e}")
            return False
    
    async def run_vlm_enhancement(self) -> bool:
        """
        Run VLM enhancement on the pipeline output.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enable_vlm:
            self.logger.info("VLM enhancement disabled, skipping")
            return True
        
        return await self.vlm_enhancer.run_vlm_enhancement()
    
    def run_semantic_chunking(self) -> bool:
        """
        Run semantic chunking on the enhanced markdown files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enable_chunking:
            self.logger.info("Semantic chunking disabled, skipping")
            return True
        
        try:
            # Import the chunker here to avoid import issues if not needed
            from ..chunker.semantic_chunker import SemanticChunker
            
            # Initialize chunker with reasonable settings for academic papers
            chunker = SemanticChunker(
                chunk_size=1024,  # Larger chunks for academic papers
                overlap=128       # Good overlap for context preservation
            )
            
            # Set up output path
            self.chunked_output_dir.mkdir(parents=True, exist_ok=True)
            chunks_output_path = self.chunked_output_dir / "semantic_chunks.json"
            
            # Process all markdown files in output directory
            results = chunker.process_output_directory(
                output_dir=self.output_directory,
                save_to=chunks_output_path
            )
            
            self.logger.info("Semantic chunking completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Semantic chunking failed: {e}")
            return False
    
    async def process(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting PDF to Enriched Markdown Pipeline")
            
            # Step 1: Find PDF files
            pdf_files = self.find_pdf_files()
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            # Step 2: Run MinerU processing
            if not self.run_mineru_processing(pdf_files):
                self.logger.error("MinerU processing failed")
                return False
            
            # Step 3: Run VLM enhancement
            if self.enable_vlm:
                vlm_result = await self.run_vlm_enhancement()
                if not vlm_result:
                    self.logger.warning("VLM enhancement failed, but continuing with pipeline")
                else:
                    self.logger.info("VLM enhancement completed successfully")
            
            # Step 4: Run semantic chunking
            if self.enable_chunking:
                if not self.run_semantic_chunking():
                    self.logger.warning("Semantic chunking failed, but pipeline completed")
            
            # Step 5: Clean up intermediate files
            mineru_output_dir = Path(__file__).parent.parent / "file_ingesting" / "output"
            cleanup_intermediate_files(mineru_output_dir, self.logger)
            
            # Success
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with exception: {e}")
            return False


# Simple function interface
async def process_pdfs_to_enriched_markdown(
    pdf_directory: str | Path,
    output_directory: str | Path,
    enable_vlm_enhancement: bool = True,
    enable_semantic_chunking: bool = True
) -> bool:
    """
    Simple function to process PDFs to enriched markdown with optional chunking.
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory for output files
        enable_vlm_enhancement: Whether to run VLM image enhancement
        enable_semantic_chunking: Whether to run semantic chunking
        
    Returns:
        bool: True if successful, False otherwise
    """
    pipeline = PDFToEnrichedMarkdownPipeline(
        pdf_directory=pdf_directory,
        output_directory=output_directory,
        enable_vlm=enable_vlm_enhancement,
        enable_chunking=enable_semantic_chunking
    )
    
    return await pipeline.process()


if __name__ == "__main__":
    # Calculate project root dynamically (LightRag_Agentic_System)
    project_root = Path(__file__).parent.parent.parent.parent
    
    processer = PDFToEnrichedMarkdownPipeline(
        pdf_directory=project_root / "ingestion_pipeline" / "pdfs_spm",
        output_directory=Path(__file__).parent.parent.parent / "output_spm",
        enable_vlm=True,
        enable_chunking=True
    )
    asyncio.run(processer.process())