"""
MinerU Processing Utilities

Handles MinerU PDF processing operations including subprocess execution
and environment configuration for the PDF pipeline.
"""

import os
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Any


class MinerUProcessingUtils:
    """Utility class for MinerU processing operations."""
    
    def __init__(self, mineru_script_path: Path, logger: logging.Logger = None):
        """
        Initialize MinerU processing utilities.
        
        Args:
            mineru_script_path: Path to the MinerU ingestor script
            logger: Optional logger instance for structured logging
        """
        self.mineru_script_path = Path(mineru_script_path)
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
    
    def _prepare_environment(self) -> Dict[str, str]:
        """
        Prepare environment variables for MinerU execution.
        
        Returns:
            Dict[str, str]: Environment variables for subprocess
        """
        env = os.environ.copy()
        
        # Remove problematic CUDA configuration if present
        if 'PYTORCH_CUDA_ALLOC_CONF' in env:
            del env['PYTORCH_CUDA_ALLOC_CONF']
        
        # Set expandable segments for better memory management
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        return env
    
    def run_mineru_processing(self) -> bool:
        """
        Execute MinerU processing script.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.mineru_script_path.exists():
                self.logger.error(f"MinerU script not found: {self.mineru_script_path}")
                return False
            
            # Ensure working directory exists
            working_dir = self.mineru_script_path.parent
            if not working_dir.exists():
                self.logger.error(f"MinerU working directory not found: {working_dir}")
                return False
            
            self.logger.info(f"Running MinerU script: {self.mineru_script_path}")
            
            # Prepare environment
            env = self._prepare_environment()
            
            # Execute MinerU script
            result = subprocess.run([
                sys.executable, str(self.mineru_script_path)
            ], cwd=self.mineru_script_path.parent, env=env, capture_output=True, text=True)
            
            self.logger.info(f"MinerU return code: {result.returncode}")
            
            if result.stdout:
                self.logger.debug(f"MinerU STDOUT: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"MinerU STDERR: {result.stderr}")
            
            if result.returncode != 0:
                self.logger.error(f"MinerU failed with return code: {result.returncode}")
                return False
            
            self.logger.info("MinerU processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"MinerU processing failed with exception: {e}")
            return False
    
    def validate_mineru_output(self, output_directory: Path) -> bool:
        """
        Validate that MinerU produced expected output.
        
        Args:
            output_directory: Directory where MinerU should have created output
            
        Returns:
            bool: True if output exists and looks valid, False otherwise
        """
        if not output_directory.exists():
            self.logger.error(f"MinerU output directory does not exist: {output_directory}")
            return False
        
        output_items = list(output_directory.iterdir())
        if not output_items:
            self.logger.error(f"MinerU output directory is empty: {output_directory}")
            return False
        
        self.logger.info(f"Found {len(output_items)} items in MinerU output")
        
        # Check for expected structure (directories with markdown files)
        valid_directories = 0
        for item in output_items:
            if item.is_dir():
                auto_dir = item / "auto"
                if auto_dir.exists():
                    md_files = list(auto_dir.glob("*.md"))
                    if md_files:
                        valid_directories += 1
                        self.logger.debug(f"Valid directory found: {item.name} ({len(md_files)} markdown files)")
        
        if valid_directories == 0:
            self.logger.warning("No valid directories with markdown files found in MinerU output")
            return False
        
        self.logger.info(f"MinerU output validation passed: {valid_directories} valid directories")
        return True

# HELPER FUNCTIONS

def cleanup_intermediate_files(directory: Path, logger: logging.Logger = None) -> None:
    """
    Clean up intermediate files after processing.
    
    Args:
        directory: Directory to clean up
        logger: Optional logger for status messages
    """
    if directory.exists():
        try:
            import shutil
            shutil.rmtree(directory)
            if logger:
                logger.info(f"Cleaned up intermediate files: {directory}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to cleanup {directory}: {e}")

def validate_mineru_dependencies() -> bool:
    """
    Validate that MinerU dependencies are available.
    
    Returns:
        bool: True if dependencies are available, False otherwise
    """
    try:
        # Check if the MinerU script exists
        script_path = Path(__file__).parent.parent / "local_mineru" / "file_ingesting" / "mineru_ingestor.py"
        return script_path.exists()
    except Exception:
        return False