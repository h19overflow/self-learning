"""
File Management Utilities

Handles file operations, path validation, and filename management for the PDF pipeline.
Includes safe filename creation and path length validation for Windows compatibility.
"""

import hashlib
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple


class FileManagementUtils:
    """Utility class for file management operations in the PDF pipeline."""
    
    def __init__(self, filename_mapping_file: Path, logger: logging.Logger = None):
        """
        Initialize file management utilities.
        
        Args:
            filename_mapping_file: Path to the filename mapping JSON file
            logger: Optional logger instance for structured logging
        """
        self.filename_mapping_file = Path(filename_mapping_file)
        self.filename_mapping = self._load_filename_mapping()
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
    
    def _load_filename_mapping(self) -> Dict[str, str]:
        """
        Load the filename mapping from JSON file.
        
        Returns:
            Dict[str, str]: Mapping of safe filenames to original filenames
        """
        if self.filename_mapping_file.exists():
            try:
                with open(self.filename_mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Could not load filename mapping: {e}")
        return {}
    
    def _save_filename_mapping(self):
        """Save the filename mapping to JSON file."""
        try:
            with open(self.filename_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.filename_mapping, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Could not save filename mapping: {e}")
    
    def create_safe_filename(self, original_path: Path, max_length: int = 50) -> str:
        """
        Create a safe filename that won't exceed Windows path limits.
        
        Args:
            original_path: Original file path
            max_length: Maximum filename length (excluding extension)
            
        Returns:
            str: Safe filename
        """
        original_name = original_path.stem
        extension = original_path.suffix
        
        # If the name is already short enough, use it
        if len(original_name) <= max_length:
            return original_path.name
        
        # Create a shortened name with hash
        hash_obj = hashlib.md5(original_name.encode('utf-8'))
        hash_suffix = hash_obj.hexdigest()[:8]
        
        # Calculate how much space we have for the readable part
        available_length = max_length - len(hash_suffix) - 1  # -1 for underscore
        
        if available_length > 0:
            readable_part = original_name[:available_length]
            readable_part = readable_part.rstrip(' .-_')
            safe_name = f"{readable_part}_{hash_suffix}"
        else:
            safe_name = hash_suffix
        
        return f"{safe_name}{extension}"
    
    def validate_path_length(self, base_path: Path, filename: str) -> bool:
        """
        Validate that the resulting path won't exceed Windows path limits.
        
        Args:
            base_path: Base directory path
            filename: Filename to check
            
        Returns:
            bool: True if path is safe, False if too long
        """
        # Calculate the full path that would be created
        potential_path = base_path / filename / "auto" / "images" / ("a" * 64 + ".jpg")
        
        # Windows path limit is 260 characters
        return len(str(potential_path)) < 250  # Leave some margin
    
    def copy_files_with_safe_names(self, pdf_files: List[Path], destination_dir: Path) -> Path:
        """
        Copy PDF files to destination directory with safe filenames.
        
        Args:
            pdf_files: List of PDF files to copy
            destination_dir: Destination directory
            
        Returns:
            Path: Destination directory path
        """
        # Ensure destination directory exists
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create destination directory {destination_dir}: {e}")
            raise
        
        if not pdf_files:
            self.logger.warning("No PDF files provided to copy")
            return destination_dir
        
        self.logger.info(f"Copying {len(pdf_files)} files to {destination_dir}")
        
        # Clear existing files
        try:
            existing_files = list(destination_dir.glob("*.pdf"))
            if existing_files:
                self.logger.info(f"Clearing {len(existing_files)} existing PDF files")
                for existing_file in existing_files:
                    existing_file.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to clear existing files: {e}")
        
        # Copy new files with safe filenames
        copied_count = 0
        for pdf_file in pdf_files:
            try:
                if not pdf_file.exists():
                    self.logger.warning(f"PDF file does not exist: {pdf_file}")
                    continue
                
                # Check if we need to create a safe filename
                if not self.validate_path_length(destination_dir.parent / "output", pdf_file.stem):
                    safe_filename = self.create_safe_filename(pdf_file)
                    self.logger.info(f"Creating safe filename: {pdf_file.name} -> {safe_filename}")
                    
                    # Store the mapping
                    self.filename_mapping[safe_filename] = pdf_file.name
                    destination = destination_dir / safe_filename
                else:
                    destination = destination_dir / pdf_file.name
                
                shutil.copy2(pdf_file, destination)
                copied_count += 1
                self.logger.debug(f"Copied: {pdf_file.name} -> {destination.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to copy {pdf_file}: {e}")
                continue
        
        # Save the filename mapping
        if self.filename_mapping:
            self._save_filename_mapping()
            self.logger.info(f"Saved filename mappings for {len(self.filename_mapping)} files")
        
        self.logger.info(f"Successfully copied {copied_count}/{len(pdf_files)} files")
        return destination_dir
    
    def copy_output_with_restored_names(self, source_dir: Path, destination_dir: Path):
        """
        Copy output files to destination with original filenames restored.
        
        Args:
            source_dir: Source directory containing processed files
            destination_dir: Final destination directory
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_dir.exists():
            self.logger.warning(f"Source directory does not exist: {source_dir}")
            return
        
        copied_count = 0
        for item in source_dir.iterdir():
            if item.is_dir():
                # Check if this directory name was mapped from a longer original name
                original_name = None
                for safe_name, orig_name in self.filename_mapping.items():
                    safe_stem = Path(safe_name).stem
                    if safe_stem == item.name:
                        original_name = Path(orig_name).stem
                        break
                
                # Use original name if available, otherwise use current name
                final_dir_name = original_name if original_name else item.name
                destination = destination_dir / final_dir_name
                
                if original_name:
                    self.logger.info(f"Restoring directory name: {item.name} -> {final_dir_name}")
                
                if destination.exists():
                    self.logger.debug(f"Removing existing destination: {destination}")
                    shutil.rmtree(destination)
                
                shutil.copytree(item, destination)
                copied_count += 1
                
                self.logger.debug(f"Copied directory: {item.name} -> {destination}")
        
        self.logger.info(f"Copied {copied_count} directories to {destination_dir}")

# HELPER FUNCTIONS

def find_pdf_files(pdf_directory: Path) -> List[Path]:
    """
    Find all PDF files in the input directory.
    
    Args:
        pdf_directory: Directory to search for PDF files
        
    Returns:
        List[Path]: List of PDF file paths
        
    Raises:
        ValueError: If directory doesn't exist or no PDFs found
    """
    if not pdf_directory.exists():
        raise ValueError(f"PDF directory does not exist: {pdf_directory}")
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_directory}")
    
    return pdf_files