"""
File validation and cleanup utilities for pipeline orchestration.

This module provides functions for cleaning filenames, validating output directories,
and managing old files to prevent contamination between pipeline runs.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any


def clean_source_filename(filename: str) -> str:
    """
    Clean source filename for consistent comparison between PDF files and ChromaDB sources.
    
    Removes file extensions, normalizes spaces, and handles special characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Cleaned filename for comparison
    """
    # Remove file extension if present
    cleaned = filename
    if cleaned.endswith('.pdf'):
        cleaned = cleaned[:-4]
    
    # Replace underscores with spaces and normalize whitespace
    cleaned = re.sub(r'[_]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove special characters and normalize
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)
    
    # Strip whitespace and convert to lowercase for case-insensitive comparison
    cleaned = cleaned.strip().lower()
    
    return cleaned


def validate_and_clean_output_directory(
    output_dir: Path, 
    files_to_process: List[str], 
    stage_name: str,
    file_extensions: List[str] = None,
    logger = None
) -> Dict[str, Any]:
    """
    Validate output directory and identify old files that should be cleaned.
    
    Args:
        output_dir: Output directory to validate
        files_to_process: List of new files that will be processed
        stage_name: Name of the processing stage for logging
        file_extensions: List of file extensions to check (e.g., ['.md', '.json'])
        logger: Logger instance
        
    Returns:
        Dict with validation results and cleanup recommendations
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    validation_result = {
        "directory_exists": False,
        "existing_files": [],
        "old_files_to_clean": [],
        "files_to_keep": [],
        "cleanup_recommended": False,
        "total_existing_files": 0
    }
    
    if not output_dir.exists():
        logger.info(f"{stage_name}: Output directory doesn't exist, will be created")
        return validation_result

    validation_result["directory_exists"] = True
    
    # Get all files in output directory
    all_files = []
    if file_extensions:
        for ext in file_extensions:
            all_files.extend(list(output_dir.glob(f"*{ext}")))
    else:
        all_files = [f for f in output_dir.iterdir() if f.is_file()]
    
    validation_result["total_existing_files"] = len(all_files)
    validation_result["existing_files"] = [f.name for f in all_files]
    
    if not all_files:
        logger.info(f"{stage_name}: Output directory is empty")
        return validation_result
    
    # Clean filenames for comparison
    files_to_process_cleaned = set(clean_source_filename(f) for f in files_to_process)
    
    # Categorize existing files
    for file_path in all_files:
        file_stem = file_path.stem
        cleaned_name = clean_source_filename(file_stem)
        
        if cleaned_name in files_to_process_cleaned:
            # This file corresponds to a new input file - keep it for reprocessing
            validation_result["files_to_keep"].append(file_path.name)
        else:
            # This file doesn't correspond to any new input - mark for cleanup
            validation_result["old_files_to_clean"].append(file_path.name)
    
    validation_result["cleanup_recommended"] = len(validation_result["old_files_to_clean"]) > 0
    
    if validation_result["cleanup_recommended"]:
        logger.warning(
            f"{stage_name}: Found {len(validation_result['old_files_to_clean'])} old files that don't match current inputs"
        )
        logger.info(f"{stage_name}: Old files to clean: {validation_result['old_files_to_clean'][:5]}...")
    
    logger.info(
        f"{stage_name}: Directory validation - {len(validation_result['files_to_keep'])} files to keep, "
        f"{len(validation_result['old_files_to_clean'])} files to clean"
    )
    
    return validation_result


def clean_old_output_files(
    output_dir: Path,
    files_to_clean: List[str],
    stage_name: str,
    dry_run: bool = False,
    logger = None
) -> Dict[str, Any]:
    """
    Clean old output files that don't correspond to current input files.
    
    Args:
        output_dir: Output directory containing files to clean
        files_to_clean: List of filenames to remove
        stage_name: Name of the processing stage for logging
        dry_run: If True, only simulate cleanup without actually deleting
        logger: Logger instance
        
    Returns:
        Dict with cleanup results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    cleanup_result = {
        "files_cleaned": [],
        "cleanup_errors": [],
        "total_files_cleaned": 0,
        "dry_run": dry_run
    }
    
    if not files_to_clean:
        logger.info(f"{stage_name}: No files to clean")
        return cleanup_result
    
    logger.info(f"{stage_name}: {'[DRY RUN] ' if dry_run else ''}Cleaning {len(files_to_clean)} old files")
    
    for filename in files_to_clean:
        file_path = output_dir / filename
        
        try:
            if file_path.exists():
                if not dry_run:
                    file_path.unlink()
                    logger.info(f"{stage_name}: Removed old file: {filename}")
                else:
                    logger.info(f"{stage_name}: [DRY RUN] Would remove: {filename}")
                cleanup_result["files_cleaned"].append(filename)
            else:
                logger.warning(f"{stage_name}: File not found for cleanup: {filename}")
                
        except Exception as e:
            error_msg = f"Failed to remove {filename}: {e}"
            cleanup_result["cleanup_errors"].append(error_msg)
            logger.error(f"{stage_name}: {error_msg}")
    
    cleanup_result["total_files_cleaned"] = len(cleanup_result["files_cleaned"])
    
    if cleanup_result["total_files_cleaned"] > 0:
        logger.info(
            f"{stage_name}: {'[DRY RUN] ' if dry_run else ''}Successfully cleaned {cleanup_result['total_files_cleaned']} files"
        )
    
    return cleanup_result