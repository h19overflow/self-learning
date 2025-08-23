"""
MinerU file ingesting module for processing PDFs to markdown.
"""
from pathlib import Path
from .mineru_ingestor import parse_doc


def process_pdfs_to_markdown(pdf_file: Path, output_directory: Path) -> bool:
    """
    Process a single PDF file to markdown using MinerU.
    
    Args:
        pdf_file: Path to the PDF file to process
        output_directory: Directory where markdown output will be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        parse_doc(
            path_list=[pdf_file],
            output_dir=str(output_directory),
            lang="en",  # Changed from 'ch' to 'en' for English papers
            backend="pipeline",
            method="auto"
        )
        return True
    except Exception as e:
        print(f"Error processing PDF {pdf_file}: {e}")
        return False