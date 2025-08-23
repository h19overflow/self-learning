"""
ChromaDB Information Extractor

This script extracts comprehensive information from ChromaDB and exports it to JSON format.
Provides detailed statistics about sources, chunks, and database configuration.

Dependencies:
- ChromaDB storage and management components
- JSON serialization for output
- Pathlib for file handling

Usage:
    python chromadb_info_extractor.py --output database_info.json
    python chromadb_info_extractor.py --output database_info.json --include_chunks
    python chromadb_info_extractor.py --collection custom_collection
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from backend.storage.chromadb_instance.chromadb_manager import ChromaDBManager
from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig


class ChromaDBInfoExtractor:
    """Extract comprehensive information from ChromaDB collections."""

    def __init__(self, persist_directory: str, collection_name: str = "academic_papers"):
        """Initialize the extractor.
        
        Args:
            persist_directory: Path to ChromaDB storage
            collection_name: Name of the collection to analyze
        """
        self.config = ChromaConfig(
            persist_directory=Path(persist_directory),
            collection_name=collection_name,
            embedding_device="cuda"  # Use CUDA for extraction
        )
        
        self.manager = ChromaDBManager(self.config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def extract_comprehensive_info(self, include_chunks: bool = False) -> Dict[str, Any]:
        """Extract comprehensive database information.
        
        Args:
            include_chunks: Whether to include full chunk content
            
        Returns:
            Dictionary containing all database information
        """
        self.logger.info("Starting comprehensive ChromaDB information extraction...")
        
        # Get basic collection info
        collection_info = self.manager.get_collection_info()
        if "error" in collection_info:
            self.logger.error(f"Failed to get collection info: {collection_info['error']}")
            return {"error": collection_info["error"]}
        
        # Get source files
        source_files = self.manager.list_source_files()
        self.logger.info(f"Found {len(source_files)} source files")
        
        # Analyze each source file
        sources_analysis = []
        total_chunks = 0
        
        for source_file in source_files:
            self.logger.info(f"Analyzing source: {source_file}")
            
            chunks = self.manager.get_chunks_from_source(source_file)
            chunk_count = len(chunks)
            total_chunks += chunk_count
            
            # Analyze chunk characteristics
            chunk_lengths = [len(chunk.get('content', '')) for chunk in chunks]
            
            source_analysis = {
                "source_file": source_file,
                "total_chunks": chunk_count,
                "chunk_statistics": {
                    "min_length": min(chunk_lengths) if chunk_lengths else 0,
                    "max_length": max(chunk_lengths) if chunk_lengths else 0,
                    "avg_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                },
                "sample_metadata": chunks[0].get('metadata', {}) if chunks else {}
            }
            
            # Include full chunks if requested
            if include_chunks:
                source_analysis["chunks"] = chunks
            
            sources_analysis.append(source_analysis)
        
        # Create comprehensive report
        comprehensive_info = {
            "extraction_metadata": {
                "extraction_date": datetime.now().isoformat(),
                "extractor_version": "1.0.0",
                "collection_analyzed": self.config.collection_name,
                "include_full_chunks": include_chunks
            },
            "collection_overview": {
                "collection_name": collection_info.get("collection_name"),
                "total_documents": collection_info.get("total_documents", 0),
                "unique_source_files": len(source_files),
                "total_chunks_verified": total_chunks,
                "configuration": collection_info.get("configuration", {})
            },
            "sources_summary": {
                "total_sources": len(source_files),
                "sources_list": source_files,
                "chunk_distribution": {
                    source["source_file"]: source["total_chunks"] 
                    for source in sources_analysis
                }
            },
            "detailed_sources_analysis": sources_analysis,
            "database_statistics": {
                "embedding_model": collection_info.get("configuration", {}).get("embedding_model"),
                "distance_function": collection_info.get("configuration", {}).get("distance_function"),
                "persist_directory": collection_info.get("configuration", {}).get("persist_directory"),
                "batch_size": collection_info.get("configuration", {}).get("batch_size")
            }
        }
        
        self.logger.info(f"Extraction completed. Total sources: {len(source_files)}, Total chunks: {total_chunks}")
        return comprehensive_info

    def export_to_json(self, output_file: str, include_chunks: bool = False, indent: int = 2) -> bool:
        """Export database information to JSON file.
        
        Args:
            output_file: Path to output JSON file
            include_chunks: Whether to include full chunk content
            indent: JSON formatting indent
            
        Returns:
            Success status
        """
        try:
            # Extract information
            info = self.extract_comprehensive_info(include_chunks=include_chunks)
            
            if "error" in info:
                self.logger.error(f"Failed to extract info: {info['error']}")
                return False
            
            # Write to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=indent, ensure_ascii=False)
            
            self.logger.info(f"Information exported to: {output_path.absolute()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            return False

    def print_summary(self) -> None:
        """Print a quick summary to console."""
        try:
            collection_info = self.manager.get_collection_info()
            if "error" in collection_info:
                print(f"Error accessing collection: {collection_info['error']}")
                return
            
            source_files = self.manager.list_source_files()
            
            print("\n" + "="*60)
            print("CHROMADB COLLECTION SUMMARY")
            print("="*60)
            print(f"Collection: {collection_info.get('collection_name', 'Unknown')}")
            print(f"Total Documents: {collection_info.get('total_documents', 0):,}")
            print(f"Unique Sources: {len(source_files)}")
            print(f"Embedding Model: {collection_info.get('configuration', {}).get('embedding_model', 'Unknown')}")
            print(f"Persist Directory: {collection_info.get('configuration', {}).get('persist_directory', 'Unknown')}")
            
            if source_files:
                print(f"\nTop 10 Sources:")
                for i, source in enumerate(source_files[:10], 1):
                    print(f"  {i:2d}. {source}")
                
                if len(source_files) > 10:
                    print(f"  ... and {len(source_files) - 10} more sources")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error generating summary: {e}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract comprehensive information from ChromaDB collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --summary
  %(prog)s --output database_info.json
  %(prog)s --output full_database.json --include_chunks
  %(prog)s --collection custom_collection --output custom_info.json
        """
    )
    
    parser.add_argument(
        "--persist_dir", 
        default="backend/storage/chromadb_storage",
        help="ChromaDB persistence directory (default: backend/storage/chromadb_storage)"
    )
    parser.add_argument(
        "--collection", 
        default="academic_papers",
        help="Collection name to analyze (default: academic_papers)"
    )
    parser.add_argument(
        "--output", 
        help="Output JSON file path"
    )
    parser.add_argument(
        "--include_chunks", 
        action="store_true",
        help="Include full chunk content in output (creates large files)"
    )
    parser.add_argument(
        "--summary", 
        action="store_true",
        help="Print quick summary to console"
    )
    parser.add_argument(
        "--indent", 
        type=int, 
        default=2,
        help="JSON formatting indent (default: 2)"
    )

    args = parser.parse_args()

    try:
        # Initialize extractor
        extractor = ChromaDBInfoExtractor(args.persist_dir, args.collection)
        
        if args.summary:
            extractor.print_summary()
            return
        
        if not args.output:
            print("Error: --output is required (or use --summary for console output)")
            parser.print_help()
            return
        
        # Export to JSON
        success = extractor.export_to_json(
            args.output, 
            include_chunks=args.include_chunks,
            indent=args.indent
        )
        
        if not success:
            sys.exit(1)
            
        print(f"\nExtraction completed successfully!")
        print(f"Output file: {Path(args.output).absolute()}")
        
        if args.include_chunks:
            print("Note: Full chunk content included - file may be very large")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()