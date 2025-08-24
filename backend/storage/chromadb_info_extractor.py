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

    def __init__(self, persist_directory: str, collection_name: str = None):
        """Initialize the extractor.
        
        Args:
            persist_directory: Path to ChromaDB storage
            collection_name: Name of the collection to analyze (None for all collections)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize ChromaDB client for collection discovery
        try:
            import chromadb
            from chromadb.config import Settings
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
        except ImportError:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        # Cache managers for each collection
        self._managers = {}

    def list_all_collections(self) -> List[str]:
        """Get list of all collections in the ChromaDB instance."""
        try:
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            self.logger.info(f"Found {len(collection_names)} collections: {collection_names}")
            return collection_names
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []

    def get_manager_for_collection(self, collection_name: str) -> ChromaDBManager:
        """Get or create a manager for a specific collection."""
        if collection_name not in self._managers:
            config = ChromaConfig(
                persist_directory=self.persist_directory,
                collection_name=collection_name,
                embedding_device="cuda"  # Use CUDA for extraction
            )
            self._managers[collection_name] = ChromaDBManager(config)
        return self._managers[collection_name]

    def extract_comprehensive_info(self, include_chunks: bool = False) -> Dict[str, Any]:
        """Extract comprehensive database information.
        
        Args:
            include_chunks: Whether to include full chunk content
            
        Returns:
            Dictionary containing all database information
        """
        self.logger.info("Starting comprehensive ChromaDB information extraction...")
        
        # Determine which collections to analyze
        if self.collection_name:
            # Analyze single collection
            collections_to_analyze = [self.collection_name]
        else:
            # Analyze all collections
            collections_to_analyze = self.list_all_collections()
            if not collections_to_analyze:
                return {"error": "No collections found in the database"}
        
        # Extract info for each collection
        collections_info = {}
        database_totals = {
            "total_documents": 0,
            "total_sources": 0,
            "total_chunks": 0
        }
        
        for collection_name in collections_to_analyze:
            self.logger.info(f"Analyzing collection: {collection_name}")
            collection_info = self._extract_single_collection_info(collection_name, include_chunks)
            
            if "error" in collection_info:
                self.logger.error(f"Failed to extract info for collection '{collection_name}': {collection_info['error']}")
                collections_info[collection_name] = collection_info
                continue
            
            collections_info[collection_name] = collection_info
            
            # Update totals
            database_totals["total_documents"] += collection_info.get("collection_overview", {}).get("total_documents", 0)
            database_totals["total_sources"] += collection_info.get("collection_overview", {}).get("unique_source_files", 0)
            database_totals["total_chunks"] += collection_info.get("collection_overview", {}).get("total_chunks_verified", 0)
        
        # Create comprehensive report
        comprehensive_info = {
            "extraction_metadata": {
                "extraction_date": datetime.now().isoformat(),
                "extractor_version": "2.0.0",
                "extraction_scope": "single_collection" if self.collection_name else "all_collections",
                "collections_analyzed": collections_to_analyze,
                "include_full_chunks": include_chunks
            },
            "database_overview": {
                "persist_directory": str(self.persist_directory),
                "total_collections": len(collections_to_analyze),
                "collections_list": collections_to_analyze,
                **database_totals
            },
            "collections": collections_info
        }
        
        self.logger.info(f"Extraction completed. Collections: {len(collections_to_analyze)}, Total chunks: {database_totals['total_chunks']}")
        return comprehensive_info

    def _extract_single_collection_info(self, collection_name: str, include_chunks: bool = False) -> Dict[str, Any]:
        """Extract information for a single collection.
        
        Args:
            collection_name: Name of the collection to analyze
            include_chunks: Whether to include full chunk content
            
        Returns:
            Dictionary containing collection information
        """
        try:
            manager = self.get_manager_for_collection(collection_name)
            
            # Get basic collection info
            collection_info = manager.get_collection_info()
            if "error" in collection_info:
                return {"error": collection_info["error"]}
            
            # Get source files
            source_files = manager.list_source_files()
            self.logger.info(f"Collection '{collection_name}': Found {len(source_files)} source files")
            
            # Analyze each source file
            sources_analysis = []
            total_chunks = 0
            
            for source_file in source_files:
                chunks = manager.get_chunks_from_source(source_file)
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
            
            # Create collection report
            return {
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
            
        except Exception as e:
            self.logger.error(f"Failed to extract info for collection '{collection_name}': {e}")
            return {"error": str(e)}

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
            # Determine which collections to analyze
            if self.collection_name:
                collections_to_analyze = [self.collection_name]
            else:
                collections_to_analyze = self.list_all_collections()
                if not collections_to_analyze:
                    print("No collections found in the database")
                    return
            
            print("\n" + "="*70)
            print("CHROMADB DATABASE SUMMARY")
            print("="*70)
            print(f"Persist Directory: {self.persist_directory}")
            print(f"Total Collections: {len(collections_to_analyze)}")
            print(f"Analysis Scope: {'Single Collection' if self.collection_name else 'All Collections'}")
            print("="*70)
            
            total_docs = 0
            total_sources = 0
            
            for collection_name in collections_to_analyze:
                try:
                    manager = self.get_manager_for_collection(collection_name)
                    collection_info = manager.get_collection_info()
                    
                    if "error" in collection_info:
                        print(f"\nCollection '{collection_name}': ERROR - {collection_info['error']}")
                        continue
                    
                    source_files = manager.list_source_files()
                    doc_count = collection_info.get('total_documents', 0)
                    source_count = len(source_files)
                    
                    total_docs += doc_count
                    total_sources += source_count
                    
                    print(f"\nCollection: {collection_name}")
                    print(f"  Documents: {doc_count:,}")
                    print(f"  Sources: {source_count}")
                    print(f"  Embedding Model: {collection_info.get('configuration', {}).get('embedding_model', 'Unknown')}")
                    
                    if source_files and len(source_files) <= 5:
                        print(f"  Sources: {', '.join(source_files[:5])}")
                    elif source_files:
                        print(f"  Top 3 Sources: {', '.join(source_files[:3])}")
                        if len(source_files) > 3:
                            print(f"  ... and {len(source_files) - 3} more")
                    
                except Exception as e:
                    print(f"\nCollection '{collection_name}': ERROR - {e}")
            
            print("\n" + "-"*70)
            print(f"TOTALS: {total_docs:,} documents across {total_sources} sources")
            print("="*70)
            
        except Exception as e:
            print(f"Error generating summary: {e}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract comprehensive information from ChromaDB collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --summary                                              # Summary of all collections
  %(prog)s --output all_collections.json                         # Export all collections
  %(prog)s --output full_database.json --include_chunks          # Export all with chunks
  %(prog)s --collection academic_papers --output papers.json     # Single collection
  %(prog)s --collection custom_collection --summary              # Single collection summary
        """
    )
    
    parser.add_argument(
        "--persist_dir", 
        default="backend/storage/chromadb_storage",
        help="ChromaDB persistence directory (default: backend/storage/chromadb_storage)"
    )
    parser.add_argument(
        "--collection", 
        help="Collection name to analyze (default: analyze all collections)"
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