"""
ChromaDB Manager - Unified Interface for Document Operations

This module provides a unified interface for both ingestion and retrieval operations
against ChromaDB, simplifying the integration with the pipeline orchestrator.

Dependencies:
- ChromaDB ingestion and retrieval components
- Configuration management
- Comprehensive error handling and logging
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig, RetrievalConfig
from backend.storage.chromadb_instance.models.search_result import RetrievalResults
from backend.storage.chromadb_instance.components.chroma_ingestion_engine import ChromaIngestionEngine
from backend.storage.chromadb_instance.components.chroma_retriever import ChromaRetriever


class ChromaDBManager:
    """Unified manager for ChromaDB operations."""

    def __init__(self, config: ChromaConfig):
        """Initialize ChromaDB manager.
        
        Args:
            config: ChromaDB configuration
        """
        self.config = config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self._ingestion_engine: Optional[ChromaIngestionEngine] = None
        self._retriever: Optional[ChromaRetriever] = None
        
        self.logger.info(f"ChromaDB Manager initialized for collection: {config.collection_name}")

    @property
    def ingestion_engine(self) -> ChromaIngestionEngine:
        """Get or create ingestion engine (lazy initialization)."""
        if self._ingestion_engine is None:
            self._ingestion_engine = ChromaIngestionEngine(self.config)
        return self._ingestion_engine

    @property
    def retriever(self) -> ChromaRetriever:
        """Get or create retriever (lazy initialization)."""
        if self._retriever is None:
            self._retriever = ChromaRetriever(self.config)
        return self._retriever

    async def ingest_chunks_from_file(self, chunks_file_path: Path) -> Dict[str, Any]:
        """Ingest semantic chunks from JSON file.
        
        Args:
            chunks_file_path: Path to chunks JSON file
            
        Returns:
            Ingestion results and statistics
        """
        self.logger.info(f"Starting ingestion from: {chunks_file_path}")
        return await self.ingestion_engine.process_chunks_file(chunks_file_path)

    def search(self, query: str, top_k: int = 5, score_threshold: Optional[float] = None) -> RetrievalResults:
        """Perform semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            Search results
        """
        retrieval_config = RetrievalConfig(
            top_k=top_k,
            score_threshold=score_threshold
        )
        return self.retriever.search(query, retrieval_config)

    def search_by_source(self, query: str, source_file: str, top_k: int = 5) -> RetrievalResults:
        """Search within specific source document.
        
        Args:
            query: Search query
            source_file: Source file name
            top_k: Number of results to return
            
        Returns:
            Filtered search results
        """
        return self.retriever.search_by_source(query, source_file, RetrievalConfig(top_k=top_k))

    def get_collection_info(self) -> Dict[str, Any]:
        """Get comprehensive collection information."""
        try:
            # Get basic stats from retriever
            stats = self.retriever.get_collection_stats()
            
            # Add configuration info
            stats.update({
                "configuration": {
                    "collection_name": self.config.collection_name,
                    "embedding_model": self.config.embedding_model,
                    "embedding_device": self.config.embedding_device,
                    "distance_function": self.config.distance_function,
                    "batch_size": self.config.batch_size,
                    "persist_directory": str(self.config.persist_directory)
                }
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def list_source_files(self) -> List[str]:
        """Get list of all source files in collection."""
        return self.retriever.list_source_files()

    def get_chunks_from_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Get all chunks from specific source file."""
        return self.retriever.get_chunks_from_source(source_file)

    def reset_collection(self) -> bool:
        """Reset (clear) the collection."""
        try:
            # Reset through ingestion engine
            success = self.ingestion_engine.reset_collection()
            
            # Reset retriever connection
            if self._retriever is not None:
                self._retriever = None  # Will be re-initialized on next access
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            return False

    async def finalize(self):
        """Clean up resources."""
        try:
            if self._ingestion_engine:
                await self._ingestion_engine.finalize()
            
            self.logger.info("ChromaDB Manager finalized")
            
        except Exception as e:
            self.logger.warning(f"Error during finalization: {e}")

    # CONVENIENCE METHODS FOR COMMON OPERATIONS

    @classmethod
    def create_default(cls, persist_directory: str, collection_name: str = "academic_papers") -> 'ChromaDBManager':
        """Create ChromaDB manager with default configuration.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
            
        Returns:
            Configured ChromaDBManager instance
        """
        config = ChromaConfig(
            persist_directory=Path(persist_directory),
            collection_name=collection_name
        )
        return cls(config)

    def quick_search(self, query: str, max_results: int = 3) -> List[str]:
        """Perform quick search returning just content strings.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of content strings
        """
        results = self.search(query, top_k=max_results)
        return [result.content for result in results.results]

    def search_with_sources(self, query: str, max_results: int = 5) -> Dict[str, List[str]]:
        """Search and group results by source file.
        
        Args:
            query: Search query  
            max_results: Maximum number of results
            
        Returns:
            Dictionary mapping source files to content lists
        """
        results = self.search(query, top_k=max_results)
        
        grouped = {}
        for result in results.results:
            source = result.source_file
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(result.content)
        
        return grouped

    # HELPER FUNCTIONS

    def _validate_file_path(self, file_path: Path) -> bool:
        """Validate that file path exists and is readable."""
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
        
        if not file_path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            return False
        
        return True