"""
ChromaDB Retriever for Semantic Search - Simplified Architecture

This module provides a clean, modular interface for semantic search operations
using ChromaDB, with components for embedding, reranking, and result formatting.

Dependencies:
- ChromaDB for vector database operations
- Modular components for embedding, reranking, and formatting
- Configuration management
"""
import weave
import logging
import time
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig, RetrievalConfig
from backend.storage.chromadb_instance.models.search_result import SearchResult, RetrievalResults
from backend.storage.chromadb_instance.components.embedding_manager import EmbeddingManager
from backend.storage.chromadb_instance.components.reranker import CrossEncoderReranker
from backend.storage.chromadb_instance.components.result_formatter import SearchResultFormatter


class ChromaRetriever:
    """Simplified semantic retriever with modular components."""

    def __init__(self, config: ChromaConfig):
        """Initialize ChromaDB retriever with modular components.
        
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
        
        # Initialize ChromaDB
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        # Initialize core components
        self._initialize_chromadb()
        self.embedding_manager = EmbeddingManager(config)
        self.reranker = CrossEncoderReranker(config)
        self.formatter = SearchResultFormatter()
        
        self.logger.info("ChromaRetriever initialized with modular components")

    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get existing collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.config.collection_name
                )
                self.logger.info(f"Connected to collection '{self.config.collection_name}'")
                self.logger.info(f"Collection contains {self.collection.count()} documents")
                
            except Exception as e:
                self.logger.error(f"Collection '{self.config.collection_name}' not found: {e}")
                self.logger.info("Creating new collection for retrieval")
                self.collection = self.chroma_client.create_collection(
                    name=self.config.collection_name,
                    metadata={"distance_function": self.config.distance_function}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    @weave.op()
    def search(self, query: str, retrieval_config: Optional[RetrievalConfig] = None) -> RetrievalResults:
        """Perform semantic search against the document collection.
        
        Args:
            query: Search query text
            retrieval_config: Optional retrieval configuration
            
        Returns:
            RetrievalResults containing search results and metadata
        """
        # Use default config if none provided
        if retrieval_config is None:
            retrieval_config = RetrievalConfig()
        
        start_time = time.time()
        self.logger.info(f"Performing search: '{query[:100]}...'")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.encode_query(query)
            
            # Calculate how many candidates to retrieve for reranking
            retrieval_k = self.reranker.calculate_retrieval_k(retrieval_config)
            
            # Prepare search parameters
            search_params = self._prepare_search_params(retrieval_config)
            
            # Perform ChromaDB search
            raw_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=retrieval_k,
                include=['documents', 'metadatas', 'distances'],
                **search_params
            )
            
            # Apply reranking if enabled and available
            if retrieval_config.enable_reranking and self.reranker.is_available():
                raw_results = self.reranker.rerank_results(query, raw_results, retrieval_config)
            
            # Format results
            search_results = self.formatter.format_search_results(
                raw_results, 
                retrieval_config,
                limit_to_top_k=True
            )
            
            # Calculate retrieval time
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            self.logger.info(f"Search completed in {retrieval_time_ms:.2f}ms, found {len(search_results)} results")
            
            # Create final results object
            return self.formatter.create_retrieval_results(
                query=query,
                search_results=search_results,
                retrieval_time_ms=retrieval_time_ms,
                collection_name=self.config.collection_name,
                config=retrieval_config,
                additional_metadata={
                    "reranker_used": retrieval_config.enable_reranking and self.reranker.is_available(),
                    "embedding_model": self.config.embedding_model,
                    "candidates_retrieved": retrieval_k
                }
            )
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            return self.formatter.create_retrieval_results(
                query=query,
                search_results=[],
                retrieval_time_ms=retrieval_time_ms,
                collection_name=self.config.collection_name,
                config=retrieval_config,
                additional_metadata={"error": str(e)}
            )

    def search_by_source(self, query: str, source_file: str, retrieval_config: Optional[RetrievalConfig] = None) -> RetrievalResults:
        """Search within a specific source document.
        
        Args:
            query: Search query text
            source_file: Name of source file to search within
            retrieval_config: Optional retrieval configuration
            
        Returns:
            RetrievalResults filtered to specific source
        """
        if retrieval_config is None:
            retrieval_config = RetrievalConfig()
        
        # Set source filter
        retrieval_config.source_filter = source_file
        
        return self.search(query, retrieval_config)

    def get_similar_chunks(self, content: str, retrieval_config: Optional[RetrievalConfig] = None) -> RetrievalResults:
        """Find chunks similar to provided content.
        
        Args:
            content: Content to find similar chunks for
            retrieval_config: Optional retrieval configuration
            
        Returns:
            RetrievalResults with similar content
        """
        return self.search(content, retrieval_config)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to understand document distribution
            sample_results = self.collection.get(limit=min(100, count), include=['metadatas'])
            
            # Analyze source files
            source_files = set()
            total_chunks = 0
            
            if sample_results.get('metadatas'):
                for metadata in sample_results['metadatas']:
                    if metadata and 'source_file' in metadata:
                        source_files.add(metadata['source_file'])
                        total_chunks += 1
            
            return {
                "collection_name": self.config.collection_name,
                "total_documents": count,
                "unique_source_files": len(source_files),
                "sample_source_files": list(source_files)[:10],  # First 10 for preview
                "embedding_model": self.config.embedding_model,
                "distance_function": self.config.distance_function,
                "persist_directory": str(self.config.persist_directory),
                "reranker_available": self.reranker.is_available(),
                "embedding_manager_ready": self.embedding_manager.is_available()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def list_source_files(self) -> List[str]:
        """Get list of all source files in the collection."""
        try:
            # Get all documents with metadata
            all_results = self.collection.get(include=['metadatas'])
            
            source_files = set()
            if all_results.get('metadatas'):
                for metadata in all_results['metadatas']:
                    if metadata and 'source_file' in metadata:
                        source_files.add(metadata['source_file'])
            
            return sorted(list(source_files))
            
        except Exception as e:
            self.logger.error(f"Failed to list source files: {e}")
            return []

    def get_chunks_from_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific source file.
        
        Args:
            source_file: Name of source file
            
        Returns:
            List of chunks with metadata
        """
        try:
            results = self.collection.get(
                where={"source_file": source_file},
                include=['documents', 'metadatas']
            )
            
            chunks = []
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            ids = results.get('ids', [])  # IDs are returned by default
            
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                chunk_data = {
                    "content": doc,
                    "metadata": metadata
                }
                if ids and i < len(ids):
                    chunk_data["id"] = ids[i]
                chunks.append(chunk_data)
            
            # Sort by chunk_index if available
            chunks.sort(key=lambda x: x.get('metadata', {}).get('chunk_index', 0))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get chunks from {source_file}: {e}")
            return []

    def _prepare_search_params(self, config: RetrievalConfig) -> Dict[str, Any]:
        """Prepare search parameters for ChromaDB query."""
        params = {}
        
        # Add metadata filters
        if config.metadata_filters or config.source_filter:
            where_clause = {}
            
            if config.source_filter:
                where_clause["source_file"] = config.source_filter
            
            if config.metadata_filters:
                where_clause.update(config.metadata_filters)
            
            params["where"] = where_clause
        
        return params

    async def finalize(self):
        """Clean up resources."""
        try:
            await self.embedding_manager.finalize()
            await self.reranker.finalize()
            self.logger.info("ChromaRetriever resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during finalization: {e}")