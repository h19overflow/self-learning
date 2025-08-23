"""
ChromaDB Ingestion Engine for Semantic Chunks

This module handles the ingestion of semantically chunked documents into ChromaDB,
providing efficient batch processing and metadata management.

Dependencies:
- ChromaDB for vector database operations
- SentenceTransformers for embeddings
- JSON chunk files from semantic chunker
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from sentence_transformers import SentenceTransformer

from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig


class ChromaIngestionEngine:
    """Engine for ingesting semantic chunks into ChromaDB."""

    def __init__(self, config: ChromaConfig):
        """Initialize ChromaDB ingestion engine.
        
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
        
        self._initialize_chromadb()
        self._initialize_embeddings()
        
        # Processing statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "successful_ingestions": 0,
            "failed_ingestions": 0,
            "processing_time": 0.0
        }

    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client with persistence
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"distance_function": self.config.distance_function}
            )
            
            self.logger.info(f"ChromaDB collection '{self.config.collection_name}' ready")
            self.logger.info(f"Collection contains {self.collection.count()} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _initialize_embeddings(self):
        """Initialize embedding model."""
        try:
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.embedding_device,
                trust_remote_code=True
            )
            self.logger.info(f"Embedding model '{self.config.embedding_model}' ready")
            
        except Exception as e:
            self.logger.warning(f"Failed to load {self.config.embedding_model}, using fallback")
            try:
                self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            except Exception as fallback_error:
                self.logger.error(f"Failed to load fallback embedding model: {fallback_error}")
                raise

    async def process_chunks_file(self, chunks_file_path: Path) -> Dict[str, Any]:
        """Process and ingest chunks from JSON file.
        
        Args:
            chunks_file_path: Path to JSON file containing chunks
            
        Returns:
            Dict containing ingestion results and statistics
        """
        start_time = datetime.now()
        self.logger.info(f"Starting ingestion from: {chunks_file_path}")
        
        try:
            # Load chunks from file
            chunks_data = self._load_chunks_from_file(chunks_file_path)
            
            # Process all documents
            all_documents = []
            for file_name, file_data in chunks_data.get("files", {}).items():
                documents = self._prepare_documents_for_ingestion(file_name, file_data)
                all_documents.extend(documents)
            
            self.stats["total_documents"] = len(chunks_data.get("files", {}))
            self.stats["total_chunks"] = len(all_documents)
            
            # Ingest in batches
            await self._ingest_documents_in_batches(all_documents)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["processing_time"] = processing_time
            
            self.logger.info(f"Ingestion completed in {processing_time:.2f}s")
            self.logger.info(f"Successfully ingested {self.stats['successful_ingestions']} chunks")
            self.logger.info(f"Failed ingestions: {self.stats['failed_ingestions']}")
            
            return {
                "success": True,
                "statistics": self.stats,
                "collection_info": {
                    "name": self.config.collection_name,
                    "total_documents": self.collection.count()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": self.stats
            }

    def _load_chunks_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load chunks data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load chunks file: {e}")
            raise

    def _prepare_documents_for_ingestion(self, file_name: str, file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare document chunks for ChromaDB ingestion.
        
        Args:
            file_name: Name of source file
            file_data: File chunk data from JSON
            
        Returns:
            List of documents prepared for ingestion
        """
        documents = []
        chunks = file_data.get("chunks", [])
        
        for chunk_index, chunk_content in enumerate(chunks):
            # Create unique document ID
            doc_id = f"{file_name}_chunk_{chunk_index:04d}_{uuid.uuid4().hex[:8]}"
            
            # Prepare metadata with enhanced video source tracking
            metadata = {
                "source_file": file_name,
                "chunk_index": chunk_index,
                "total_chunks": len(chunks),
                "word_count": len(chunk_content.split()),
                "char_count": len(chunk_content),
                "ingestion_timestamp": datetime.now().isoformat(),
                "content_type": "video_transcript" if file_name.startswith("WPgG_PlOsYs") or "youtube.com" in chunk_content else "document"
            }
            
            # Extract video URL from content for credibility tracking
            import re
            video_url_match = re.search(r'https://www\.youtube\.com/watch\?v=[\w-]+', chunk_content)
            if video_url_match:
                metadata["video_url"] = video_url_match.group()
                metadata["content_type"] = "video_transcript"
            
            # Extract video ID if present in filename or content
            video_id_match = re.search(r'[A-Za-z0-9_-]{11}', file_name)
            if video_id_match and len(video_id_match.group()) == 11:
                metadata["video_id"] = video_id_match.group()
            
            documents.append({
                "id": doc_id,
                "content": chunk_content,
                "metadata": metadata
            })
        
        return documents

    async def _ingest_documents_in_batches(self, documents: List[Dict[str, Any]]):
        """Ingest documents in batches for better performance."""
        total_docs = len(documents)
        batch_size = self.config.batch_size
        
        # Create batches
        batches = [
            documents[i:i + batch_size] 
            for i in range(0, total_docs, batch_size)
        ]
        
        self.logger.info(f"Processing {total_docs} documents in {len(batches)} batches")
        
        # Process batches with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        tasks = []
        
        for batch_idx, batch in enumerate(batches):
            task = self._process_batch_with_semaphore(semaphore, batch_idx, batch)
            tasks.append(task)
        
        # Wait for all batches to complete
        await asyncio.gather(*tasks)

    async def _process_batch_with_semaphore(self, semaphore: asyncio.Semaphore, batch_idx: int, batch: List[Dict[str, Any]]):
        """Process a single batch with concurrency control."""
        async with semaphore:
            await self._process_batch(batch_idx, batch)

    async def _process_batch(self, batch_idx: int, batch: List[Dict[str, Any]]):
        """Process a single batch of documents."""
        self.logger.info(f"Processing batch {batch_idx + 1} ({len(batch)} documents)")
        
        try:
            # Extract data for ChromaDB
            ids = [doc["id"] for doc in batch]
            documents = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            
            # Generate embeddings
            embeddings = await self._generate_embeddings_async(documents)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.stats["successful_ingestions"] += len(batch)
            self.logger.info(f"Batch {batch_idx + 1} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Batch {batch_idx + 1} failed: {e}")
            self.stats["failed_ingestions"] += len(batch)
            
            if not self.config.continue_on_error:
                raise

    async def _generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.embedding_model.encode(texts).tolist()
        )

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.config.collection_name,
                "document_count": count,
                "persist_directory": str(self.config.persist_directory),
                "embedding_model": self.config.embedding_model,
                "distance_function": self.config.distance_function
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def reset_collection(self) -> bool:
        """Reset (clear) the current collection."""
        try:
            self.chroma_client.delete_collection(name=self.config.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.config.collection_name,
                metadata={"distance_function": self.config.distance_function}
            )
            self.logger.info(f"Collection '{self.config.collection_name}' reset successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            return False

    async def finalize(self):
        """Clean up resources."""
        try:
            # ChromaDB handles persistence automatically
            self.logger.info("ChromaDB ingestion engine finalized")
        except Exception as e:
            self.logger.warning(f"Error during finalization: {e}")

    # HELPER FUNCTIONS
    
    def _validate_chunk_content(self, content: str) -> bool:
        """Validate chunk content before ingestion."""
        if not content or not content.strip():
            return False
        if len(content) < 10:  # Too short to be meaningful
            return False
        return True