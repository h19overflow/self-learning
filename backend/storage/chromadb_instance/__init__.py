"""
ChromaDB Instance for Document Ingestion and Retrieval

A comprehensive ChromaDB integration for the document processing pipeline,
providing semantic search capabilities for academic papers and documents.

Components:
- ChromaDB ingestion engine for document chunks
- Semantic retriever for query-based document search
- Embedding management using SentenceTransformers
- Configurable collection management

Architecture:
This module provides a pluggable ChromaDB solution that integrates seamlessly
with the existing pipeline orchestrator and semantic chunker.
"""

from .models.chroma_config import ChromaConfig, RetrievalConfig
from .models.search_result import SearchResult, RetrievalResults
from .components.chroma_ingestion_engine import ChromaIngestionEngine  
from .components.chroma_retriever import ChromaRetriever
from .chromadb_manager import ChromaDBManager

__all__ = [
    'ChromaConfig',
    'RetrievalConfig', 
    'SearchResult',
    'RetrievalResults',
    'ChromaIngestionEngine',
    'ChromaRetriever',
    'ChromaDBManager'
]