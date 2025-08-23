"""
Concise Semantic Chunker with LangChain and Nomic Model

A minimal implementation using LangChain's SemanticChunker for intelligent text chunking.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

try:
    from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
    from langchain_core.embeddings import Embeddings
    LANGCHAIN_SEMANTIC_AVAILABLE = True
except ImportError:
    LANGCHAIN_SEMANTIC_AVAILABLE = False
    # Fallback dummy classes
    class Embeddings:
        pass

from .utils.page_mapping_utils import PageMappingUtils
from .utils.file_processing_utils import FileProcessingUtils


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer models to work with LangChain."""
    
    def __init__(self, model_name: str = 'nomic-ai/nomic-embed-text-v1.5'):
        """Initialize the SentenceTransformer embeddings wrapper.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception:
            # Fallback to a more standard model
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.
        
        Args:
            texts: List of text to embed.
            
        Returns:
            List of embeddings.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding.
        """
        embedding = self.model.encode([text])
        return embedding[0].tolist()

class SemanticChunker:
    """Concise semantic chunker using LangChain's SemanticChunker and Nomic model."""

    def __init__(self, chunk_size: int = 8192, overlap: int = 500, threshold: float = 0.75):
        self.chunk_size = chunk_size
        self.threshold = threshold

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize semantic chunker
        try:
            if LANGCHAIN_SEMANTIC_AVAILABLE:
                # Create embeddings wrapper
                embeddings = SentenceTransformerEmbeddings('nomic-ai/nomic-embed-text-v1.5')
                
                # Calculate min_chunk_size based on chunk_size
                min_chunk_size = max(500, self.chunk_size // 5)  # At least 500 chars, or 1/5 of chunk_size
                
                # Pass the embeddings wrapper into SemanticChunker
                self.semantic_chunker = LangChainSemanticChunker(
                    embeddings=embeddings, 
                    breakpoint_threshold_amount=self.threshold,
                    min_chunk_size=min_chunk_size
                )
                self.logger.info(f"LangChain semantic chunker ready with Nomic embeddings (chunk_size={chunk_size}, min_chunk_size={min_chunk_size})")
            else:
                self.semantic_chunker = None
                self.logger.warning("langchain_experimental not available, using fallback chunking")
        except Exception as e:
            self.logger.warning(f"Semantic chunker failed to initialize, using fallback: {e}")
            self.semantic_chunker = None
        
        # Initialize utilities
        self.page_mapper = PageMappingUtils(logger=self.logger)
        self.file_processor = FileProcessingUtils(logger=self.logger)
    
    def chunk_text(self, text: str, source_file: str) -> List[str]:
        """Split text into semantic chunks."""
        if self.semantic_chunker:
            try:
                # Create semantic chunks using LangChain's method
                docs = self.semantic_chunker.create_documents([text])   # returns List[Document]
                chunks = [d.page_content for d in docs]   # extract plain text
                
                if not chunks:
                    self.logger.warning(f"Semantic chunker produced no chunks for {source_file}. Using fallback.")
                    return self._fallback_chunker(text)

                self.logger.info(f"Created {len(chunks)} semantic chunks for {source_file}")
                return chunks
            except Exception as e:
                self.logger.error(f"Semantic chunking failed during execution for {source_file}: {e}. Using fallback.")
                return self._fallback_chunker(text)
        else:
            self.logger.info(f"Using fallback chunker for {source_file}.")
            return self._fallback_chunker(text)
    
    def _fallback_chunker(self, text: str) -> List[str]:
        """Robust fallback chunker that respects chunk_size and overlap parameters."""
        if not text.strip():
            return []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk_size
            test_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                    
                    # Handle overlap if configured
                    if self.overlap > 0 and len(current_chunk) > self.overlap:
                        overlap_text = current_chunk[-self.overlap:]
                        # Find a good break point in the overlap
                        break_point = overlap_text.rfind('. ')
                        if break_point > self.overlap // 2:
                            overlap_text = overlap_text[break_point + 2:]
                        current_chunk = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Paragraph is too long, split it by sentences
                    if len(paragraph) > self.chunk_size:
                        sentences = paragraph.split('. ')
                        temp_chunk = ""
                        for sentence in sentences:
                            test_sentence = f"{temp_chunk}. {sentence}" if temp_chunk else sentence
                            if len(test_sentence) <= self.chunk_size:
                                temp_chunk = test_sentence
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk)
                                temp_chunk = sentence
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # If no chunks were created, return the original text split to chunk_size
        if not chunks:
            words = text.split()
            current_chunk = ""
            for word in words:
                test_chunk = f"{current_chunk} {word}" if current_chunk else word
                if len(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks if chunks else [text]

  
    
    def process_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """Process single markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        source_name = file_path.stem
        chunks = self.chunk_text(text, source_name)
        
        return {
            "source_file": source_name,
            "total_chunks": len(chunks),
            "chunks": chunks  # No need for a list comprehension here
        }
    
    def process_output_directory(self, output_dir: Path, save_to: Path) -> Dict[str, Any]:
        """Process all markdown files in directory."""
        markdown_files = list(output_dir.rglob("*.md"))
        file_results = {}
        
        for md_file in markdown_files:
            file_result = self.process_markdown_file(md_file)
            file_results[file_result["source_file"]] = file_result
        
        total_chunks = sum(r["total_chunks"] for r in file_results.values())
        
        summary = {
            "summary": {
                "total_files_processed": len(file_results),
                "total_chunks_generated": total_chunks
            },
            "files": file_results
        }
        
        # Save results
        save_to.parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Processed {len(file_results)} files, {total_chunks} chunks")
        return summary