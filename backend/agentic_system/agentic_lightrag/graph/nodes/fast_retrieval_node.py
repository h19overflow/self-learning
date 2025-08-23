"""
Fast Retrieval Node - Optimized for Speed

This node bypasses expensive model loading by using pre-cached components.
"""

from typing import Dict, Any
from backend.agentic_system.agentic_lightrag.graph.state import AgenticLightRAGState
from backend.storage.chromadb_instance.models.chroma_config import RetrievalConfig
import weave

class FastRetrievalNode:
    """
    Fast retrieval node that uses pre-cached ChromaRetriever.
    """
    
    def __init__(self, cached_retriever=None):
        """Initialize with cached retriever for speed."""
        self.cached_retriever = cached_retriever
    
    async def process(self, state: AgenticLightRAGState) -> Dict[str, Any]:
        """
        Process retrieval using cached components for maximum speed.
        
        Args:
            state: Current state with user question
            
        Returns:
            Dict[str, Any]: Updated state fields with context and sources
        """
        if not self.cached_retriever:
            print("[FastRetrievalNode] No cached retriever available, skipping retrieval")
            return {
                "context": "No cached retriever available for fast retrieval.",
                "sources": []
            }
        
        try:
            print(f"[FastRetrievalNode] Fast retrieval for: {state.question[:50]}...")
            
            # Use query analysis if available, otherwise use defaults
            if not state.query_analysis:
                retrieval_config = RetrievalConfig(top_k=8, enable_reranking=True)
                query = state.question
            else:
                query = state.query_analysis.enhanced_query or state.question
                strategy = state.query_analysis.strategy
                retrieval_config = RetrievalConfig(top_k=strategy.chunk_top_k, enable_reranking=True)
            
            # Perform fast retrieval with cached components
            retrieval_results = self.cached_retriever.search(query, retrieval_config)
            
            # Format context from retrieval results
            if retrieval_results.has_results:
                # Combine top results into context string
                context_parts = []
                for i, result in enumerate(retrieval_results.results, 1):
                    context_parts.append(
                        f"[Source {i}: {result.source_file} | Score: {result.score:.3f}]\n"
                        f"{result.content}\n"
                    )
                
                context = "\n" + "-" * 80 + "\n".join(context_parts)
                sources = retrieval_results.unique_sources
                
                print(f"[FastRetrievalNode] Retrieved {len(retrieval_results.results)} results in {retrieval_results.retrieval_time_ms:.1f}ms")
                
                return {
                    "context": context,
                    "sources": sources
                }
            else:
                print(f"[FastRetrievalNode] No results found")
                return {
                    "context": f"No relevant context found for query: '{query}'",
                    "sources": []
                }
            
        except Exception as e:
            print(f"[FastRetrievalNode] Error during fast retrieval: {e}")
            return {
                "context": f"Fast retrieval failed: {e}",
                "sources": []
            }

# HELPER FUNCTIONS
@weave.op()
async def run_fast_retrieval_node(state: AgenticLightRAGState, cached_retriever=None) -> Dict[str, Any]:
    """
    Standalone function to run fast retrieval node with cached components.
    
    Args:
        state: Input state
        cached_retriever: Pre-loaded ChromaRetriever instance
        
    Returns:
        Dict[str, Any]: Updated state fields
    """
    node = FastRetrievalNode(cached_retriever)
    return await node.process(state)