"""
Retrieval Node - Second node in the agentic workflow

This node uses the modular ChromaRetriever for high-quality semantic search
with cross-encoder reranking and intelligent parameter mapping.
"""

from pathlib import Path
from ..state import AgenticLightRAGState
from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig, RetrievalConfig
from backend.storage.chromadb_instance.components.chroma_retriever import ChromaRetriever
import weave

        

class RetrievalNode:
    """
    Node that retrieves context from ChromaDB using intelligent query analysis parameters.
    
    Maps query analysis strategies to ChromaRetriever configurations for optimal results.
    """
    
    def __init__(self):
        """Initialize the retrieval node with ChromaRetriever."""
        self.retriever = None
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the ChromaRetriever with optimal configuration."""
        try:
            # Configure ChromaDB for the agentic system
            storage_path = Path(__file__).parent.parent.parent.parent.parent / "storage" / "chromadb_storage"
            
            config = ChromaConfig(
                persist_directory=storage_path,
                collection_name="academic_papers",
                embedding_model="BAAI/bge-large-en-v1.5",  # High-quality embeddings
                embedding_device="cuda",
                enable_reranking_by_default=True  # Enable reranking for better precision
            )
            
            self.retriever = ChromaRetriever(config)
            print(f"[RetrievalNode] ChromaRetriever initialized with {config.collection_name}")
            
        except Exception as e:
            print(f"[RetrievalNode] Failed to initialize ChromaRetriever: {e}")
            self.retriever = None
       
    async def process(self, state: AgenticLightRAGState) -> AgenticLightRAGState:
        """
        Process the query analysis and retrieve relevant context using ChromaRetriever.
        
        Args:
            state: Current state with query analysis
            
        Returns:
            AgenticLightRAGState: Updated state with retrieved context
        """
        try:
            # Check if retriever is available
            if not self.retriever:
                print("[RetrievalNode] ChromaRetriever not available, attempting to reinitialize")
                self._initialize_retriever()
                if not self.retriever:
                    raise Exception("ChromaRetriever initialization failed")
            
            # Check if query analysis is available
            if not state.query_analysis:
                print("[RetrievalNode] No query analysis found, using default parameters")
                retrieval_config = RetrievalConfig(top_k=5, enable_reranking=True)
                query = state.question
            else:
                # Use enhanced query if available
                query = state.query_analysis.enhanced_query or state.question
                strategy = state.query_analysis.strategy
                retrieval_config = RetrievalConfig(top_k=strategy.chunk_top_k,enable_reranking=True)
            # Perform retrieval with ChromaRetriever
            retrieval_results = self.retriever.search(query, retrieval_config)
            
            # Format context from retrieval results
            if retrieval_results.has_results:
                # Combine top results into context string
                context_parts = []
                for i, result in enumerate(retrieval_results.results, 1):
                    context_parts.append(
                        f"[Source {i}: {result.source_file} | Score: {result.score:.3f}]\n"
                        f"{result.content}\n"
                    )
                
                state.context = "\n" + "-" * 80 + "\n".join(context_parts)
                
                # Store unique source files in state
                state.sources = retrieval_results.unique_sources
                
                print(f"[RetrievalNode] Retrieved {len(retrieval_results.results)} results")
                print(f"[RetrievalNode] Average score: {retrieval_results.average_score:.3f}")
                print(f"[RetrievalNode] Context length: {len(state.context)} characters")
                print(f"[RetrievalNode] Retrieval time: {retrieval_results.retrieval_time_ms:.1f}ms")
                print(f"[RetrievalNode] Sources: {state.sources}")
            else:
                # No results found
                state.context = f"No relevant context found for query: '{query}'. The system will provide a general response."
                print(f"[RetrievalNode] No results found for query")
            
            return state
            
        except Exception as e:
            print(f"[RetrievalNode] Error during retrieval: {e}")
            import traceback
            print(f"[RetrievalNode] Full traceback: {traceback.format_exc()}")
            
            # Provide fallback context
            state.context = f"""Unable to retrieve specific context for the query: "{state.question}". 
            
This may be due to system connectivity issues or configuration problems. The system will attempt to provide a general response based on the question asked."""
            
            return state

# HELPER FUNCTIONS
@weave.op()
async def run_retrieval_node(state: AgenticLightRAGState) -> AgenticLightRAGState:
    """
    Standalone function to run ChromaRetriever-based retrieval node.
    
    Args:
        state: Input state with query analysis
        
    Returns:
        AgenticLightRAGState: Updated state with retrieved context
    """
    node = RetrievalNode()
    return await node.process(state)