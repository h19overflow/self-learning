"""
Ultra-Fast Agentic LightRAG Workflow - Maximum Speed Version

This workflow prioritizes speed by minimizing AI calls and using aggressive caching.
"""

from .state import AgenticLightRAGState
from .nodes.answering_node import run_answering_node
from .nodes.fast_retrieval_node import run_fast_retrieval_node
from backend.storage.chromadb_instance.models.chroma_config import RetrievalConfig
import weave
import hashlib

weave.init('LIGHT_RAG_AGENTIC_SYSTEM')

class UltraFastAgenticLightRAGWorkflow:
    """
    Ultra-fast workflow that minimizes AI calls and uses aggressive caching.
    """
    
    def __init__(self, cached_retriever=None, query_cache=None):
        """Initialize with cached components for maximum speed."""
        self.cached_retriever = cached_retriever
        self.query_cache = query_cache or {}
    
    def _get_query_hash(self, question: str) -> str:
        """Get a hash for the question to use as cache key."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()[:12]
    
    def _determine_retrieval_params(self, question: str) -> RetrievalConfig:
        """Fast heuristic-based parameter selection instead of AI analysis."""
        question_lower = question.lower()
        
        # Simple heuristics for retrieval parameters
        if any(word in question_lower for word in ['what is', 'define', 'definition']):
            # Simple definitions - fewer results needed
            return RetrievalConfig(top_k=5, enable_reranking=True)
        elif any(word in question_lower for word in ['how', 'explain', 'describe']):
            # Explanations - moderate results
            return RetrievalConfig(top_k=8, enable_reranking=True)
        elif any(word in question_lower for word in ['compare', 'difference', 'vs', 'versus']):
            # Comparisons - more comprehensive results
            return RetrievalConfig(top_k=12, enable_reranking=True)
        else:
            # Default balanced approach
            return RetrievalConfig(top_k=8, enable_reranking=True)
    
    async def process_question(self, question: str) -> AgenticLightRAGState:
        """
        Process a user question through the ultra-fast workflow.
        
        Args:
            question: User's question to process
            
        Returns:
            AgenticLightRAGState: Final state with answer
        """
        print("=" * 50)
        print(f"⚡ ULTRA-FAST WORKFLOW")
        print("=" * 50)
        print(f"📝 Question: {question}")
        print("-" * 50)
       
        # Initialize state
        state = AgenticLightRAGState(question=question, messages=[])
        
        try:
            # Skip query analysis - use fast heuristics instead
            print("\n⚡ STEP 1: Fast Parameter Selection (No AI)")
            retrieval_config = self._determine_retrieval_params(question)
            print(f"   ✅ Heuristic Top-K: {retrieval_config.top_k}")
            
            # Step 2: Ultra-fast retrieval with cached components
            print("\n🚄 STEP 2: Ultra-Fast Retrieval")
            if self.cached_retriever:
                retrieval_results = self.cached_retriever.search(question, retrieval_config)
                
                if retrieval_results.has_results:
                    # Format context quickly
                    context_parts = []
                    for i, result in enumerate(retrieval_results.results, 1):
                        context_parts.append(
                            f"[Source {i}: {result.source_file} | Score: {result.score:.3f}]\n"
                            f"{result.content}\n"
                        )
                    
                    state.context = "\n" + "-" * 80 + "\n".join(context_parts)
                    state.sources = retrieval_results.unique_sources
                    
                    print(f"   ⚡ Retrieved {len(retrieval_results.results)} results in {retrieval_results.retrieval_time_ms:.1f}ms")
                else:
                    state.context = f"No relevant context found for query: '{question}'"
                    state.sources = []
            else:
                state.context = "No cached retriever available."
                state.sources = []
            
            # Step 3: Fast answer generation
            print("\n🎓 STEP 3: Fast Answer Generation")
            state = await run_answering_node(state)
            
            if state.answer:
                print(f"   ✅ Answer: {len(state.answer)} characters")
            
            print("\n" + "=" * 50)
            print("⚡ ULTRA-FAST COMPLETE")
            print("=" * 50)
            
            return state
            
        except Exception as e:
            print(f"\n❌ ULTRA-FAST ERROR: {type(e).__name__}: {e}")
            state.answer = f"I encountered an error: {str(e)}. Please try rephrasing your question."
            return state