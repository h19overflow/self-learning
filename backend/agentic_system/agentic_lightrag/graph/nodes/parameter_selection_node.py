"""
Parameter Selection Node - Fast Heuristic-Based Parameter Selection

This node uses simple heuristics instead of AI analysis for ultra-fast parameter selection.
"""

from typing import Dict, Any
from ..state import AgenticLightRAGState
from backend.storage.chromadb_instance.models.chroma_config import RetrievalConfig
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from ...agents.query_agent.query_schema import QueryAnalysis, QueryStrategy


class ParameterSelectionNode:
    """
    Node that performs fast heuristic-based parameter selection for retrieval.
    """
    
    def __init__(self):
        """Initialize the parameter selection node."""
        pass
    
    def _determine_retrieval_params(self, question: str) -> RetrievalConfig:
        """Fast heuristic-based parameter selection instead of AI analysis."""
        question_lower = question.lower()
        
        # Simple heuristics for retrieval parameters
        if any(word in question_lower for word in ['what is', 'define', 'definition']):
            return RetrievalConfig(top_k=5, enable_reranking=True)
        elif any(word in question_lower for word in ['how', 'explain', 'describe']):
            return RetrievalConfig(top_k=8, enable_reranking=True)
        elif any(word in question_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return RetrievalConfig(top_k=12, enable_reranking=True)
        else:
            return RetrievalConfig(top_k=8, enable_reranking=True)
    
    async def process(self, state: AgenticLightRAGState) -> Dict[str, Any]:
        """
        Process fast heuristic parameter selection.
        
        Args:
            state: Current state with user question
            
        Returns:
            Dict[str, Any]: Updated state fields
        """
        print("\n⚡ STEP 1: Fast Parameter Selection (No AI)")
        
        # Ensure messages are initialized
        if not state.messages:
            state.messages = []
        
        # Add human message to conversation
        human_message = HumanMessage(content=state.question)
        updated_messages = add_messages(state.messages, [human_message])
        
        # Determine retrieval config using heuristics
        retrieval_config = self._determine_retrieval_params(state.question)
        print(f"   ✅ Heuristic Top-K: {retrieval_config.top_k}")
        
        # Create proper QueryAnalysis and QueryStrategy instances
        query_strategy = QueryStrategy(
            enhanced_query=state.question,
            analysis=f"Heuristic analysis: determined top_k={retrieval_config.top_k} based on question type",
            chunk_top_k=retrieval_config.top_k,
            reasoning=f"Selected {retrieval_config.top_k} chunks based on question pattern analysis"
        )
        
        query_analysis = QueryAnalysis(
            query=state.question,
            enhanced_query=state.question,
            strategy=query_strategy
        )
        
        return {
            "query_analysis": query_analysis, 
            "messages": updated_messages
        }


# HELPER FUNCTIONS

async def run_parameter_selection_node(state: AgenticLightRAGState) -> Dict[str, Any]:
    """
    Standalone function to run parameter selection node.
    
    Args:
        state: Input state
        
    Returns:
        Dict[str, Any]: Updated state fields
    """
    node = ParameterSelectionNode()
    return await node.process(state)