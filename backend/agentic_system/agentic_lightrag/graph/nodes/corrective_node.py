"""
Corrective Node - Intelligently analyzes and rewrites queries when initial answer indicates insufficient context
"""

import re
from ..state import AgenticLightRAGState
from .retrieval_node import run_retrieval_node
from .query_analysis_node import run_query_analysis_node
from ...agents.corrective_agent.corrective_agent import CorrectiveAgent


class CorrectiveNode:
    """
    Node that detects insufficient context responses and intelligently rewrites queries for better retrieval.
    """
    
    def __init__(self):
        """Initialize the corrective node."""
        self.max_retries = 2
        self.corrective_agent = CorrectiveAgent()
    
    def _needs_correction(self, answer: str) -> bool:
        """
        Detect if the answer indicates insufficient context and needs correction.
        
        Args:
            answer: The generated answer to analyze
            
        Returns:
            bool: True if correction is needed
        """
        if not answer:
            return True
        
        # Patterns that indicate insufficient context
        insufficient_patterns = [
            r"no information found",
            r"not provided in the current context",
            r"context does not contain",
            r"unable to retrieve specific",
            r"not detailed in the context",
            r"information.*not available",
            r"context.*insufficient",
            r"not elaborated.*in.*context",
            r"cannot provide.*based on.*context",
            r"details.*are not provided",
            r"specific.*information.*not found"
        ]
        
        answer_lower = answer.lower()
        
        for pattern in insufficient_patterns:
            if re.search(pattern, answer_lower):
                return True
        
        return False
    
    async def _rewrite_query_intelligently(self, state: AgenticLightRAGState, retry_attempt: int) -> AgenticLightRAGState:
        """
        Use LLM to analyze why retrieval failed and rewrite the query intelligently.
        
        Args:
            state: Current state with insufficient context
            retry_attempt: Current retry attempt (1 or 2)
            
        Returns:
            AgenticLightRAGState: State with rewritten query
        """
        try:
            print(f"[CorrectiveNode] Analyzing insufficient context with LLM...")
            
            # Get the original query (before any enhancement)
            original_query = state.query_analysis.query if state.query_analysis else state.question
            
            # Analyze and get corrected query
            corrected_query = await self.corrective_agent.get_corrected_query(
                original_query=original_query,
                insufficient_answer=state.answer if hasattr(state, 'answer') else "Insufficient context retrieved",
                retrieved_context=state.context if state.context else "No context retrieved"
            )
            
            print(f"[CorrectiveNode] Query rewrite complete:")
            print(f"   Original: {original_query}")
            print(f"   Rewritten: {corrected_query}")
            
            # Update the state with the rewritten query (this will trigger new query analysis)
            state.question = corrected_query
            
            # Clear previous query analysis to force re-analysis with new query
            state.query_analysis = None
            
            # Re-run query analysis with the new query
            print(f"[CorrectiveNode] Re-analyzing rewritten query...")
            state = await run_query_analysis_node(state)
            
            if state.query_analysis:
                strategy = state.query_analysis.strategy
                print(f"[CorrectiveNode] New analysis complete:")
                print(f"   Mode: {strategy.mode}")
                print(f"   Top-K: {strategy.top_k}")
                print(f"   Chunk-Top-K: {strategy.chunk_top_k}")
            
            return state
            
        except Exception as e:
            print(f"[CorrectiveNode] Error in intelligent rewrite: {e}")
            print(f"[CorrectiveNode] Falling back to simple query expansion...")
            
            # Fallback to simple expansion
            original_query = state.question
            expanded_query = f"{original_query} including procedures, processes, steps, workflows, persons-in-charge, responsible roles, implementation details, and operational requirements"
            
            state.question = expanded_query
            state.query_analysis = None
            state = await run_query_analysis_node(state)
            
            return state
    
    async def process(self, state: AgenticLightRAGState, retry_attempt: int = 1) -> AgenticLightRAGState:
        """
        Process correction by intelligently rewriting the query and refetching.
        
        Args:
            state: Current state with insufficient answer
            retry_attempt: Current retry attempt number
            
        Returns:
            AgenticLightRAGState: Updated state with enhanced retrieval
        """
        try:
            print(f"[CorrectiveNode] Starting intelligent correction attempt {retry_attempt}")
            
            # Use LLM to analyze and rewrite the query intelligently
            state = await self._rewrite_query_intelligently(state, retry_attempt)
            
            # Re-run retrieval with the rewritten query
            print(f"[CorrectiveNode] Re-running retrieval with rewritten query...")
            state = await run_retrieval_node(state)
            
            if state.context:
                print(f"[CorrectiveNode] Enhanced retrieval completed:")
                print(f"   Retrieved: {len(state.context)} characters")
                print(f"   Preview: {state.context[:200]}...")
            else:
                print(f"[CorrectiveNode] Enhanced retrieval failed - no context retrieved")
            
            return state
            
        except Exception as e:
            print(f"[CorrectiveNode] Error during correction: {e}")
            return state


# HELPER FUNCTIONS

async def run_corrective_node(state: AgenticLightRAGState, retry_attempt: int = 1) -> AgenticLightRAGState:
    """
    Standalone function to run corrective node.
    
    Args:
        state: Input state
        retry_attempt: Current retry attempt number
        
    Returns:
        AgenticLightRAGState: Updated state
    """
    node = CorrectiveNode()
    return await node.process(state, retry_attempt)


def needs_correction(answer: str) -> bool:
    """
    Check if an answer needs correction due to insufficient context.
    
    Args:
        answer: The answer to check
        
    Returns:
        bool: True if correction is needed
    """
    node = CorrectiveNode()
    return node._needs_correction(answer)