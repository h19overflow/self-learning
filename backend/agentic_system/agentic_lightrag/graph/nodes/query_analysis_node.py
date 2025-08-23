"""
Query Analysis Node - First node in the agentic workflow
"""

from backend.agentic_system.agentic_lightrag.graph.state import AgenticLightRAGState
from backend.agentic_system.agentic_lightrag.agents.query_agent.query_agent import QueryAgent
import weave
import asyncio

class QueryAnalysisNode:
    """
    Node that analyzes the user question and determines optimal retrieval parameters.
    """
    
    def __init__(self):
        """Initialize the query analysis node with query agent."""
        self.query_agent = QueryAgent()
    
    async def process(self, state: AgenticLightRAGState) -> AgenticLightRAGState:
        """
        Process the user question and populate query analysis in state.
        
        Args:
            state: Current state with user question
            
        Returns:
            AgenticLightRAGState: Updated state with query analysis
        """
        try:
            print(f"[QueryAnalysisNode] Processing question: {state.question}")
            
            # Analyze the question using query agent
            query_analysis = await self.query_agent.analyze_query(state.question)
            
            # Update state with analysis results
            state.query_analysis = query_analysis
            state.question = query_analysis.enhanced_query
            
            print(f"[QueryAnalysisNode] Original Query: {query_analysis.query}")
            print(f"[QueryAnalysisNode] Enhanced Query: {query_analysis.enhanced_query}")
            print(f"Chunk-Top-K: {query_analysis.strategy.chunk_top_k}")
            
            return state
            
        except Exception as e:
            print(f"[QueryAnalysisNode] Error during analysis: {e}")
            # Return state unchanged if analysis fails
            return state

# HELPER FUNCTIONS
async def run_query_analysis_node(state: AgenticLightRAGState) -> AgenticLightRAGState:
    """
    Standalone function to run query analysis node.
    
    Args:
        state: Input state
        
    Returns:
        AgenticLightRAGState: Updated state
    """
    node = QueryAnalysisNode()
    return await node.process(state)