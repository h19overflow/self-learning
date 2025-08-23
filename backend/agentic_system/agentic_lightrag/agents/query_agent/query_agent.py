"""
Query Agent for analyzing queries and determining optimal LightRAG retrieval parameters
"""

import os
from pydantic_ai import Agent, RunContext  
from pydantic import BaseModel  
from typing import Optional, Dict, Any

import weave

from backend.agentic_system.agentic_lightrag.agents.query_agent.query_schema import QueryStrategy, QueryAnalysis, QueryDeps
from backend.agentic_system.agentic_lightrag.agents.query_agent.query_prompt import QUERY_STRATEGY_PROMPT
from pydantic_ai.settings import ModelSettings

from dotenv import load_dotenv
load_dotenv()

query_agent = Agent(
    'gemini-2.0-flash',
    output_type=QueryStrategy,
    deps_type=QueryDeps
)

@query_agent.system_prompt
def dynamic_system_prompt(ctx: RunContext[QueryDeps]) -> str:
    """
    Create custom instructions for the query agent.
    
    Args:
        ctx: The context containing the query to analyze
        
    Returns:
        str: The complete instructions for the AI agent
    """
    return QUERY_STRATEGY_PROMPT.format(query=ctx.deps.query)

class QueryAgent:
    """
    The main query agent class for analyzing queries and determining optimal LightRAG parameters.
    """
    
    def __init__(self):
        """
        Initialize the query agent.
        """
    @weave.op()     
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query and determine optimal LightRAG parameters.
        
        Args:
            query: The user query to analyze
            
        Returns:
            QueryAnalysis: Complete analysis with query and strategy
        """
        try:
            # Create dependencies
            deps = QueryDeps(query=query)
            
            # Run the agent analysis with temperature control
            result = await query_agent.run(
                "Analyze this query and determine the optimal LightRAG mode and parameters", 
                deps=deps,
                model_settings=ModelSettings(temperature=0.1)
            )
            
            print(f"[QueryAgent] Original Query: {query}")
            print(f"[QueryAgent] Enhanced Query: {result.output.enhanced_query}")
            print(f"[QueryAgent] Reasoning: {result.output.reasoning}")
            
            # Package the results
            return QueryAnalysis(
                query=query,
                enhanced_query=result.output.enhanced_query,
                strategy=result.output
            )
            
        except Exception as e:
            # Return default strategy on error
            default_strategy = QueryStrategy(
                enhanced_query=query,  # Use original query as enhanced query fallback
                analysis=f"Error analyzing query: {e}. Using default hybrid strategy.",
                chunk_top_k=11,
                reasoning="Default fallback due to analysis error",
            )
            return QueryAnalysis(
                query=query,
                enhanced_query=default_strategy.enhanced_query,
                strategy=default_strategy
            )
    
    async def get_query_parameters(self, query: str) -> Dict[str, Any]:
        """
        Get just the query parameters for a query (simplified interface).
        
        Args:
            query: The user query to analyze
            
        Returns:
            Dict with mode, top_k and chunk_top_k parameters
        """
        analysis = await self.analyze_query(query)
        return {
            "chunk_top_k": analysis.strategy.chunk_top_k,
            "analysis": analysis.strategy.analysis,
            "reasoning": analysis.strategy.reasoning
        }


if __name__ == "__main__":
    import asyncio
    
    async def test_query_agent():
        """Test the QueryAgent with various types of queries."""
        agent = QueryAgent()
        
        test_queries = [
            "What is the capital of France?",
            "Tell me about Einstein's contributions to physics",
            "How do economic policies affect global climate change?", 
            "Compare machine learning approaches for natural language processing",
            "What are the key relationships between quantum mechanics and information theory?",
            "Explain the historical development of neural networks and their applications in modern AI systems"
        ]
        
        print("=== Query Agent Test Results ===\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Test {i}: {query}")
            print("-" * 80)
            
            try:
                # Full analysis
                analysis = await agent.analyze_query(query)
                print(f"Chunk Top-K: {analysis.strategy.chunk_top_k}")
                print(f"Analysis: {analysis.strategy.analysis}")
                print(f"Reasoning: {analysis.strategy.reasoning}")
                
                # Simplified parameters
                params = await agent.get_query_parameters(query)
                print(f"Simplified Params: {params}")
                
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n" + "="*80 + "\n")
    
    # Run the test
    asyncio.run(test_query_agent()) 