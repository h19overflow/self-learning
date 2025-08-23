"""
Corrective Agent for analyzing insufficient context and rewriting queries intelligently
"""

import os
from pydantic_ai import Agent, RunContext  
from pydantic import BaseModel  
from typing import Optional, Dict, Any

from .corrective_schema import CorrectiveStrategy, CorrectiveAnalysis, CorrectionDeps
from .corrective_prompt import CORRECTIVE_ANALYSIS_PROMPT
from pydantic_ai.settings import ModelSettings
import weave

from dotenv import load_dotenv
load_dotenv()

corrective_agent = Agent(
    'gemini-2.0-flash',
    output_type=CorrectiveStrategy,
    deps_type=CorrectionDeps
)

@corrective_agent.system_prompt
def dynamic_system_prompt(ctx: RunContext[CorrectionDeps]) -> str:
    """
    Create custom instructions for the corrective agent.
    
    Args:
        ctx: The context containing the query, answer, and context to analyze
        
    Returns:
        str: The complete instructions for the AI agent
    """
    return CORRECTIVE_ANALYSIS_PROMPT.format(
        original_query=ctx.deps.original_query,
        insufficient_answer=ctx.deps.insufficient_answer,
        retrieved_context=ctx.deps.retrieved_context
    )

class CorrectiveAgent:
    """
    The main corrective agent class for analyzing why retrieval failed and rewriting queries.
    """
    
    def __init__(self):
        """
        Initialize the corrective agent.
        """
        pass
    @weave.op()
    async def analyze_and_correct(self, original_query: str, insufficient_answer: str, retrieved_context: str) -> CorrectiveAnalysis:
        """
        Analyze why the retrieval was insufficient and rewrite the query aggressively.
        
        Args:
            original_query: The original query that failed
            insufficient_answer: The insufficient answer received
            retrieved_context: The context that was retrieved but insufficient
            
        Returns:
            CorrectiveAnalysis: Complete analysis with corrected query and strategy
        """
        try:
            # Create dependencies
            deps = CorrectionDeps(
                original_query=original_query,
                insufficient_answer=insufficient_answer,
                retrieved_context=retrieved_context
            )
            
            # Run the agent analysis with temperature control
            result = await corrective_agent.run(
                "Analyze why the retrieval was insufficient and rewrite the query to get better results", 
                deps=deps,
                model_settings=ModelSettings(temperature=0.1)
            )
            
            print(f"[CorrectiveAgent] Analysis complete:")
            print(f"   Original Query: {original_query}")
            print(f"   Problem Identified: {result.output.problem_analysis}")
            print(f"   Rewritten Query: {result.output.rewritten_query}")
            print(f"   Strategy: {result.output.correction_strategy}")
            
            # Package the results
            return CorrectiveAnalysis(
                original_query=original_query,
                insufficient_answer=insufficient_answer,
                retrieved_context=retrieved_context,
                correction=result.output
            )
            
        except Exception as e:
            # Return fallback correction on error
            fallback_strategy = CorrectiveStrategy(
                problem_analysis=f"Error analyzing query: {e}. Using fallback query expansion.",
                rewritten_query=f"{original_query} including procedures, steps, processes, responsibilities, persons-in-charge, workflow, and detailed implementation",
                correction_strategy="Fallback: Broad query expansion with key business terms",
                confidence_level="Low",
                expected_improvement="May retrieve more relevant information through expanded terminology"
            )
            return CorrectiveAnalysis(
                original_query=original_query,
                insufficient_answer=insufficient_answer,
                retrieved_context=retrieved_context,
                correction=fallback_strategy
            )
    
    async def get_corrected_query(self, original_query: str, insufficient_answer: str, retrieved_context: str) -> str:
        """
        Get just the corrected query (simplified interface).
        
        Args:
            original_query: The original query that failed
            insufficient_answer: The insufficient answer received
            retrieved_context: The context that was retrieved but insufficient
            
        Returns:
            str: The rewritten query
        """
        analysis = await self.analyze_and_correct(original_query, insufficient_answer, retrieved_context)
        return analysis.correction.rewritten_query

# HELPER FUNCTIONS

def create_fallback_correction(original_query: str, error_msg: str = "") -> CorrectiveStrategy:
    """
    Create a fallback corrective strategy when analysis fails.
    
    Args:
        original_query: The original query
        error_msg: Optional error message to include
        
    Returns:
        CorrectiveStrategy: Fallback correction strategy
    """
    return CorrectiveStrategy(
        problem_analysis=f"Fallback correction for query: {original_query[:50]}..." + (f" Error: {error_msg}" if error_msg else ""),
        rewritten_query=f"{original_query} including detailed procedures, processes, responsibilities, persons-in-charge, workflows, and implementation steps",
        correction_strategy="Fallback: Broad expansion with business process terminology",
        confidence_level="Medium",
        expected_improvement="General query expansion may capture more relevant information"
    )

if __name__ == "__main__":
    import asyncio
    
    async def test_corrective_agent():
        """Test the CorrectiveAgent with various insufficient answer scenarios."""
        agent = CorrectiveAgent()
        
        test_cases = [
            {
                "original_query": "What are the procedures of SIS - TMG Critical process and their Person-In-Charge respectively?",
                "insufficient_answer": "The provided context does not contain specific procedures for the SIS - TMG Critical Process or designated Persons-In-Charge for each procedure, workflow, documentation, or escalation points.",
                "retrieved_context": "SIS - TMG Critical Process is a sales incentive framework. The scope includes approval and disbursement processes."
            }
        ]
        
        print("=== Corrective Agent Test Results ===\n")
        
        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['original_query']}")
            print("-" * 80)
            
            try:
                # Full analysis
                analysis = await agent.analyze_and_correct(
                    case['original_query'], 
                    case['insufficient_answer'],
                    case['retrieved_context']
                )
                print(f"Problem Analysis: {analysis.correction.problem_analysis}")
                print(f"Rewritten Query: {analysis.correction.rewritten_query}")
                print(f"Strategy: {analysis.correction.correction_strategy}")
                print(f"Confidence: {analysis.correction.confidence_level}")
                print(f"Expected Improvement: {analysis.correction.expected_improvement}")
                
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n" + "="*80 + "\n")
    
    # Run the test
    asyncio.run(test_corrective_agent())