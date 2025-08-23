"""
Schema definitions for the corrective agent
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class CorrectiveStrategy(BaseModel):
    """Strategy for correcting insufficient retrieval"""
    
    problem_analysis: str = Field(
        ...,
        description="Analysis of why the original query failed to retrieve sufficient context"
    )
    
    rewritten_query: str = Field(
        ...,
        description="Aggressively rewritten query designed to capture the missing information"
    )
    
    correction_strategy: str = Field(
        ...,
        description="Description of the correction approach used (terminology expansion, semantic variation, etc.)"
    )
    
    confidence_level: str = Field(
        ...,
        description="Confidence level in the correction (High, Medium, Low)"
    )
    
    expected_improvement: str = Field(
        ...,
        description="What improvement is expected from this query rewrite"
    )


class CorrectiveAnalysis(BaseModel):
    """Complete analysis result from corrective agent"""
    
    original_query: str = Field(
        ...,
        description="The original query that failed"
    )
    
    insufficient_answer: str = Field(
        ...,
        description="The insufficient answer that triggered correction"
    )
    
    retrieved_context: str = Field(
        ...,
        description="The context that was retrieved but insufficient"
    )
    
    correction: CorrectiveStrategy = Field(
        ...,
        description="The correction strategy and rewritten query"
    )


class CorrectionDeps(BaseModel):
    """Dependencies for corrective agent"""
    
    original_query: str = Field(
        ...,
        description="The original query that failed to retrieve sufficient context"
    )
    
    insufficient_answer: str = Field(
        ...,
        description="The insufficient answer that was generated"
    )
    
    retrieved_context: str = Field(
        ...,
        description="The context that was retrieved but found insufficient"
    )