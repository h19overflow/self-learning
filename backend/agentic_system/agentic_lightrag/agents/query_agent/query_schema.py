from pydantic import BaseModel, Field
from typing import Literal, Optional


class QueryDeps(BaseModel):
    """
    Dependencies for the query agent.
    """
    query: str  # The user query to analyze


class QueryStrategy(BaseModel):
    """Strategy analysis and parameters for LightRAG query execution."""
    
    enhanced_query: str = Field(..., description="The enhanced version of the user query")
    analysis: str = Field(..., description="Analysis of the query characteristics and requirements")
    chunk_top_k: int = Field(..., description="Number of text chunks to retrieve and rerank")
    reasoning: str = Field(..., description="Justification for parameter selections")

class QueryAnalysis(BaseModel):
    """Complete query analysis with original query and strategy."""
    
    query: str = Field(..., description="The original user query")
    enhanced_query: str = Field(..., description="The enhanced version of the user query")
    strategy: QueryStrategy = Field(..., description="The determined strategy and parameters")
