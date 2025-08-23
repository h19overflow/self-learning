"""
Graph State for Agentic LightRAG System with Conversation Memory
"""

from pydantic import BaseModel, Field
from typing import Optional, Annotated,List
from ..agents.query_agent.query_schema import QueryAnalysis
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgenticLightRAGState(BaseModel):
    """
    Enhanced state for the Agentic LightRAG workflow with conversation memory.
    
    This state manages the flow from question analysis through retrieval to final answer,
    while maintaining conversation history for context-aware follow-ups.
    """
    
    question: str = Field(..., description="The user's current question")
    answer: Optional[str] = Field(None, description="The final educational answer")
    query_analysis: Optional[QueryAnalysis] = Field(None, description="Analysis and parameters from query agent")
    context: Optional[str] = Field(None, description="Retrieved context from knowledge base")
    # Source information
    sources: Optional[List[str]] = Field(None, description="List of source files used for answer")
    # Conversation memory with reducer
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list, 
        description="Conversation history with reducer for message management"
    )
    # Session management
    session_id: Optional[str] = Field(None, description="Unique session identifier for conversation tracking")
    
   
