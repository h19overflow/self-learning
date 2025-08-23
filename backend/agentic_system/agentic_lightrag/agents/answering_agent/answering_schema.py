from pydantic import BaseModel, Field
from typing import Optional,List
from langchain_core.messages import BaseMessage


class AnsweringDeps(BaseModel):
    """
    Dependencies for the answering agent.
    """
    question: str  # The user's question
    context: str   # The context to answer from
    messages:List[BaseMessage] = []  # Conversation history for context-aware responses


class EducationalResponse(BaseModel):
    """Flexible, well-formatted response that adapts to any question type."""
    
    answer: str = Field(..., description="Well-structured answer using clear formatting like headers, bullet points, numbered lists, or sections as appropriate for the content. Should be readable and organized, not a single paragraph.")
    confidence_level: str = Field(..., description="Confidence level in the answer (High/Medium/Low)")  
    learning_notes: Optional[str] = Field(None, description="Additional context, caveats, or helpful notes if relevant")
    

class AnswerAnalysis(BaseModel):
    """Complete analysis with question, context, and factual response."""
    
    question: str = Field(..., description="The original user question")
    context_used: str = Field(..., description="The context provided for answering")
    response: EducationalResponse = Field(..., description="The direct factual response")