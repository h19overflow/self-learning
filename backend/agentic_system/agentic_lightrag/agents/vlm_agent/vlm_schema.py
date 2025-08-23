from pydantic import BaseModel, Field
from typing import Optional


class VLMDeps(BaseModel):
    """
    Dependencies for the VLM agent.
    """
    image_data: bytes  # The image data to analyze
    context: str       # The document context surrounding the image
    image_filename: str  # The filename for reference


class VLMDescription(BaseModel):
    """Enhanced image description from the VLM agent."""
    
    semantic_understanding: str = Field(..., description="What the image represents conceptually and its main purpose")
    content_interpretation: str = Field(..., description="Processes, concepts, relationships, or systems being shown")
    knowledge_extraction: str = Field(..., description="Key takeaways, patterns, and insights from the image")
    contextual_relevance: str = Field(..., description="How the image fits within the document's narrative")
    accessible_description: str = Field(..., description="Clear, comprehensive explanation for readers")
    confidence_level: str = Field(..., description="Confidence level in the analysis (High/Medium/Low)")
    

class VLMAnalysis(BaseModel):
    """Complete VLM analysis with image context and description."""
    
    image_filename: str = Field(..., description="The image filename")
    context_preview: str = Field(..., description="Preview of the document context")
    description: VLMDescription = Field(..., description="The detailed VLM analysis")
    processing_success: bool = Field(..., description="Whether the analysis was successful")