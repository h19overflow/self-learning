"""
VLM Agent for analyzing images and generating content-focused descriptions
"""

import os
from pydantic_ai import Agent, RunContext  

from .vlm_schema import VLMDescription, VLMAnalysis, VLMDeps
from .vlm_prompt import VLM_SYSTEM_PROMPT, VLM_USER_PROMPT

from dotenv import load_dotenv
load_dotenv()

vlm_agent = Agent(
    model='gemini-2.5-flash',
    output_type=VLMDescription,
    deps_type=VLMDeps, 
)

@vlm_agent.system_prompt
def dynamic_system_prompt(ctx: RunContext[VLMDeps]) -> str:
    """
    Create system instructions for the VLM agent.
    
    Args:
        ctx: The context (not used for system prompt)
        
    Returns:
        str: The system prompt for VLM analysis
    """
    return VLM_SYSTEM_PROMPT

# No user_prompt decorator needed - we'll handle the user prompt in the run call

class VLMAgent:
    """
    VLM agent for analyzing images and generating educational descriptions.
    """
    
    def __init__(self):
        """
        Initialize the VLM agent.
        """
        # Set up API key
        os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        
    async def analyze_image(self, image_data: bytes, context: str, image_filename: str) -> VLMAnalysis:
        """
        Analyze an image and generate a comprehensive description.
        
        Args:
            image_data: The raw image data
            context: The document context surrounding the image
            image_filename: The filename for reference
            
        Returns:
            VLMAnalysis: Complete analysis with structured description
        """
        try:
            # Create dependencies
            deps = VLMDeps(
                image_data=image_data,
                context=context,
                image_filename=image_filename
            )
            
            print(f"      🤖 Analyzing image: {image_filename}")
            print(f"      📊 Image data size: {len(image_data)} bytes")
            print(f"      📝 Context preview: {context[:200]}{'...' if len(context) > 200 else ''}")
            
            # Run the agent analysis with image data
            # For pydantic-ai with images, use BinaryContent
            from pydantic_ai.messages import BinaryContent
            
            # Format the user prompt with context
            formatted_prompt = VLM_USER_PROMPT.format(
                context=context,
                image_filename=image_filename
            )
            
            user_content = [
                formatted_prompt,
                BinaryContent(data=image_data, media_type="image/jpeg")
            ]
            
            result = await vlm_agent.run(
                user_prompt=user_content,
                deps=deps
            )
            
            print(f"      ✅ Analysis complete for {image_filename}")
            
            # Package the results
            return VLMAnalysis(
                image_filename=image_filename,
                context_preview=context[:200] + ("..." if len(context) > 200 else ""),
                description=result.output,
                processing_success=True
            )
            
        except Exception as e:
            print(f"      ❌ Error analyzing {image_filename}: {e}")
            
            # Return error analysis
            return VLMAnalysis(
                image_filename=image_filename,
                context_preview=context,
                description=VLMDescription(
                    semantic_understanding=f"Error analyzing image {image_filename}: {e}",
                    content_interpretation="Analysis failed due to technical issues.",
                    knowledge_extraction="Unable to extract insights from image due to error.",
                    contextual_relevance="Image analysis could not be completed.",
                    accessible_description="Image description unavailable due to processing error.",
                    confidence_level="Low"
                ),
                processing_success=False
            )
    
    async def get_simple_description(self, image_data: bytes, context: str, image_filename: str) -> str:
        """
        Get a simple combined description (simplified interface).
        
        Args:
            image_data: The raw image data
            context: The document context
            image_filename: The filename
            
        Returns:
            str: Combined description text
        """
        analysis = await self.analyze_image(image_data, context, image_filename)
        
        if not analysis.processing_success:
            return analysis.description.semantic_understanding
        
        # Combine all sections into a comprehensive description
        desc = analysis.description
        combined = f"""## Image Analysis: {image_filename}

**Conceptual Understanding:**
{desc.semantic_understanding}

**Content Interpretation:**
{desc.content_interpretation}

**Key Insights:**
{desc.knowledge_extraction}

**Document Context:**
{desc.contextual_relevance}

**Summary:**
{desc.accessible_description}
"""
        return combined.strip()

# HELPER FUNCTIONS

def create_default_description(image_filename: str, error_msg: str = "") -> VLMDescription:
    """
    Create a default VLM description when analysis fails.
    
    Args:
        image_filename: The image filename
        error_msg: Optional error message
        
    Returns:
        VLMDescription: Default description
    """
    return VLMDescription(
        semantic_understanding=f"Unable to analyze image {image_filename}" + (f": {error_msg}" if error_msg else ""),
        content_interpretation="Image analysis could not be completed due to technical issues.",
        knowledge_extraction="No insights could be extracted from this image.",
        contextual_relevance="Image context analysis unavailable.",
        accessible_description="This image could not be processed for description generation.",
        confidence_level="Low"
    )

if __name__ == "__main__":
    import asyncio
    
    async def test_vlm_agent():
        """Test the VLMAgent with sample data."""
        agent = VLMAgent()
        
        # Sample test data
        sample_image_data = b"fake_image_data_for_testing"
        sample_context = "This is a research paper about machine learning architectures, specifically focusing on transformer models and attention mechanisms."
        sample_filename = "transformer_architecture.jpg"
        
        print("=== VLM Agent Test ===\n")
        
        try:
            # Full analysis
            analysis = await agent.analyze_image(
                image_data=sample_image_data,
                context=sample_context,
                image_filename=sample_filename
            )
            
            print(f"Filename: {analysis.image_filename}")
            print(f"Success: {analysis.processing_success}")
            print(f"Context: {analysis.context_preview}")
            print(f"Semantic Understanding: {analysis.description.semantic_understanding[:100]}...")
            print(f"Confidence: {analysis.description.confidence_level}")
            
            # Simple description
            simple_desc = await agent.get_simple_description(
                image_data=sample_image_data,
                context=sample_context,
                image_filename=sample_filename
            )
            print(f"Simple Description Length: {len(simple_desc)} characters")
            
        except Exception as e:
            print(f"Test Error: {e}")
    
    # Run the test
    asyncio.run(test_vlm_agent())