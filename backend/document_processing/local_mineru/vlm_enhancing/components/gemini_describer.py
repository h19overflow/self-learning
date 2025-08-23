"""
Gemini VLM integration component - Uses VLMAgent exclusively.

This module handles image description generation using the VLMAgent
for structured, high-quality image analysis.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import sys

# Import models
sys.path.append(str(Path(__file__).parent.parent))
from models import ImageContext, DescriptionResult

# Import VLM Agent
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from agentic_lightrag.agents.vlm_agent.vlm_agent import VLMAgent


class GeminiDescriber:
    """
    Enhanced image description generator using VLMAgent.
    
    This class uses the new VLMAgent exclusively for structured,
    high-quality image analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-lite"):
        """
        Initialize the Gemini describer.
        
        Args:
            api_key: Gemini API key (will load from environment if not provided)
            model: Model name to use (defaults to gemini-2.5-flash-lite)
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be provided or set in environment")
        
        # Initialize VLM Agent
        print("🤖 Initializing VLMAgent for image analysis")
        self.vlm_agent = VLMAgent()
    
    async def describe_image(self, image_context: ImageContext) -> DescriptionResult:
        """
        Generate a description for an image using its context.
        
        Args:
            image_context: ImageContext object containing image and surrounding text
            
        Returns:
            DescriptionResult with the generated description or error information
        """
        image_path = image_context.image_ref.image_path
        filename = image_context.image_ref.filename
        print(f"      🖼️  Processing image: {filename}")
        
        # Validate image exists
        if not image_context.image_ref.exists():
            error_msg = f"Image file not found: {image_path}"
            print(f"      ❌ {error_msg}")
            return DescriptionResult.failed(error_msg)
        
        try:
            # Read image data
            print(f"      📖 Reading image data...")
            image_data = image_context.image_ref.image_path.read_bytes()
            print(f"      📊 Image size: {len(image_data)} bytes")
            
            # Prepare context
            prompt_context = image_context.create_prompt_context()
            print(f"      📝 Context preview: {prompt_context[:100]}{'...' if len(prompt_context) > 100 else ''}")
            
            # Generate description using VLMAgent
            print(f"      🤖 Analyzing with VLMAgent...")
            description = await self.vlm_agent.get_simple_description(
                image_data=image_data,
                context=prompt_context,
                image_filename=filename
            )
            
            if description and description.strip():
                print(f"      ✅ VLMAgent analysis complete ({len(description)} chars)")
                return DescriptionResult.success(description.strip())
            else:
                error_msg = "Empty response from VLMAgent"
                print(f"      ❌ {error_msg}")
                return DescriptionResult.failed(error_msg)
                
        except Exception as e:
            error_msg = f"Error processing image {filename}: {str(e)}"
            print(f"      ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return DescriptionResult.failed(error_msg)
    
    async def describe_multiple_images(self, image_contexts: list[ImageContext]) -> list[tuple[ImageContext, DescriptionResult]]:
        """
        Generate descriptions for multiple images concurrently.
        
        Args:
            image_contexts: List of ImageContext objects
            
        Returns:
            List of tuples containing (ImageContext, DescriptionResult)
        """
        if not image_contexts:
            return []
        
        # Process images concurrently with reasonable limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_single_image(context: ImageContext) -> tuple[ImageContext, DescriptionResult]:
            async with semaphore:
                print(f"🤖 Processing {context.image_ref.filename}...")
                result = await self.describe_image(context)
                
                # Log result
                if hasattr(result, 'success') and result.success:
                    print(f"   ✅ {context.image_ref.filename}: Success")
                else:
                    error_msg = getattr(result, 'error_message', 'Unknown error')
                    print(f"   ❌ {context.image_ref.filename}: Failed - {error_msg}")
                
                return context, result
        
        # Execute all tasks concurrently
        print(f"🚀 Starting VLMAgent batch processing of {len(image_contexts)} images...")
        tasks = [process_single_image(context) for context in image_contexts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        success_count = 0
        error_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_count += 1
                print(f"❌ Task {i+1} exception: {str(result)}")
                error_result = DescriptionResult.failed(f"VLMAgent processing failed: {str(result)}")
                processed_results.append((image_contexts[i], error_result))
            else:
                context, desc_result = result
                if hasattr(desc_result, 'success') and desc_result.success:
                    success_count += 1
                else:
                    error_count += 1
                processed_results.append(result)
        
        print(f"📊 VLMAgent batch processing complete: {success_count} successful, {error_count} failed")
        return processed_results

# HELPER FUNCTIONS

def create_describer(api_key: Optional[str] = None) -> GeminiDescriber:
    """
    Factory function to create a GeminiDescriber instance.
    
    Args:
        api_key: Optional API key override
        
    Returns:
        GeminiDescriber: Configured describer instance
    """
    return GeminiDescriber(api_key=api_key)

# Backward compatibility alias
GeminiVLM = GeminiDescriber