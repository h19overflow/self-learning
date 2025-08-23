"""
Educational Answering Agent for providing tutoring-style responses to user questions
"""

import os
from typing import List
from pydantic_ai import Agent, RunContext  
from pydantic import BaseModel  
from langchain_core.messages import BaseMessage
import weave
from .answering_schema import EducationalResponse, AnswerAnalysis, AnsweringDeps
from .answering_prompt import EDUCATIONAL_TUTORING_PROMPT
from typing import List
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from pydantic_ai.settings import ModelSettings
load_dotenv()

answering_agent = Agent(
    'gemini-2.0-flash',
    output_type=EducationalResponse,
    deps_type=AnsweringDeps
)

@answering_agent.system_prompt
def dynamic_system_prompt(ctx: RunContext[AnsweringDeps]) -> str:
    """
    Create custom instructions for the answering agent.
    
    Args:
        ctx: The context containing the question and context to use
        
    Returns:
        str: The complete instructions for the AI agent
    """
    # Format messages for display in prompt
    messages_text = ""
    if ctx.deps.messages:
        messages_text = "## Conversation History:\\n"
        for msg in ctx.deps.messages[-5:]:  # Last 5 messages for context
            role = msg.__class__.__name__.replace("Message", "").lower()
            messages_text += f"**{role.capitalize()}**: {msg.content}\\n"
        messages_text += "\\n"
    else:
        messages_text = "## New Conversation\\n\\n"
    
    return EDUCATIONAL_TUTORING_PROMPT.format(
        question=ctx.deps.question,
        context=ctx.deps.context,
        messages=messages_text
    )

class AnsweringAgent:
    """
    Educational answering agent that provides tutoring-style responses.
    """
    
    def __init__(self):
        """
        Initialize the answering agent.
        """
        # Set up API key
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    @weave.op() 
    async def answer_question(self, question: str, context: str, messages: List[BaseMessage] = None) -> AnswerAnalysis:
        """
        Answer a user question based on provided context in an educational manner.
        
        Args:
            question: The user's question
            context: The context to answer from
            messages: Previous conversation history for context-aware responses
            
        Returns:
            AnswerAnalysis: Complete analysis with question, context, and response
        """
        try:
            # Ensure messages is a list
            if messages is None:
                messages = []
            
            # Create dependencies with proper message formatting
            deps = AnsweringDeps(question=question, context=context, messages=messages)
            
            # Run the agent with temperature control
            result = await answering_agent.run(
                "Provide an educational answer to this question using the given context", 
                deps=deps,
                model_settings=ModelSettings(temperature=0.1)
            )
            
            # Package the results
            return AnswerAnalysis(
                question=question,
                context_used=context,
                response=result.output
            )
            
        except Exception as e:
            # Return default response on error
            return AnswerAnalysis(
                question=question,
                context_used=context,
                response=EducationalResponse(
                    answer=f"I apologize, but I encountered an error while processing your question: {e}. Please try rephrasing your question or provide more context.",
                    confidence_level="Low",
                    learning_notes="Unable to process the question due to technical issues."
                )
            )
    
    async def get_simple_answer(self, question: str, context: str) -> str:
        """
        Get just the answer text (simplified interface).
        
        Args:
            question: The user's question
            context: The context to answer from
            
        Returns:
            str: The educational answer
        """
        analysis = await self.answer_question(question, context)
        return analysis.response.answer

# HELPER FUNCTIONS

def create_default_response(question: str, context: str, error_msg: str = "") -> EducationalResponse:
    """
    Create a default educational response when analysis fails.
    
    Args:
        question: The original question
        context: The provided context
        error_msg: Optional error message
        
    Returns:
        EducationalResponse: Default response
    """
    return EducationalResponse(
        answer=f"I understand you're asking about: {question[:100]}..." + 
               (f" However, I encountered an issue: {error_msg}" if error_msg else 
                " I'll do my best to help based on the available information."),
        confidence_level="Medium",
        learning_notes="This is a general response. For more specific help, please provide additional context or rephrase your question."
    )

# if __name__ == "__main__":
#     import asyncio
    
#     async def test_answering_agent():
#         """Test the AnsweringAgent with various educational scenarios."""
#         agent = AnsweringAgent()
        
#         test_cases = [
#             {
#                 "question": "What is photosynthesis?",
#                 "context": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. It occurs primarily in the leaves using chlorophyll."
#             },
#             {
#                 "question": "How does machine learning work?",
#                 "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions."
#             },
#             {
#                 "question": "Explain the quarterly revenue trends",
#                 "context": "Q1 revenue was $2.5M, Q2 was $3.1M showing 24% growth, Q3 reached $3.8M with 23% growth, and Q4 closed at $4.2M with 11% growth. The company shows consistent growth throughout the year."
#             }
#         ]
        
#         print("=== Educational Answering Agent Test Results ===\n")
        
#         for i, case in enumerate(test_cases, 1):
#             print(f"Test {i}: {case['question']}")
#             print("-" * 60)
#             print(f"Context: {case['context'][:100]}...")
#             print()
            
#             try:
#                 # Full analysis
#                 analysis = await agent.answer_question(case['question'], case['context'])
#                 print(f"Answer: {analysis.response.answer}")
#                 print(f"Confidence: {analysis.response.confidence_level}")
#                 if analysis.response.learning_notes:
#                     print(f"Learning Notes: {analysis.response.learning_notes}")
                
#                 # Simple answer
#                 simple = await agent.get_simple_answer(case['question'], case['context'])
#                 print(f"Simple Answer Length: {len(simple)} characters")
                
#             except Exception as e:
#                 print(f"Error: {e}")
            
#             print("\n" + "="*60 + "\n")
    
#     # Run the test
#     asyncio.run(test_answering_agent())