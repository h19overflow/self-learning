"""
Answering Node - Final node in the agentic workflow
"""

from ..state import AgenticLightRAGState
from ...agents.answering_agent.answering_agent import AnsweringAgent
from langchain_core.messages import HumanMessage, AIMessage
import weave
class AnsweringNode:
    """
    Node that generates educational answers using retrieved context and the original question.
    """
    
    def __init__(self):
        """Initialize the answering node with answering agent."""
        self.answering_agent = AnsweringAgent()
    
    async def process(self, state: AgenticLightRAGState) -> AgenticLightRAGState:
        """
        Process the question and context to generate an educational answer.
        
        Args:
            state: Current state with question and context
            
        Returns:
            AgenticLightRAGState: Updated state with final answer
        """
        try:
            print(f"[AnsweringNode] Generating answer for: {state.question}")
            print(f"🔍 [AnsweringNode] DEBUG: Messages type: {type(state.messages)}")
            print(f"🔍 [AnsweringNode] DEBUG: Messages length: {len(state.messages) if state.messages else 0}")
            
            # Ensure messages is initialized
            if state.messages is None:
                state.messages = []
            
            # Add human message to conversation
            human_question = HumanMessage(content=state.question)
            state.messages.append(human_question)
            
            # Check if we have context
            if not state.context:
                print("[AnsweringNode] No context available, using minimal context")
                state.context = f"Question context: {state.question}"
            
            # Generate educational answer using the answering agent
            print(f"🔍 [AnsweringNode] DEBUG: Calling answering agent with {len(state.messages)} messages")
            answer_analysis = await self.answering_agent.answer_question(
                question=state.question,
                context=state.context,
                messages=state.messages
            )
            
            # Update state with the final answer
            state.answer = answer_analysis.response.answer
            
            # Add AI response to conversation
            ai_message = AIMessage(content=state.answer)
            state.messages.append(ai_message)
            
            print(f"[AnsweringNode] Answer generated - Length: {len(state.answer)} characters")
            print(f"[AnsweringNode] Confidence: {answer_analysis.response.confidence_level}")
            
            if answer_analysis.response.learning_notes:
                print(f"[AnsweringNode] Learning notes available")
            
            return state
            
        except Exception as e:
            print(f"[AnsweringNode] Error during answer generation: {e}")
            
            # Provide fallback answer
            state.answer = f"""I apologize, but I encountered a technical issue while generating an answer for your question: "{state.question}". 

Please try rephrasing your question or asking again. The system is designed to provide educational explanations on a wide range of topics."""
            
            return state

# HELPER FUNCTIONS

async def run_answering_node(state: AgenticLightRAGState) -> AgenticLightRAGState:
    """
    Standalone function to run answering node.
    
    Args:
        state: Input state
        
    Returns:
        AgenticLightRAGState: Updated state
    """
    node = AnsweringNode()
    return await node.process(state)