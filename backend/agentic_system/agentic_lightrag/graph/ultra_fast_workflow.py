"""
Ultra-Fast Agentic LightRAG Workflow - Clean LangGraph Implementation

This workflow provides a clean graph structure with all node logic separated into individual files.
"""

from .state import AgenticLightRAGState
from .nodes.parameter_selection_node import ParameterSelectionNode
from .nodes.fast_retrieval_node import FastRetrievalNode
from .nodes.answering_node import AnsweringNode

import weave
from typing import AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END, START

weave.init('self_learning')


class UltraFastAgenticLightRAGWorkflow:
    """
    Ultra-fast LangGraph workflow with streaming capabilities and cached components.
    
    This class focuses purely on graph orchestration - all node logic is in separate files.
    """
    
    def __init__(self, cached_retriever=None, query_cache=None):
        """Initialize with cached components for maximum speed."""
        self.cached_retriever = cached_retriever
        self.query_cache = query_cache or {}
        
        # Initialize nodes with cached components
        self.parameter_selection_node = ParameterSelectionNode()
        self.fast_retrieval_node = FastRetrievalNode(cached_retriever)
        self.answering_node = AnsweringNode()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with clean node separation."""
        workflow = StateGraph(AgenticLightRAGState)
        
        # Add nodes - each node handles its own logic
        workflow.add_node("parameter_selection", self.parameter_selection_node.process)
        workflow.add_node("fast_retrieval", self.fast_retrieval_node.process)
        workflow.add_node("answer_generation", self.answering_node.process)
        
        # Define edges - simple linear flow
        workflow.add_edge(START, "parameter_selection")
        workflow.add_edge("parameter_selection", "fast_retrieval")  
        workflow.add_edge("fast_retrieval", "answer_generation")
        workflow.add_edge("answer_generation", END)
        
        return workflow.compile()
    
    async def process_question(self, question: str) -> AgenticLightRAGState:
        """
        Process a user question through the LangGraph workflow.
        
        Args:
            question: User's question to process
            
        Returns:
            AgenticLightRAGState: Final state with answer
        """
        print("=" * 50)
        print(f"⚡ ULTRA-FAST LANGGRAPH WORKFLOW")
        print("=" * 50)
        print(f"📝 Question: {question}")
        print("-" * 50)
       
        try:
            # Initialize state
            initial_state = AgenticLightRAGState(question=question, messages=[])
            
            # Run through the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            
            return final_state
            
        except Exception as e:
            print(f"\n❌ ULTRA-FAST ERROR: {type(e).__name__}: {e}")
            error_state = AgenticLightRAGState(
                question=question,
                answer=f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                messages=[]
            )
            return error_state
    
    async def stream_question(self, question: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a question with streaming updates for real-time responses.
        
        Args:
            question: User's question to process
            
        Yields:
            Dict[str, Any]: Streaming updates with partial results
        """
        try:
            # Initialize state
            initial_state = AgenticLightRAGState(question=question, messages=[])
            
            # Stream through the graph
            async for chunk in self.graph.astream(initial_state):
                # Extract the current node and state
                for node_name, node_state in chunk.items():
                    yield {
                        "node": node_name,
                        "state": node_state,
                        "question": question
                    }
                    
        except Exception as e:
            print(f"\n❌ STREAMING ERROR: {type(e).__name__}: {e}")
            yield {
                "node": "error",
                "state": {
                    "answer": f"I encountered an error: {str(e)}. Please try rephrasing your question."
                },
                "question": question
            }