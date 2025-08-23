"""
Ultra-Fast Gradio Chat Interface for Agentic LightRAG

This interface uses cached components and optimized processing for maximum speed.
No async dependencies, with pre-loaded models and heuristic-based query processing.
"""

import gradio as gr
import time
from backend.agentic_system.agentic_lightrag.graph.ultra_fast_workflow import UltraFastAgenticLightRAGWorkflow


class SimplifiedWorkflowManager:
    """Clean synchronous interface with persistent event loop and cached components."""
    
    def __init__(self):
        self.workflow = None
        self.stats = {
            "total_requests": 0,
            "total_time": 0.0,
            "avg_response_time": 0.0
        }
        # Create a persistent event loop for async operations
        import asyncio
        import threading
        self.loop = None
        self.loop_thread = None
        self._start_event_loop()
        
        # Cache for expensive components
        self._cached_retriever = None
        self._cached_embeddings = None
        self._query_cache = {}  # Simple cache for query analysis
        self._initialize_cached_components()
    
    def _initialize_cached_components(self):
        """Initialize and cache expensive components for reuse."""
        try:
            print("🚀 Pre-loading expensive components for faster responses...")
            
            # Pre-load ChromaDB connection and models
            from backend.storage.chromadb_instance.components.chroma_retriever import ChromaRetriever
            from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig
            from pathlib import Path
            
            # Create config for ChromaDB (matching original retrieval node)
            storage_path = Path(__file__).parent.parent.parent / "storage" / "chromadb_storage"
            chroma_config = ChromaConfig(
                persist_directory=storage_path,
                collection_name="academic_papers",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_device="cuda",
                enable_reranking_by_default=True
            )
            
            # Initialize and cache retriever (this loads models once)
            self._cached_retriever = ChromaRetriever(chroma_config)
            print("✅ ChromaDB retriever cached and ready")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not pre-load components: {e}")
            self._cached_retriever = None
    
    def _start_event_loop(self):
        """Start a persistent event loop in a background thread."""
        import asyncio
        import threading
        
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # Wait for the loop to be ready
        while self.loop is None:
            time.sleep(0.01)
    
    def _run_async_task(self, coro):
        """Run an async coroutine in the persistent event loop."""
        import asyncio
        import concurrent.futures
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
    
    def get_workflow(self):
        """Get or create workflow instance (reuse the same instance)."""
        if self.workflow is None:
            print("🔄 Creating Ultra-Fast AgenticLightRAG workflow instance...")
            self.workflow = UltraFastAgenticLightRAGWorkflow(
                cached_retriever=self._cached_retriever,
                query_cache=self._query_cache
            )
        return self.workflow
    
    def process_question(self, question: str) -> str:
        """Process question using synchronous workflow execution."""
        start_time = time.time()
        
        try:
            print(f"🔍 Processing question synchronously: {question[:100]}...")
            
            # Get workflow instance
            workflow = self.get_workflow()
            
            # Call async workflow using persistent event loop
            result_state = self._run_async_task(workflow.process_question(question))
            
            # Extract the answer and sources
            if hasattr(result_state, 'answer') and result_state.answer:
                response = result_state.answer
                
                # Add source information if available
                if hasattr(result_state, 'sources') and result_state.sources:
                    sources_text = "\n\n---\n**📚 Sources:**\n"
                    for i, source in enumerate(result_state.sources, 1):
                        sources_text += f"{i}. {source}\n"
                    response += sources_text
            else:
                response = "I apologize, but I couldn't generate an answer for your question."
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_time"] += processing_time
            self.stats["avg_response_time"] = self.stats["total_time"] / self.stats["total_requests"]
            
            print(f"⚡ Request processed in {processing_time:.2f}s (avg: {self.stats['avg_response_time']:.2f}s)")
            
            return response
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            import traceback
            traceback.print_exc()
            return f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."


# Global workflow manager
workflow_manager = SimplifiedWorkflowManager()


def process_message(message, history):
    """
    Synchronous function to process user messages.
    No async wrappers - just simple synchronous execution.
    
    Args:
        message: User input message
        history: Chat history (not used in current implementation)
    
    Returns:
        str: Generated response from the workflow
    """
    try:
        print(f"🚀 GRADIO: Processing question: {message[:100]}...")
        
        # Use the simplified synchronous method
        response = workflow_manager.process_question(message)
        
        print(f"✅ GRADIO: Answer generated - {len(response)} characters")
        
        return response
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        print(f"❌ GRADIO: Error processing message: {e}")
        print(f"❌ GRADIO: Error type: {type(e).__name__}")
        
        # Return a user-friendly error message
        return f"{error_msg}. Please try rephrasing your question or provide more context."


def create_interface():
    """
    Create and configure the Gradio chat interface.
    
    Returns:
        gr.Interface: Configured Gradio interface
    """
    
    # Create the chat interface with purely synchronous function
    interface = gr.ChatInterface(
        fn=process_message,  # Purely synchronous - no async anywhere!
        title="🤖 Agentic LightRAG Assistant (Clean Execution)",
        examples=[
            "What is machine learning and how does it work?",
            "How does neural attention work in transformers?",
            "Explain the difference between supervised and unsupervised learning",
            "What are the applications of federated learning?",
            "How do large language models generate text?",
            "What is the transformer architecture?"
        ]
    )
    
    return interface


def launch_interface():
    """
    Launch the simplified Gradio interface.
    """
    print("🚀 Starting Ultra-Fast Agentic LightRAG Chat Interface...")
    print("⚡ Speed-optimized with cached components and heuristic processing!")
    
    # Create interface
    interface = create_interface()
    
    print("📚 Ready to provide ultra-fast educational explanations!")
    print("⚡ Optimized workflow with cached models and smart heuristics!")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    launch_interface()