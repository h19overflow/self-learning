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
    
    def get_available_sources(self):
        """Get list of available sources from ChromaDB."""
        if self._cached_retriever is None:
            return "⚠️ ChromaDB retriever not available"
        
        try:
            sources = self._cached_retriever.list_source_files()
            if not sources:
                return "📭 No sources found in the collection"
            
            # Format sources nicely with numbers and stats
            collection_stats = self._cached_retriever.get_collection_stats()
            total_docs = collection_stats.get('total_documents', 0)
            
            formatted_sources = [
                f"📚 **Available Knowledge Sources** ({len(sources)} files, {total_docs:,} total chunks)",
                "",
                "**Source Files:**"
            ]
            
            for i, source in enumerate(sources, 1):
                formatted_sources.append(f"{i:2d}. {source}")
            
            formatted_sources.extend([
                "",
                f"🧠 **Embedding Model**: {collection_stats.get('embedding_model', 'Unknown')}",
                f"📊 **Collection**: {collection_stats.get('collection_name', 'Unknown')}",
                f"🔄 **Reranker Available**: {'✅' if collection_stats.get('reranker_available', False) else '❌'}"
            ])
            
            return "\n".join(formatted_sources)
            
        except Exception as e:
            return f"❌ Error retrieving sources: {str(e)}"
    
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

    async def stream_question(self, question: str):
        """Process question with streaming updates for real-time responses."""
        start_time = time.time()
        
        try:
            print(f"🌊 Processing question with streaming: {question[:100]}...")
            
            # Get workflow instance
            workflow = self.get_workflow()
            
            # Stream through the workflow
            current_response = ""
            node_status = {}
            
            async for chunk in workflow.stream_question(question):
                node_name = chunk.get("node", "unknown")
                node_state = chunk.get("state", {})
                
                # Update node status
                node_status[node_name] = "completed"
                
                # If we have an answer, include it
                if "answer" in node_state and node_state["answer"]:
                    current_response = node_state["answer"]
                    
                    # Add sources if available
                    if hasattr(workflow, '_cached_retriever') and workflow._cached_retriever:
                        # Try to get sources from the latest state
                        if "sources" in node_state and node_state["sources"]:
                            sources_text = "\n\n---\n**📚 Sources:**\n"
                            for i, source in enumerate(node_state["sources"], 1):
                                sources_text += f"{i}. {source}\n"
                            current_response += sources_text
                
                # Yield intermediate result
                if current_response:
                    yield status_text + "\n\n---\n\n" + current_response
                else:
                    yield status_text
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_time"] += processing_time
            self.stats["avg_response_time"] = self.stats["total_time"] / self.stats["total_requests"]
            
            print(f"🌊 Streaming request processed in {processing_time:.2f}s (avg: {self.stats['avg_response_time']:.2f}s)")
                
        except Exception as e:
            print(f"❌ Error in streaming question: {e}")
            import traceback
            traceback.print_exc()
            yield f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."


# Global workflow manager
workflow_manager = SimplifiedWorkflowManager()


def refresh_sources():
    """Function to refresh the sources display."""
    print("🔄 Refreshing sources list...")
    return workflow_manager.get_available_sources()


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


async def stream_message(message, history):
    """
    Streaming function to process user messages with real-time updates.
    
    Args:
        message: User input message
        history: Chat history
        
    Yields:
        str: Streaming response updates
    """
    try:
        print(f"🌊 GRADIO: Streaming question: {message[:100]}...")
        
        # Use the streaming method
        async for response_chunk in workflow_manager.stream_question(message):
            yield response_chunk
        
        print(f"✅ GRADIO: Streaming completed for question")
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        print(f"❌ GRADIO: Error streaming message: {e}")
        print(f"❌ GRADIO: Error type: {type(e).__name__}")
        
        # Return a user-friendly error message
        yield f"{error_msg}. Please try rephrasing your question or provide more context."


def create_interface():
    """
    Create and configure the Gradio chat interface with sources display.
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    
    with gr.Blocks(
        title="🤖 Ultra-Fast Agentic LightRAG Assistant",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("# 🤖 Ultra-Fast Agentic LightRAG Assistant")
        gr.Markdown("*Speed-optimized RAG system with cached components, smart heuristics, and **streaming responses***")
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(
                    height=600,
                    placeholder="Start a conversation by typing a question below...",
                    show_copy_button=True,
                    type="messages"
                )
                
                with gr.Row():
                    with gr.Column(scale=7):
                        msg_input = gr.Textbox(
                            placeholder="Ask me anything about machine learning, AI, or the available knowledge sources...",
                            show_label=False,
                            container=False
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Send", variant="primary")
                    with gr.Column(scale=1):
                        stream_btn = gr.Button("🌊 Stream", variant="secondary")
                
                # Example questions
                with gr.Row():
                    gr.Examples(
                        examples=[
                            "What is machine learning and how does it work?",
                            "How does neural attention work in transformers?", 
                            "Explain the difference between supervised and unsupervised learning",
                            "What are the applications of federated learning?",
                            "How do large language models generate text?",
                            "What is the transformer architecture?"
                        ],
                        inputs=msg_input,
                        label="💡 Example Questions"
                    )
            
            # Knowledge Sources Tab  
            with gr.Tab("📚 Knowledge Sources"):
                gr.Markdown("### Available Resources in ChromaDB Collection")
                
                with gr.Row():
                    with gr.Column(scale=9):
                        gr.Markdown("*This shows all the source documents available for answering your questions*")
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                
                sources_display = gr.Markdown(
                    value=workflow_manager.get_available_sources(),
                    label="Available Sources"
                )
        
        # Event handlers
        def respond(message, history):
            """Process user message and return chat history."""
            if not message.strip():
                return history, ""
            
            # Add user message to history (OpenAI format)
            history.append({"role": "user", "content": message})
            
            # Get AI response
            response = process_message(message, history)
            
            # Add AI response to history (OpenAI format)
            history.append({"role": "assistant", "content": response})
            
            return history, ""
        
        def stream_respond(message, history):
            """Process user message with streaming updates."""
            if not message.strip():
                return history, ""
            
            # Add user message to history (OpenAI format)
            history.append({"role": "user", "content": message})
            
            # Start with empty assistant response
            history.append({"role": "assistant", "content": ""})
            
            # Use the existing event loop to run async streaming
            import asyncio
            
            # Create a simple wrapper to handle streaming
            def run_streaming():
                try:
                    # Get the event loop from workflow manager
                    loop = workflow_manager.loop
                    if loop is None:
                        return "Error: Event loop not available"
                    
                    # Run the streaming coroutine
                    async def stream_coro():
                        final_response = ""
                        async for chunk in workflow_manager.stream_question(message):
                            final_response = chunk
                        return final_response
                    
                    future = asyncio.run_coroutine_threadsafe(stream_coro(), loop)
                    return future.result()
                    
                except Exception as e:
                    print(f"❌ Streaming error: {e}")
                    return f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."
            
            # Get streaming response
            response = run_streaming()
            
            # Update the last message in history
            history[-1]["content"] = response
            
            return history, ""
        
        # Connect events
        msg_input.submit(respond, [msg_input, chatbot], [chatbot, msg_input])
        submit_btn.click(respond, [msg_input, chatbot], [chatbot, msg_input])
        stream_btn.click(stream_respond, [msg_input, chatbot], [chatbot, msg_input])
        refresh_btn.click(refresh_sources, outputs=sources_display)
    
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
        share=True,
        inbrowser=True
    )


if __name__ == "__main__":
    launch_interface()