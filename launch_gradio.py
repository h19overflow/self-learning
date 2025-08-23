"""
Launch Script for Agentic LightRAG with ChromaRetriever Integration

This script launches the Gradio interface for the enhanced agentic RAG system
that uses ChromaRetriever for high-quality semantic search with reranking.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.agentic_system.agentic_lightrag.gradio_interface import launch_interface

if __name__ == "__main__":
    print("🚀 Starting Enhanced Agentic LightRAG with ChromaRetriever")
    print("=" * 60)
    print("🎯 Features:")
    print("  • Intelligent query analysis and enhancement")
    print("  • High-quality semantic search with BGE embeddings") 
    print("  • Cross-encoder reranking for precision")
    print("  • Corrective retrieval with LLM-based query rewriting")
    print("  • Educational answer generation")
    print("  • 9,989 academic documents available")
    print("=" * 60)
    print()
    
    try:
        launch_interface()
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Interface stopped.")
    except Exception as e:
        print(f"\n❌ Error launching interface: {e}")
        sys.exit(1)