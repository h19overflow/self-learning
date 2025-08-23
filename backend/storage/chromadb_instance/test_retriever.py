"""
Simple test script for the modular ChromaRetriever
"""

import sys
from pathlib import Path
import weave
# Path for storage directory
current_dir = Path(__file__).parent

from backend.storage.chromadb_instance.models.chroma_config import ChromaConfig, RetrievalConfig
from backend.storage.chromadb_instance.components.chroma_retriever import ChromaRetriever
weave.init('self_learning')
def test_basic_retrieval():
    """Test basic retrieval functionality."""
    print("Testing ChromaRetriever modular architecture...")
    print("-" * 50)
    
    # Configure ChromaDB
    config = ChromaConfig(
        persist_directory=current_dir.parent / "chromadb_storage",
        collection_name="academic_papers",
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_device="cuda"
    )
    
    # Initialize retriever
    retriever = ChromaRetriever(config)
    
    # Get collection stats
    print("Collection Statistics:")
    stats = retriever.get_collection_stats()
    print(f"  Collection: {stats.get('collection_name', 'Unknown')}")
    print(f"  Documents: {stats.get('total_documents', 0)}")
    print(f"  Sources: {stats.get('unique_source_files', 0)}")
    print(f"  Reranker available: {stats.get('reranker_available', False)}")
    print(f"  Embedding manager ready: {stats.get('embedding_manager_ready', False)}")
    print()
    
    # Test basic search
    query = "What is attention mechanism?"
    print(f"Testing basic search: '{query}'")
    
    basic_config = RetrievalConfig(top_k=3, enable_reranking=False)
    results = retriever.search(query, basic_config)
    
    print(f"Found {results.total_results} results in {results.retrieval_time_ms:.1f}ms")
    if results.has_results:
        print(f"Average score: {results.average_score:.3f}")
        print("Results:")
        for i, result in enumerate(results.results, 1):
            print(f"  {i}. Score: {result.score:.3f} | {result.source_file}")
            print(f"     Content: {result.content[:80]}...")
    print()
    
    # Test with reranking
    print("Testing with reranking enabled...")
    advanced_config = RetrievalConfig(
        top_k=5,
        enable_reranking=True,
        rerank_top_k_multiplier=2,
        enable_diversity=True
    )
    
    results_reranked = retriever.search(query, advanced_config)
    print(f"Reranked: {results_reranked.total_results} results in {results_reranked.retrieval_time_ms:.1f}ms")
    if results_reranked.has_results:
        print(f"Average score: {results_reranked.average_score:.3f}")
        print("Top reranked result:")
        top_result = results_reranked.results[0]
        print(f"  Score: {top_result.score:.3f} | {top_result.source_file}")
        print(f"  Content: {top_result.content[:120]}...")
    
    print("\n✅ ChromaRetriever modular architecture test completed!")


if __name__ == "__main__":
    try:
       
        test_basic_retrieval()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()