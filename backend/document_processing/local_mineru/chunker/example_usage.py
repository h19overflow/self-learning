"""
Example usage of the semantic chunker for inspecting chunking behavior.

This script demonstrates how to use the SemanticChunker to process
specific markdown files and inspect the chunking results.
"""

import asyncio
from pathlib import Path

try:
    from .semantic_chunker import SemanticChunker
except ImportError:
    from semantic_chunker import SemanticChunker


def inspect_chunks(chunks, source_file, max_preview=3):
    """Inspect and display chunk details."""
    print(f"\n📄 INSPECTING: {source_file}")
    print(f"📊 Total chunks: {len(chunks)}")
    print("="*60)
    
    for i, chunk in enumerate(chunks[:max_preview]):
        print(f"\n🧩 CHUNK {i+1}/{len(chunks)}:")
        print(f"📏 Length: {len(chunk)} characters")
        print(f"📝 Preview (first 200 chars):")
        print("-" * 40)
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print("-" * 40)
        
        # Count words and lines
        word_count = len(chunk.split())
        line_count = len(chunk.splitlines())
        print(f"📊 Words: {word_count}, Lines: {line_count}")
    
    if len(chunks) > max_preview:
        print(f"\n... and {len(chunks) - max_preview} more chunks")
    
    # Overall statistics
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    print(f"\n📈 STATISTICS:")
    print(f"   • Total characters: {total_chars:,}")
    print(f"   • Average chunk size: {avg_chunk_size:.0f} chars")
    print(f"   • Min chunk size: {min(len(chunk) for chunk in chunks) if chunks else 0}")
    print(f"   • Max chunk size: {max(len(chunk) for chunk in chunks) if chunks else 0}")


async def main():
    """Main example function for inspecting chunking."""
    # Define the specific file path
    target_file = Path(r"C:\Users\User\Projects\LIGHT_RAG_AGENTIC_SYSTEM\ingestion_pipeline\output_spm\IM-HSBA-SOP_updated 30 Jul 2021\auto\IM-HSBA-SOP_updated 30 Jul 2021.md")
    chunks_output = Path(__file__).parent / "chunked_output" / "inspection_results.json"
    
    print("🔍 SEMANTIC CHUNKER INSPECTION")
    print("="*50)
    print(f"📂 Target file: {target_file}")
    print(f"💾 Output file: {chunks_output}")
    print(f"📁 File exists: {target_file.exists()}")
    
    if not target_file.exists():
        print("❌ Target file not found!")
        return
    
    # Initialize chunker with 8K chunks (updated settings)
    chunker = SemanticChunker(
        threshold= 0.9,  # Adjust threshold for chunking
    )
    
    
    # Process the specific file
    try:
        print(f"\n🚀 Processing file...")
        
        # Read the file directly to inspect content
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📖 File content: {len(content):,} characters")
        print(f"📄 File lines: {len(content.splitlines()):,}")
        
        # Chunk the text directly
        source_name = target_file.stem
        chunks = chunker.chunk_text(content, source_name)
        
        # Inspect the chunks
        inspect_chunks(chunks, source_name)
        
        # Also process using the full directory method for comparison
        print(f"\n🔄 Processing via directory method...")
        results = chunker.process_output_directory(target_file.parent, chunks_output)
        
        print(f"\n📊 DIRECTORY METHOD RESULTS:")
        print(f"Files processed: {results['summary']['total_files_processed']}")
        print(f"Total chunks: {results['summary']['total_chunks_generated']}")
        
        for file_name, file_data in results['files'].items():
            if file_name == source_name:
                print(f"\n📁 {file_name}:")
                print(f"   • Chunks: {file_data['total_chunks']}")
                print(f"   • File chunks match: {len(chunks) == file_data['total_chunks']}")
        
        print(f"\n✅ Inspection complete! Results saved to: {chunks_output}")
        
    except Exception as e:
        print(f"❌ Error during chunking: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
