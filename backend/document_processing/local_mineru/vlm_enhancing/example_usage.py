"""
Example usage of the VLM Enhancement Pipeline.

This script demonstrates how to use the pipeline to enhance MinerU output
with AI-generated image descriptions using Gemini 2.0 Flash.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our pipeline components
from .vlm_pipeline import VLMPipeline, PipelineConfig


async def enhance_all_documents():
    """Enhance all documents in the MinerU output directory."""
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        print("Please set your Gemini API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        return
    
    print("🚀 Starting VLM Enhancement Pipeline")
    print("=" * 60)
    
    # Configure the pipeline
    config = PipelineConfig(
        gemini_model="gemini-2.5-flash-lite",  # Latest Gemini 2.0 Flash
        backup_original_files=True,           # Always backup originals
        max_concurrent_requests=3,            # Don't overwhelm the API
        skip_existing_descriptions=True,      # Skip already processed images
        log_level="INFO"                      # Detailed logging
    )
    
    # Initialize the pipeline
    pipeline = VLMPipeline(config)
    
    # Define the output directory
    output_dir = Path("C:/Users/User/Projects/paper_verifier/backend/ingestion_pipeline/mineru/file_ingesting/output")
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    try:
        # Process all documents in the directory
        results = await pipeline.process_directory(output_dir)
        
        # Display results
        print("\n" + "=" * 60)
        print("🎉 Processing Complete!")
        print("=" * 60)
        print(f"📄 Files processed: {results['processed_files']}")
        print(f"🖼️  Total images: {results['total_images']}")
        print(f"✅ Successful descriptions: {results['successful_descriptions']}")
        print(f"❌ Failed descriptions: {results['failed_descriptions']}")
        print(f"⏭️  Skipped descriptions: {results['skipped_descriptions']}")
        
        if results['total_images'] > 0:
            success_rate = results['successful_descriptions'] / results['total_images']
            print(f"📊 Success rate: {success_rate:.1%}")
        
        # Show any errors
        if results['errors']:
            print(f"\n⚠️  Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"   • {error}")
            if len(results['errors']) > 5:
                print(f"   ... and {len(results['errors']) - 5} more")
        
        print("\n💡 Tips:")
        print("   • Backup files are created automatically (.backup extension)")
        print("   • Re-run the script to process any failed images")
        print("   • Check the enhanced markdown files for AI descriptions")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


async def enhance_single_document(document_path: str):
    """Enhance a single document."""
    
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        return
    
    print(f"🎯 Processing single document: {document_path}")
    print("=" * 60)
    
    # Configure the pipeline
    config = PipelineConfig(
        gemini_model="gemini-2.0-flash-exp",
        backup_original_files=True,
        log_level="INFO"
    )
    
    # Initialize the pipeline
    pipeline = VLMPipeline(config)
    
    # Process the document
    document = Path(document_path)
    
    if not document.exists():
        print(f"❌ Document not found: {document}")
        return
    
    try:
        result = await pipeline.process_single_document(document)
        
        print(f"\n📊 Results for {document.name}:")
        print(f"   • Original images: {result.original_image_count}")
        print(f"   • Processed: {result.processed_image_count}")
        print(f"   • Successful: {result.successful_descriptions}")
        print(f"   • Failed: {result.failed_descriptions}")
        print(f"   • Skipped: {result.skipped_descriptions}")
        print(f"   • Success rate: {result.success_rate():.1%}")
        
        print(f"\n✅ {result.summary()}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")


if __name__ == "__main__":
    print("VLM Enhancement Pipeline for MinerU Output")
    print("Powered by Gemini 2.0 Flash Vision Model")
    print()
    
    import sys
    
    if len(sys.argv) > 1:
        # Process single document if path provided
        document_path = sys.argv[1]
        asyncio.run(enhance_single_document(document_path))
    else:
        # Process all documents
        print("Processing all documents in output directory...")
        print("To process a single document, run: python example_usage.py <path-to-markdown-file>")
        print()
        asyncio.run(enhance_all_documents())