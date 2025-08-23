#!/usr/bin/env python3
"""
Pipeline Stage Runner

A convenient script to run specific stages of the document processing pipeline.
Useful for resuming from interruptions or running individual stages.

Usage:
    python run_pipeline_stage.py <stage> [options]

Examples:
    python run_pipeline_stage.py vlm
    python run_pipeline_stage.py chunking --no-rag
    python run_pipeline_stage.py rag --chunks-file custom_chunks.json
    python run_pipeline_stage.py help
"""

import asyncio
import argparse
from pathlib import Path
import sys

from prefect_pipeline_orchestrator import run_pipeline_from_stage, print_stage_help


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run specific stages of the document processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Resume from VLM enhancement stage after disconnection
    python run_pipeline_stage.py vlm
    
    # Skip to chunking stage without RAG ingestion
    python run_pipeline_stage.py chunking --no-rag
    
    # Only run RAG ingestion with custom chunks file
    python run_pipeline_stage.py rag --chunks-file custom_chunks.json
    
    # Show detailed help about stages
    python run_pipeline_stage.py help
        """
    )
    
    parser.add_argument(
        'stage',
        choices=['pdf', 'full', 'complete', 'vlm', 'vlm_enhancement', 'enhancement', 
                'chunking', 'chunk', 'semantic_chunking', 'rag', 'ingestion', 'chromadb', 'help'],
        help='Pipeline stage to run'
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='local_mineru/books',
        help='Directory containing PDF files (default: local_mineru/books)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_books',
        help='Directory for processed output (default: output_books)'
    )
    
    parser.add_argument(
        '--chunks-file',
        type=str,
        help='File path for chunked results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--no-vlm',
        action='store_true',
        help='Disable VLM enhancement'
    )
    
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG ingestion'
    )
    
    parser.add_argument(
        '--enable-video',
        action='store_true',
        help='Enable video transcription'
    )
    
    parser.add_argument(
        '--max-vlm-requests',
        type=int,
        default=5,
        help='Maximum concurrent VLM requests (default: 5)'
    )
    
    return parser


async def main():
    """Main function to run pipeline stages."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if requested
    if args.stage == 'help':
        print_stage_help()
        parser.print_help()
        return
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    
    # Convert relative paths to absolute
    pdf_directory = script_dir / args.pdf_dir
    output_directory = script_dir / args.output_dir
    
    if args.chunks_file:
        chunks_file = args.chunks_file
    else:
        chunks_file = str(script_dir / "chunked_output_books" / "semantic_chunks.json")
    
    # Validate directories exist for certain stages
    if args.stage in ['vlm', 'vlm_enhancement', 'enhancement'] and not output_directory.exists():
        print(f"Error: Output directory does not exist: {output_directory}")
        print("Make sure PDF processing has been completed first.")
        sys.exit(1)
    
    if args.stage in ['rag', 'ingestion', 'chromadb'] and not Path(chunks_file).exists():
        print(f"Error: Chunks file does not exist: {chunks_file}")
        print("Make sure semantic chunking has been completed first.")
        sys.exit(1)
    
    print(f"Running pipeline from stage: {args.stage}")
    print(f"PDF directory: {pdf_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Chunks file: {chunks_file}")
    print(f"VLM enabled: {not args.no_vlm}")
    print(f"RAG enabled: {not args.no_rag}")
    print(f"Video transcription enabled: {args.enable_video}")
    print("-" * 50)
    
    try:
        # Run the pipeline stage
        results = await run_pipeline_from_stage(
            stage=args.stage,
            pdf_directory=str(pdf_directory),
            output_directory=str(output_directory),
            chunked_output_file=chunks_file,
            enable_vlm=not args.no_vlm,
            enable_rag=not args.no_rag,
            enable_video_transcription=args.enable_video
        )
        
        # Print results
        print("\n" + "=" * 50)
        print("PIPELINE RESULTS")
        print("=" * 50)
        print(f"Status: {results['pipeline_status']}")
        print(f"Starting Stage: {results.get('starting_stage', 'Full Pipeline')}")
        print(f"Duration: {results['pipeline_duration_seconds']:.2f} seconds")
        
        # Print stage-specific results
        summary = results.get('summary', {})
        if 'total_files_processed' in summary:
            print(f"Files Processed: {summary['total_files_processed']}")
        if 'total_videos_transcribed' in summary:
            print(f"Videos Transcribed: {summary['total_videos_transcribed']}")
        if 'total_images_enhanced' in summary:
            print(f"Images Enhanced: {summary['total_images_enhanced']}")
        if 'total_chunks_generated' in summary:
            print(f"Chunks Generated: {summary['total_chunks_generated']}")
        if 'total_chunks_ingested' in summary:
            print(f"Chunks Ingested: {summary['total_chunks_ingested']}")
        
        print("=" * 50)
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError: Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())