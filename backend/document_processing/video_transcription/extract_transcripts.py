"""
Simple Playlist Transcript Extractor

A simple script to extract transcripts from playlists defined in playlist_sources.json.
Perfect for testing the playlist transcription functionality.

Usage:
    python extract_transcripts.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

from .video_transcription_manager import VideoTranscriptionManager
from .models import TranscriptResult


def save_results_to_json(results: List[TranscriptResult], output_file: str):
    """Save transcript results to JSON file."""
    output_data = {
        "transcripts": [result.to_dict() for result in results],
        "summary": {
            "total_videos": len(results),
            "successful_extractions": sum(1 for r in results if r.success),
            "failed_extractions": sum(1 for r in results if not r.success)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")


def save_transcripts_as_text(results: List[TranscriptResult], output_dir: str):
    """Save each transcript as a separate text file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for result in results:
        if result.success and result.transcript_text:
            filename = f"{result.video_id}.txt"
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Video: {result.video_title}\n")
                f.write(f"URL: {result.video_url}\n")
                f.write(f"Language: {result.language}\n")
                f.write(f"Duration: {result.duration} seconds\n")
                f.write("-" * 50 + "\n\n")
                f.write(result.transcript_text)
    
    successful_count = sum(1 for r in results if r.success)
    print(f"📝 Saved {successful_count} transcript text files to: {output_dir}")


async def main():
    """Main function to extract transcripts from playlist sources."""
    print("🎬 Simple Playlist Transcript Extractor")
    print("=" * 50)
    
    # Check if playlist_sources.json exists
    sources_file = Path(__file__).parent / "playlist_sources.json"
    if not sources_file.exists():
        print(f"❌ Error: playlist_sources.json not found at: {sources_file}")
        print("Please create the playlist_sources.json file with your playlist URLs.")
        return
    
    try:
        # Initialize manager and process playlists
        print("🚀 Initializing video transcription manager...")
        manager = VideoTranscriptionManager()
        
        print("📋 Processing playlists from playlist_sources.json...")
        results = await manager.extract_all_playlists()
        
        if not results:
            print("⚠️  No results returned. Check your playlist_sources.json file.")
            return
        
        # Print results summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n✅ Successfully extracted: {len(successful)} transcripts")
        print(f"❌ Failed extractions: {len(failed)} transcripts")
        
        if failed and len(failed) <= 10:
            print("\nFailed videos:")
            for result in failed:
                print(f"  - {result.video_url}: {result.error_message}")
        elif failed:
            print(f"\nFirst 10 failed videos:")
            for result in failed[:10]:
                print(f"  - {result.video_url}: {result.error_message}")
            print(f"  ... and {len(failed) - 10} more failures")
        
        # Save results
        if results:
            # Save JSON
            json_file = "extracted_transcripts.json"
            save_results_to_json(results, json_file)
            
            # Save text files
            text_dir = "extracted_transcripts"
            save_transcripts_as_text(results, text_dir)
            
            # Show preview
            print(f"\n📋 Transcript Preview (first 2 successful extractions):")
            print("-" * 60)
            preview_count = 0
            for result in results:
                if result.success and preview_count < 2:
                    preview = result.transcript_text[:200] + "..." if len(result.transcript_text) > 200 else result.transcript_text
                    print(f"\n🎥 {result.video_title}")
                    print(f"🔗 {result.video_url}")
                    print(f"📄 {preview}")
                    preview_count += 1
        
        print(f"\n🎉 Extraction complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())