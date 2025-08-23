"""
Transcription tasks for the pipeline orchestration system.

This module contains tasks for video transcription and processing.
"""

from typing import Dict, Any
from datetime import datetime
from prefect import task
from prefect.logging import get_run_logger

from backend.document_processing.video_transcription import VideoTranscriptionManager
from ..utils.pipeline_config import PipelineConfiguration


@task(
    name="Video Transcription",
    description="Extract transcripts from YouTube videos",
    retries=2,
    retry_delay_seconds=15
)
async def video_transcription_task(config: PipelineConfiguration) -> Dict[str, Any]:
    """
    Extract transcripts from YouTube videos and save to file.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dict[str, Any]: Video transcription results
    """
    logger = get_run_logger()
    
    if not config.enable_video_transcription:
        logger.info("Video transcription disabled, skipping")
        return {"skipped": True, "reason": "Video transcription disabled"}
    
    logger.info("Starting video transcription from playlist sources")
    
    try:
        # Initialize video transcription manager
        manager = VideoTranscriptionManager()
        
        # Extract transcripts from playlist sources
        logger.info("Extracting transcripts from playlist_sources.json")
        results = await manager.extract_all_playlists()
        
        # Save transcripts to file for semantic chunking
        transcripts_file = config.output_directory / "video_transcripts.json"
        transcripts_data = {
            "transcripts": [result.to_dict() for result in results],
            "summary": {
                "total_videos": len(results),
                "successful_extractions": sum(1 for r in results if r.success),
                "failed_extractions": sum(1 for r in results if not r.success)
            }
        }
        
        # Save to JSON file
        import json
        with open(transcripts_file, 'w', encoding='utf-8') as f:
            json.dump(transcripts_data, f, indent=2, ensure_ascii=False)
        
        # Create enhanced markdown files for semantic chunking with video metadata
        # Save directly to output_directory for semantic chunker processing
        transcript_texts_dir = config.output_directory
        transcript_texts_dir.mkdir(exist_ok=True, parents=True)
        
        for result in results:
            if result.success and result.transcript_text:
                # Create rich markdown file with structured metadata for better chunking
                text_file = transcript_texts_dir / f"{result.video_id}.md"
                with open(text_file, 'w', encoding='utf-8') as f:
                    # Header with video metadata
                    f.write(f"# {result.video_title}\\n\\n")
                    
                    # Metadata section for credibility and traceability
                    f.write("## Video Information\\n\\n")
                    f.write(f"- **Video ID**: {result.video_id}\\n")
                    f.write(f"- **Source URL**: {result.video_url}\\n")
                    f.write(f"- **Language**: {result.language}\\n")
                    f.write(f"- **Duration**: {result.duration:.1f} seconds ({result.duration/60:.1f} minutes)\\n")
                    f.write(f"- **Transcript Source**: {result.source}\\n\\n")
                    
                    # Enhanced content section
                    f.write("## Video Content\\n\\n")
                    
                    # Add structured transcript with better formatting for semantic chunking
                    # Split into logical paragraphs for better semantic understanding
                    transcript_paragraphs = result.transcript_text.split('. ')
                    current_paragraph = ""
                    paragraph_length = 0
                    
                    for sentence in transcript_paragraphs:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        # Add sentence to current paragraph
                        if current_paragraph:
                            current_paragraph += ". " + sentence
                        else:
                            current_paragraph = sentence
                        
                        paragraph_length += len(sentence)
                        
                        # Break into new paragraph after ~300-500 characters for better chunking
                        if paragraph_length > 300 or "okay so" in sentence.lower() or "now" in sentence.lower()[:10]:
                            f.write(current_paragraph)
                            if not current_paragraph.endswith('.'):
                                f.write('.')
                            f.write("\\n\\n")
                            current_paragraph = ""
                            paragraph_length = 0
                    
                    # Write remaining content
                    if current_paragraph:
                        f.write(current_paragraph)
                        if not current_paragraph.endswith('.'):
                            f.write('.')
                        f.write("\\n\\n")
                    
                    # Footer with source attribution for credibility
                    f.write("---\\n\\n")
                    f.write(f"*Transcript extracted from: [{result.video_url}]({result.video_url})*\\n")
                    f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\\n")
        
        logger.info(f"Video transcription completed: {transcripts_data['summary']['successful_extractions']} successful")
        return {
            "success": True,
            "total_videos": transcripts_data['summary']['total_videos'],
            "successful_extractions": transcripts_data['summary']['successful_extractions'],
            "failed_extractions": transcripts_data['summary']['failed_extractions'],
            "transcripts_file": str(transcripts_file),
            "transcript_texts_directory": str(transcript_texts_dir)
        }
        
    except Exception as e:
        logger.error(f"Video transcription failed: {e}")
        if not config.continue_on_errors:
            raise
        return {"failed": True, "error": str(e)}