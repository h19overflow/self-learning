# VLM Enhancement Pipeline for MinerU Output

This pipeline enriches MinerU-generated markdown files with AI-generated image descriptions using Google's Gemini 2.0 Flash Vision Language Model.

## Features

- 🔍 **Smart Image Extraction**: Automatically finds image references in markdown files
- 📝 **Context Analysis**: Extracts surrounding text to provide context for descriptions  
- 🤖 **Gemini 2.0 Flash Integration**: Uses state-of-the-art VLM for accurate descriptions
- 📄 **Markdown Enhancement**: Seamlessly integrates descriptions into existing files
- 🔄 **Batch Processing**: Processes multiple documents concurrently
- 🔐 **Backup Safety**: Automatically backs up original files
- 📊 **Progress Tracking**: Detailed logging and statistics

## Architecture

```
vlm_enhancing/
├── components/                 # Core processing components
│   ├── image_extractor.py     # Extract image refs from markdown
│   ├── context_analyzer.py    # Analyze surrounding text
│   ├── gemini_describer.py    # Gemini 2.0 Flash integration
│   └── markdown_enricher.py   # Insert descriptions into markdown
├── models/                    # Data structures
│   ├── image_context.py       # Image and context models
│   └── description_result.py  # Result and status models
├── vlm_pipeline.py           # Main orchestrator
├── example_usage.py          # Usage examples
└── test_pipeline.py         # Test suite
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install google-genai python-dotenv
   ```

2. **Set API Key**:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

3. **Test Installation**:
   ```bash
   python test_pipeline.py
   ```

## Usage

### Process All Documents
```python
from vlm_pipeline import VLMPipeline, PipelineConfig
import asyncio

config = PipelineConfig(
    gemini_model="gemini-2.0-flash-exp",
    backup_original_files=True,
    max_concurrent_requests=3
)

pipeline = VLMPipeline(config)
results = await pipeline.process_directory("path/to/mineru/output")
```

### Command Line Usage
```bash
# Process all documents
python example_usage.py

# Process single document  
python example_usage.py "path/to/document.md"
```

## How It Works

1. **Image Detection**: Scans markdown files for `![](images/filename.jpg)` patterns
2. **Context Extraction**: Gets surrounding paragraphs and section titles
3. **Description Generation**: Sends image + context to Gemini 2.0 Flash
4. **Markdown Enhancement**: Replaces simple tags with enhanced versions:

   **Before:**
   ```markdown
   ![](images/29e695197893607df2ac75e853c53c33926dd8dd8d2b47486a30282705a38b61.jpg)
   ```

   **After:**
   ```markdown
   ![A bar chart showing head-to-head win rates across different conditions and datasets](images/29e695197893607df2ac75e853c53c33926dd8dd8d2b47486a30282705a38b61.jpg)
   
   *AI-generated description: A bar chart showing head-to-head win rates across different conditions and datasets, with GraphRAG conditions (C0-C3) consistently outperforming naive RAG on comprehensiveness and diversity metrics.*
   ```

## Configuration Options

```python
PipelineConfig(
    gemini_api_key=None,           # API key (or set GEMINI_API_KEY env var)
    gemini_model="gemini-2.0-flash-exp",  # Model to use
    backup_original_files=True,    # Create .backup files
    max_concurrent_requests=5,     # Concurrent API calls
    skip_existing_descriptions=True, # Skip already processed images
    log_level="INFO"              # Logging level
)
```

## Error Handling

- **Missing Images**: Skips references to non-existent image files
- **API Failures**: Retries with exponential backoff
- **Processing Errors**: Logs errors but continues with other images
- **Backup Recovery**: Use `restore_from_backup()` if needed

## Testing

The test suite validates core functionality without requiring API calls:

```bash
python test_pipeline.py
```

Tests include:
- Image extraction from markdown files
- Context analysis around image references  
- Component integration
- Error handling scenarios

## Requirements

- Python 3.8+
- `google-genai` for Gemini API
- `python-dotenv` for environment variables
- MinerU output directory with markdown files and images

## Performance

- **Speed**: ~2-5 seconds per image (depends on API response time)
- **Concurrency**: Configurable concurrent requests (default: 5)
- **Memory**: Lightweight - processes images one at a time
- **Cost**: Varies by Gemini API usage (vision model calls)