# 🚀 Prefect Pipeline Instructions

## What is Prefect?

Prefect is a workflow orchestration tool that provides beautiful monitoring, automatic retries, and professional pipeline management for your document processing system.

## 📋 Quick Start Guide

### Step 1: Start the Prefect UI Server

Open a terminal and run:

```bash
prefect server start
```

**What this does:** Starts the Prefect monitoring dashboard on your local machine.

**Expected output:** You'll see a message like:
```
Check out the dashboard at http://127.0.0.1:4200
```

### Step 2: Open the Beautiful Dashboard

1. Open your web browser
2. Go to: **http://localhost:4200**
3. You'll see the Prefect dashboard with a clean, modern interface

### Step 3: Run Your Pipeline

In a **new terminal window** (keep the server running), navigate to your project:

```bash
cd backend/ingestion_pipeline
python run_prefect_pipeline.py
```

**What this does:** Runs your complete document processing pipeline with monitoring.

### Step 4: Watch the Magic ✨

Go back to your browser (http://localhost:4200) and you'll see:

- **Live pipeline execution** with real-time status updates
- **Task progress bars** showing each stage (PDF → VLM → Chunking → RAG)
- **Detailed logs** for debugging
- **Performance metrics** and timing information
- **Automatic retry attempts** if something fails

## 🎛️ Understanding the Dashboard

### Main Dashboard
- **Flow Runs**: Your pipeline executions (green = success, red = failed, yellow = running)
- **Recent Activity**: Timeline of all pipeline runs
- **Quick Stats**: Success rates and performance metrics

### Flow Details (Click on any pipeline run)
- **Visual Graph**: See your pipeline stages as connected boxes
- **Task Status**: Each box shows if a task is pending, running, completed, or failed
- **Logs**: Click any task to see detailed logs and error messages
- **Timing**: See how long each stage takes

### Key Features
- **🔄 Auto Retries**: Failed tasks automatically retry (you'll see this in the UI)
- **📊 Progress Tracking**: Watch your pipeline progress in real-time
- **🚨 Error Details**: Click failed tasks to see exactly what went wrong
- **⏱️ Performance**: See which stages are slow and need optimization

## ⚙️ Configuration Options

### Simple Usage (Recommended)
```python
# Edit run_prefect_pipeline.py and update these paths:
pdf_directory="your/pdf/folder"
output_directory="your/output/folder"
enable_vlm=True      # Set to False to skip image descriptions
enable_rag=True      # Set to False to skip RAG ingestion
```

### Advanced Configuration
Edit the `PipelineConfiguration` in `run_prefect_pipeline.py`:

```python
config = PipelineConfiguration(
    # Required paths
    pdf_input_directory=Path("input_pdfs"),
    output_directory=Path("output"),
    
    # Processing settings
    chunk_size=512,                    # Size of text chunks
    overlap=100,                       # Overlap between chunks
    max_concurrent_vlm_requests=3,     # How many AI requests at once
    
    # Feature toggles
    enable_vlm_enhancement=True,       # AI image descriptions
    enable_rag_ingestion=True,         # Knowledge graph creation
    
    # Error handling
    continue_on_errors=True,           # Keep going if something fails
    max_retries=2,                     # How many times to retry failures
)
```

## 🔧 Common Commands

### Start/Stop Prefect Server
```bash
# Start the server
prefect server start

# Stop the server (Ctrl+C in the terminal where it's running)
```

### Run Different Pipeline Configurations
```bash
# Run with default settings
python run_prefect_pipeline.py

# For debugging, you can also run individual components
python prefect_pipeline_orchestrator.py
```

### Check Pipeline Status
```bash
# See recent pipeline runs
prefect flow-run ls

# Get details about a specific run
prefect flow-run inspect <flow-run-id>
```

## 🎯 What Each Stage Does

1. **📄 PDF Processing**: Converts your PDFs to markdown using MinerU
2. **🖼️ VLM Enhancement**: AI analyzes images and adds descriptions
3. **🧩 Semantic Chunking**: Breaks text into smart, meaningful pieces
4. **🧠 RAG Ingestion**: Builds a searchable knowledge graph

## 🚨 Troubleshooting

### Pipeline Won't Start
- Make sure Prefect server is running (`prefect server start`)
- Check that your input PDF directory exists
- Verify all file paths in `run_prefect_pipeline.py`

### Tasks Keep Failing
- Check the Prefect UI for detailed error messages
- Look at the logs for the specific failed task
- Common issues: missing API keys, incorrect file paths, insufficient disk space

### UI Won't Load
- Make sure you're going to http://localhost:4200
- Check that the Prefect server is still running
- Try refreshing the page

### Performance Issues
- Reduce `max_concurrent_vlm_requests` if you're hitting API limits
- Increase `chunk_size` if you want fewer, larger chunks
- Disable VLM or RAG if you want faster processing

## 💡 Pro Tips

1. **Keep the server running** in one terminal while running pipelines in another
2. **Bookmark the dashboard** (http://localhost:4200) for easy access
3. **Check the logs first** when something goes wrong - they're very detailed
4. **Use the visual graph** to understand your pipeline flow
5. **Monitor performance** to optimize your settings over time

## 🎉 That's It!

You now have a professional, monitored pipeline for processing academic papers. The Prefect UI gives you enterprise-level workflow management with beautiful visualizations and automatic error handling.

**Happy processing!** 🚀