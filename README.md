# Self-Learning RAG Pipeline

**Agentic document processing and intelligent question answering system**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow%20Engine-purple.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## Architecture Overview

### Document Processing Pipeline

```mermaid
graph TD
    A[PDF Input Directory] --> B[PDF Processing Task]
    C[playlist_sources.json] --> D[Video Transcription Task]
    
    B --> E[Output Directory]
    D --> E[Video Transcripts]
    
    E --> F[VLM Enhancement Task]
    F --> G[Semantic Chunking Task]
    G --> H[semantic_chunks.json]
    H --> I[ChromaDB RAG Ingestion Task]
    I --> J[Vector Database]
    
    style B fill:#e1f5fe
    style D fill:#e1f5fe
    style F fill:#f3e5f5
    style G fill:#e8f5e8
    style I fill:#e8f5e8
```

### Agentic Query System

```mermaid
graph LR
    A[User] --> B[Gradio Interface]
    B --> C[UltraFast Workflow]
    C --> D[Parameter Selection Node]
    C --> E[Fast Retrieval Node]  
    C --> F[Answering Node]
    
    D --> G[Query Agent]
    E --> H[ChromaDB System]
    F --> I[Answering Agent]
    
    J[VLM Agent] --> K[Document Processing Pipeline]
    L[Corrective Agent] --> H
    
    M[AgenticLightRAGState] --> C
    
    style C fill:#fff3e0
    style G fill:#e3f2fd
    style I fill:#e3f2fd
```

## Key Features

- **Multi-Modal Processing**: PDFs, videos, and images with AI enhancement
- **Agentic Workflow**: Query analysis, retrieval optimization, and answer generation agents  
- **Fast Retrieval**: ChromaDB vector database with semantic chunking
- **State Management**: Session-aware conversation handling
- **Educational Focus**: Specialized for learning and knowledge extraction

## Performance Metrics

| Component | Performance |
|-----------|-------------|
| Knowledge Base | 9,989+ semantic chunks |
| Query Response | < 2 seconds average |
| Source Materials | 27 academic documents |
| Processing Pipeline | Parallel PDF + Video ingestion |

## Quick Start

### Prerequisites

```bash
# Python 3.8+
# UV package manager
# 8GB+ RAM recommended
```

### Installation

```bash
# Clone repository
git clone https://github.com/h19overflow/self-learning.git
cd self-learning

# Environment setup
uv sync

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Usage

```bash
# Explore knowledge base
python -m backend.storage.chromadb_info_extractor --summary

# Start interface
python -m backend.agentic_system.agentic_lightrag.gradio_interface_simplified

# Monitor pipelines
prefect server start
```

## System Components

### Document Pipeline Tasks

- **PDF Processing**: Converts PDFs to enriched Markdown
- **Video Transcription**: Extracts YouTube video transcripts
- **VLM Enhancement**: AI-powered image and diagram analysis
- **Semantic Chunking**: Creates meaningful text segments
- **ChromaDB Ingestion**: Stores vectors with metadata

### Agentic Workflow Nodes

- **Parameter Selection**: Optimizes retrieval parameters
- **Fast Retrieval**: Vector search with context assembly
- **Answering**: Educational response generation with source citations

### AI Agents

- **Query Agent**: Analyzes and categorizes user questions
- **Answering Agent**: Generates educational responses
- **VLM Agent**: Processes visual content
- **Corrective Agent**: Refines search queries

## Example Queries

- "Explain the attention mechanism in transformers"
- "How does LoRA improve fine-tuning efficiency?"
- "Compare different RAG architectures"
- "What are the key principles of atomic habits?"

---

**Built for educational AI research and knowledge extraction**