# Streamlit Agentic Application

A production-ready conversational AI application built with Streamlit, featuring advanced language model capabilities, document analysis, and multimodal interactions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Available Tools](#available-tools)
- [Development](#development)
- [API Reference](#api-reference)
- [Production Deployment](#production-deployment)

## Overview

This application provides a sophisticated chat interface powered by NVIDIA's language models, with support for:

- 🤖 Real-time streaming responses with tool calling capabilities
- 📄 Intelligent PDF document analysis with progress tracking
- 🎨 AI-powered image generation and analysis
- 🔍 Semantic search through knowledge bases
- 🌐 Web content extraction and search
- 🏗️ Production-grade architecture using MVC patterns

## Features

### Core Capabilities

- **Multimodal Chat Interface**: Support for text, images, and document-based conversations
- **Advanced Tool System**: 11 specialized tools for various tasks
- **Smart Context Management**: Automatic conversation context injection
- **Real-time Streaming**: Smooth response streaming with progress indicators
- **Session Management**: Isolated user sessions with external file storage
- **Batch Processing**: Efficient handling of large documents through intelligent chunking

### Available Tools

1. **Text Assistant** (`text_assistant`) - Comprehensive text processing including analysis, summarization, translation, and code development
2. **Image Generation** (`generate_image`) - AI-powered image creation with style control and aspect ratio options
3. **Image Analysis** (`analyze_image`) - Vision-capable LLM for analyzing uploaded images
4. **PDF Summary** (`retrieve_pdf_summary`) - Intelligent document summarization with hierarchical processing
5. **PDF Text Processor** (`process_pdf_text`) - Advanced PDF text extraction and processing
6. **Web Search** (`tavily_internet_search`) - General internet search capabilities
7. **News Search** (`tavily_news_search`) - Real-time news and current events search
8. **Weather** (`weather`) - Current weather information for any location
9. **Web Extract** (`extract_web_content`) - Extract and parse content from web pages
10. **Retriever** (`retrieval_search`) - Semantic search through knowledge bases
11. **Conversation Context** (`conversation_context`) - Analyze conversation history and patterns

## Architecture

The application follows a Model-View-Controller (MVC) pattern with clear separation of concerns:

```
streamlit-chatbot/
├── docker/
│   └── app/
│       ├── main.py                    # Application entry point
│       ├── controllers/               # Business logic controllers
│       │   ├── session_controller.py  # Session state management
│       │   ├── message_controller.py  # Message processing and validation
│       │   ├── file_controller.py     # File upload and processing
│       │   ├── image_controller.py    # Image upload and processing
│       │   └── response_controller.py # LLM response orchestration
│       ├── services/                  # Core service layer
│       │   ├── llm_service.py        # LLM interaction service
│       │   ├── llm_client_service.py # LLM client management
│       │   ├── chat_service.py       # Chat processing service
│       │   ├── pdf_context_service.py # PDF context injection
│       │   ├── pdf_analysis_service.py # Intelligent PDF analysis
│       │   ├── streaming_service.py   # Response streaming
│       │   ├── tool_execution_service.py # Tool orchestration
│       │   └── file_storage_service.py # External file storage
│       ├── tools/                     # LLM tool implementations
│       │   ├── base.py               # Base tool interface
│       │   ├── registry.py           # Tool registry system
│       │   ├── assistant.py          # Text processing assistant
│       │   ├── pdf_summary.py        # PDF summarization tool
│       │   ├── pdf_text_processor.py # PDF text processing tool
│       │   ├── image_gen.py          # Image generation tool
│       │   ├── image_analysis_tool.py # Image analysis tool
│       │   └── ...                   # Other tool implementations
│       ├── models/                    # Data models and schemas
│       ├── ui/                        # UI components
│       └── utils/                     # Utilities and configuration
│           └── config.py             # Centralized configuration
├── docker-compose.yml                 # Docker orchestration
├── .env.example                      # Environment variables template
└── README.md                         # This file
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA API Key
- (Optional) Tavily API Key for web search
- (Optional) Image generation endpoint

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd streamlit-chatbot
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Build and run with Docker Compose**

   ```bash
   ./rebuild.sh
   ```

   Or manually:

   ```bash
   docker compose build app
   docker compose up app nginx -d
   docker compose logs -f app
   ```

4. **Access the application**

   Open your browser and navigate to `http://localhost:80`

## Configuration

All configuration is centralized in `utils/config.py`. The system validates environment variables on startup.

### Required Environment Variables

```bash
# NVIDIA API Configuration
NVIDIA_API_KEY=your_api_key_here

# Model Endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Model Names
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Vision Language Model (for image analysis)
VLM_ENDPOINT=https://integrate.api.nvidia.com/v1
VLM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
```

### Optional Services

```bash
# Image Generation
IMAGE_ENDPOINT=your_image_generation_endpoint

# Web Search (Tavily)
TAVILY_API_KEY=your_tavily_api_key

# PDF Processing
NVINGEST_ENDPOINT=http://localhost:7670

# Vector Database (for retrieval)
EMBEDDING_ENDPOINT=https://integrate.api.nvidia.com/v1
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
COLLECTION_NAME=your_collection_name
DATABASE_URL=your_milvus_url
DEFAULT_DB=default
```

## Available Tools

### Text Processing

The `text_assistant` tool supports multiple task types:

- **analyze** - Document analysis and insights
- **summarize** - Create concise summaries
- **proofread** - Grammar and style corrections
- **rewrite** - Enhance clarity and impact
- **critic** - Provide constructive feedback
- **translate** - Convert between languages
- **develop** - Programming assistance
- **qa** - Answer questions about documents

### Image Capabilities

- **Generate Images**: Create AI-generated images with customizable styles, moods, and aspect ratios
- **Analyze Images**: Use vision models to describe, identify objects, or answer questions about uploaded images

### Document Processing

- **PDF Analysis**: Intelligent document analysis with query-aware processing
- **Batch Processing**: Handle large PDFs (>100 pages) efficiently
- **Hierarchical Summarization**: Multi-level summarization for comprehensive documents

### Information Retrieval

- **Web Search**: General internet search using Tavily
- **News Search**: Specialized search for current events
- **Web Extraction**: Extract clean content from URLs
- **Semantic Search**: Query knowledge bases using vector similarity

## Development

### Adding New Controllers

Controllers handle business logic and should follow this pattern:

```python
from utils.config import config

class NewController:
    def __init__(self, config_obj):
        self.config_obj = config_obj
        # Initialize with dependencies

    def process_action(self, data):
        # Implement business logic
        # Use config for settings
        # Return processed result
```

### Creating New Tools

Tools extend LLM capabilities and must implement the `BaseTool` interface:

```python
from tools.base import BaseTool, BaseToolResponse

class NewToolResponse(BaseToolResponse):
    # Define response fields
    result: str

class NewTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "tool_name"
        self.description = "Tool description"
        self.llm_type = "fast"  # or "llm" or "intelligent"

    def to_openai_format(self):
        # Return OpenAI function definition
        pass

    def execute(self, params: Dict[str, Any]) -> NewToolResponse:
        # Implement tool logic
        pass
```

### Service Integration

Services provide reusable functionality:

```python
class NewService:
    def __init__(self, config: ChatConfig):
        self.config = config
        # Initialize service

    async def process(self, data):
        # Implement async processing
        # Handle errors appropriately
        # Return results
```

## API Reference

### Query Flow

1. **User Input** → Message validation and processing
2. **Context Enhancement** → Automatic PDF/conversation context injection
3. **Tool Selection** → LLM determines required tools
4. **Tool Execution** → Parallel or sequential execution
5. **Response Generation** → LLM synthesizes final response
6. **Streaming Output** → Real-time display to user

### Tool Execution Strategies

- **Parallel Execution**: Default for independent tools
- **Sequential Execution**: For tools with dependencies
- **Direct Response**: For tools that provide final answers

### PDF Processing Strategies

| Document Size | Pages | Strategy                        |
| ------------- | ----- | ------------------------------- |
| Small         | ≤5    | Single-pass processing          |
| Medium        | 6-15  | Batch processing with synthesis |
| Large         | >15   | Two-phase intelligent analysis  |

## Production Deployment

### Performance Optimization

- **Streaming Responses**: Better user experience with real-time feedback
- **Parallel Tool Execution**: Minimize latency for multiple tool calls
- **Intelligent Document Analysis**: Efficient processing of large PDFs
- **External File Storage**: Prevent memory exhaustion

### Reliability Features

- **Error Handling**: Comprehensive error handling at all layers
- **Fallback Mechanisms**: Automatic fallbacks for tool failures
- **Session Isolation**: Prevent cross-user data leakage
- **Resource Management**: Automatic cleanup of old files

### Scalability Considerations

- **Stateless Design**: Enables horizontal scaling
- **External Storage**: Files stored outside application memory
- **Configurable Models**: Switch between model sizes for cost optimization
- **Batch Processing**: Handle large documents without memory issues

### Monitoring and Logging

- Structured logging throughout the application
- Performance metrics for tool execution
- Error tracking and alerting capabilities
- Session-based log correlation
