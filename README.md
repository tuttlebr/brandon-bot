# Streamlit Chat Application

A production-ready conversational AI application built with Streamlit, featuring advanced language model capabilities, document analysis, and multimodal interactions.

## Overview

This application provides a sophisticated chat interface powered by NVIDIA's language models, with support for:

- Real-time streaming responses with tool calling capabilities
- Intelligent PDF document analysis with progress tracking
- AI-powered image generation
- Semantic search through knowledge bases
- Production-grade architecture using controller patterns

## Architecture

The application follows a Model-View-Controller (MVC) pattern with clear separation of concerns:

```
docker/app/
├── main.py                    # Application entry point
├── controllers/               # Business logic controllers
│   ├── session_controller.py  # Session state management
│   ├── message_controller.py  # Message processing and validation
│   ├── file_controller.py     # File upload and processing
│   └── response_controller.py # LLM response orchestration
├── services/                  # Core service layer
│   ├── llm_service.py        # LLM interaction service
│   ├── chat_service.py       # Chat processing service
│   ├── pdf_context_service.py # PDF context injection
│   ├── pdf_analysis_service.py # Intelligent PDF analysis
│   ├── streaming_service.py   # Response streaming
│   ├── tool_execution_service.py # Tool orchestration
│   └── file_storage_service.py # External file storage
├── tools/                     # LLM tool implementations
│   ├── assistant.py          # Text processing assistant
│   ├── pdf_tool.py           # PDF processing tool
│   ├── image_tool.py         # Image generation tool
│   └── search_tools.py       # Search capabilities
├── models/                    # Data models and schemas
├── ui/                        # UI components
└── utils/                     # Utilities and configuration
    └── config.py             # Centralized configuration
```

## How It Works

### Query Flow

1. **User Input**: User enters a query through the Streamlit chat interface
2. **Message Processing**: `MessageController` validates and processes the input
3. **Context Enhancement**: `PDFContextService` automatically injects relevant PDF content if documents are loaded
4. **Tool Selection**: `LLMService` determines if specialized tools are needed
5. **Tool Execution**: `ToolExecutionService` orchestrates parallel tool calls
6. **Response Generation**: LLM generates response based on tool results and context
7. **Streaming Output**: Response is streamed back to the user in real-time

### Intelligent PDF Analysis

For PDF documents, the system employs a multi-strategy approach:

- **Small Documents (≤5 pages)**: Processes entire content in a single pass
- **Medium Documents (6-15 pages)**: Batch processing with synthesis
- **Large Documents (>15 pages)**: Two-phase intelligent analysis:
  - Phase 1: Rapid relevance scanning across all pages
  - Phase 2: Deep analysis of relevant sections only

Progress is tracked in real-time with estimated completion times.

## Core Services

### LLM Service

Manages interactions with language models, including:

- Model selection (fast, standard, intelligent)
- Streaming response generation
- Tool call orchestration
- Context window management

### PDF Analysis Service

Provides intelligent document analysis with:

- Query-aware page selection
- Multi-level processing strategies
- Progress tracking and status updates
- Automatic fallback mechanisms

### Tool Execution Service

Orchestrates tool execution with:

- Parallel execution strategies
- Error handling and retries
- Result aggregation
- Direct response routing

### File Storage Service

Manages external file storage to prevent memory issues:

- Session-based file isolation
- Automatic cleanup
- Storage limit enforcement
- Metadata tracking

## Configuration

All configuration is centralized in `utils/config.py`. The system validates environment variables on startup.

### Required Environment Variables

```bash
# NVIDIA API Configuration
NVIDIA_API_KEY=your_api_key

# Model Endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Model Names
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Optional Services
IMAGE_ENDPOINT=your_image_generation_endpoint
TAVILY_API_KEY=your_tavily_api_key
```

## Installation and Setup

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd streamlit-chatbot

# Create .env file with required variables
cp .env.example .env
# Edit .env with your configuration

#!/bin/bash
clear
set -e  # Exit on any error
export COMPOSE_BAKE=true
docker run -it --rm -v ./docker:/docker --workdir /docker --entrypoint uv ghcr.io/astral-sh/uv:python3.13-bookworm-slim sync
docker compose build app
docker compose up app nginx -d
docker compose logs -f app
```

## Development Guide

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
from tools.base import BaseTool

class NewTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "tool_name"
        self.description = "Tool description"
        self.llm_type = "fast"  # or "llm" or "intelligent"

    def to_openai_format(self):
        # Return OpenAI function definition

    def _run(self, **kwargs):
        # Implement tool logic
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

## Production Considerations

### Performance

- Implements streaming responses for better user experience
- Uses parallel tool execution to minimize latency
- Employs intelligent document analysis to handle large PDFs efficiently

### Reliability

- Comprehensive error handling at all layers
- Automatic fallback mechanisms for tool failures
- Session isolation prevents cross-user data leakage

### Scalability

- Stateless design allows horizontal scaling
- External file storage prevents memory exhaustion
- Configurable model selection for cost optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
