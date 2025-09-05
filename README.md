# Streamlit Agentic Chatbot

A production-ready conversational AI application built with Streamlit, featuring advanced language model capabilities, intelligent document analysis, multimodal interactions, and a sophisticated **Model-View-Controller (MVC)** architecture with framework abstraction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [REST API](#rest-api)
- [Available Tools](#available-tools)
- [MVC Implementation](#mvc-implementation)
- [Services Layer](#services-layer)
- [Development Guide](#development-guide)
- [Production Deployment](#production-deployment)

## Overview

This application provides a sophisticated agentic chatbot interface powered by NVIDIA's language models, available both as a web UI and a RESTful API, with a **production-grade MVC architecture** that ensures:

- 🏗️ **Clean separation of concerns** between Models, Views, and Controllers
- 🎯 **Framework abstraction** through view interfaces and helpers
- 🔗 **Service-oriented architecture** with specialized business logic services
- ✅ **Domain-driven design** with rich models and validation
- 🧪 **High testability** through dependency injection and interfaces
- 🚀 **Production scalability** with external file storage and efficient processing

### Core Capabilities

- 🤖 **Real-time streaming responses** with advanced tool orchestration
- 📄 **Intelligent PDF analysis** with query-aware processing and batch handling
- 🎨 **AI-powered image generation and analysis** with VLM support
- 🔍 **Multi-source information retrieval** (web search, news, weather)
- 💬 **Smart context management** with automatic conversation and PDF context injection
- 📁 **Robust session management** with isolated user sessions and file storage
- ⚡ **Optimized processing** for large documents and multimodal content

## Features

### Advanced Tool System

The application includes **12 specialized tools** providing comprehensive AI capabilities:

1. **Text Assistant** (`text_assistant`) - Comprehensive text processing with multiple capabilities:
   - Document analysis and insights extraction
   - Intelligent summarization
   - Grammar and style proofreading
   - Content rewriting and enhancement
   - Constructive criticism and feedback
   - Multi-language translation
   - Code development assistance

1. **Image Generation** (`generate_image`) - AI-powered image creation with:
   - Context-aware prompt enhancement
   - Multiple style presets (photorealistic, digital art, oil painting, etc.)
   - Aspect ratio control
   - Conversation context integration

1. **Image Analysis** (`analyze_image`) - Vision-capable LLM for:
   - Image description and understanding
   - Object identification
   - Visual question answering
   - Automatic image optimization for VLM processing

1. **PDF Summary** (`retrieve_pdf_summary`) - Intelligent document summarization with:
   - Recursive summarization for large documents
   - Query-aware processing
   - Batch processing for efficiency

1. **PDF Text Processor** (`process_pdf_text`) - Advanced PDF analysis:
   - Query-specific document analysis
   - Intelligent page selection
   - Context-aware responses

1. **Web Search** (`serpapi_internet_search`) - General web search integration

1. **News Search** (`serpapi_news_search`) - Real-time news and current events

1. **Weather** (`get_weather`) - Current weather information

1. **Web Extract** (`extract_web_content`) - Clean content extraction from URLs

1. **Retriever** (`retrieval_search`) - Vector-based semantic search for knowledge bases

1. **Conversation Context** (`conversation_context`) - Intelligent conversation analysis:
   - Conversation summaries
   - Topic extraction
   - Task continuity tracking
   - Document analysis context

1. **Generalist Conversation** (`generalist_conversation`) - General discussion and explanations

1. **Context Generation** (`context_generation`) - Generate or modify images based on an existing image and text

### Production Features

- **RESTful API**: OpenAI-compatible REST API endpoint for programmatic access
- **Streaming Architecture**: Efficient real-time response streaming with progress indicators
- **Smart Context Management**: Automatic injection of conversation and document context
- **External File Storage**: Prevents memory exhaustion with disk-based file management
- **Batch Processing**: Efficient handling of large PDFs through intelligent batching
- **Session Isolation**: Secure, isolated user sessions with proper cleanup
- **Error Recovery**: Comprehensive error handling with graceful degradation
- **View Abstraction**: Framework-agnostic UI layer for easy testing and migration
- **Flexible Tool Configuration**: Enable/disable tools dynamically via configuration
- **Per-Model API Keys**: Support for different API keys per model type for better resource management

## Architecture

This application implements a **sophisticated MVC pattern** with service-oriented architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                           VIEW LAYER                            │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │   UI Components     │  │      View Abstraction            │  │
│  │   - ChatHistory     │  │  - IViewInterface                │  │
│  │   - Streamlit UI    │  │  - StreamlitViewInterface        │  │
│  └─────────────────────┘  │  - ViewHelperFactory             │  │
│                           │  - Framework Independence        │  │
│                           └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                        CONTROLLER LAYER                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Controllers                             │ │
│  │   - SessionController: Session state & lifecycle           │ │
│  │   - MessageController: Message validation & processing     │ │
│  │   - FileController: File upload coordination               │ │
│  │   - ImageController: Image processing coordination         │ │
│  │   - ResponseController: LLM response orchestration         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER                           │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │   Core Services     │  │    Specialized Services          │  │
│  │   - LLMService      │  │  - PDFAnalysisService            │  │
│  │   - ChatService     │  │  - PDFContextService             │  │
│  │   - StreamingService│  │  - PDFSummarizationService       │  │
│  │   - FileStorage     │  │  - DocumentAnalyzerService       │  │
│  └─────────────────────┘  │  - TextProcessorService          │  │
│                           │  - TranslationService            │  │
│                           │  - ConversationContextService    │  │
│                           │  - ToolExecutionService          │  │
│                           └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                             │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │   Domain Models     │  │        Tool System               │  │
│  │   - User            │  │  - BaseTool (MVC pattern)        │  │
│  │   - Session         │  │  - Tool Registry                 │  │
│  │   - ChatMessage     │  │  - Tool Controllers              │  │
│  │   - ChatConfig      │  │  - Tool Views                    │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
./
├── docker/
│   └── app/
│       ├── main.py                      # Streamlit application entry point
│       ├── api.py                       # FastAPI REST API server
│       │
│       ├── controllers/                 # 🎮 CONTROLLER LAYER
│       │   ├── session_controller.py    # Session management
│       │   ├── message_controller.py    # Message processing
│       │   ├── file_controller.py       # File upload handling
│       │   ├── image_controller.py      # Image processing
│       │   └── response_controller.py   # Response generation
│       │
│       ├── models/                      # 📊 MODEL LAYER
│       │   ├── user.py                  # User domain model
│       │   ├── session.py               # Session domain model
│       │   ├── chat_message.py          # Message model
│       │   ├── chat_config.py           # Configuration wrapper
│       │   └── validation.py            # Validation service
│       │
│       ├── ui/                          # 🖥️ VIEW LAYER
│       │   ├── components.py            # UI components
│       │   ├── view_helpers.py          # View abstraction helpers
│       │   └── view_interfaces.py       # Interface definitions
│       │
│       ├── services/                    # 🔧 SERVICE LAYER
│       │   ├── llm_service.py          # LLM orchestration
│       │   ├── chat_service.py         # Chat processing
│       │   ├── streaming_service.py     # Response streaming
│       │   ├── tool_execution_service.py # Tool execution
│       │   ├── file_storage_service.py  # File management
│       │   ├── pdf_analysis_service.py  # PDF analysis
│       │   ├── pdf_context_service.py   # PDF context injection
│       │   ├── document_analyzer_service.py # Document analysis
│       │   ├── text_processor_service.py # Text processing
│       │   └── translation_service.py   # Translation service
│       │
│       ├── tools/                       # 🛠️ TOOL SYSTEM
│       │   ├── base.py                 # Base tool with MVC
│       │   ├── registry.py             # Tool registry
│       │   ├── assistant.py            # Text assistant tool
│       │   ├── image_gen.py            # Image generation
│       │   ├── image_analysis_tool.py  # Image analysis
│       │   ├── conversation_context.py # Context analysis
│       │   ├── generalist.py           # General conversation
│       │   └── ...                     # Other tools
│       │
│       └── utils/                       # 🔧 UTILITIES
│           ├── config.py               # Centralized config
│           ├── animated_loading.py     # UI animations
│           └── ...                     # Other utilities
│
├── docker-compose.yml                   # Container orchestration
├── .env.example                        # Environment template
└── README.md                           # This documentation
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
   git clone https://github.com/tuttlebr/streamlit-agent.git
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
   - Streamlit UI: `http://localhost:80`
   - REST API: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

## Configuration

All configuration is managed through environment variables and centralized in `utils/config.py`.

### Required Environment Variables

```bash
# Global NVIDIA API Configuration (used as default for all models)
NVIDIA_API_KEY=your_api_key_here

# LLM Model Configuration
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct

FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct

INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Vision Language Model (for image analysis)
VLM_ENDPOINT=https://integrate.api.nvidia.com/v1
VLM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1

# Application Settings
BOT_TITLE=Streamlit Agentic Chatbot
```

### Per-Model API Keys (Optional)

The application now supports **individual API keys for each model type**, providing greater flexibility in resource management and usage tracking. If not specified, all models will fall back to using the global `NVIDIA_API_KEY`.

```bash
# Individual model API keys (optional - defaults to NVIDIA_API_KEY if not set)
FAST_LLM_API_KEY=your_fast_llm_api_key          # For quick operations (weather, search)
LLM_API_KEY=your_standard_llm_api_key           # For general text processing
INTELLIGENT_LLM_API_KEY=your_intelligent_key    # For complex reasoning & tool selection
VLM_API_KEY=your_vision_model_api_key           # For image analysis
EMBEDDING_API_KEY=your_embedding_api_key        # For vector embeddings
RERANKER_API_KEY=your_reranker_api_key         # For search result reranking
IMAGE_API_KEY=your_image_generation_api_key     # For image generation
```

#### Benefits of Per-Model API Keys:

- **Multi-Provider Support**: Use different API providers for different models (e.g., NVIDIA for LLMs, custom endpoint for embeddings)
- **Resource Management**: Apply different rate limits or quotas per model type
- **Cost Tracking**: Monitor usage and costs separately for each model type
- **Fallback Support**: Automatically falls back to `NVIDIA_API_KEY` if individual keys aren't set
- **Backward Compatibility**: Existing setups with just `NVIDIA_API_KEY` continue to work

### Optional Services

```bash
# Image Generation
IMAGE_ENDPOINT=your_image_generation_endpoint

# Web Search (Tavily)
TAVILY_API_KEY=your_tavily_api_key

# PDF Processing (NVIngest)
NVINGEST_ENDPOINT=http://localhost:7670

# Vector Database (for retrieval tool)
EMBEDDING_ENDPOINT=https://integrate.api.nvidia.com/v1
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
COLLECTION_NAME=your_collection_name
DATABASE_URL=your_milvus_url
DEFAULT_DB=default
```

### Model Usage by Tool

The application intelligently selects the appropriate LLM model based on the tool:

- **Fast Model** (`FAST_LLM_MODEL_NAME`): Used for quick operations
  - Weather queries
  - Web searches
  - News searches
  - Image generation prompts

- **Standard Model** (`LLM_MODEL_NAME`): Used for general tasks
  - Conversation context analysis
  - Web content extraction
  - Text processing tasks

- **Intelligent Model** (`INTELLIGENT_LLM_MODEL_NAME`): Used for complex reasoning
  - Tool selection and orchestration

- **Vision Model** (`VLM_MODEL_NAME`): Used for image analysis
  - Image understanding and description

### Tool Configuration

The application supports flexible tool configuration through environment variables or configuration files:

```bash
# Enable/disable specific tools via environment variables
TOOL_ENABLE_TEXT_ASSISTANT=true
TOOL_ENABLE_CONVERSATION_CONTEXT=true
TOOL_ENABLE_SERPAPI_INTERNET_SEARCH=false
TOOL_ENABLE_PDF_ASSISTANT=true

# Use enhanced tool descriptions
USE_ENHANCED_TOOL_DESCRIPTIONS=true

# Or use a configuration file
TOOL_CONFIG_FILE=/path/to/tool_config.json
```

Example tool configuration file (`tool_config.json`):

```json
{
  "enabled_tools": {
    "text_assistant": true,
    "conversation_context": true,
    "serpapi_internet_search": false,
    "pdf_assistant": true,
    "analyze_image": true,
    "generate_image": true
  }
}
```

## REST API

The application provides a RESTful API that is fully compatible with the OpenAI Chat Completions API format, making it easy to integrate with existing tools and libraries.

### API Endpoints

#### Chat Completion

```
POST /agent
```

Send chat completion requests in OpenAI format:

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "model": "fast",
    "temperature": 0.7,
    "stream": false
  }'
```

#### Streaming Response

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a story about a robot"}
    ],
    "stream": true
  }'
```

#### With Session ID

```
POST /agent/{session_id}
```

Maintain conversation continuity with session IDs:

```bash
curl -X POST http://localhost:8000/agent/my-session-123 \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Continue our previous discussion"}
    ]
  }'
```

### API Features

- **OpenAI Compatibility**: Works with any OpenAI-compatible client library
- **Streaming Support**: Real-time streaming responses for better UX
- **Session Management**: Maintain conversation context across requests
- **Tool Integration**: Full access to all enabled tools via API
- **Model Selection**: Specify which model to use (fast, standard, intelligent, vlm)

### Running the API

To run the API server alongside the Streamlit app:

```bash
# Using Docker Compose (recommended)
docker compose up app api nginx -d

# Or run directly
python docker/app/api.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### API Response Format

Responses follow the OpenAI Chat Completion format:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "meta/llama-3.1-70b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The weather in San Francisco is currently 65°F with partly cloudy skies."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 18,
    "total_tokens": 30
  }
}
```

## MVC Implementation

The application follows a strict Model-View-Controller pattern with additional service layer:

### Controllers

Controllers coordinate between models, services, and views:

```python
# Example: SessionController manages session lifecycle
class SessionController:
    def __init__(self, config_obj: ChatConfig):
        self.config = config_obj
        self.file_storage = FileStorageService()

    def get_current_session(self) -> Session:
        """Get or create current session with proper initialization"""
        # Returns domain model with business logic
```

### Services Layer

Services encapsulate complex business logic:

- **LLMService**: Orchestrates LLM interactions, tool execution, and response streaming
- **PDFAnalysisService**: Handles intelligent PDF analysis with query-aware processing
- **ConversationContextService**: Manages automatic context injection
- **FileStorageService**: Singleton service for external file management
- **ToolExecutionService**: Coordinates tool execution strategies (parallel/sequential)

### View Abstraction

The view layer is abstracted through interfaces:

```python
# View interface for framework independence
class IViewInterface(ABC):
    @abstractmethod
    def show_message(self, message: UIMessage) -> None:
        """Display a message to the user"""

# Streamlit implementation
class StreamlitViewInterface(IViewInterface):
    def show_message(self, message: UIMessage) -> None:
        if message.message_type == "success":
            st.success(message.content)
```

### Tool System with MVC

Each tool follows the MVC pattern:

```python
class BaseTool(ABC):
    def __init__(self):
        self._controller = None  # Tool business logic
        self._view = None       # Tool response formatting
        self._initialize_mvc()  # Set up components
```

## Services Layer

The service layer provides specialized business logic:

### PDF Processing Services

- **PDFAnalysisService**: Intelligent query-specific analysis
- **PDFSummarizationService**: Recursive summarization for large documents
- **PDFContextService**: Automatic PDF context injection
- **PDFBatchProcessor**: Efficient batch processing for large files

### Text Processing Services

- **TextProcessorService**: Handles all text processing tasks
- **TranslationService**: Multi-language translation
- **DocumentAnalyzerService**: Deep document analysis

### Core Services

- **StreamingService**: Manages LLM streaming responses
- **ResponseParsingService**: Parses LLM responses and extracts tool calls
- **LLMClientService**: Singleton service for LLM client management

### Context Services

- **ConversationContextService**: Automatic conversation context injection
- **FileStorageService**: External file storage management

### Controllers

Controllers coordinate between models, services, and views:

```python
from models import User, Session, validation_service
from ui.view_helpers import view_factory
from services import YourBusinessService

class NewController:
    """Controller following MVC principles"""

    def __init__(self, config_obj: ChatConfig):
        self.config_obj = config_obj
        self.business_service = YourBusinessService(config_obj)

        # Initialize view helpers for UI operations
        self.message_helper = view_factory.create_message_helper()
        self.form_helper = view_factory.create_form_helper()

        # Cache for domain models
        self._current_model: Optional[YourModel] = None

    def process_user_action(self, action_data: Dict[str, Any]) -> None:
        """Process user action using domain models"""

        # 1. Validate input using domain models
        model = YourModel(**action_data)
        validation_result = validation_service.validate_model(model)

        if not validation_result.is_valid:
            self.message_helper.show_error("Validation failed")
            return

        # 2. Execute business logic through services
        try:
            result = self.business_service.process(model)

            # 3. Update domain models
            model.update_from_result(result)
            self._current_model = model

            # 4. Display results using view helpers
            self.message_helper.show_success("Action completed successfully")

        except Exception as e:
            logging.error(f"Error processing action: {e}")
            self.message_helper.show_error(f"Processing failed: {str(e)}")
```

### Services Layer

Services handle business logic and can utilize domain models:

```python
from models import User, Session, validation_service

class EnhancedService:
    """Service with domain model integration"""

    def __init__(self, config: ChatConfig):
        self.config = config

    async def process_with_validation(self, user: User, session: Session) -> Dict[str, Any]:
        """Process with comprehensive validation"""

        # Validate individual models
        user_validation = validation_service.validate_user(user)
        session_validation = validation_service.validate_session(session)

        # Validate cross-model consistency
        consistency_validation = validation_service.validate_user_session_pair(user, session)

        # Combine validation results
        if not all([user_validation.is_valid, session_validation.is_valid, consistency_validation.is_valid]):
            errors = []
            errors.extend(user_validation.get_error_messages())
            errors.extend(session_validation.get_error_messages())
            errors.extend(consistency_validation.get_error_messages())
            raise ValueError(f"Validation failed: {errors}")

        # Execute business logic with validated models
        result = await self._execute_business_logic(user, session)

        # Update domain models
        user.update_activity()
        session.update_timestamp()

        return result
```

## Development Guide

### Adding New Tools

Create a new tool following the MVC pattern:

```python
from tools.base import BaseTool, BaseToolResponse, ToolController, ToolView
from typing import Dict, Any, Type

class MyToolResponse(BaseToolResponse):
    """Response model for the tool"""
    result: str
    metadata: Dict[str, Any] = {}

class MyToolController(ToolController):
    """Business logic for the tool"""
    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implement tool logic
        return {"result": "processed"}

class MyToolView(ToolView):
    """Response formatting for the tool"""
    def format_response(self, data: Dict[str, Any], response_type: Type[BaseToolResponse]) -> BaseToolResponse:
        return response_type(success=True, result=data["result"])

class MyTool(BaseTool):
    """Main tool class"""
    def __init__(self):
        super().__init__()
        self.name = "my_tool"
        self.description = "Tool description"
        self.llm_type = "fast"  # or "llm", "intelligent", "vlm"

    def _initialize_mvc(self):
        self._controller = MyToolController()
        self._view = MyToolView()

    def get_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Parameter description"}
                    },
                    "required": ["param1"]
                }
            }
        }
```

### Creating New Services

Services encapsulate business logic:

```python
from typing import Dict, Any, Optional
import asyncio

class MyBusinessService:
    """Service for specific business logic"""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.llm_client = self._init_client()

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Async processing with proper error handling"""
        try:
            # Business logic here
            result = await self._perform_operation(data)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {"success": False, "error": str(e)}
```

### Working with Controllers

Controllers coordinate the application flow:

```python
from controllers import BaseController
from services import MyBusinessService
from ui.view_helpers import ViewHelperFactory

class MyController:
    """Controller for managing specific functionality"""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.service = MyBusinessService(config)
        self.view_factory = ViewHelperFactory()
        self.message_helper = self.view_factory.create_message_helper()

    def handle_user_action(self, action_data: Dict[str, Any]):
        """Handle user action with proper MVC flow"""
        try:
            # Validate input
            if not self._validate_input(action_data):
                self.message_helper.show_error("Invalid input")
                return

            # Process through service
            result = self.service.process(action_data)

            # Update UI through view helpers
            if result["success"]:
                self.message_helper.show_success("Operation completed")
            else:
                self.message_helper.show_error(result["error"])

        except Exception as e:
            logger.error(f"Controller error: {e}")
            self.message_helper.show_error("An error occurred")
```

### Best Practices

1. **Separation of Concerns**: Keep business logic in services, UI logic in views
2. **Use View Abstractions**: Always use view helpers instead of direct Streamlit calls
3. **Error Handling**: Implement comprehensive error handling at each layer
4. **Async Operations**: Use async/await for I/O operations
5. **Tool Configuration**: Set appropriate `llm_type` for each tool based on requirements
6. **Service Patterns**: Use singleton pattern for stateful services (e.g., FileStorageService)
7. **Testing**: Write tests for controllers and services independently of views

## Production Deployment

### Docker Deployment

The application is containerized for easy deployment:

```bash
# Build and run with Docker Compose
docker compose up -d

# View logs
docker compose logs -f app api

# Access the services
# Streamlit UI: http://localhost:80
# REST API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
# App Documentation: http://localhost:8001
```

### Environment Configuration

Create a `.env` file with your configuration:

```bash
# Copy the example file
cp env.example .env

# Edit with your API keys
nano .env
```

Basic configuration:

```bash
# Required
NVIDIA_API_KEY=your_key_here

# Optional services
TAVILY_API_KEY=your_key_here
IMAGE_ENDPOINT=your_endpoint_here

# Optional per-model API keys (see env.example for full list)
# FAST_LLM_API_KEY=your_key_here
# LLM_API_KEY=your_key_here
```

### Production Features

1. **External File Storage**: Files stored on disk to prevent memory issues
2. **Session Isolation**: Each user has isolated session state
3. **Error Recovery**: Comprehensive error handling at all layers
4. **Resource Management**: Efficient streaming and batch processing
5. **Monitoring**: Built-in logging and error tracking

### Kubernetes Deployment

The application includes complete Kubernetes deployment configurations:

#### 1. Create the namespace and secrets

```bash
# Create namespace
kubectl create namespace streamlit-chatbot

# Create app secrets with per-model API keys
./create-app-secrets.sh
```

The `create-app-secrets.sh` script now supports per-model API keys:

- First, it prompts for the global NVIDIA_API_KEY (used as default)
- Then optionally allows you to set individual API keys for each model
- Press Enter to skip any individual API key and use the global default

#### 2. Deploy the application

```bash
# Apply the Kubernetes manifests
kubectl apply -f kubernetes-deployment.yaml

# Check deployment status
kubectl get pods -n streamlit-chatbot
kubectl get services -n streamlit-chatbot
```

#### 3. Managing API Keys in Kubernetes

To update API keys after deployment:

```bash
# Delete existing secret
kubectl delete secret app-secrets -n streamlit-chatbot

# Re-run the script with new keys
./create-app-secrets.sh

# Restart the deployment to pick up new keys
kubectl rollout restart deployment/app -n streamlit-chatbot
```

#### 4. Using env-to-k8s.sh for bulk configuration

If you have a `.env` file with all your settings:

```bash
# Convert .env to Kubernetes ConfigMap and Secret
./env-to-k8s.sh .env

# The script automatically identifies sensitive values (API keys)
# and creates appropriate ConfigMap and Secret resources
```

### Performance Considerations

- **LLM Model Selection**: Different models for different tasks optimize cost/performance
- **Parallel Tool Execution**: Tools execute in parallel when possible
- **Streaming Responses**: Efficient memory usage with streaming
- **Batch Processing**: Large PDFs processed in configurable batches
- **Context Management**: Automatic context injection with configurable limits

### Scaling Considerations

- **Stateless Design**: Application can be horizontally scaled
- **External Storage**: File storage can be moved to S3/cloud storage
- **Database Integration**: Vector database for knowledge retrieval
- **Load Balancing**: Nginx included for reverse proxy setup

### Security Notes

- **API Key Management**: Use environment variables, never commit keys
- **File Validation**: All uploads validated before processing
- **Session Security**: Sessions isolated by user
- **Input Validation**: All user inputs sanitized

This architecture ensures the application is production-ready with proper separation of concerns, comprehensive error handling, and scalable design.
