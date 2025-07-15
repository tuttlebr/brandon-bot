# Streamlit Agentic Application

A production-ready conversational AI application built with Streamlit, featuring advanced language model capabilities, document analysis, multimodal interactions, and a sophisticated **Model-View-Controller (MVC)** architecture.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Enhanced MVC Architecture](#enhanced-mvc-architecture)
- [Domain Models](#domain-models)
- [View Abstraction Layer](#view-abstraction-layer)
- [Validation System](#validation-system)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Available Tools](#available-tools)
- [Development Guide](#development-guide)
- [API Reference](#api-reference)
- [Production Deployment](#production-deployment)

## Overview

This application provides a sophisticated agentic interface powered by NVIDIA's language models, implemented with a **gold-standard MVC architecture** that ensures:

- 🏗️ **Clean separation of concerns** between Models, Views, and Controllers
- 🎯 **Domain-driven design** with rich business logic encapsulation
- 🔗 **Framework abstraction** enabling easy UI framework switching
- ✅ **Comprehensive validation** with cross-model consistency checks
- 🧪 **High testability** through dependency injection and interfaces
- 🚀 **Production-ready scalability** with proper error handling

### Core Capabilities

- 🤖 **Real-time streaming responses** with tool calling capabilities
- 📄 **Intelligent PDF document analysis** with progress tracking
- 🎨 **AI-powered image generation and analysis**
- 🔍 **Semantic search** through knowledge bases
- 🌐 **Web content extraction and search**
- 💬 **Smart context management** with automatic conversation context injection
- 📁 **Session management** with isolated user sessions and external file storage
- ⚡ **Batch processing** for efficient handling of large documents

## Features

### Advanced Tool System

11 specialized tools providing comprehensive AI capabilities:

1. **Text Assistant** (`text_assistant`) - Analysis, summarization, translation, and code development
2. **Image Generation** (`generate_image`) - AI-powered image creation with style control
3. **Image Analysis** (`analyze_image`) - Vision-capable LLM for image understanding
4. **PDF Summary** (`retrieve_pdf_summary`) - Intelligent document summarization
5. **PDF Text Processor** (`process_pdf_text`) - Advanced PDF text extraction
6. **Web Search** (`tavily_internet_search`) - General internet search capabilities
7. **News Search** (`tavily_news_search`) - Real-time news and current events
8. **Weather** (`weather`) - Current weather information for any location
9. **Web Extract** (`extract_web_content`) - Clean content extraction from web pages
10. **Retriever** (`retrieval_search`) - Semantic search through knowledge bases
11. **Conversation Context** (`conversation_context`) - Conversation history analysis

### Production Features

- **Real-time Streaming**: Smooth response streaming with progress indicators
- **Multimodal Interface**: Support for text, images, and document-based conversations
- **External File Storage**: Prevents memory exhaustion in production environments
- **Intelligent Document Processing**: Hierarchical processing with query-aware analysis
- **Session Isolation**: Secure, isolated user sessions with proper cleanup
- **Error Recovery**: Comprehensive error handling with automatic fallbacks

## Enhanced MVC Architecture

This application implements a **sophisticated MVC pattern** with modern enhancements:

```
┌─────────────────────────────────────────────────────────────────┐
│                           VIEW LAYER                            │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │   UI Components     │  │        View Interfaces           │  │
│  │   - ChatHistory     │  │  - IChatDisplayInterface         │  │
│  │   - Streamlit UI    │  │  - IFileManagementInterface      │  │
│  │   - Components      │  │  - ISessionManagementInterface   │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
│           │                           │                         │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │    View Helpers     │  │      View Abstractions           │  │
│  │   - MessageHelper   │  │  - StreamlitViewInterface        │  │
│  │   - FormHelper      │  │  - ViewHelperFactory             │  │
│  │   - FileHelper      │  │  - Framework Independence        │  │
│  │   - ProgressHelper  │  │  - Easy Testing & Mocking        │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                        CONTROLLER LAYER                         │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │    Controllers      │  │       Enhanced Features          │  │
│  │   - SessionContro...│  │  - Domain Model Integration      │  │
│  │   - MessageContro...│  │  - View Helper Usage             │  │
│  │   - FileController  │  │  - Reduced Framework Coupling    │  │
│  │   - ImageController │  │  - Comprehensive Validation      │  │
│  │   - ResponseContr...│  │  - Dependency Injection          │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                             │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │   Domain Models     │  │       Validation Layer           │  │
│  │   - User            │  │  - ValidationService             │  │
│  │   - Session         │  │  - Cross-Model Validation        │  │
│  │   - ToolEntity      │  │  - Business Rule Enforcement     │  │
│  │   - FileInfo        │  │  - Comprehensive Error Reporting │  │
│  │   - ChatMessage     │  │  - Batch Validation Support      │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
./
├── docker/
│   └── app/
│       ├── main.py                      # Application entry point & orchestration
│       │
│       ├── controllers/                 # 🎮 CONTROLLER LAYER
│       │   ├── session_controller.py    # Session state & lifecycle management
│       │   ├── message_controller.py    # Message processing & validation
│       │   ├── file_controller.py       # File upload & processing coordination
│       │   ├── image_controller.py      # Image upload & processing coordination
│       │   └── response_controller.py   # LLM response generation & streaming
│       │
│       ├── models/                      # 📊 MODEL LAYER
│       │   ├── user.py                  # User domain model with business logic
│       │   ├── session.py               # Session domain model with state management
│       │   ├── tool_entity.py           # Tool domain model with execution tracking
│       │   ├── chat_message.py          # Message domain model with transformations
│       │   ├── chat_config.py           # Configuration wrapper model
│       │   └── validation.py            # Comprehensive validation service
│       │
│       ├── ui/                          # 🖥️ VIEW LAYER
│       │   ├── components.py            # UI component implementations
│       │   ├── view_helpers.py          # Framework abstraction helpers
│       │   └── view_interfaces.py       # UI operation contracts & interfaces
│       │
│       ├── services/                    # 🔧 SERVICE LAYER
│       │   ├── llm_service.py          # LLM interaction orchestration
│       │   ├── chat_service.py         # Chat processing business logic
│       │   ├── streaming_service.py     # Response streaming management
│       │   ├── tool_execution_service.py # Tool orchestration & execution
│       │   ├── file_storage_service.py  # External file storage management
│       │   ├── pdf_analysis_service.py  # Intelligent PDF processing
│       │   └── ...                     # Additional specialized services
│       │
│       ├── tools/                       # 🛠️ TOOL SYSTEM
│       │   ├── base.py                 # Abstract tool interface
│       │   ├── registry.py             # Tool registry & management
│       │   ├── assistant.py            # Text processing tool
│       │   ├── image_gen.py            # Image generation tool
│       │   └── ...                     # Additional tool implementations
│       │
│       └── utils/                       # 🔧 UTILITIES
│           ├── config.py               # Centralized configuration management
│           └── ...                     # Additional utilities
│
├── docker-compose.yml                   # 🐳 Container orchestration
├── .env.example                        # 📝 Environment configuration template
└── README.md                           # 📚 This documentation
```

## Domain Models

The application uses **rich domain models** that encapsulate business logic and provide type safety:

### User Model (`models/user.py`)

```python
from models import User, UserRole, UserPreferences

# Create user with validation
user = User(
    user_id="user_123",
    role=UserRole.USER,
    preferences=UserPreferences(
        language="en",
        theme="dark",
        message_limit=100
    )
)

# Business logic methods
user.increment_message_count()
user.can_send_message()  # Checks against message limit
user.get_display_name()  # Returns formatted display name
```

### Session Model (`models/session.py`)

```python
from models import Session, SessionStatus, ProcessingStatus, FileInfo

# Create session with comprehensive state tracking
session = Session(
    session_id="session_456",
    user_id="user_123",
    status=SessionStatus.ACTIVE,
    processing_status=ProcessingStatus.IDLE
)

# File management
file_info = session.add_uploaded_file("doc.pdf", "file_789", "pdf", 1024)
session.mark_file_processed("file_789")
pdf_files = session.get_files_by_type("pdf")

# Context management
session.set_context("current_topic", "AI Architecture")
topic = session.get_context("current_topic")
```

### Tool Entity Model (`models/tool_entity.py`)

```python
from models import ToolEntity, ToolStatus, ToolType, ExecutionMetrics

# Create tool with metadata and tracking
tool = ToolEntity(
    tool_id="text_assistant",
    name="Text Assistant",
    display_name="AI Text Assistant",
    description="Comprehensive text processing tool",
    tool_type=ToolType.TEXT_PROCESSING,
    status=ToolStatus.AVAILABLE
)

# Execution tracking
tool.record_execution(success=True, execution_time=1.5)
success_rate = tool.metrics.get_success_rate()  # Returns percentage
```

## View Abstraction Layer

The **view abstraction layer** provides framework independence and improved testability:

### View Interfaces (`ui/view_interfaces.py`)

```python
from ui import IChatDisplayInterface, IFileManagementInterface

# Abstract interfaces define contracts for UI operations
class IChatDisplayInterface(ABC):
    @abstractmethod
    def display_message(self, role: str, content: str) -> None: ...

    @abstractmethod
    def get_user_message(self, placeholder: str) -> Optional[str]: ...
```

### View Helpers (`ui/view_helpers.py`)

```python
from ui import view_factory

# Use framework-agnostic helpers in controllers
message_helper = view_factory.create_message_helper()
file_helper = view_factory.create_file_helper()
progress_helper = view_factory.create_progress_helper()

# Display operations without framework coupling
message_helper.show_success("Operation completed successfully!")
file_result = file_helper.upload_pdf("Upload your document")
with progress_helper.show_indeterminate_progress("Processing..."):
    # Long-running operation
    pass
```

## Validation System

The **comprehensive validation system** ensures data integrity and business rule compliance:

### Validation Service (`models/validation.py`)

```python
from models import validation_service, User, Session

# Validate individual models
user = User(user_id="test", preferences=UserPreferences(message_limit=-1))
result = validation_service.validate_user(user)

if not result.is_valid:
    for error in result.errors:
        print(f"Field {error.field}: {error.message}")

# Cross-model validation
session = Session(session_id="test", user_id="different_user")
consistency_result = validation_service.validate_user_session_pair(user, session)

# Batch validation
models = [user1, user2, session1, session2]
batch_results = validation_service.validate_batch(models)
```

### Business Rule Validation

- **User validation**: Message limits, preferences, role consistency
- **Session validation**: File integrity, context serialization, timestamp logic
- **Tool validation**: Parameter requirements, execution metrics
- **Cross-model validation**: User-session consistency, file-session alignment

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

All configuration is centralized in `utils/config.py` with comprehensive validation.

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

### Text Processing Capabilities

The `text_assistant` tool supports multiple task types:

- **analyze** - Document analysis and insights extraction
- **summarize** - Create concise summaries of long content
- **proofread** - Grammar and style corrections
- **rewrite** - Enhance clarity and impact while preserving meaning
- **critic** - Provide constructive feedback and improvement suggestions
- **translate** - Convert text between languages with context awareness
- **develop** - Programming assistance and code development
- **qa** - Answer questions about document content with citations

### Multimodal Capabilities

- **Image Generation**: AI-powered image creation with customizable styles, moods, and aspect ratios
- **Image Analysis**: Vision-capable LLM for describing, identifying objects, or answering questions about images
- **Document Processing**: Intelligent PDF analysis with query-aware processing and hierarchical summarization
- **Web Integration**: Real-time web search, news retrieval, and clean content extraction

### Information Retrieval

- **Semantic Search**: Query knowledge bases using vector similarity matching
- **Web Search**: General internet search capabilities using Tavily
- **News Search**: Specialized search for current events and trending topics
- **Weather Information**: Current weather data for any global location

## Development Guide

### Adding New Domain Models

Create rich domain models with validation and business logic:

```python
from models import BaseModel, Field, validator
from datetime import datetime
from typing import Optional

class NewDomainModel(BaseModel):
    """Rich domain model with business logic"""

    entity_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Entity name")
    created_at: datetime = Field(default_factory=datetime.now)

    @validator('entity_id')
    def validate_entity_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Entity ID cannot be empty")
        return v.strip()

    def update_timestamp(self) -> None:
        """Business logic method"""
        self.updated_at = datetime.now()

    def to_display_format(self) -> Dict[str, Any]:
        """View formatting method"""
        return {
            "id": self.entity_id,
            "display_name": self.name,
            "created": self.created_at.isoformat()
        }
```

### Creating Controllers with MVC Pattern

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

### Implementing View Abstractions

Create framework-agnostic UI operations:

```python
from ui.view_interfaces import IApplicationInterface
from ui.view_helpers import UIMessage, FileUploadResult

class CustomViewImplementation(IApplicationInterface):
    """Custom view implementation for different frameworks"""

    def show_message(self, message: UIMessage) -> None:
        """Display message using custom framework"""
        # Framework-specific implementation
        pass

    def show_file_uploader(self, accepted_types: List[str], **kwargs) -> FileUploadResult:
        """Show file uploader using custom framework"""
        # Framework-specific implementation
        pass
```

### Creating Tools with Domain Integration

Tools can leverage domain models for better type safety:

```python
from tools.base import BaseTool, BaseToolResponse
from models import ToolEntity, ToolStatus

class EnhancedToolResponse(BaseToolResponse):
    """Type-safe tool response"""
    result: str
    metadata: Dict[str, Any]
    execution_time: float

class NewTool(BaseTool):
    """Tool with domain model integration"""

    def __init__(self):
        super().__init__()
        self.name = "new_tool"
        self.description = "Enhanced tool with domain models"

        # Create tool entity for tracking
        self.tool_entity = ToolEntity(
            tool_id=self.name,
            name="New Tool",
            description=self.description,
            tool_type=ToolType.CUSTOM,
            status=ToolStatus.AVAILABLE
        )

    def execute(self, params: Dict[str, Any]) -> EnhancedToolResponse:
        """Execute with comprehensive tracking"""
        start_time = time.time()

        try:
            # Validate parameters using domain model
            validation_errors = self.tool_entity.validate_parameters(params)
            if validation_errors:
                raise ValueError(f"Invalid parameters: {validation_errors}")

            # Execute tool logic
            result = self._process_request(params)

            # Record successful execution
            execution_time = time.time() - start_time
            self.tool_entity.record_execution(True, execution_time)

            return EnhancedToolResponse(
                success=True,
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            # Record failed execution
            execution_time = time.time() - start_time
            self.tool_entity.record_execution(False, execution_time, str(e))

            return EnhancedToolResponse(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
```

### Service Layer Best Practices

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

## API Reference

### Request Flow in MVC Architecture

1. **User Input** → View Layer captures input using view helpers
2. **Controller Processing** → Controllers validate input and coordinate actions
3. **Domain Model Validation** → Comprehensive validation using validation service
4. **Service Layer Execution** → Business logic processing with domain models
5. **Tool Execution** → LLM determines and executes required tools
6. **Response Generation** → LLM synthesizes final response with streaming
7. **View Layer Display** → Results displayed using view abstractions

### Domain Model Integration

```python
# Example of complete MVC flow
from controllers import SessionController
from models import User, Session, validation_service

# 1. Controller coordinates the flow
controller = SessionController(config)

# 2. Domain models provide type safety and business logic
session = controller.get_current_session()  # Returns Session domain model
user = controller.get_current_user()        # Returns User domain model

# 3. Validation ensures data integrity
validation_result = validation_service.validate_user_session_pair(user, session)

# 4. Business logic executed through domain models
user.increment_message_count()
session.set_context("processing_status", "active")

# 5. View helpers provide framework-agnostic UI
controller.display_session_info(show_details=True)
controller.show_processing_status("Processing your request...")
```

### Tool Execution with Domain Models

```python
# Tools can access and update domain models
from tools.registry import tool_registry
from models import ToolEntity

# Get tool with domain model tracking
tool_name = "text_assistant"
tool = tool_registry.get_tool(tool_name)

# Execute with comprehensive tracking
result = tool.execute({"task": "analyze", "text": "Sample text"})

# Access execution metrics through domain model
tool_entity = tool.tool_entity
success_rate = tool_entity.metrics.get_success_rate()
avg_execution_time = tool_entity.metrics.average_execution_time
```

## Production Deployment

### MVC Architecture Benefits in Production

- **Maintainability**: Clear separation of concerns makes debugging and updates easier
- **Testability**: Domain models and view abstractions enable comprehensive testing
- **Scalability**: Framework-agnostic design allows for easy technology migrations
- **Reliability**: Comprehensive validation prevents runtime errors
- **Monitoring**: Domain models provide built-in metrics and tracking

### Performance Optimization

- **Domain Model Caching**: Controllers cache domain models for efficient access
- **Validation Caching**: Validation results cached to prevent redundant checks
- **View Helper Optimization**: Framework abstractions minimize UI framework overhead
- **Service Layer Efficiency**: Business logic optimized through domain model methods

### Reliability Features

- **Comprehensive Validation**: All data validated through domain models and validation service
- **Error Handling**: Multi-layer error handling with graceful degradation
- **Session Isolation**: Domain models ensure proper session boundaries
- **Data Integrity**: Cross-model validation prevents inconsistent states

### Monitoring and Observability

- **Domain Model Metrics**: Built-in tracking for users, sessions, and tools
- **Validation Monitoring**: Track validation failures and business rule violations
- **Controller Performance**: Monitor request processing times across controllers
- **Service Layer Metrics**: Track business logic execution and error rates

### Deployment Considerations

The MVC architecture provides several deployment advantages:

```yaml
# docker-compose.yml optimized for MVC architecture
services:
  app:
    image: streamlit-chatbot:latest
    environment:
      # Domain model validation
      - VALIDATION_STRICT_MODE=true
      # View layer configuration
      - UI_FRAMEWORK=streamlit
      # Controller optimization
      - CONTROLLER_CACHE_SIZE=1000
    volumes:
      # External file storage for domain models
      - ./storage:/app/storage
    healthcheck:
      # Health check using domain model validation
      test:
        [
          "CMD",
          "python",
          "-c",
          "from models import validation_service; print('healthy')",
        ]
```

This enhanced MVC architecture ensures your application is production-ready, maintainable, and scalable while providing excellent developer experience through clear patterns and comprehensive validation.
