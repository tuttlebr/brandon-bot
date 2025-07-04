# Services Architecture

This document describes the service layer architecture and patterns used throughout the application.

## Service Layer Overview

The service layer implements the core business logic and encapsulates external dependencies. Services are designed to be:

- **Stateless**: Services don't maintain state between calls
- **Reusable**: Can be used by multiple controllers
- **Testable**: Easy to unit test in isolation
- **Configurable**: Accept configuration through dependency injection

## Core Design Patterns

### 1. Dependency Injection

All services receive configuration through constructor injection:

```python
class ChatService:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.llm_client = self._initialize_client()
```

### 2. Service Factory Pattern

Complex services use factory methods for initialization:

```python
class LLMService:
    @classmethod
    def create_from_config(cls, config: ChatConfig) -> 'LLMService':
        client = OpenAI(api_key=config.api_key)
        return cls(client, config)
```

### 3. Async Support

Services support both sync and async operations:

```python
class StreamingService:
    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self.client.stream(prompt):
            yield chunk
```

## Service Categories

### 1. Core Services

#### LLMService
- Manages LLM client connections
- Handles model selection and switching
- Implements retry logic and error handling

#### ChatService
- Orchestrates conversation flow
- Manages message history
- Coordinates with tools and streaming

#### StreamingService
- Handles real-time response streaming
- Manages backpressure
- Implements chunking strategies

### 2. Document Services

#### PDFAnalysisService
- Extracts text from PDFs
- Performs document analysis
- Generates summaries

#### PDFContextService
- Manages PDF context switching
- Maintains document state
- Handles multi-document queries

#### DocumentAnalyzerService
- Analyzes document structure
- Extracts metadata
- Identifies key sections

### 3. Processing Services

#### TextProcessorService
- Text cleaning and normalization
- Token counting and splitting
- Format conversion

#### ResponseParsingService
- Parses LLM responses
- Extracts structured data
- Handles tool calls

#### TranslationService
- Multi-language support
- Context-aware translation
- Language detection

### 4. Storage Services

#### FileStorageService
- Manages file uploads
- Handles file persistence
- Implements cleanup policies

#### ConversationContextService
- Stores conversation history
- Manages context windows
- Implements compression

### 5. Integration Services

#### ToolExecutionService
- Executes registered tools
- Manages tool dependencies
- Handles parallel execution

#### ImageService
- Image generation integration
- Image processing
- Format conversion

## Service Communication

### Internal Communication

Services communicate through well-defined interfaces:

```python
# Service A calling Service B
class PDFAnalysisService:
    def __init__(self, text_processor: TextProcessorService):
        self.text_processor = text_processor

    def analyze(self, pdf_content: bytes) -> AnalysisResult:
        text = self.extract_text(pdf_content)
        processed = self.text_processor.process(text)
        return self.generate_analysis(processed)
```

### Event System

Services can emit and listen to events:

```python
from events import EventBus

class ChatService:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def send_message(self, message: str):
        # Process message
        self.event_bus.emit('message.sent', {'content': message})
```

## Error Handling

### Service-Level Errors

Each service defines its own exception types:

```python
class PDFAnalysisError(ServiceError):
    """Raised when PDF analysis fails"""
    pass

class ChatService:
    def process_message(self, message: str):
        try:
            return self._process(message)
        except LLMError as e:
            raise ChatServiceError(f"Failed to process: {e}")
```

### Error Recovery

Services implement recovery strategies:

```python
class LLMService:
    @retry(max_attempts=3, backoff=exponential_backoff)
    def generate_response(self, prompt: str) -> str:
        return self.client.complete(prompt)
```

## Performance Considerations

### Caching

Services use caching for expensive operations:

```python
from functools import lru_cache

class PDFAnalysisService:
    @lru_cache(maxsize=100)
    def analyze_cached(self, pdf_hash: str) -> AnalysisResult:
        return self._perform_analysis(pdf_hash)
```

### Connection Pooling

Services reuse connections:

```python
class LLMService:
    def __init__(self):
        self.client_pool = ClientPool(max_size=10)

    def get_client(self) -> LLMClient:
        return self.client_pool.acquire()
```

### Batch Processing

Services support batch operations:

```python
class TextProcessorService:
    def process_batch(self, texts: List[str]) -> List[ProcessedText]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(self.process, texts))
```

## Testing Services

### Unit Testing

```python
# tests/test_chat_service.py
def test_chat_service():
    mock_llm = Mock()
    config = ChatConfig(model="test")
    service = ChatService(config, llm_client=mock_llm)

    result = service.process_message("Hello")

    assert mock_llm.complete.called
    assert result.role == "assistant"
```

### Integration Testing

```python
# tests/integration/test_pdf_flow.py
def test_pdf_analysis_flow():
    pdf_service = PDFAnalysisService(config)
    text_service = TextProcessorService(config)

    with open("test.pdf", "rb") as f:
        result = pdf_service.analyze(f.read())

    processed = text_service.process(result.text)
    assert len(processed.chunks) > 0
```

## Service Lifecycle

### Initialization

Services are initialized at application startup:

```python
# utils/startup.py
def initialize_services(config: ChatConfig) -> ServiceRegistry:
    registry = ServiceRegistry()

    # Core services
    registry.register('llm', LLMService(config))
    registry.register('chat', ChatService(config))

    # Dependent services
    text_processor = TextProcessorService(config)
    registry.register('pdf', PDFAnalysisService(config, text_processor))

    return registry
```

### Cleanup

Services implement cleanup methods:

```python
class FileStorageService:
    def cleanup(self):
        """Clean up temporary files and close connections"""
        self.temp_dir.cleanup()
        self.connection_pool.close()
```

## Best Practices

1. **Single Responsibility**: Each service should have one clear purpose
2. **Interface Segregation**: Define minimal interfaces
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **Error Context**: Provide meaningful error messages
5. **Logging**: Log important operations and errors
6. **Metrics**: Track service performance
7. **Documentation**: Document service contracts

## Adding New Services

To add a new service:

1. Create service class in `services/` directory
2. Define service interface and exceptions
3. Implement core functionality
4. Add unit tests
5. Register in service initialization
6. Document in API reference

Example:

```python
# services/new_service.py
from models.chat_config import ChatConfig
from utils.exceptions import ServiceError
import logging

logger = logging.getLogger(__name__)

class NewServiceError(ServiceError):
    """Raised when new service operation fails"""
    pass

class NewService:
    """Service description"""

    def __init__(self, config: ChatConfig):
        self.config = config
        logger.info("NewService initialized")

    def process(self, data: str) -> str:
        """Process data and return result"""
        try:
            # Implementation
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise NewServiceError(f"Failed to process: {e}")
```

## Next Steps

- Review [Controllers Architecture](controllers.md)
- See [API Reference](../api/services.md)
- Check [Testing Guide](../development/testing.md)
