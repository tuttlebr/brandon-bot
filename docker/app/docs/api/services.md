# Services API Reference

This document provides detailed API documentation for all service classes in the Streamlit Chat Application.

## LLMService

The core service for interacting with language models.

### Class: `LLMService`

```python
class LLMService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `create_llm_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]`

Converts chat messages to LLM-compatible format.

**Parameters:**

- `messages`: List of message dictionaries with 'role' and 'content'

**Returns:**

- List of formatted messages for LLM API

**Example:**

```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
formatted = llm_service.create_llm_messages(messages)
```

##### `generate_streaming_response(messages: List[Dict], model_type: str = "llm") -> Generator`

Generates streaming response from LLM.

**Parameters:**

- `messages`: Formatted messages for LLM
- `model_type`: One of "fast", "llm", or "intelligent"

**Returns:**

- Generator yielding response chunks

**Example:**

```python
for chunk in llm_service.generate_streaming_response(messages, "fast"):
    print(chunk, end="")
```

##### `handle_tool_calls(tool_calls: List[Dict]) -> List[Dict]`

Processes and executes tool calls from LLM.

**Parameters:**

- `tool_calls`: List of tool call specifications

**Returns:**

- List of tool execution results

## ChatService

Manages chat interactions and message processing.

### Class: `ChatService`

```python
class ChatService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `process_message(message: str, role: str = "user") -> Dict[str, Any]`

Processes and validates chat messages.

**Parameters:**

- `message`: The message content
- `role`: Message role ("user" or "assistant")

**Returns:**

- Processed message dictionary

**Example:**

```python
processed = chat_service.process_message("Hello!", "user")
```

##### `format_message_for_display(message: Dict[str, Any]) -> str`

Formats message for UI display.

**Parameters:**

- `message`: Message dictionary

**Returns:**

- Formatted message string

## PDFContextService

Manages PDF context injection for queries.

### Class: `PDFContextService`

```python
class PDFContextService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `inject_pdf_context(messages: List[Dict], query: str) -> List[Dict]`

Injects relevant PDF context into messages.

**Parameters:**

- `messages`: Current conversation messages
- `query`: User query to match against PDF content

**Returns:**

- Messages with injected PDF context

**Example:**

```python
enhanced_messages = pdf_context_service.inject_pdf_context(
    messages,
    "What does the document say about revenue?"
)
```

##### `get_pdf_info_for_display() -> Optional[str]`

Gets formatted PDF information for display.

**Returns:**

- Formatted string with PDF details or None

## PDFAnalysisService

Provides intelligent PDF document analysis.

### Class: `PDFAnalysisService`

```python
class PDFAnalysisService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `analyze_pdf_intelligent(pdf_document: Dict, query: str, update_progress: Callable) -> str`

Performs intelligent analysis of PDF documents.

**Parameters:**

- `pdf_document`: PDF document dictionary with pages
- `query`: Analysis query
- `update_progress`: Progress callback function

**Returns:**

- Analysis result string

**Example:**

```python
def progress_callback(progress, message):
    print(f"{progress}%: {message}")

result = pdf_service.analyze_pdf_intelligent(
    pdf_doc,
    "Summarize key findings",
    progress_callback
)
```

##### `extract_relevant_content(pages: List[Dict], query: str, max_pages: int = 10) -> List[Dict]`

Extracts query-relevant pages from document.

**Parameters:**

- `pages`: List of page dictionaries
- `query`: Search query
- `max_pages`: Maximum pages to return

**Returns:**

- List of relevant pages

## StreamingService

Handles response streaming for real-time updates.

### Class: `StreamingService`

```python
class StreamingService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `stream_response(response_generator: Generator, container: Any) -> str`

Streams response chunks to UI container.

**Parameters:**

- `response_generator`: Generator yielding response chunks
- `container`: Streamlit container for output

**Returns:**

- Complete response text

**Example:**

```python
container = st.empty()
full_response = streaming_service.stream_response(
    llm_response_generator,
    container
)
```

## ToolExecutionService

Orchestrates tool execution and result handling.

### Class: `ToolExecutionService`

```python
class ToolExecutionService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `execute_tools_parallel(tool_calls: List[Dict]) -> List[Dict]`

Executes multiple tools in parallel.

**Parameters:**

- `tool_calls`: List of tool call specifications

**Returns:**

- List of execution results

**Example:**

```python
tool_calls = [
    {"name": "search", "arguments": {"query": "AI trends"}},
    {"name": "calculator", "arguments": {"expression": "2+2"}}
]
results = tool_service.execute_tools_parallel(tool_calls)
```

##### `should_direct_return(tool_name: str) -> bool`

Determines if tool result should be returned directly.

**Parameters:**

- `tool_name`: Name of the tool

**Returns:**

- Boolean indicating direct return

## FileStorageService

Manages external file storage for session isolation.

### Class: `FileStorageService`

```python
class FileStorageService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `store_file(file_data: bytes, filename: str, session_id: str) -> str`

Stores file in external storage.

**Parameters:**

- `file_data`: File content as bytes
- `filename`: Original filename
- `session_id`: User session identifier

**Returns:**

- Storage path or identifier

**Example:**

```python
path = storage_service.store_file(
    pdf_bytes,
    "document.pdf",
    session_id
)
```

##### `retrieve_file(file_path: str, session_id: str) -> Optional[bytes]`

Retrieves file from storage.

**Parameters:**

- `file_path`: Storage path
- `session_id`: User session identifier

**Returns:**

- File bytes or None if not found

##### `cleanup_session_files(session_id: str) -> int`

Removes all files for a session.

**Parameters:**

- `session_id`: User session identifier

**Returns:**

- Number of files removed

## ImageService

Handles AI-powered image generation.

### Class: `ImageService`

```python
class ImageService:
    def __init__(self, config: ChatConfig)
```

#### Methods

##### `generate_image(prompt: str, size: str = "1024x1024") -> Optional[str]`

Generates image from text prompt.

**Parameters:**

- `prompt`: Text description for image
- `size`: Image dimensions

**Returns:**

- Base64 encoded image or None

**Example:**

```python
image_b64 = image_service.generate_image(
    "A serene mountain landscape at sunset",
    "1024x1024"
)
```

## Usage Examples

### Complete Service Integration

```python
from services import (
    LLMService,
    ChatService,
    PDFContextService,
    StreamingService
)
from models import ChatConfig

# Initialize configuration
config = ChatConfig.from_environment()

# Initialize services
llm_service = LLMService(config)
chat_service = ChatService(config)
pdf_service = PDFContextService(config)
streaming_service = StreamingService(config)

# Process user message
user_message = "Tell me about the uploaded document"
processed = chat_service.process_message(user_message)

# Get conversation history
messages = st.session_state.messages

# Inject PDF context
enhanced_messages = pdf_service.inject_pdf_context(
    messages,
    user_message
)

# Generate streaming response
response_gen = llm_service.generate_streaming_response(
    enhanced_messages,
    model_type="intelligent"
)

# Stream to UI
container = st.empty()
full_response = streaming_service.stream_response(
    response_gen,
    container
)
```

### Error Handling

```python
try:
    response = llm_service.generate_streaming_response(messages)
except RateLimitError as e:
    st.error("Rate limit exceeded. Please try again later.")
except APIError as e:
    st.error(f"API Error: {str(e)}")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    st.error("An unexpected error occurred.")
```

## Best Practices

1. **Always use dependency injection** for service initialization
2. **Handle errors gracefully** with appropriate user feedback
3. **Use streaming** for long responses to improve UX
4. **Inject context** before sending to LLM for better results
5. **Clean up resources** (files, connections) after use
6. **Log important operations** for debugging
7. **Validate inputs** before processing
8. **Use appropriate model types** based on task complexity

## Next Steps

- See [Controllers API](controllers.md) for controller documentation
- Review [Tools API](tools.md) for tool system reference
- Check [Models](models.md) for data model documentation
