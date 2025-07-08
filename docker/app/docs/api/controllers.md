# Controllers API Reference

This document provides detailed API documentation for all controller classes in the Streamlit Chat Application.

## Overview

Controllers orchestrate the application flow and coordinate between the UI layer and service layer. They follow the MVC pattern and handle:

- User input validation
- Service coordination
- State management
- UI updates

## SessionController

Manages user session state and application-wide settings.

### Class: `SessionController`

```python
class SessionController:
    def __init__(self, config_obj: ChatConfig)
```

#### Key Methods

##### `initialize_session_state()`

Initializes all session state variables on first run.

**Example:**

```python
session_controller = SessionController(config)
session_controller.initialize_session_state()
```

##### `store_pdf_document(filename: str, pdf_data: dict) -> str`

Stores PDF document reference in session state.

**Parameters:**

- `filename`: Name of the PDF file
- `pdf_data`: Processed PDF data dictionary

**Returns:**

- PDF reference ID

##### `get_model_name(model_type: str = "fast") -> str`

Gets the configured model name for the specified type.

**Parameters:**

- `model_type`: One of "fast", "llm", or "intelligent"

**Returns:**

- Model name string

## MessageController

Handles message validation, processing, and chat history management.

### Class: `MessageController`

```python
class MessageController:
    def __init__(self, config_obj: ChatConfig, chat_service: ChatService,
                 session_controller=None)
```

#### Key Methods

##### `validate_prompt(prompt: str) -> bool`

Validates user input before processing.

**Parameters:**

- `prompt`: User input string

**Returns:**

- Boolean indicating if prompt is valid

**Validation checks:**

- Non-empty
- Within length limits
- No tool call instructions

##### `safe_add_message_to_history(role: str, content: Any) -> bool`

Safely adds a message to chat history with validation.

**Parameters:**

- `role`: Message role ("user" or "assistant")
- `content`: Message content (string or dict)

**Returns:**

- Success status

## FileController

Manages file uploads, particularly PDF processing.

### Class: `FileController`

```python
class FileController:
    def __init__(self, config_obj: ChatConfig,
                 message_controller: MessageController,
                 session_controller=None)
```

#### Key Methods

##### `process_pdf_upload(uploaded_file) -> bool`

Processes uploaded PDF files through the pipeline.

**Parameters:**

- `uploaded_file`: Streamlit UploadedFile object

**Returns:**

- Success status

**Processing steps:**

1. Validation (size, type)
2. Duplicate detection
3. API processing
4. Storage
5. Optional summarization

##### `is_new_upload(uploaded_file) -> bool`

Checks if file is a new upload or re-upload.

**Parameters:**

- `uploaded_file`: Streamlit UploadedFile object

**Returns:**

- True if new upload

## ResponseController

Orchestrates LLM response generation and streaming display.

### Class: `ResponseController`

```python
class ResponseController:
    def __init__(self, config_obj: ChatConfig, llm_service: LLMService,
                 message_controller: MessageController,
                 session_controller: SessionController,
                 chat_history_component: ChatHistoryComponent)
```

#### Key Methods

##### `generate_and_display_response(prepared_messages: List[Dict[str, Any]])`

Main method for generating and displaying AI responses.

**Parameters:**

- `prepared_messages`: Formatted messages for LLM

**Features:**

- Streaming response display
- Tool execution handling
- Image generation detection
- Error recovery

##### `_check_for_image_generation_response() -> Dict[str, Any]`

Detects and extracts image generation results from responses.

**Returns:**

- Image response data or empty dict

## Usage Examples

### Complete Controller Integration

```python
from controllers import (
    SessionController,
    MessageController,
    FileController,
    ResponseController
)
from services import ChatService, LLMService
from components import ChatHistoryComponent
from models import ChatConfig

# Initialize configuration
config = ChatConfig.from_environment()

# Initialize controllers with dependencies
session_controller = SessionController(config)
session_controller.initialize_session_state()

chat_service = ChatService(config)
message_controller = MessageController(config, chat_service, session_controller)

file_controller = FileController(config, message_controller, session_controller)

llm_service = LLMService(config)
chat_history = ChatHistoryComponent()
response_controller = ResponseController(
    config, llm_service, message_controller,
    session_controller, chat_history
)

# Process user input
if message_controller.validate_prompt(user_input):
    message_controller.safe_add_message_to_history("user", user_input)
    prepared_messages = message_controller.prepare_messages_for_processing(
        session_controller.get_messages()
    )
    response_controller.generate_and_display_response(prepared_messages)
```

### PDF Upload Flow

```python
# Handle PDF upload
uploaded_file = st.file_uploader("Choose PDF", type=['pdf'])
if uploaded_file:
    if file_controller.is_new_upload(uploaded_file):
        success = file_controller.process_pdf_upload(uploaded_file)
        if success:
            st.success("PDF processed successfully!")
```

## Error Handling

Controllers implement comprehensive error handling:

```python
try:
    response_controller.generate_and_display_response(messages)
except StreamingError as e:
    st.error("Streaming failed. Retrying...")
    response_controller.generate_response_with_cleanup_separation(messages)
except Exception as e:
    logger.error(f"Response generation failed: {e}")
    st.error("An error occurred. Please try again.")
```

## Best Practices

1. **Always validate input** through MessageController
2. **Check session state** before operations
3. **Handle errors gracefully** with user feedback
4. **Use dependency injection** for testability
5. **Log important operations** for debugging
6. **Clean up resources** after operations

## State Management

Controllers coordinate state through SessionController:

```python
# Check processing state
if session_controller.is_processing():
    st.warning("Please wait for current operation to complete")
    return

# Set processing state
session_controller.set_processing(True)
try:
    # Perform operation
    process_data()
finally:
    session_controller.set_processing(False)
```

## Next Steps

- See [Services API](services.md) for service layer documentation
- Review [Architecture Overview](../architecture/overview.md) for system design
- Check [Tools API](tools.md) for tool system reference
