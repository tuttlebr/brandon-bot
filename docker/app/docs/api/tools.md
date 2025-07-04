# Tools API Reference

This document provides detailed API documentation for the tool system and all available tools.

## Tool System Overview

The tool system extends LLM capabilities with specialized functions. Tools follow a consistent interface and can be executed in parallel or sequentially based on dependencies.

## Base Tool Interface

All tools inherit from `BaseTool`:

```python
class BaseTool(ABC):
    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.llm_type: Literal["fast", "llm", "intelligent"] = "fast"

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> BaseToolResponse:
        pass
```

## Available Tools

### 1. Text Assistant (`text_assistant`)

Advanced text processing capabilities.

```python
class AssistantTool(BaseTool):
    name = "text_assistant"
    llm_type = "intelligent"
```

**Supported Tasks:**
- `summarize` - Create concise summaries
- `proofread` - Grammar and style corrections
- `rewrite` - Improve clarity and flow
- `critic` - Provide constructive feedback
- `translate` - Language translation

**Parameters:**
```python
{
    "task_type": "summarize|proofread|rewrite|critic|translate",
    "text": "Text to process",
    "instructions": "Optional specific instructions",
    "source_language": "For translation",
    "target_language": "For translation"
}
```

**Example:**
```python
response = assistant_tool.execute({
    "task_type": "summarize",
    "text": long_article,
    "instructions": "Focus on key findings, max 200 words"
})
```

### 2. PDF Summary Tool (`retrieve_pdf_summary`)

Generate or retrieve document summaries.

```python
class PDFSummaryTool(BaseTool):
    name = "retrieve_pdf_summary"
    llm_type = "llm"
```

**Parameters:**
```python
{
    "filename": "Optional specific filename",
    "summary_type": "document|pages|all"
}
```

### 3. PDF Text Processor (`process_pdf_text`)

Process specific PDF pages with text operations.

```python
class PDFTextProcessorTool(BaseTool):
    name = "process_pdf_text"
    llm_type = "llm"
```

**Parameters:**
```python
{
    "task_type": "summarize|proofread|rewrite|critic|translate",
    "page_numbers": [1, 2, 3],  # Optional specific pages
    "instructions": "Optional instructions"
}
```

### 4. Image Generation (`generate_image`)

AI-powered image creation.

```python
class ImageGenerationTool(BaseTool):
    name = "generate_image"
    llm_type = "fast"  # Uses fast model for prompt enhancement
```

**Parameters:**
```python
{
    "user_prompt": "Original user request",
    "subject": "Main subject of image",
    "style": "photorealistic|digital art|oil painting|minimalist|fantasy",
    "mood": "bright and cheerful|dark and moody|serene|dramatic",
    "details": "Additional details",
    "aspect_ratio": "square|portrait|landscape",
    "cfg_scale": 3.5,  # 1.5-4.5
    "use_conversation_context": true
}
```

**Example:**
```python
response = image_tool.execute({
    "user_prompt": "Create a sunset scene",
    "subject": "mountain landscape",
    "style": "photorealistic",
    "mood": "serene",
    "aspect_ratio": "landscape"
})
```

### 5. Web Search (`tavily_internet_search`)

General web search capabilities.

```python
class TavilyTool(BaseTool):
    name = "tavily_internet_search"
    llm_type = "fast"
```

**Parameters:**
```python
{
    "query": "Search query string"
}
```

### 6. News Search (`tavily_news_search`)

Specialized news search from trusted sources.

```python
class NewsTool(BaseTool):
    name = "tavily_news_search"
    llm_type = "fast"
```

**Parameters:**
```python
{
    "query": "News search query"
}
```

### 7. Weather Tool (`get_weather`)

Real-time weather information.

```python
class WeatherTool(BaseTool):
    name = "get_weather"
    llm_type = "fast"
```

**Parameters:**
```python
{
    "location": "City name or ZIP code"
}
```

### 8. Web Extract (`extract_web_content`)

Extract content from URLs.

```python
class WebExtractTool(BaseTool):
    name = "extract_web_content"
    llm_type = "fast"
```

**Parameters:**
```python
{
    "url": "URL to extract content from"
}
```

### 9. Retrieval Search (`retrieval_search`)

Semantic search in vector database.

```python
class RetrieverTool(BaseTool):
    name = "retrieval_search"
    llm_type = "llm"
```

**Parameters:**
```python
{
    "query": "Search query",
    "use_reranker": true,  # Optional
    "max_results": 10      # Optional
}
```

### 10. Conversation Context (`conversation_context`)

Analyze conversation history (internal tool).

```python
class ConversationContextTool(BaseTool):
    name = "conversation_context"
    llm_type = "intelligent"
```

**Parameters:**
```python
{
    "query": "conversation_summary|recent_topics|user_preferences|task_continuity",
    "max_messages": 20,
    "include_document_content": true
}
```

## Tool Registry

Tools are managed through a central registry:

```python
from tools.registry import tool_registry

# Get all tool definitions
tools = tool_registry.get_all_definitions()

# Execute a tool
result = tool_registry.execute_tool("text_assistant", {
    "task_type": "summarize",
    "text": content
})

# Get specific tool
tool = tool_registry.get_tool("generate_image")
```

## Tool Response Format

All tools return a response inheriting from `BaseToolResponse`:

```python
class BaseToolResponse(BaseModel):
    success: bool = True
    error_message: Optional[str] = None
    direct_response: bool = False  # If True, return directly to user
```

**Example Response:**
```python
{
    "success": true,
    "result": "Processed content...",
    "task_type": "summarize",
    "processing_notes": "Summarized from 500 to 100 words",
    "direct_response": true
}
```

## Tool Execution Strategies

Tools can be executed in different ways:

### Parallel Execution
Default for independent tools:
```python
tool_calls = [
    {"name": "get_weather", "arguments": {"location": "NYC"}},
    {"name": "tavily_news_search", "arguments": {"query": "AI news"}}
]
# Both execute simultaneously
```

### Sequential Execution
For tools with dependencies:
```python
# conversation_context must complete before others
tool_calls = [
    {"name": "conversation_context", "arguments": {...}},
    {"name": "generate_image", "arguments": {...}}
]
```

## Creating Custom Tools

To create a new tool:

```python
from tools.base import BaseTool, BaseToolResponse
from pydantic import Field

class CustomResponse(BaseToolResponse):
    data: str = Field(description="Processed data")
    metadata: dict = Field(description="Additional metadata")

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "custom_tool"
        self.description = "Description of what this tool does"
        self.llm_type = "fast"  # or "llm" or "intelligent"

    def to_openai_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input parameter"
                        }
                    },
                    "required": ["input"]
                }
            }
        }

    def execute(self, params: Dict[str, Any]) -> CustomResponse:
        # Tool implementation
        input_data = params["input"]

        # Process input
        result = process_data(input_data)

        return CustomResponse(
            success=True,
            data=result,
            metadata={"processed_at": datetime.now()}
        )
```

## Tool Selection

The LLM automatically selects appropriate tools based on:

1. **Tool descriptions** - Clear, specific descriptions
2. **User intent** - What the user is asking for
3. **Context** - Current conversation state
4. **Availability** - Which tools are registered

## Error Handling

Tools implement standardized error handling:

```python
try:
    result = tool.execute(params)
except ValidationError as e:
    return BaseToolResponse(
        success=False,
        error_message=f"Invalid parameters: {e}"
    )
except ToolExecutionError as e:
    return BaseToolResponse(
        success=False,
        error_message=f"Execution failed: {e}"
    )
```

## Best Practices

1. **Clear Descriptions**: Write specific tool descriptions
2. **Parameter Validation**: Validate all inputs
3. **Error Messages**: Provide helpful error messages
4. **Appropriate LLM**: Choose the right model type
5. **Response Format**: Follow consistent response structure
6. **Logging**: Log tool execution for debugging
7. **Timeout Handling**: Implement timeouts for external calls

## Configuration

Tool behavior can be configured:

```python
# In tool_llm_config.py
TOOL_LLM_TYPES = {
    "conversation_context": "intelligent",
    "text_assistant": "intelligent",
    "retrieval_search": "llm",
    "weather": "fast",
    # ... etc
}
```

## Next Steps

- See [Controllers API](controllers.md) for tool orchestration
- Review [Services API](services.md) for underlying services
- Check [Architecture Overview](../architecture/overview.md) for system design
