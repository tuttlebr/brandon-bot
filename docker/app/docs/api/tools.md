# Tools API Reference

The Streamlit Chat Application includes a comprehensive tool system that extends the LLM's capabilities. This document provides detailed information about each available tool.

## Overview

The application includes 11 specialized tools that can be automatically invoked by the LLM based on user queries:

1. **Text Processing Tools**: Text assistant for various language tasks
2. **Image Tools**: Generation and analysis of images
3. **Document Tools**: PDF processing and summarization
4. **Search Tools**: Web search, news, and knowledge retrieval
5. **Utility Tools**: Weather, web extraction, and conversation analysis

## Tool Architecture

### Base Tool Class

All tools inherit from `BaseTool` and implement:

```python
class BaseTool(ABC):
    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.llm_type: Literal["fast", "llm", "intelligent", "vlm"] = "fast"

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> BaseToolResponse:
        pass

    def to_openai_format(self) -> Dict[str, Any]:
        pass
```

### Tool Registry

Tools are managed by a singleton registry that handles:

- Tool registration and discovery
- Execution orchestration
- Definition management

## Available Tools

### 1. Text Assistant (`text_assistant`)

Comprehensive text processing tool for various language tasks.

**Capabilities:**

- **analyze**: Document insights and analysis
- **summarize**: Condense content into key points
- **proofread**: Grammar and style corrections
- **rewrite**: Enhance clarity and impact
- **critic**: Constructive feedback
- **translate**: Language translation
- **develop**: Programming assistance
- **generalist**: General discussion

**Parameters:**

```python
{
    "user_message": str,  # Original user request
    "task": str,         # Task type (analyze, summarize, etc.)
    "target_language": str,  # For translation tasks
    "focus_areas": List[str],  # Optional focus areas
    "formatting_requirements": str,  # Optional formatting
    "expertise_level": str,  # casual, professional, technical
    "additional_instructions": str  # Extra instructions
}
```

**LLM Type:** `intelligent`

### 2. Image Generation (`generate_image`)

AI-powered image generation with style control.

**Parameters:**

```python
{
    "user_prompt": str,      # Original request
    "subject": str,          # Main subject
    "style": str,           # Artistic style (default: "photorealistic")
    "mood": str,            # Atmosphere (default: "natural")
    "details": str,         # Additional details
    "aspect_ratio": str,    # "square", "portrait", "landscape"
    "cfg_scale": float,     # Guidance scale (1.5-4.5, default: 3.5)
    "use_conversation_context": bool  # Use chat context
}
```

**LLM Type:** `fast`

### 3. Image Analysis (`analyze_image`)

Vision-capable LLM for analyzing uploaded images.

**Parameters:**

```python
{
    "question": str  # Question about the uploaded image
}
```

**Note:** Requires an image to be uploaded via the sidebar first.

**LLM Type:** `vlm`

### 4. PDF Summary (`pdf_summary`)

Intelligent document summarization for uploaded PDFs.

**Parameters:**

```python
{
    "query": str,            # Summary focus/question
    "summary_type": str,     # "brief", "detailed", "key-points"
    "max_length": int,       # Maximum summary length
    "focus_sections": List[str]  # Specific sections to focus on
}
```

**LLM Type:** `llm`

### 5. PDF Text Processor (`pdf_text_processor`)

Advanced PDF text extraction and processing.

**Parameters:**

```python
{
    "query": str,           # Processing request
    "page_range": str,      # e.g., "1-5", "all"
    "processing_type": str, # "extract", "analyze", "search"
    "search_terms": List[str]  # For search operations
}
```

**LLM Type:** `llm`

### 6. Web Search (`tavily_internet_search`)

General internet search for current information.

**Parameters:**

```python
{
    "query": str  # Search query
}
```

**Features:**

- Real-time web search
- Source attribution
- Content summarization

**LLM Type:** `fast`

### 7. News Search (`news`)

Specialized search for current news and events.

**Parameters:**

```python
{
    "query": str,          # News search query
    "time_range": str,     # "today", "week", "month"
    "category": str        # Optional news category
}
```

**LLM Type:** `llm`

### 8. Weather (`weather`)

Current weather information for any location.

**Parameters:**

```python
{
    "location": str  # City name or coordinates
}
```

**Returns:**

- Current conditions
- Temperature
- Forecast summary

**LLM Type:** `fast`

### 9. Web Extract (`extract_web_content`)

Extract and parse content from web pages.

**Parameters:**

```python
{
    "url": str,              # Web page URL
    "extract_type": str,     # "text", "links", "images", "all"
    "include_metadata": bool # Include page metadata
}
```

**LLM Type:** `fast`

### 10. Retriever (`retriever`)

Semantic search through knowledge bases.

**Parameters:**

```python
{
    "query": str,           # Search query
    "k": int,              # Number of results (default: 5)
    "score_threshold": float # Minimum similarity score
}
```

**Features:**

- Vector similarity search
- Context-aware retrieval
- Relevance scoring

**LLM Type:** `llm`

### 11. Conversation Context (`conversation_context`)

Analyze conversation history and patterns.

**Parameters:**

```python
{
    "query": str,          # Analysis type
    "max_messages": int,   # Messages to analyze
    "focus_query": str     # Specific focus area
}
```

**Query Types:**

- `"conversation_summary"`: Overall summary
- `"user_intent"`: Intent analysis
- `"key_topics"`: Topic extraction
- `"action_items"`: Extract tasks

**LLM Type:** `intelligent`

## Tool Execution

### Automatic Tool Selection

The LLM automatically selects appropriate tools based on:

1. User query analysis
2. Tool descriptions and capabilities
3. Context awareness
4. Tool availability

### Parallel Execution

Multiple tools can be executed in parallel for efficiency:

- Independent tools run simultaneously
- Results are aggregated
- Errors are handled gracefully

### Direct Response Tools

Some tools provide direct responses to users:

- Image generation results
- Image analysis outputs
- Weather information
- Search results

## Error Handling

### Common Error Types

1. **Parameter Validation Errors**
   - Missing required parameters
   - Invalid parameter types
   - Out-of-range values

2. **Execution Errors**
   - API failures
   - Network timeouts
   - Resource limitations

3. **Tool Not Found**
   - Invalid tool name
   - Tool not registered

### Error Response Format

```python
{
    "success": False,
    "error_message": str,
    "error_type": str,
    "tool_name": str
}
```

## Configuration

### Tool LLM Types

Tools are configured to use specific LLM types:

```python
TOOL_LLM_TYPES = {
    "text_assistant": "intelligent",
    "generate_image": "fast",
    "analyze_image": "vlm",
    "pdf_summary": "llm",
    "pdf_text_processor": "llm",
    "tavily_internet_search": "fast",
    "news": "llm",
    "weather": "fast",
    "extract_web_content": "fast",
    "retriever": "llm",
    "conversation_context": "intelligent"
}
```

### Environment Variables

Required for specific tools:

```bash
# For image generation
IMAGE_ENDPOINT=your_endpoint

# For web search
TAVILY_API_KEY=your_api_key

# For image analysis
VLM_ENDPOINT=https://integrate.api.nvidia.com/v1
VLM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
```

## Best Practices

### Tool Development

1. **Clear Descriptions**: Write precise tool descriptions
2. **Parameter Validation**: Validate all inputs
3. **Error Messages**: Provide helpful error messages
4. **Resource Management**: Clean up resources properly
5. **Logging**: Log important operations

### Tool Usage

1. **Specific Queries**: Be clear about what you need
2. **Tool Limitations**: Understand each tool's capabilities
3. **Context Provision**: Provide necessary context
4. **Error Recovery**: Handle failures gracefully

## Examples

### Text Processing

```python
# Summarize a document
response = text_assistant.execute({
    "user_message": "Summarize this document focusing on key findings",
    "task": "summarize",
    "expertise_level": "professional"
})
```

### Image Generation

```python
# Generate an image
response = generate_image.execute({
    "user_prompt": "Create a sunset over mountains",
    "subject": "mountain sunset",
    "style": "photorealistic",
    "mood": "serene",
    "aspect_ratio": "landscape"
})
```

### Web Search

```python
# Search the web
response = tavily_search.execute({
    "query": "latest developments in quantum computing"
})
```

## See Also

- [Services API](services.md) - Core service implementations
- [Controllers API](controllers.md) - Controller patterns
- [Architecture Overview](../architecture/overview.md) - System design
