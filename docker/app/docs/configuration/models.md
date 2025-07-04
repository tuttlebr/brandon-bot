# Model Configuration Guide

This guide covers how to configure and use different language models in the Streamlit Chat Application.

## Available Model Types

The application supports three model categories:

### 1. Fast Models (`FAST_LLM_MODEL_NAME`)
- Used for: Quick responses, simple queries, tool selection
- Examples: `meta/llama-3.1-8b-instruct`, `mistralai/mistral-7b-instruct-v0.3`
- Characteristics: Lower latency, lower cost, good for routing

### 2. Standard Models (`LLM_MODEL_NAME`)
- Used for: General conversations, standard queries
- Examples: `meta/llama-3.1-70b-instruct`, `mistralai/mixtral-8x7b-instruct-v0.1`
- Characteristics: Balanced performance and quality

### 3. Intelligent Models (`INTELLIGENT_LLM_MODEL_NAME`)
- Used for: Complex reasoning, document analysis, code generation
- Examples: `nvidia/llama-3.3-nemotron-70b-instruct`, `google/gemma-2-27b-it`
- Characteristics: Highest quality, slower response times

## Configuration

### Environment Variables

```bash
# Model endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
IMAGE_ENDPOINT=https://api.segmind.com/v1/sdxl1.0-txt2img  # Optional

# Model selection
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Model parameters
TEMPERATURE=0.7
MAX_TOKENS=16380
TOP_P=0.95
FREQUENCY_PENALTY=0.0
PRESENCE_PENALTY=0.0
```

### Dynamic Model Configuration

You can also configure models programmatically:

```python
from models.chat_config import ChatConfig

config = ChatConfig(
    llm_model_name="meta/llama-3.1-70b-instruct",
    fast_llm_model_name="meta/llama-3.1-8b-instruct",
    intelligent_llm_model_name="nvidia/llama-3.3-nemotron-70b-instruct",
    temperature=0.7
)
```

## Model Selection Logic

### Automatic Selection

The application automatically selects the appropriate model based on:

1. **Query Complexity**
   - Simple queries → Fast model
   - Standard queries → Default model
   - Complex queries → Intelligent model

2. **Tool Requirements**
   - Tool selection → Fast model
   - Tool execution → Model specified by tool

3. **User Override**
   - Users can manually select models in the sidebar

### Selection Algorithm

```python
def select_model(query: str, context: Dict) -> str:
    # Check for user override
    if context.get("user_model_preference"):
        return context["user_model_preference"]

    # Check query complexity
    complexity = analyze_complexity(query)

    if complexity == "simple":
        return config.fast_llm_model_name
    elif complexity == "complex":
        return config.intelligent_llm_model_name
    else:
        return config.llm_model_name
```

## Available Models

### NVIDIA Models

**Fast Models (8K Context)**
- `meta/llama-3.1-8b-instruct` - Quick responses, routing
- `mistralai/mistral-7b-instruct-v0.3` - Long context (32K), fast performance

**Standard Models (8K Context)**
- `meta/llama-3.1-70b-instruct` - General conversation
- `mistralai/mixtral-8x7b-instruct-v0.1` - Balanced performance (32K context)

**Intelligent Models (16K Context)**
- `nvidia/llama-3.3-nemotron-70b-instruct` - Complex reasoning, advanced tasks

### Custom Model Support

To add support for a custom model:

1. Add model configuration:
```python
CUSTOM_MODEL_CONFIG = {
    "model_name": "custom/model-name",
    "endpoint": "https://custom-endpoint.com/v1",
    "max_tokens": 4096,
    "supports_streaming": True,
    "supports_tools": False
}
```

2. Register with model service:
```python
from services.llm_service import LLMService

llm_service = LLMService(config)
llm_service.register_model(CUSTOM_MODEL_CONFIG)
```

## Model Parameters

### Temperature
Controls randomness in responses:
- `0.0`: Deterministic, focused
- `0.7`: Balanced (default)
- `1.0`: Creative, varied

### Top-P (Nucleus Sampling)
Controls diversity:
- `0.1`: Very focused
- `0.95`: Balanced (default)
- `1.0`: Maximum diversity

### Max Tokens
Maximum response length:
- Fast models: 1000-2000
- Standard models: 2000-4000
- Intelligent models: 4000-16000

### Frequency/Presence Penalty
Controls repetition:
- `0.0`: No penalty (default)
- `1.0`: Strong penalty
- `2.0`: Maximum penalty

## Context Management

### Context Window Sizes

Different models have different context limits:

```python
MODEL_CONTEXT_LIMITS = {
    "meta/llama-3.1-8b-instruct": 8192,
    "meta/llama-3.1-70b-instruct": 8192,
    "nvidia/llama-3.3-nemotron-70b-instruct": 16384,
    "mistralai/mixtral-8x7b-instruct-v0.1": 32768
}
```

### Sliding Window

For long conversations:

```bash
# Configure sliding window
SLIDING_WINDOW_MAX_TURNS=20
MAX_CONTEXT_LENGTH=1000000
```

### Context Compression

The application automatically compresses context when approaching limits:

```python
def compress_context(messages: List[Dict], model: str) -> List[Dict]:
    max_tokens = MODEL_CONTEXT_LIMITS.get(model, 8192)
    current_tokens = count_tokens(messages)

    if current_tokens > max_tokens * 0.8:
        # Summarize older messages
        return compress_messages(messages)

    return messages
```

## Performance Optimization

### Model Caching

Enable caching for repeated queries:

```bash
ENABLE_MODEL_CACHE=true
CACHE_TTL=3600  # 1 hour
```

### Batch Processing

For multiple queries:

```python
responses = llm_service.batch_generate([
    "Query 1",
    "Query 2",
    "Query 3"
], model="meta/llama-3.1-8b-instruct")
```

### Streaming Configuration

```python
# Enable streaming for better UX
stream_config = {
    "stream": True,
    "chunk_size": 10,  # tokens per chunk
    "buffer_size": 100  # buffer before display
}
```

## Cost Management

### Token Usage Tracking

Monitor token usage:

```python
from utils.token_counter import TokenCounter

counter = TokenCounter()
usage = counter.track_usage(model, messages, response)
print(f"Tokens used: {usage['total']}")
print(f"Estimated cost: ${usage['cost']}")
```

### Cost Optimization Strategies

1. **Use appropriate models**
   - Simple queries → Fast models
   - Only use intelligent models when needed

2. **Implement caching**
   - Cache common responses
   - Use semantic similarity for cache hits

3. **Optimize prompts**
   - Keep system prompts concise
   - Remove unnecessary context

## Troubleshooting

### Common Issues

1. **Model Not Available**
   ```python
   Error: Model 'xyz' not found
   Solution: Check model name and availability
   ```

2. **Context Length Exceeded**
   ```python
   Error: Context length (X) exceeds limit (Y)
   Solution: Enable sliding window or compression
   ```

3. **Rate Limiting**
   ```python
   Error: Rate limit exceeded
   Solution: Implement exponential backoff
   ```

### Debug Mode

Enable detailed model logging:

```bash
MODEL_DEBUG=true
LOG_TOKEN_USAGE=true
```

## Best Practices

1. **Model Selection**
   - Start with fast models
   - Upgrade only when needed
   - Let users override when appropriate

2. **Parameter Tuning**
   - Lower temperature for factual tasks
   - Higher temperature for creative tasks
   - Adjust based on user feedback

3. **Context Management**
   - Implement sliding windows
   - Compress old messages
   - Clear context periodically

4. **Error Handling**
   - Implement fallback models
   - Retry with backoff
   - Provide clear error messages

## Advanced Configuration

### Multi-Model Routing

```python
ROUTING_CONFIG = {
    "code_generation": "nvidia/llama-3.3-nemotron-70b-instruct",
    "translation": "meta/llama-3.1-8b-instruct",
    "analysis": "meta/llama-3.1-70b-instruct",
    "creative": "mistralai/mixtral-8x7b-instruct-v0.1"
}
```

### A/B Testing

```python
AB_TEST_CONFIG = {
    "enabled": True,
    "models": {
        "control": "meta/llama-3.1-70b-instruct",
        "variant": "nvidia/llama-3.3-nemotron-70b-instruct"
    },
    "split": 0.5
}
```

## Next Steps

- Review [Environment Configuration](environment.md)
- See [Performance Tuning](../deployment/scaling.md)
- Check [API Reference](../api/services.md)
