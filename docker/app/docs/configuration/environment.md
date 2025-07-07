# Environment Configuration

This guide covers all environment variables used by the Streamlit Chat Application.

## Required Variables

These environment variables must be set for the application to function:

### API Credentials

```bash
# NVIDIA API key for language models
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxx
```

### Model Endpoints

```bash
# Standard LLM endpoint
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Fast model endpoint (for quick responses)
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Intelligent model endpoint (for complex reasoning)
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
```

### Model Names

```bash
# Standard language model
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct

# Fast response model
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct

# Advanced reasoning model
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct
```

## Optional Variables

### Vision Language Model (VLM)

For image analysis capabilities:

```bash
# VLM endpoint (defaults to LLM_ENDPOINT if not set)
VLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# VLM model name (defaults to nvidia/llama-3.1-nemotron-nano-vl-8b-v1)
VLM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
```

### Image Generation

For AI image creation:

```bash
# Image generation endpoint
IMAGE_ENDPOINT=https://your-image-generation-endpoint.com

# Image generation API key (if different from NVIDIA_API_KEY)
IMAGE_API_KEY=your-image-api-key
```

### Web Search

For internet search capabilities:

```bash
# Tavily API key for web search
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxx
```

### Vector Database

For semantic search and retrieval:

```bash
# Embedding model endpoint
EMBEDDING_ENDPOINT=https://integrate.api.nvidia.com/v1

# Embedding model name
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# Database configuration
DATABASE_URL=http://localhost:19530
DEFAULT_DB=default
COLLECTION_NAME=documents
```

### Application Settings

```bash
# Application title
BOT_TITLE="Streamlit Chat Assistant"

# File upload limits (in MB)
PDF_MAX_SIZE_MB=200
IMAGE_MAX_SIZE_MB=10

# Session configuration
SESSION_TIMEOUT_MINUTES=60
MAX_MESSAGES_PER_SESSION=1000

# Memory management
MAX_CONTEXT_TOKENS=8192
SLIDING_WINDOW_SIZE=50
```

### UI Configuration

```bash
# Brand colors (hex values)
BRAND_COLOR=#76B900

# Avatar paths
ASSISTANT_AVATAR_PATH=./assets/assistant_avatar.png
USER_AVATAR_PATH=./assets/user_avatar.png

# Display settings
MESSAGES_PER_PAGE=25
SHOW_TOOL_CONTEXT=true
```

### Performance Tuning

```bash
# LLM parameters
DEFAULT_TEMPERATURE=0.0
DEFAULT_TOP_P=1.0
DEFAULT_PRESENCE_PENALTY=0.0
DEFAULT_FREQUENCY_PENALTY=0.0
DEFAULT_MAX_TOKENS=2048

# Streaming settings
STREAM_CHUNK_SIZE=10
STREAM_DELAY_MS=0

# Tool execution
TOOL_EXECUTION_TIMEOUT=30
MAX_PARALLEL_TOOLS=5
```

### Debug and Logging

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Enable debug mode
DEBUG_MODE=false

# Tool execution logging
LOG_TOOL_EXECUTION=true

# Performance monitoring
ENABLE_PERFORMANCE_MONITORING=false
```

## Environment File Example

Create a `.env` file in your project root:

```bash
# === REQUIRED SETTINGS ===
# NVIDIA Configuration
NVIDIA_API_KEY=nvapi-your-key-here

# Model Endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Model Names
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# === OPTIONAL FEATURES ===
# Vision Language Model
VLM_ENDPOINT=https://integrate.api.nvidia.com/v1
VLM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1

# Image Generation
IMAGE_ENDPOINT=https://your-endpoint.com
IMAGE_API_KEY=your-key

# Web Search
TAVILY_API_KEY=tvly-your-key

# === APPLICATION SETTINGS ===
BOT_TITLE="My AI Assistant"
PDF_MAX_SIZE_MB=200
IMAGE_MAX_SIZE_MB=10

# === PERFORMANCE ===
DEFAULT_TEMPERATURE=0.0
MAX_CONTEXT_TOKENS=8192
```

## Configuration Priority

The application loads configuration in this order:

1. Environment variables
2. `.env` file
3. Default values in `utils/config.py`

## Validation

The application validates configuration on startup:

```python
# Automatic validation checks:
- Required API keys are present
- Endpoints are valid URLs
- Model names follow expected format
- Numeric values are within valid ranges
```

## Docker Configuration

When using Docker, pass environment variables via:

### Docker Compose

```yaml
services:
  app:
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - LLM_ENDPOINT=${LLM_ENDPOINT}
      # ... other variables
```

### Docker Run

```bash
docker run -e NVIDIA_API_KEY=your-key \
           -e LLM_ENDPOINT=https://endpoint \
           your-image
```

## Security Best Practices

1. **Never commit `.env` files** - Add to `.gitignore`
2. **Use environment-specific files** - `.env.development`, `.env.production`
3. **Rotate API keys regularly**
4. **Use secrets management** in production (AWS Secrets Manager, etc.)
5. **Limit API key permissions** to required scopes

## Troubleshooting

### Missing Required Variables

```
ConfigurationError: Missing required environment variable: NVIDIA_API_KEY
```

**Solution**: Ensure all required variables are set in your `.env` file.

### Invalid Endpoints

```
ConfigurationError: Invalid endpoint URL: not-a-url
```

**Solution**: Verify all endpoints are valid HTTPS URLs.

### Model Not Found

```
Error: Model 'unknown-model' not found
```

**Solution**: Check model names match available models from your provider.

## Advanced Configuration

### Custom Model Providers

To use non-NVIDIA models:

```bash
# OpenAI Configuration
LLM_ENDPOINT=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4
NVIDIA_API_KEY=sk-your-openai-key

# Anthropic Configuration
LLM_ENDPOINT=https://api.anthropic.com/v1
LLM_MODEL_NAME=claude-3-opus
NVIDIA_API_KEY=your-anthropic-key
```

### Proxy Configuration

For corporate environments:

```bash
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080
NO_PROXY=localhost,127.0.0.1
```

### Resource Limits

```bash
# Memory limits
MAX_FILE_STORAGE_MB=1000
MAX_SESSION_MEMORY_MB=500

# Rate limiting
REQUESTS_PER_MINUTE=60
TOKENS_PER_MINUTE=100000
```

## See Also

- [Model Configuration](models.md) - Detailed model setup
- [Docker Deployment](../deployment/docker.md) - Container configuration
- [FAQ](../faq.md) - Common configuration questions
