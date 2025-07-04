# Environment Configuration

Complete guide to configuring Nano Chat Application through environment variables.

## Overview

Nano uses environment variables for configuration, following the 12-factor app methodology. All configuration is centralized in the `utils/config.py` module.

## Required Environment Variables

These variables MUST be set for the application to start:

### NVIDIA API Configuration

```bash
# Your NVIDIA API key for accessing language models
NVIDIA_API_KEY=nvapi-your-key-here

# Base endpoints for different model types
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Model identifiers
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct
```

## Optional Environment Variables

### Branding and UI

```bash
# Application title (default: "Nano")
BOT_TITLE=Nano

# User reference term (default: "human")
META_USER=human

# Brand color in hex (default: "#76b900")
BRAND_COLOR=#76b900
```

### Image Generation

```bash
# Endpoint for image generation service
IMAGE_ENDPOINT=https://your-image-api.com/v1/generate
```

### Web Search

```bash
# Tavily API key for web search functionality
TAVILY_API_KEY=tvly-your-api-key-here
```

### PDF Processing

```bash
# NVIDIA Ingest endpoint for PDF processing
NVINGEST_ENDPOINT=http://nvingest:7670/v1/extract_text
```

### Vector Database (Milvus)

```bash
# Database configuration for semantic search
DATABASE_URL=http://milvus-standalone:19530
COLLECTION_NAME=milvus
PARTITION_NAME=milvus
DEFAULT_DB=milvus

# Embedding configuration
EMBEDDING_ENDPOINT=https://integrate.api.nvidia.com/v1
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# Reranker configuration
RERANKER_ENDPOINT=https://integrate.api.nvidia.com/v1/ranking
RERANKER_MODEL=nvidia/nv-rerankqa-mistral-4b-v3
```

### Performance Tuning

```bash
# LLM parameters
DEFAULT_TEMPERATURE=0.3
DEFAULT_TOP_P=0.95
DEFAULT_FREQUENCY_PENALTY=0.0
DEFAULT_PRESENCE_PENALTY=0.0

# Context window settings
SLIDING_WINDOW_MAX_TURNS=20  # Number of conversation turns to keep
MAX_CONTEXT_LENGTH=1000000   # Maximum context length in characters

# Conversation context injection
AUTO_INJECT_CONVERSATION_CONTEXT=true
MIN_TURNS_FOR_CONTEXT_INJECTION=1

# API timeouts (in seconds)
DEFAULT_REQUEST_TIMEOUT=3600
LLM_REQUEST_TIMEOUT=3600
IMAGE_REQUEST_TIMEOUT=3600
PDF_PROCESSING_TIMEOUT=6000
```

### Storage Configuration

```bash
# File processing limits
MAX_PDF_SIZE=16777216  # 16MB in bytes
SUPPORTED_PDF_TYPES=pdf

# Session limits
MAX_IMAGES_IN_SESSION=50
MAX_PDFS_IN_SESSION=3

# Storage paths
FILE_STORAGE_PATH=/tmp/chatbot_storage
```

### System Settings

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Suppress Streamlit warnings
SUPPRESS_STREAMLIT_WARNINGS=true
```

## Configuration File Examples

### Development Configuration (.env.development)

```bash
# Development settings
NVIDIA_API_KEY=nvapi-dev-key
BOT_TITLE=Nano Dev
LOG_LEVEL=DEBUG

# Use faster models for development
LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=meta/llama-3.1-70b-instruct

# Local endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Reduced limits for faster testing
MAX_PDF_SIZE=5242880  # 5MB
SLIDING_WINDOW_MAX_TURNS=10
```

### Production Configuration (.env.production)

```bash
# Production settings
NVIDIA_API_KEY=${NVIDIA_API_KEY_PROD}
BOT_TITLE=Nano
LOG_LEVEL=INFO

# Production models
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Production endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Production settings
SLIDING_WINDOW_MAX_TURNS=20
AUTO_INJECT_CONVERSATION_CONTEXT=true
SUPPRESS_STREAMLIT_WARNINGS=true

# Production services
NVINGEST_ENDPOINT=http://nvingest:7670/v1/extract_text
DATABASE_URL=http://milvus-standalone:19530
```

## Using Environment Files

### Loading Environment Files

```bash
# Using docker-compose
docker compose --env-file .env.production up

# Using Python directly
python -m python-dotenv run python main.py

# Export to shell
export $(cat .env | grep -v '^#' | xargs)
```

### Environment File Priority

1. System environment variables (highest priority)
2. `.env` file in project root
3. Default values in `utils/config.py` (lowest priority)

## Configuration Validation

Nano validates configuration on startup:

```python
# The application checks for required variables
missing_vars = config.env.validate_required_env_vars()
if missing_vars:
    logging.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
```

## Dynamic Configuration

Some settings can be changed at runtime:

### Model Selection

Users can switch models during conversation:
- "Use the fast model" - Switches to FAST_LLM_MODEL_NAME
- "Use the standard model" - Switches to LLM_MODEL_NAME
- "Use intelligent mode" - Switches to INTELLIGENT_LLM_MODEL_NAME

## Security Best Practices

### 1. Never Commit Secrets

```bash
# .gitignore
.env
.env.*
!.env.example
```

### 2. Use Secret Management

For production, use proper secret management:

```bash
# AWS Secrets Manager
NVIDIA_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id prod/nvidia-api-key \
  --query SecretString --output text)

# Kubernetes Secrets
kubectl create secret generic app-secrets \
  --from-literal=NVIDIA_API_KEY=$NVIDIA_API_KEY
```

### 3. Rotate Keys Regularly

```bash
# Update keys without downtime
docker compose --env-file .env.new up -d --no-deps app
```

## Troubleshooting Configuration

### Common Issues

1. **Variable Not Loading**
   ```bash
   # Check if variable is set
   echo $NVIDIA_API_KEY

   # Check in container
   docker compose exec app env | grep NVIDIA
   ```

2. **PDF Processing Fails**
   ```bash
   # Ensure NVINGEST_ENDPOINT is accessible
   curl http://nvingest:7670/health
   ```

3. **Model Not Found**
   ```bash
   # Verify model names are correct
   # Check NVIDIA API documentation for valid models
   ```

### Debug Configuration

```bash
# Print all configuration
docker compose exec app python -c "
from utils.config import config
print(config.__dict__)
"

# Validate configuration
docker compose exec app python -c "
from utils.config import config
config.validate_environment()
"
```

## Configuration Schema

### Complete Variable Reference

**Core Settings**
- `NVIDIA_API_KEY` (string, required) - NVIDIA API authentication key
- `BOT_TITLE` (string, default: "Nano") - Application name
- `META_USER` (string, default: "human") - User reference term

**Model Configuration**
- `LLM_ENDPOINT` (string, required) - Main LLM endpoint URL
- `LLM_MODEL_NAME` (string, required) - Main model identifier
- `FAST_LLM_ENDPOINT` (string, required) - Fast model endpoint URL
- `FAST_LLM_MODEL_NAME` (string, required) - Fast model identifier
- `INTELLIGENT_LLM_ENDPOINT` (string, required) - Intelligent model endpoint URL
- `INTELLIGENT_LLM_MODEL_NAME` (string, required) - Intelligent model identifier

**Optional Services**
- `IMAGE_ENDPOINT` (string) - Image generation endpoint
- `TAVILY_API_KEY` (string) - Tavily search API key
- `NVINGEST_ENDPOINT` (string) - PDF processing endpoint
- `DATABASE_URL` (string) - Milvus database URL

**Performance**
- `SLIDING_WINDOW_MAX_TURNS` (int, default: 20) - Conversation history limit
- `MAX_CONTEXT_LENGTH` (int, default: 1000000) - Maximum context size
- `AUTO_INJECT_CONVERSATION_CONTEXT` (bool, default: true) - Auto-inject context

**Limits**
- `MAX_PDF_SIZE` (int, default: 16777216) - Max PDF size in bytes
- `MAX_IMAGES_IN_SESSION` (int, default: 50) - Max images per session
- `MAX_PDFS_IN_SESSION` (int, default: 3) - Max PDFs per session

**PDF Batch Processing**
- `PDF_BATCH_PROCESSING_THRESHOLD` (int, default: 50) - Pages threshold to trigger batch processing
- `PDF_PAGES_PER_BATCH` (int, default: 20) - Maximum pages per batch
- `PDF_CONTEXT_MAX_PAGES` (int, default: 30) - Maximum pages to include in context at once
- `PDF_CONTEXT_MAX_CHARS` (int, default: 100000) - Maximum characters per context injection

## Next Steps

- Review [Model Configuration](models.md) for model-specific settings
- See [UI Configuration](ui.md) for interface customization
- Check [Deployment Guide](../deployment/docker.md) for production setup
