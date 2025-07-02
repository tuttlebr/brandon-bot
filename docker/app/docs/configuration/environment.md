# Environment Configuration

Complete guide to configuring the Streamlit Chat Application through environment variables.

## Overview

The application uses environment variables for configuration, following the 12-factor app methodology. All configuration is centralized in the `utils/config.py` module.

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

### Image Generation

```bash
# Endpoint for image generation service
IMAGE_ENDPOINT=https://your-image-api.com/v1/generate

# API key for image service (if different from NVIDIA)
IMAGE_API_KEY=your-image-api-key
```

### Web Search

```bash
# Tavily API key for web search functionality
TAVILY_API_KEY=tvly-your-api-key-here
```

### Application Settings

```bash
# Maximum file upload size in MB (default: 200)
MAX_FILE_SIZE_MB=500

# Session timeout in minutes (default: 60)
SESSION_TIMEOUT_MINUTES=120

# Enable debug logging (default: INFO)
LOGGING_LEVEL=DEBUG

# Custom branding
APP_TITLE="My Custom Chat Application"
BRAND_COLOR="#FF6B6B"
```

### Performance Tuning

```bash
# Maximum tokens for context (default: 4096)
MAX_CONTEXT_TOKENS=8192

# Streaming chunk size (default: 10)
STREAM_CHUNK_SIZE=20

# Request timeout in seconds (default: 300)
REQUEST_TIMEOUT=600

# Maximum retries for API calls (default: 3)
MAX_RETRIES=5
```

### Storage Configuration

```bash
# External storage path (default: /tmp/file_storage)
FILE_STORAGE_PATH=/persistent/storage

# Maximum storage per session in MB (default: 1000)
MAX_SESSION_STORAGE_MB=2000

# Storage cleanup interval in hours (default: 24)
STORAGE_CLEANUP_INTERVAL=12
```

## Configuration File Examples

### Development Configuration (.env.development)

```bash
# Development settings
NVIDIA_API_KEY=nvapi-dev-key
LOGGING_LEVEL=DEBUG
MAX_FILE_SIZE_MB=50
SESSION_TIMEOUT_MINUTES=30

# Use faster models for development
LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=meta/llama-3.1-70b-instruct

# Local endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
```

### Production Configuration (.env.production)

```bash
# Production settings
NVIDIA_API_KEY=${NVIDIA_API_KEY_PROD}
LOGGING_LEVEL=INFO
MAX_FILE_SIZE_MB=200
SESSION_TIMEOUT_MINUTES=60

# Production models
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Production endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Performance settings
MAX_CONTEXT_TOKENS=8192
REQUEST_TIMEOUT=300
MAX_RETRIES=5

# Storage settings
FILE_STORAGE_PATH=/persistent/storage
MAX_SESSION_STORAGE_MB=1000
```

### Testing Configuration (.env.test)

```bash
# Test settings
NVIDIA_API_KEY=test-api-key
LOGGING_LEVEL=WARNING
MAX_FILE_SIZE_MB=10

# Mock endpoints for testing
LLM_ENDPOINT=http://localhost:8000/mock/v1
FAST_LLM_ENDPOINT=http://localhost:8000/mock/v1
INTELLIGENT_LLM_ENDPOINT=http://localhost:8000/mock/v1

# Test models
LLM_MODEL_NAME=test-model
FAST_LLM_MODEL_NAME=test-model-fast
INTELLIGENT_LLM_MODEL_NAME=test-model-intelligent
```

## Using Environment Files

### Loading Environment Files

```bash
# Using docker-compose
docker compose --env-file .env.production up

# Using Python directly
python -m dotenv -f .env.development run python main.py

# Export to shell
export $(cat .env | grep -v '^#' | xargs)
```

### Environment File Priority

1. System environment variables (highest priority)
2. `.env` file in project root
3. Default values in code (lowest priority)

## Configuration Validation

The application validates configuration on startup:

```python
# In utils/config.py
def validate_environment():
    """Validate all required environment variables"""
    required_vars = [
        'NVIDIA_API_KEY',
        'LLM_ENDPOINT',
        'LLM_MODEL_NAME'
    ]

    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
```

## Dynamic Configuration

Some settings can be changed at runtime:

### Model Selection

Users can switch models during conversation:
- "Use the fast model" - Switches to FAST_LLM_MODEL_NAME
- "Use intelligent mode" - Switches to INTELLIGENT_LLM_MODEL_NAME

### Debug Mode

Enable debug mode temporarily:
```python
# In the chat
"Enable debug mode for this session"

# Via API
POST /api/session/debug
{"enabled": true}
```

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

### 4. Limit Scope

Use different API keys for different environments:
- Development: Limited quota, test data only
- Staging: Production-like but isolated
- Production: Full access, monitoring enabled

## Monitoring Configuration

### Health Check Variables

```bash
# Enable health endpoint
ENABLE_HEALTH_CHECK=true
HEALTH_CHECK_PATH=/health

# Health check intervals
HEALTH_CHECK_INTERVAL=30
```

### Metrics Collection

```bash
# Enable metrics
ENABLE_METRICS=true
METRICS_PORT=9090

# Prometheus endpoint
PROMETHEUS_ENDPOINT=http://prometheus:9090
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

2. **Wrong Variable Format**
   ```bash
   # Correct
   NVIDIA_API_KEY=nvapi-abc123

   # Wrong (quotes not needed in .env)
   NVIDIA_API_KEY="nvapi-abc123"
   ```

3. **Path Issues**
   ```bash
   # Use absolute paths in production
   FILE_STORAGE_PATH=/app/storage  # Good
   FILE_STORAGE_PATH=./storage     # May cause issues
   ```

### Debug Configuration

```bash
# Print all configuration
docker compose exec app python -c "
from utils.config import config
config.print_configuration()
"

# Validate configuration
docker compose exec app python -c "
from utils.config import config
config.validate_environment()
"
```

## Configuration Schema

### Complete Variable Reference

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| NVIDIA_API_KEY | string | - | Yes | NVIDIA API authentication key |
| LLM_ENDPOINT | string | - | Yes | Main LLM endpoint URL |
| LLM_MODEL_NAME | string | - | Yes | Main model identifier |
| FAST_LLM_ENDPOINT | string | - | Yes | Fast model endpoint URL |
| FAST_LLM_MODEL_NAME | string | - | Yes | Fast model identifier |
| INTELLIGENT_LLM_ENDPOINT | string | - | Yes | Intelligent model endpoint URL |
| INTELLIGENT_LLM_MODEL_NAME | string | - | Yes | Intelligent model identifier |
| IMAGE_ENDPOINT | string | - | No | Image generation endpoint |
| IMAGE_API_KEY | string | NVIDIA_API_KEY | No | Image service API key |
| TAVILY_API_KEY | string | - | No | Tavily search API key |
| MAX_FILE_SIZE_MB | int | 200 | No | Maximum upload size |
| SESSION_TIMEOUT_MINUTES | int | 60 | No | Session timeout |
| LOGGING_LEVEL | string | INFO | No | Log level |
| APP_TITLE | string | "Streamlit Chat" | No | Application title |
| BRAND_COLOR | string | "#FF4B4B" | No | Brand color |
| MAX_CONTEXT_TOKENS | int | 4096 | No | Context window size |
| STREAM_CHUNK_SIZE | int | 10 | No | Streaming chunk size |
| REQUEST_TIMEOUT | int | 300 | No | API request timeout |
| MAX_RETRIES | int | 3 | No | API retry attempts |

## Next Steps

- Review [Model Configuration](models.md) for model-specific settings
- See [UI Configuration](ui.md) for interface customization
- Check [Deployment Guide](../deployment/docker.md) for production setup
