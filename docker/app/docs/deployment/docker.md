# Docker Deployment Guide

This guide covers deploying the Streamlit Chat Application using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB available RAM
- Required API keys (NVIDIA, Tavily, etc.)

## Quick Deployment

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/tuttlebr/streamlit-chatbot.git
cd streamlit-chatbot

# Create environment file
cp .env.example .env
# Edit .env with your API keys
```

### 2. Build and Deploy

```bash
# Build all services
docker compose build

# Start the application
docker compose up -d

# Check status
docker compose ps
```

### 3. Access the Application

- Main Application: `http://localhost:8080`
- Streamlit Direct: `http://localhost:8050`

## Docker Compose Configuration

### Service Architecture

```yaml
services:
  app:
    build:
      context: ./docker
      dockerfile: Dockerfile
    ports:
      - "8050:8501"
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - LLM_ENDPOINT=${LLM_ENDPOINT}
    volumes:
      - ./docker/app:/app
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app

  milvus-standalone:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus

  nvingest:
    build:
      context: ./nvingest
    ports:
      - "7670:7670"
```

## Environment Configuration

### Required Variables

```bash
# API Keys
NVIDIA_API_KEY=nvapi-xxx
TAVILY_API_KEY=tvly-xxx  # Optional

# Model Configuration
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Service Endpoints
NVINGEST_ENDPOINT=http://nvingest:7670/v1/extract_text
DATABASE_URL=http://milvus-standalone:19530
```

### Optional Variables

```bash
# Performance Tuning
SLIDING_WINDOW_MAX_TURNS=20

# File Limits
MAX_PDF_SIZE=16777216  # 16MB
MAX_IMAGES_IN_SESSION=50

# Logging
LOG_LEVEL=INFO
```

## Volume Management

### Persistent Data

```yaml
volumes:
  milvus_data:
    driver: local
  file_storage:
    driver: local
```

### Backup Strategy

```bash
# Backup Milvus data
docker run --rm -v milvus_data:/source -v $(pwd):/backup alpine \
  tar czf /backup/milvus_backup_$(date +%Y%m%d).tar.gz -C /source .

# Backup file storage
docker run --rm -v file_storage:/source -v $(pwd):/backup alpine \
  tar czf /backup/files_backup_$(date +%Y%m%d).tar.gz -C /source .
```

## Network Configuration

### Default Network

```yaml
networks:
  default:
    driver: bridge
```

### Custom Network

```yaml
networks:
  chatbot_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Resource Limits

### Memory and CPU

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G
```

## Health Checks

### Application Health

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Logging Configuration

### Log Rotation

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Centralized Logging

```yaml
logging:
  driver: "syslog"
  options:
    syslog-address: "tcp://192.168.0.42:123"
```

## Security Considerations

### 1. Secret Management

```bash
# Use Docker secrets
echo "nvapi-xxx" | docker secret create nvidia_api_key -

# Reference in compose
services:
  app:
    secrets:
      - nvidia_api_key
```

### 2. Network Isolation

```yaml
services:
  app:
    networks:
      - frontend
      - backend

  milvus:
    networks:
      - backend
```

### 3. Read-Only Filesystem

```yaml
services:
  app:
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
```

## Monitoring

### Container Metrics

```bash
# Real-time stats
docker stats

# Container logs
docker compose logs -f app

# Specific time range
docker compose logs --since 1h app
```

### Health Monitoring

```bash
# Check health status
docker compose ps

# Detailed health info
docker inspect app | jq '.[0].State.Health'
```

## Troubleshooting

### Common Issues

1. **Container Won't Start**

   ```bash
   # Check logs
   docker compose logs app

   # Validate compose file
   docker compose config
   ```

2. **Network Issues**

   ```bash
   # Test connectivity
   docker compose exec app ping milvus-standalone

   # Check network
   docker network ls
   docker network inspect streamlit-chatbot_default
   ```

3. **Volume Permissions**
   ```bash
   # Fix permissions
   docker compose exec app chown -R app:app /tmp/file_storage
   ```

### Debug Mode

```yaml
services:
  app:
    environment:
      - LOG_LEVEL=DEBUG
    command: ["python", "-u", "main.py"]
```

## Scaling

### Horizontal Scaling

```bash
# Scale app instances
docker compose up -d --scale app=3

# With load balancer
services:
  nginx:
    depends_on:
      - app
    links:
      - app
```

### Vertical Scaling

Adjust resource limits in docker-compose.yaml:

```yaml
deploy:
  resources:
    limits:
      cpus: "4"
      memory: 8G
```

## Maintenance

### Updates

```bash
# Pull latest images
docker compose pull

# Rebuild with no cache
docker compose build --no-cache

# Rolling update
docker compose up -d --no-deps app
```

### Cleanup

```bash
# Remove stopped containers
docker compose down

# Remove all including volumes
docker compose down -v

# Prune unused resources
docker system prune -a
```

## Production Checklist

- [ ] Set strong API keys
- [ ] Configure HTTPS (reverse proxy)
- [ ] Enable log aggregation
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Set resource limits
- [ ] Enable health checks
- [ ] Configure restart policies
- [ ] Document deployment process
- [ ] Test disaster recovery

## Next Steps

- Review [Production Setup](production.md) for advanced configurations
- See [Scaling Guide](scaling.md) for high-availability setup
- Check [Environment Configuration](../configuration/environment.md) for all options
