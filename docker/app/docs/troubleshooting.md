# Troubleshooting Guide

This guide helps you resolve common issues with the Streamlit Chat Application.

## Quick Diagnostics

Before diving into specific issues, run these quick checks:

```bash
# Check if Docker is running
docker --version
docker ps

# Check if the application is running
docker compose ps

# View application logs
docker compose logs app -n 100

# Check port availability
sudo lsof -i :8080
sudo lsof -i :8050
```

## Common Issues

### Application Won't Start

#### Symptoms
- Cannot access `http://localhost:8080`
- Docker container exits immediately
- Error messages in logs

#### Solutions

1. **Check Docker Status**
   ```bash
   # Ensure Docker daemon is running
   sudo systemctl status docker

   # Start Docker if needed
   sudo systemctl start docker
   ```

2. **Verify Environment Variables**
   ```bash
   # Check if .env file exists
   ls -la .env

   # Verify required variables
   cat .env | grep NVIDIA_API_KEY
   ```

3. **Port Conflicts**
   ```bash
   # Kill processes using the ports
   sudo kill -9 $(sudo lsof -t -i:8080)
   sudo kill -9 $(sudo lsof -i:8050)

   # Restart the application
   docker compose down
   docker compose up -d
   ```

4. **Clean Rebuild**
   ```bash
   # Remove all containers and rebuild
   docker compose down -v
   docker compose build --no-cache
   docker compose up -d
   ```

### NVIDIA API Errors

#### Symptoms
- "API key is invalid" error
- "Rate limit exceeded" message
- Model not found errors

#### Solutions

1. **Verify API Key**
   ```bash
   # Test API key directly
   curl -H "Authorization: Bearer $NVIDIA_API_KEY" \
        https://integrate.api.nvidia.com/v1/models
   ```

2. **Check Model Names**
   Ensure model names in `.env` match available models:
   ```
   LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
   FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
   INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct
   ```

3. **Rate Limit Issues**
   - Wait 1-2 minutes before retrying
   - Switch to a different model temporarily
   - Check your API quota on NVIDIA dashboard

### PDF Upload Issues

#### Symptoms
- PDF upload fails
- "File too large" error
- Processing hangs indefinitely

#### Solutions

1. **Check File Size**
   Default limit is 200MB. For larger files:
   ```bash
   # Update docker-compose.yaml
   environment:
     - MAX_FILE_SIZE_MB=500
   ```

2. **Verify PDF Format**
   ```bash
   # Check if PDF is valid
   file your_document.pdf

   # Try with a simple PDF first
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory allocation
   # Docker Desktop: Settings > Resources > Memory

   # Or in docker-compose.yaml:
   services:
     app:
       deploy:
         resources:
           limits:
             memory: 4G
   ```

### Chat Interface Issues

#### Symptoms
- Messages not sending
- Responses cut off
- Interface frozen

#### Solutions

1. **Browser Issues**
   - Clear browser cache: `Ctrl/Cmd + Shift + R`
   - Try a different browser
   - Disable browser extensions

2. **Session State Issues**
   ```python
   # Add to your code for debugging
   import streamlit as st
   if st.button("Clear Session"):
       for key in st.session_state.keys():
           del st.session_state[key]
       st.rerun()
   ```

3. **WebSocket Connection**
   - Check firewall settings
   - Ensure WebSocket support in reverse proxy
   - Try direct connection without proxy

### Performance Issues

#### Symptoms
- Slow response times
- High CPU/memory usage
- Timeouts

#### Solutions

1. **Optimize Model Selection**
   ```python
   # Use fast model for simple queries
   "Please use the fast model"

   # Use intelligent model only when needed
   "Switch to intelligent model for this complex analysis"
   ```

2. **Resource Monitoring**
   ```bash
   # Monitor container resources
   docker stats app

   # Check system resources
   htop
   ```

3. **Scaling Options**
   ```yaml
   # docker-compose.yaml
   services:
     app:
       deploy:
         replicas: 2
   ```

## Error Messages Reference

### Common Error Messages

**"Configuration error: NVIDIA_API_KEY not found"**
- **Cause**: Missing environment variable
- **Solution**: Add `NVIDIA_API_KEY` to `.env` file

**"Rate limit exceeded"**
- **Cause**: Too many API requests
- **Solution**: Wait 60 seconds, use fast model

**"Failed to process PDF"**
- **Cause**: Corrupted or unsupported PDF
- **Solution**: Try a different PDF, check format

**"Connection refused"**
- **Cause**: Service not running
- **Solution**: Run `docker compose up -d`

**"Out of memory"**
- **Cause**: Large file or response
- **Solution**: Increase Docker memory limits

**"Model not found"**
- **Cause**: Invalid model name
- **Solution**: Check model name in configuration

**"WebSocket connection failed"**
- **Cause**: Network or proxy issue
- **Solution**: Check firewall, try direct connection

## Debugging Tools

### 1. Enable Debug Logging

```python
# In utils/config.py
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")

# Set in .env
LOGGING_LEVEL=DEBUG
```

### 2. View Detailed Logs

```bash
# Follow all logs
docker compose logs -f

# View specific service logs
docker compose logs app -n 1000

# Search for errors
docker compose logs app | grep -i error
```

### 3. Interactive Debugging

```bash
# Enter container shell
docker compose exec app bash

# Run Python shell
python

# Test components
from services import LLMService
from models import ChatConfig
config = ChatConfig.from_environment()
llm = LLMService(config)
```

### 4. Health Checks

```python
# Add health check endpoint
@app.route('/health')
def health_check():
    checks = {
        'llm_service': check_llm_connection(),
        'pdf_service': check_pdf_service(),
        'storage': check_storage_access()
    }
    return jsonify(checks)
```

## Advanced Troubleshooting

### Network Issues

1. **DNS Resolution**
   ```bash
   # Test DNS
   nslookup integrate.api.nvidia.com

   # Add custom DNS
   docker compose exec app cat /etc/resolv.conf
   ```

2. **Proxy Configuration**
   ```bash
   # Set proxy in docker-compose.yaml
   environment:
     - HTTP_PROXY=http://proxy.company.com:8080
     - HTTPS_PROXY=http://proxy.company.com:8080
   ```

### Memory Leaks

1. **Monitor Memory Usage**
   ```python
   import psutil
   import os

   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

2. **Force Garbage Collection**
   ```python
   import gc
   gc.collect()
   ```

### Database/Storage Issues

1. **Clear Session Storage**
   ```bash
   # Remove session files
   docker compose exec app rm -rf /tmp/sessions/*
   ```

2. **Reset File Storage**
   ```bash
   # Clear uploaded files
   docker compose exec app rm -rf /tmp/file_storage/*
   ```

## Getting Help

### Before Asking for Help

1. **Collect Information**
   ```bash
   # System info
   uname -a
   docker --version
   docker compose version

   # Application logs
   docker compose logs app > app_logs.txt

   # Environment (sanitized)
   cat .env | grep -v API_KEY
   ```

2. **Create Minimal Reproduction**
   - Identify steps to reproduce
   - Test with minimal configuration
   - Document error messages

### Where to Get Help

1. **Documentation**
   - Review this troubleshooting guide
   - Check [Architecture Overview](architecture/overview.md)
   - Read [Configuration Guide](configuration/environment.md)

2. **Community Support**
   - GitHub Issues
   - Discord/Slack community
   - Stack Overflow with appropriate tags

3. **Professional Support**
   - Contact development team
   - Submit detailed bug report
   - Request feature enhancement

## Preventive Measures

### Regular Maintenance

1. **Update Dependencies**
   ```bash
   # Update Docker images
   docker compose pull

   # Rebuild with latest
   docker compose build --no-cache
   ```

2. **Monitor Resources**
   - Set up alerts for high CPU/memory
   - Track API usage and limits
   - Monitor disk space

3. **Backup Configuration**
   ```bash
   # Backup important files
   cp .env .env.backup
   cp docker-compose.yaml docker-compose.yaml.backup
   ```

### Best Practices

1. **Use Version Control**
   - Track configuration changes
   - Document modifications
   - Tag stable versions

2. **Test Changes**
   - Test in development first
   - Gradual rollout
   - Have rollback plan

3. **Monitor Logs**
   - Regular log review
   - Set up log aggregation
   - Alert on errors

## Quick Reference Card

```bash
# Restart everything
./rebuild.sh

# View logs
docker compose logs -f app

# Enter container
docker compose exec app bash

# Check status
docker compose ps

# Stop everything
docker compose down

# Clean restart
docker compose down -v && docker compose up -d
```

Remember: Most issues can be resolved with a clean rebuild or checking the logs!
