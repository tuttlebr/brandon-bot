# Frequently Asked Questions (FAQ)

## General Questions

### What is the Streamlit Chat Application?

The Streamlit Chat Application is a production-ready conversational AI application that provides:
- Advanced language model integration (NVIDIA, OpenAI)
- PDF document analysis and context-aware Q&A
- Image generation capabilities
- Multi-tool support (weather, news, search, etc.)
- Real-time streaming responses

### What are the system requirements?

**Minimum Requirements:**
- Python 3.10+
- 4GB RAM
- 10GB disk space
- Docker (for containerized deployment)

**Recommended:**
- 8GB+ RAM
- 20GB+ disk space
- GPU support for faster inference (optional)

### Which language models are supported?

The application supports multiple model providers:
- **NVIDIA NIM**: LLaMA, Mistral, Nemotron models
- **OpenAI**: GPT-3.5, GPT-4 (with configuration)
- **Custom models**: Via API endpoint configuration

## Setup and Installation

### How do I get an NVIDIA API key?

1. Visit [NVIDIA AI](https://www.nvidia.com/en-us/ai/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key
5. Add to your `.env` file: `NVIDIA_API_KEY=nvapi-xxx`

### Why is my Docker build failing?

Common causes and solutions:

1. **Network issues**:
   ```bash
   # Use buildkit for better caching
   DOCKER_BUILDKIT=1 docker compose build
   ```

2. **Memory limitations**:
   ```bash
   # Increase Docker memory limit
   docker system prune -a  # Clean up first
   ```

3. **Port conflicts**:
   ```bash
   # Check port usage
   lsof -i :8050  # Streamlit port
   lsof -i :19530  # Milvus port
   ```

### How do I run without Docker?

```bash
# Install dependencies
pip install -r docker/app/requirements.txt

# Set environment variables
export NVIDIA_API_KEY=your-key
export LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Run application
cd docker/app
streamlit run main.py
```

## Features and Usage

### How do I switch between different language models?

You can switch models in three ways:

1. **Sidebar Selection**: Use the model dropdown in the sidebar
2. **Environment Variables**: Set default models in `.env`
3. **Programmatic**: Specify in your query (e.g., "Using the intelligent model, analyze...")

### Why is PDF analysis not working?

Check these common issues:

1. **File size**: PDFs must be under 16MB (configurable)
2. **File format**: Ensure it's a valid PDF, not encrypted
3. **NVIngest service**: Check if the service is running:
   ```bash
   docker compose ps nvingest
   ```
4. **Permissions**: Ensure write access to temp directory

### How do I enable image generation?

1. Add image endpoint to `.env`:
   ```bash
   IMAGE_ENDPOINT=https://api.segmind.com/v1/sdxl1.0-txt2img
   ```

2. Restart the application

3. Use natural language: "Generate an image of a sunset over mountains"

### Can I use multiple PDFs simultaneously?

Yes! The application supports:
- Uploading multiple PDFs
- Switching context between documents
- Asking questions across documents
- Comparing document contents

### How do I clear the chat history?

Three options:
1. Click "Clear Chat" button in the sidebar
2. Refresh the page (F5)
3. Start a new session (close and reopen tab)

## Performance and Optimization

### Why are responses slow?

Response time depends on:

1. **Model selection**:
   - Fast models: 1-2 seconds
   - Standard models: 2-5 seconds
   - Intelligent models: 5-10 seconds

2. **Network latency**: Check your connection to API endpoints

3. **Context length**: Longer conversations take more time

**Optimization tips:**
- Use fast models for simple queries
- Clear old messages periodically
- Enable response caching

### How can I reduce token usage/costs?

1. **Use appropriate models**:
   ```python
   # For simple queries
   FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
   ```

2. **Enable sliding window**:
   ```bash
   SLIDING_WINDOW_MAX_TURNS=10
   ```

3. **Optimize prompts**: Keep system prompts concise

4. **Implement caching**: Reuse responses for similar queries

### Memory usage is high. What can I do?

1. **Limit concurrent uploads**:
   ```bash
   MAX_IMAGES_IN_SESSION=20
   MAX_PDF_SIZE=8388608  # 8MB
   ```

2. **Enable garbage collection**:
   ```python
   ENABLE_GC=true
   GC_INTERVAL=300  # seconds
   ```

3. **Clear session state**: Remove processed files after use

## Troubleshooting

### "API Key Invalid" error

1. Check key format: Should start with `nvapi-`
2. Verify key in [NVIDIA dashboard](https://nvidia.com)
3. Ensure no extra spaces in `.env`
4. Try regenerating the key

### "Model not found" error

This happens when:
- Model name is misspelled
- Model is not available in your region
- API endpoint is incorrect

Solution:
```bash
# List available models
curl -H "Authorization: Bearer $NVIDIA_API_KEY" \
  https://integrate.api.nvidia.com/v1/models
```

### Streamlit connection errors

1. **"Please wait..." forever**:
   - Clear browser cache
   - Try incognito mode
   - Check browser console for errors

2. **WebSocket errors**:
   - Check firewall settings
   - Ensure ports 8050-8051 are open
   - Try different browser

### Docker container keeps restarting

Check logs:
```bash
docker compose logs app -f
```

Common fixes:
- Ensure all required env vars are set
- Check file permissions
- Verify Docker resources (memory/CPU)

## Advanced Configuration

### How do I add a custom tool?

1. Create tool class:
```python
# tools/custom_tool.py
from tools.base import BaseTool

class CustomTool(BaseTool):
    def __init__(self):
        self.name = "custom_tool"
        self.description = "My custom tool"

    def execute(self, params):
        # Implementation
        return result
```

2. Register in `tools/initialize_tools.py`

### Can I use a different vector database?

Yes, the application supports:
- Milvus (default)
- ChromaDB
- Pinecone
- Weaviate

Configure in `.env`:
```bash
VECTOR_DB_TYPE=chroma
VECTOR_DB_URL=http://localhost:8000
```

### How do I enable authentication?

Basic authentication example:

```python
# utils/auth.py
import streamlit as st

def check_auth():
    if 'authenticated' not in st.session_state:
        password = st.text_input("Password", type="password")
        if password == os.getenv("APP_PASSWORD"):
            st.session_state.authenticated = True
        else:
            st.stop()
```

### Can I deploy to production?

Yes! See [Deployment Guide](deployment/docker.md) for:
- Docker Compose production setup
- Kubernetes deployment
- Cloud provider guides (AWS, GCP, Azure)
- SSL/TLS configuration
- Load balancing

## Data and Privacy

### Is my data stored?

By default:
- Chat history: Session only (cleared on refresh)
- Uploaded files: Temporary storage (auto-deleted)
- No permanent storage unless configured

### Can I use this offline?

Partial offline support:
- UI runs offline
- Requires internet for:
  - LLM API calls
  - Tool execution (weather, news)
  - Image generation

### How do I export chat history?

Add export functionality:
```python
# In sidebar
if st.button("Export Chat"):
    chat_json = json.dumps(st.session_state.messages)
    st.download_button(
        "Download",
        chat_json,
        "chat_history.json"
    )
```

## Contributing and Development

### How do I contribute?

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

See [Contributing Guide](development/contributing.md)

### Where do I report bugs?

- GitHub Issues: [Create issue](https://github.com/tuttlebr/streamlit-chatbot/issues)
- Include:
  - Steps to reproduce
  - Error messages
  - Environment details

### How do I request features?

1. Check existing issues first
2. Create feature request with:
   - Use case description
   - Expected behavior
   - Alternative solutions

## Common Error Messages

### "RuntimeError: Session state key already exists"

Clear session state:
```python
for key in list(st.session_state.keys()):
    del st.session_state[key]
```

### "Connection timeout"

1. Check internet connection
2. Verify API endpoint accessibility
3. Increase timeout:
   ```bash
   REQUEST_TIMEOUT=60
   ```

### "CUDA out of memory"

If using GPU:
1. Reduce batch size
2. Use smaller models
3. Clear GPU cache:
   ```python
   torch.cuda.empty_cache()
   ```

## Best Practices

### For Users

1. **Start simple**: Use fast models first
2. **Be specific**: Clear, detailed prompts get better results
3. **Use tools wisely**: Let the AI choose appropriate tools
4. **Manage context**: Clear old conversations periodically

### For Developers

1. **Follow conventions**: Use existing patterns
2. **Write tests**: Maintain >80% coverage
3. **Document changes**: Update relevant docs
4. **Performance first**: Profile before optimizing

## Still Need Help?

- 📖 Read the [full documentation](index.md)
- 💬 Join our [Discord community](#)
- 📧 Email support: support@example.com
- 🐛 Report issues on [GitHub](https://github.com/tuttlebr/streamlit-chatbot/issues)
