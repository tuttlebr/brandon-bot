# Quick Start Guide

Get the Streamlit Chat Application running in under 5 minutes!

## Prerequisites

Before you begin, ensure you have:

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- NVIDIA API key (get one from [NVIDIA AI Foundation](https://www.nvidia.com/en-us/ai/))
- (Optional) Tavily API key for web search functionality

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tuttlebr/streamlit-chatbot.git
cd streamlit-chatbot
```

### 2. Create Environment File

Create a `.env` file in the root directory:

```bash
# NVIDIA API Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here

# Model Endpoints
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
FAST_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
INTELLIGENT_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Model Names
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
FAST_LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
INTELLIGENT_LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-70b-instruct

# Optional: Image Generation
IMAGE_ENDPOINT=https://your-image-endpoint.com
IMAGE_API_KEY=your_image_api_key

# Optional: Web Search
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Build and Run

Use the provided rebuild script for a clean setup:

```bash
chmod +x rebuild.sh
./rebuild.sh
```

Or manually with Docker Compose:

```bash
# Build the application
docker compose build app

# Start the services
docker compose up -d app nginx

# View logs
docker compose logs -f app
```

### 4. Access the Application

Open your browser and navigate to:

```
http://localhost:8080
```

You should see the chat interface ready to use!

## First Conversation

Try these sample prompts to explore the features:

1. **Basic Chat**: "Hello! What can you help me with today?"
2. **Code Generation**: "Write a Python function to calculate fibonacci numbers"
3. **PDF Analysis**: Upload a PDF and ask "Summarize this document"
4. **Image Generation**: "Create an image of a futuristic city at sunset"

## Quick Configuration

### Using Different Models

You can switch between models by asking:

- "Use the fast model" - For quick responses
- "Use the intelligent model" - For complex reasoning

### Adjusting Memory

The application maintains conversation history. To clear it:

- Refresh the page
- Or click "Clear Chat" if available

## Next Steps

- 📖 Read the [Installation Guide](installation.md) for detailed setup options
- 👤 Check the [User Guide](../user-guide/chat-interface.md) for feature details
- ⚙️ See [Configuration](../configuration/environment.md) for customization
- 🚀 Review [Production Setup](../deployment/production.md) for deployment

## Troubleshooting Quick Fixes

### Application Won't Start

```bash
# Check if ports are in use
sudo lsof -i :8080
sudo lsof -i :8050

# Stop existing containers
docker compose down
```

### NVIDIA API Errors

- Verify your API key is correct
- Check your API quota at NVIDIA dashboard
- Ensure the model names are correct

### PDF Upload Issues

- Check file size (default limit: 200MB)
- Ensure PDF is not corrupted
- Try with a smaller PDF first

Need more help? See our [Troubleshooting Guide](../troubleshooting.md).
