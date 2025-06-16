# Streamlit AI Chatbot

A sophisticated AI chatbot built with Streamlit that supports both text-to-text conversations and text-to-image generation. Features include retrieval-augmented generation (RAG), vector search, reranking, and a modern dark-themed UI.

## 🚀 Features

### 💬 Conversational AI

- **Multi-modal Chat**: Text-to-text conversations and text-to-image generation
- **NVIDIA-powered**: Uses NVIDIA's AI APIs for LLM and image generation
- **Streaming Responses**: Real-time response generation with typing indicators
- **Chat History**: Persistent conversation memory with pagination

### 🔍 Advanced RAG (Retrieval-Augmented Generation)

- **Vector Search**: Semantic search using embeddings
- **Reranking**: Intelligent result reranking for better context relevance
- **Context Integration**: Automatically enhances prompts with relevant information
- **Milvus Integration**: Scalable vector database support

### 🎨 Modern UI/UX

- **Dark Theme**: NVIDIA-branded dark interface
- **Responsive Design**: Works on desktop and mobile
- **Custom Styling**: Beautiful, modern chat interface
- **Image Display**: Integrated image viewing with captions

### 🔧 Enterprise Ready

- **Docker Support**: Containerized deployment
- **Environment Configuration**: Flexible configuration via environment variables
- **Authentication**: Optional password protection
- **Logging**: Comprehensive error handling and debugging
- **Health Checks**: Built-in health monitoring

## 📋 Prerequisites

- Docker and Docker Compose
- NVIDIA API key
- Access to:
  - NVIDIA LLM endpoint
  - NVIDIA embedding endpoint
  - NVIDIA image generation endpoint
  - Milvus vector database (optional)

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd streamlit-chatbot
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# Bot Configuration
BOT_TITLE=Nano
META_USER=Your Name
AUTH_USERNAME=admin
AUTH_KEY=your-auth-key

# NVIDIA API Configuration
NVIDIA_API_KEY=your-nvidia-api-key
LLM_MODEL_NAME=meta/llama-3.1-405b-instruct
LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
EMBEDDING_ENDPOINT=https://integrate.api.nvidia.com/v1
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
IMAGE_ENDPOINT=https://integrate.api.nvidia.com/v1/images/generations

# Vector Database (Optional)
DATABASE_URL=your-milvus-url
COLLECTION_NAME=chatbot_context
DEFAULT_DB=default
PARTITION_NAME=default

# Reranker (Optional)
RERANKER_ENDPOINT=https://integrate.api.nvidia.com/v1
RERANKER_MODEL=nvidia/nv-rerankqa-mistral-4b-v3

# Authentication (Optional)
# Uncomment to enable password protection
# password=your-secure-password
```

### 3. Launch with Docker

```bash
# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f app
```

### 4. Access the Application

Open your browser and navigate to:

- **Local**: http://localhost:8501
- **With Nginx**: http://localhost:8017

## 📚 Configuration Guide

### Environment Variables

| Variable             | Description                   | Required | Default   |
| -------------------- | ----------------------------- | -------- | --------- |
| `BOT_TITLE`          | Name of the AI assistant      | Yes      | "Nano"    |
| `META_USER`          | Default user name             | Yes      | "Brandon" |
| `NVIDIA_API_KEY`     | NVIDIA API authentication key | Yes      | -         |
| `LLM_MODEL_NAME`     | NVIDIA LLM model to use       | Yes      | -         |
| `LLM_ENDPOINT`       | NVIDIA LLM API endpoint       | Yes      | -         |
| `EMBEDDING_ENDPOINT` | Embedding API endpoint        | Yes      | -         |
| `EMBEDDING_MODEL`    | Embedding model name          | Yes      | -         |
| `IMAGE_ENDPOINT`     | Image generation endpoint     | Yes      | -         |
| `DATABASE_URL`       | Milvus vector database URL    | Yes      | -         |
| `COLLECTION_NAME`    | Milvus collection name        | Yes      | "milvus"  |
| `RERANKER_ENDPOINT`  | Reranker API endpoint         | Yes      | -         |
| `RERANKER_MODEL`     | Reranker model name           | Yes      | -         |

### Model Configuration

The application supports various NVIDIA models:

**LLM Models:**

- `nvidia/llama-3.3-nemotron-super-49b-v1`
- `meta/llama-3.1-70b-instruct`
- `microsoft/phi-3-medium-128k-instruct`
- `mistralai/mixtral-8x7b-instruct-v0.1`

**Embedding Models:**

- `nvidia/llama-3.2-nv-embedqa-1b-v2`
- `nvidia/nv-embed-v1`

**Reranking Models:**

- nvidia/llama-3.2-nv-rerankqa-1b-v2

**Image Generation:**

- black-forest-labs/flux.1-schnell

## 🎯 Usage

### Text Conversations

Simply type your message in the chat input and press Enter. The AI will respond with contextually relevant information.

### Image Generation

Use phrases like:

- "Generate an image of..."
- "Create a picture of..."
- "Draw a..."
- "Show me a visual of..."

Example: _"Generate an image of a futuristic cityscape at sunset"_

### RAG Search

When enabled, the chatbot automatically searches your knowledge base for relevant context to enhance responses.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Chat Service   │    │  NVIDIA APIs    │
│                 │───▶│                  │───▶│                 │
│ • Chat Interface│    │ • Message Proc.  │    │ • LLM           │
│ • Image Display │    │ • Context Enhancement│  │ • Embeddings   │
│ • History Mgmt  │    │ • RAG Integration│    │ • Image Gen     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │ Vector Database  │
         │              │                  │
         └──────────────│ • Milvus        │
                        │ • Similarity    │
                        │ • Reranking     │
                        └──────────────────┘
```

### Key Components

- **Streamlit App**: Main UI and orchestration
- **Chat Service**: Message processing and RAG integration
- **Image Service**: Text-to-image generation handling
- **LLM Service**: Streaming response generation
- **UI Components**: Reusable interface elements
- **Utils**: Helper functions for various operations

## 🔧 Development

### Local Development Setup

```bash
# Navigate to app directory
cd docker/app

# Install dependencies (if running locally)
pip install -r requirements.txt

# Run directly with Streamlit
streamlit run streamlit_app.py
```

### Project Structure

```
docker/app/
├── streamlit_app.py          # Main application entry point
├── models/                   # Data models and configuration
│   ├── chat_config.py       # Application configuration
│   └── chat_message.py      # Message handling
├── services/                 # Business logic services
│   ├── chat_service.py      # Chat processing and RAG
│   ├── image_service.py     # Image generation
│   └── llm_service.py       # LLM interaction
├── ui/                      # User interface components
│   ├── components.py        # Reusable UI components
│   └── styles.py           # Custom CSS and styling
├── utils/                   # Utility functions
│   ├── auth.py             # Authentication helpers
│   ├── environment.py      # Environment configuration
│   ├── image.py            # Image processing utilities
│   ├── retrieval.py        # Vector search and RAG
│   ├── split_context.py    # Context processing
│   └── system_prompt.py    # AI system prompts
└── assets/                 # Static assets (images, etc.)
```

### Code Style

The project uses:

- **Black** for code formatting
- **isort** for import sorting
- **Type hints** for better code documentation
- **Pydantic** for data validation

## 🐛 Troubleshooting

### Common Issues

**No LLM Response**

- Verify `NVIDIA_API_KEY` is set correctly
- Check `LLM_ENDPOINT` and `LLM_MODEL_NAME` configuration
- Review application logs for API errors

**Image Generation Fails**

- Ensure `IMAGE_ENDPOINT` is configured
- Verify the model supports image generation
- Check for content policy violations

**Vector Search Not Working**

- Confirm `DATABASE_URL` and `COLLECTION_NAME` are set
- Verify Milvus database connectivity
- Check if embedding endpoint is configured

**UI Issues**

- Clear browser cache and cookies
- Try accessing via incognito/private mode
- Check for JavaScript console errors

### Debugging

```bash
# View application logs
docker-compose logs -f app

# Check container status
docker-compose ps

# Restart services
docker-compose restart
```

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:

- Check the [Issues](../../issues) section
- Review the troubleshooting guide above
- Consult NVIDIA's API documentation

---

Built with ❤️ using Streamlit and NVIDIA AI APIs
