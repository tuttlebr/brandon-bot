# Nano Chat Application

Welcome to the documentation for Nano - a production-ready conversational AI platform powered by NVIDIA's most advanced language models.

## Overview

Nano provides a sophisticated chat interface with cutting-edge AI capabilities, featuring:

- 🚀 **Real-time Streaming**: Experience fluid conversations with streaming responses
- 📄 **Intelligent PDF Analysis**: Upload and analyze documents with context-aware processing
- 🎨 **AI Image Generation**: Create stunning images from text descriptions
- 🔍 **Smart Search**: Web search and semantic knowledge base retrieval
- 🛠️ **Extensible Tool System**: Advanced tools for specialized tasks
- 🏗️ **Production Architecture**: Built with MVC pattern and enterprise-grade design
- 🧠 **Auto Context Injection**: Automatic PDF and conversation context awareness

## Key Features

### 🤖 Advanced Language Models

Nano leverages three specialized NVIDIA models optimized for different use cases:

- **Fast Model** (`meta/llama-3.1-8b-instruct`): Quick responses for simple queries
- **Standard Model** (`meta/llama-3.1-70b-instruct`): Balanced performance for general use
- **Intelligent Model** (`nvidia/llama-3.3-nemotron-70b-instruct`): Advanced reasoning for complex tasks

### 📊 Smart Document Processing

Our intelligent PDF analysis system automatically adapts to document size:

- **Small Documents (≤5 pages)**: Instant full analysis
- **Medium Documents (6-15 pages)**: Efficient batch processing
- **Large Documents (>15 pages)**: Two-phase intelligent scanning with relevance detection

### 🔧 Built-in Tool System

Nano includes powerful tools that extend its capabilities:

- **Text Assistant**: Advanced text processing (summarization, proofreading, translation)
- **PDF Tools**: Document analysis and content extraction
- **Image Generation**: AI-powered image creation with style control
- **Web Search**: Real-time web information via Tavily
- **News Search**: Current events and news articles
- **Weather**: Real-time weather information
- **Conversation Context**: Intelligent conversation analysis

### 🏢 Enterprise Features

- **Session Management**: Isolated user sessions with secure file handling
- **External Storage**: Efficient file storage preventing memory issues
- **Error Recovery**: Comprehensive error handling and graceful degradation
- **Scalability**: Stateless design for horizontal scaling
- **Configuration**: Centralized environment-based configuration
- **Progress Tracking**: Real-time progress updates for long operations

## System Requirements

- Docker and Docker Compose
- NVIDIA API credentials
- 4GB+ RAM recommended
- Python 3.10+ (for development)
- Optional: Tavily API key for web search

## Quick Navigation

- 🚀 **[Quick Start](getting-started/quickstart.md)** - Get running in 5 minutes
- 📖 **[User Guide](user-guide/chat-interface.md)** - Learn all features
- 🏗️ **[Architecture](architecture/overview.md)** - Technical deep dive
- 🔧 **[Configuration](configuration/environment.md)** - Setup guide
- 🐛 **[Troubleshooting](troubleshooting.md)** - Common issues

## What's New

### Recent Updates

- **Auto PDF Context**: PDFs are automatically injected into context when relevant
- **Conversation Memory**: Sliding window context maintains conversation flow
- **Enhanced Tools**: Improved tool selection and parallel execution
- **Dark Theme**: Modern dark interface matching Nano's brand
- **Session Cleanup**: Automatic cleanup of temporary files

## License

This project is licensed under the MIT License. See the LICENSE file for details.
