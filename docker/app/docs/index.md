# Nemotron Chat Application Documentation

Welcome to the comprehensive documentation for the Nemotron Chat Application - a production-ready conversational AI platform with multimodal capabilities. Powered by the [NVIDIA Nemotron family of models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/).

## What Is NVIDIA Nemotron?

The NVIDIA Nemotron™ family of [multimodal models](https://www.nvidia.com/en-us/ai-data-science/foundation-models) provides state-of-the-art agentic reasoning for graduate-level scientific reasoning, advanced math, coding, instruction following, tool calling, and visual reasoning.

The models are optimized for different computing platforms: Nano for cost-efficiency and edge deployment, Super for balanced accuracy and compute efficiency on a single GPU, and Ultra for maximum accuracy in data centers.

The Nemotron models are commercially viable with an open license that allows for customization and data control.

### 🤖 NVIDIA Nemotron Models

Build enterprise agentic AI with benchmark-winning open reasoning and multimodal foundation models. This app is powered by NVIDIA's state-of-the-art language models, offering specialized models for different use cases:

- **llama-3.1-nemotron-nano-8b-v1 (fast)**: Leading reasoning and agentic AI accuracy model for PC and edge.
- **llama-3.3-nemotron-super-49b-v1 (llm)**: High efficiency model with leading accuracy for reasoning, tool calling, chat, and instruction following.
- **llama-3.1-nemotron-ultra-253b-v1 (intelligent)**: Superior inference efficiency with highest accuracy for scientific and complex math reasoning, coding, tool calling, and instruction following.
- **llama-3.1-nemotron-nano-vl-8b-v1 (vlm)**: Multi-modal vision-language model that understands text/img and creates informative responses.

## Overview

This application provides a sophisticated chat interface powered by NVIDIA's language models, featuring:

- **Multimodal Support**: Text, image, and document-based conversations
- **11+ Specialized Tools**: From web search to image generation
- **Smart Context Management**: Automatic conversation and PDF context injection
- **Production Architecture**: MVC pattern with clear separation of concerns
- **Real-time Streaming**: Smooth response delivery with progress indicators

## Key Features

### 🤖 Conversational AI

- Multiple language model support (Fast, Standard, Intelligent, VLM)
- Real-time streaming responses
- Context-aware conversations
- Tool-augmented responses

### 📄 Document Intelligence

- PDF analysis with progress tracking
- Intelligent document summarization
- Batch processing for large documents
- Context-aware Q&A

### 🖼️ Multimodal Capabilities

- **Image Generation**: Create AI-generated images with style control
- **Image Analysis**: Analyze uploaded images using vision models
- Support for various image formats

### 🔧 Advanced Tool System

- Text processing and analysis
- Web search and information retrieval
- Weather and news updates
- Code generation and development assistance
- Translation services

### 🏗️ Production-Ready Architecture

- Controller-based design pattern
- Service layer abstraction
- Comprehensive error handling
- Session isolation and security

## Quick Navigation

### Getting Started

- [Quick Start Guide](getting-started/quickstart.md) - Get up and running in 5 minutes
- [Installation](getting-started/installation.md) - Detailed setup instructions
- [First Steps](getting-started/first-steps.md) - Your first conversation

### User Guides

- [Chat Interface](user-guide/chat-interface.md) - Master the chat interface
- [PDF Analysis](user-guide/pdf-analysis.md) - Document processing features
- [Image Generation](user-guide/image-generation.md) - Create AI-powered images
- [Image Analysis](user-guide/image-upload-vlm.md) - Analyze images with vision models
- [Search Features](user-guide/search-features.md) - Web and knowledge base search

### Technical Documentation

- [Architecture Overview](architecture/overview.md) - System design and patterns
- [Services](architecture/services.md) - Core service implementations
- [API Reference](api/services.md) - Detailed API documentation

### Configuration & Deployment

- [Environment Variables](configuration/environment.md) - Configuration options
- [Model Configuration](configuration/models.md) - LLM setup guide
- [Docker Deployment](deployment/docker.md) - Production deployment

## Need Help?

- Check the [FAQ](faq.md) for common questions
- Visit [Troubleshooting](troubleshooting.md) for solving issues
- Review the [GitHub Repository](https://github.com/tuttlebr/streamlit-chatbot) for source code

## What's New

### Recent Updates

- **Auto PDF Context**: PDFs are automatically injected into context when relevant
- **Conversation Memory**: Sliding window context maintains conversation flow
- **Enhanced Tools**: Improved tool selection and parallel execution
- **Dark Theme**: Modern dark interface matching Nano's brand
- **Session Cleanup**: Automatic cleanup of temporary files

## License

This project is licensed under the MIT License. See the LICENSE file for details.
