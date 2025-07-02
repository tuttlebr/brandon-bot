# Installation Guide

This guide covers detailed installation instructions for the Streamlit Chat Application.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for development)
- Git

## Installation Methods

### 1. Docker Installation (Recommended)

See the [Quick Start Guide](quickstart.md) for the fastest way to get started.

### 2. Manual Installation

For development or custom deployments:

```bash
# Clone the repository
git clone https://github.com/tuttlebr/streamlit-chatbot.git
cd streamlit-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r docker/pyproject.toml
```

### 3. Cloud Deployment

The application can be deployed to various cloud platforms. See the [Deployment Guide](../deployment/docker.md) for details.

## Next Steps

- [First Steps](first-steps.md) - Configure and run your first chat
- [Configuration](../configuration/environment.md) - Set up environment variables
