# Streamlit Chat Application

## 🚀 **Production-Ready Application**

This is a fully refactored, production-ready Streamlit chatbot application with advanced LLM capabilities, PDF processing, and image generation.

## ✨ **Key Features**

- **Advanced LLM Chat** - Streaming responses with tool calling capabilities
- **PDF Document Analysis** - Upload and analyze PDF documents
- **Image Generation** - AI-powered image creation
- **Vector Search** - Semantic search through knowledge bases
- **Production Architecture** - Controller pattern with centralized configuration

## 🏃 **Quick Start**

### **Run the Application**

```bash
# Single production application
streamlit run main.py
```

**Single Entry Point**: Clean, streamlined codebase with only one application file.

### **Environment Setup**

Required environment variables:

```bash
NVIDIA_API_KEY=your_nvidia_api_key
FAST_LLM_MODEL_NAME=your_model_name
FAST_LLM_ENDPOINT=your_endpoint
# ... see utils/config.py for complete list
```

## 🏗️ **Architecture Overview**

```
docker/app/
├── main.py                    # 🎯 Main application entry point
├── controllers/               # 🎮 Controller pattern implementation
│   ├── session_controller.py  # Session state management
│   ├── message_controller.py  # Message processing
│   ├── file_controller.py     # File operations
│   └── response_controller.py # LLM response handling
├── utils/
│   └── config.py             # 🔧 Centralized configuration
├── services/                 # 🛠️ Core services
├── tools/                    # 🔨 LLM tools and integrations
├── ui/                       # 🎨 UI components
└── models/                   # 📝 Data models
```

## 🎯 **Production Benefits**

### **Reduced Complexity**

- **Controller pattern** for separation of concerns
- **Single responsibility** principle throughout

### **Centralized Configuration**

- **All settings** in one location (`utils/config.py`)
- **Environment validation** on startup
- **Type-safe configuration** with dataclasses

### **Improved Maintainability**

- **Isolated components** for easy testing
- **Clear separation** of concerns
- **Production logging** and error handling

## 🔧 **Configuration**

All configuration is centralized in `utils/config.py`:

```python
from utils.config import config

# Access environment variables
api_key = config.env.NVIDIA_API_KEY

# Access UI settings
spinner_icons = config.ui.SPINNER_ICONS

# Get LLM parameters
llm_params = config.get_llm_parameters()

# Validate environment
config.validate_environment()
```

## 🧪 **Development**

### **Testing Configuration**

```python
from utils.config import config
config.validate_environment()  # Check required env vars
```

### **Adding New Controllers**

```python
# controllers/new_controller.py
from utils.config import config

class NewController:
    def __init__(self, config_obj):
        self.config_obj = config_obj
        # Use centralized config: config.env.*, config.ui.*, etc.
```

## 🏆 **Key Achievements**

- ✅ **Production-ready architecture** with controller pattern
- ✅ **Zero configuration redundancy** - single source of truth
- ✅ **Environment validation** with startup checks

## 🎉 **Ready for Production**

This application demonstrates best practices for:

- Separation of concerns via controller pattern
- Centralized configuration management
- Production-ready error handling and logging
- Scalable architecture for future development
