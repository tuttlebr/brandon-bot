# Image Upload and Analysis

The Streamlit Chat Application includes powerful image analysis capabilities using Vision Language Models (VLMs), allowing you to upload images and ask questions about their content.

## Overview

The image analysis feature uses NVIDIA's vision-capable language models to:
- Describe image content
- Answer specific questions about images
- Identify objects, people, and scenes
- Analyze visual elements like colors, composition, and style
- Provide insights about technical diagrams or illustrations

## How to Use

### 1. Upload an Image

1. Look for the **"📷 Image Upload"** section in the sidebar
2. Click **"Choose image file"** to select your image
3. Supported formats:
   - PNG
   - JPG/JPEG
   - GIF
   - BMP
   - WebP
4. Maximum file size: 10MB (configurable)
5. Wait for the confirmation message

### 2. Ask Questions

Once your image is uploaded, you can ask various types of questions:

#### General Description
- "What do you see in this image?"
- "Describe this image in detail"
- "What's happening in this picture?"

#### Specific Analysis
- "What objects are visible in this image?"
- "What is the main subject of this photo?"
- "Describe the colors and lighting"
- "What's the mood or atmosphere?"

#### Technical Questions
- "Is this a diagram? What does it show?"
- "What type of chart is this?"
- "Explain this technical illustration"

#### Creative Questions
- "Is this an illustration or a photograph?"
- "What art style is this?"
- "What story does this image tell?"

## Features

### Session Persistence
- Images remain available throughout your chat session
- You can ask multiple questions about the same image
- Upload a new image to replace the current one

### Intelligent Analysis
- The VLM provides detailed, context-aware responses
- Understands complex scenes and relationships
- Can identify text within images
- Recognizes technical diagrams and charts

### Privacy and Security
- Images are stored only in your session
- Automatic cleanup when session ends
- No permanent storage of uploaded images

## Configuration

The image analysis feature uses these environment variables:

```bash
# Vision Language Model Endpoint (Optional)
# Defaults to LLM_ENDPOINT if not specified
VLM_ENDPOINT=https://integrate.api.nvidia.com/v1

# Vision Language Model Name (Optional)
# Defaults to nvidia/llama-3.1-nemotron-nano-vl-8b-v1
VLM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1

# Image Upload Settings (Optional)
IMAGE_MAX_SIZE_MB=10  # Maximum upload size in MB
```

## Technical Details

### How It Works

1. **Upload Processing**:
   - Image is validated for format and size
   - Converted to base64 encoding
   - Stored in session state

2. **Analysis Flow**:
   - When you ask about the image, the `analyze_image` tool is triggered
   - Image data is sent to the VLM along with your question
   - Each request is a one-shot call (no conversation history)
   - Response is streamed back in real-time

3. **Session Management**:
   - Images are tied to your session
   - Only one image active at a time
   - Clear option available to remove current image

### Best Practices

1. **Image Quality**:
   - Use clear, well-lit images for best results
   - Higher resolution provides more detail for analysis
   - Avoid heavily compressed images

2. **Question Clarity**:
   - Be specific about what you want to know
   - Ask one question at a time for focused answers
   - Follow up with more specific questions based on initial analysis

3. **Use Cases**:
   - Product identification and description
   - Technical diagram explanation
   - Art and design analysis
   - Scene understanding
   - Text extraction from images
   - Visual problem solving

## Limitations

- One image at a time per session
- No image editing capabilities
- Analysis quality depends on image clarity
- Large images may take longer to process
- Some specialized content may require domain expertise

## Troubleshooting

### Image Won't Upload
- Check file size (under 10MB)
- Verify file format is supported
- Try a different browser if issues persist

### Analysis Not Working
- Ensure image uploaded successfully
- Check for confirmation message
- Try re-uploading the image

### Slow Response
- Larger images take more time
- Complex scenes require more processing
- Network speed affects upload time

## Examples

### Example 1: Product Analysis
```
User: [Uploads product photo]
User: "What product is this and what are its key features?"
Assistant: "This appears to be a wireless mechanical keyboard with..."
```

### Example 2: Technical Diagram
```
User: [Uploads circuit diagram]
User: "Explain this circuit diagram"
Assistant: "This is a basic LED circuit showing..."
```

### Example 3: Art Analysis
```
User: [Uploads painting]
User: "What art style is this and what period might it be from?"
Assistant: "This painting exhibits characteristics of Impressionism..."
```

## See Also

- [Image Generation](image-generation.md) - Create AI-generated images
- [Chat Interface](chat-interface.md) - General chat features
- [Configuration Guide](../configuration/environment.md) - Environment setup
