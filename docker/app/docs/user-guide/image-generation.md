# Image Generation Guide

Create stunning AI-generated images directly from the chat interface.

## Getting Started

To generate an image, simply describe what you want to see:

```
"Generate an image of a sunset over mountains"
"Create a picture of a futuristic cityscape"
"Draw a cute robot holding flowers"
```

## Configuration Required

Image generation requires additional setup:

1. Set the `IMAGE_ENDPOINT` environment variable
2. Provide `IMAGE_API_KEY` if different from NVIDIA key
3. Restart the application

See [Environment Configuration](../configuration/environment.md) for details.

## Best Practices

### Be Descriptive

- Include details about style, colors, mood
- Specify the setting and atmosphere
- Mention specific elements you want

### Example Prompts

**Simple**:

- "A red rose in a garden"

**Detailed**:

- "A vibrant red rose with morning dew drops, surrounded by green leaves in a sunny garden, photorealistic style"

**Artistic**:

- "Abstract representation of joy using warm colors and flowing shapes, modern art style"

## Image Specifications

- Default size: 1024x1024
- Format: PNG
- Display: Inline in chat

## Limitations

- One image per request
- Processing time varies
- Some prompts may be filtered

## Troubleshooting

### No Image Generated

- Check API configuration
- Verify endpoint is accessible
- Review error messages

### Poor Quality Results

- Add more descriptive details
- Specify art style or technique
- Try different phrasings

## Next Steps

- Explore [Search Features](search-features.md)
- Learn about [PDF Analysis](pdf-analysis.md)
- Configure [API Settings](../configuration/environment.md)
