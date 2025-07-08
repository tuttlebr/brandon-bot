# Search Features Guide

Access real-time web information and search capabilities through the chat interface.

## Web Search

Get current information from the web:

```
"What's the latest news about renewable energy?"
"Search for recent AI developments"
"Find information about [company] stock price"
```

## Configuration

Web search requires:

- `TAVILY_API_KEY` environment variable
- Active internet connection

See [Environment Configuration](../configuration/environment.md) for setup.

## Search Types

### Current Events

- News and recent developments
- Real-time information
- Trending topics

### Research

- Academic information
- Technical documentation
- Historical data

### Fact Checking

- Verify claims
- Get updated statistics
- Confirm current information

## Best Practices

### Be Specific

Instead of: "Tell me about AI"
Try: "What are the latest breakthroughs in AI language models in 2024?"

### Specify Timeframe

- "Recent developments in..."
- "Latest updates about..."
- "Current status of..."

### Ask Follow-ups

Build on search results with additional questions for deeper insights.

## Limitations

- Requires API key configuration
- Subject to rate limits
- May not access all websites

## Combining with Other Features

### With PDF Analysis

1. Upload a research paper
2. Search for recent related work
3. Compare findings

### With Chat

1. Get search results
2. Ask for analysis
3. Request summaries

## Tips for Better Results

1. **Use Keywords**: Include specific terms you're looking for
2. **Be Current**: Specify if you need recent information
3. **Verify Sources**: Ask about the credibility of sources

## Troubleshooting

### No Results

- Check API key configuration
- Verify internet connectivity
- Try rephrasing query

### Outdated Information

- Explicitly ask for recent/current data
- Specify date ranges
- Request latest updates

## Next Steps

- Master the [Chat Interface](chat-interface.md)
- Explore [PDF Analysis](pdf-analysis.md)
- Learn about [Image Generation](image-generation.md)
