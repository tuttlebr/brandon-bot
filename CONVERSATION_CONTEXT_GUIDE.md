# Conversation Context Tool - Efficient History Integration

## Overview

The Conversation Context Tool provides an efficient way to include user history between turns for tool calls that benefit from prior conversation context. Instead of passing the entire conversation history to every tool (which would be inefficient and expensive), this tool intelligently analyzes and summarizes recent conversation history for other tools to use.

## Key Features

- **Efficient Context Summarization**: Generates concise summaries instead of passing raw conversation history
- **Multiple Context Types**: Supports different types of context analysis (summary, topics, preferences, task continuity)
- **Automatic Message Injection**: Conversation messages are automatically injected when the tool is called
- **Configurable History Window**: Control how many recent messages to analyze
- **LLM-Powered Analysis**: Uses the same LLM infrastructure as your chatbot for consistent analysis

## Architecture

### 1. Conversation Context Tool (`conversation_context.py`)

The core tool that analyzes conversation history and generates context summaries:

```python
# Tool definition
{
    "name": "conversation_context",
    "parameters": {
        "context_type": "conversation_summary|recent_topics|user_preferences|task_continuity",
        "message_count": 6,  # Number of recent messages to analyze
        "focus_query": "optional specific focus"
    }
}
```

### 2. Automatic Message Injection

The LLM service automatically injects conversation messages when the context tool is called:

```python
# In llm_service.py - execute_single_tool function
elif tool_name == "conversation_context":
    modified_args = tool_args.copy()
    modified_args["messages"] = messages  # Full conversation history
    tool_args = modified_args
```

### 3. Context Types

#### `conversation_summary`
- **Purpose**: General conversation summary
- **Use Case**: When tools need overall context about what's been discussed
- **Output**: Concise summary under 200 words

#### `recent_topics`
- **Purpose**: Extract main topics and themes
- **Use Case**: When tools need to understand subject matter
- **Output**: List of key topics and themes

#### `user_preferences`
- **Purpose**: Analyze user communication patterns and preferences
- **Use Case**: When tools need to adapt to user style
- **Output**: User behavior analysis and preferences

#### `task_continuity`
- **Purpose**: Understand ongoing tasks and goals
- **Use Case**: When tools need to continue or build upon previous work
- **Output**: Task analysis and next steps

## Implementation Details

### Integration with Existing Tools

The conversation context tool integrates seamlessly with your existing tool ecosystem:

1. **Added to Tool Registry**: Automatically available to the LLM
2. **Message Injection**: Conversation history is injected automatically
3. **Context Sharing**: Results can be used by subsequent tool calls in the same turn

### Example Usage Scenarios

#### Scenario 1: Weather with Location Memory
```
User: "What's the weather like?"
Assistant: [Calls conversation_context to check if user previously mentioned a location]
Assistant: [Calls weather tool with inferred location from context]
```

#### Scenario 2: Continued Task Analysis
```
User: "Can you continue with the analysis we were doing?"
Assistant: [Calls conversation_context with task_continuity to understand what analysis was being done]
Assistant: [Calls appropriate tools to continue the task]
```

#### Scenario 3: Preference-Aware Responses
```
User: "Summarize this article"
Assistant: [Calls conversation_context with user_preferences to understand preferred summary style]
Assistant: [Calls text_assistant with appropriate style parameters]
```

## Usage Examples

### Basic Context Summary
```python
# LLM decides to call context tool
{
    "name": "conversation_context",
    "arguments": {
        "context_type": "conversation_summary",
        "message_count": 6
    }
}

# Response provides summary of last 6 messages
{
    "context_type": "conversation_summary",
    "summary": "User has been asking about Python programming best practices...",
    "relevant_messages_count": 6,
    "key_topics": ["Python", "best practices", "code review"],
    "user_intent": "seeking_information"
}
```

### Topic-Focused Analysis
```python
# LLM calls context tool with specific focus
{
    "name": "conversation_context",
    "arguments": {
        "context_type": "recent_topics",
        "message_count": 10,
        "focus_query": "machine learning projects"
    }
}

# Response focuses on ML-related topics
{
    "context_type": "recent_topics",
    "summary": "Discussion has covered several ML topics including...",
    "key_topics": ["neural networks", "data preprocessing", "model evaluation"],
    "user_intent": "project_planning"
}
```

## How Other Tools Can Benefit

### 1. Enhanced Tool Descriptions
Update tool descriptions to mention when context might be helpful:

```python
self.description = """Get weather information. If no location is provided,
use conversation_context to check if user previously mentioned a location."""
```

### 2. Multi-Tool Workflows
The LLM can chain tools together:

1. Call `conversation_context` to understand what's been discussed
2. Use context results to make better decisions for subsequent tool calls
3. Provide more relevant and personalized responses

### 3. Context-Aware Parameters
Tools can be enhanced to accept context parameters:

```python
def enhanced_tool_call(self, location=None, context_summary=None):
    if not location and context_summary:
        # Extract location from context summary
        location = self.extract_location_from_context(context_summary)

    return self.execute_with_location(location)
```

## Performance Considerations

### Efficiency Features

1. **Configurable Message Window**: Only analyze recent messages (default: 6 messages)
2. **Focused Analysis**: Target specific aspects with `focus_query` parameter
3. **Concurrent Tool Execution**: Context analysis runs in parallel with other tools
4. **Cached Results**: Context results available to subsequent tools in same turn

### Resource Management

- **Token Efficiency**: Summaries are much shorter than full conversation history
- **Selective Usage**: Only called when context would be beneficial
- **Temperature Control**: Uses lower temperature (0.3) for consistent analysis

## Best Practices

### 1. When to Use Context Analysis

**Good Use Cases:**
- User asks vague questions that need context ("continue with that", "what about the other one")
- Tools need location/preference information not explicitly provided
- Multi-step workflows where previous steps matter
- Personalization based on user communication style

**Avoid When:**
- User provides all necessary information explicitly
- One-off questions without context dependency
- Simple factual queries

### 2. Choosing Context Types

- **conversation_summary**: General purpose, good default choice
- **recent_topics**: When understanding subject matter is key
- **user_preferences**: For personalization and style adaptation
- **task_continuity**: For multi-step or ongoing projects

### 3. Integration Patterns

```python
# Pattern 1: Context-first approach
# 1. Check context first
# 2. Use context to enhance main tool call

# Pattern 2: Parallel approach
# 1. Call both context and main tool simultaneously
# 2. Combine results intelligently

# Pattern 3: Fallback approach
# 1. Try main tool call
# 2. If insufficient info, use context to retry
```

## Configuration

### Environment Variables
The conversation context tool uses the same LLM configuration as your main chatbot:

```bash
# Uses existing config from ChatConfig.from_environment()
API_KEY=your_api_key
LLM_ENDPOINT=your_llm_endpoint
LLM_MODEL_NAME=your_model_name
```

### Customization Options

You can customize the tool by modifying `conversation_context.py`:

- **Analysis Prompts**: Modify `_get_system_prompt()` for different analysis styles
- **Topic Extraction**: Enhance `_extract_key_topics()` for better topic identification
- **Intent Detection**: Improve `_extract_user_intent()` for better intent recognition
- **Message Filtering**: Customize `_format_messages_for_analysis()` for different message handling

## Troubleshooting

### Common Issues

1. **Empty Context Results**
   - Check if conversation has enough messages
   - Verify message_count parameter is appropriate
   - Ensure messages contain meaningful content

2. **Context Tool Not Called**
   - Verify tool is properly registered in `ALL_TOOLS`
   - Check LLM understands when context would be helpful
   - Consider updating tool descriptions to mention context usage

3. **Poor Context Quality**
   - Adjust `message_count` parameter
   - Use `focus_query` to target specific aspects
   - Check if conversation history is being cleaned properly

### Debug Information

Enable debug logging to see context tool execution:

```python
import logging
logging.getLogger("tools.conversation_context").setLevel(logging.DEBUG)
```

This will show:
- Which messages are being analyzed
- Context generation process
- Topic extraction results
- Intent detection outcomes

## Future Enhancements

### Potential Improvements

1. **Semantic Chunking**: Group related messages together for better context
2. **User Modeling**: Build persistent user preference profiles
3. **Context Caching**: Cache context results across conversation turns
4. **Tool-Specific Context**: Generate context tailored for specific tools
5. **Multi-Modal Context**: Handle image and other content types in context analysis

### Integration Opportunities

- **Vector Search**: Combine with retrieval tool for document-aware context
- **External Knowledge**: Integrate with knowledge bases for enhanced context
- **Conversation Analytics**: Track conversation patterns over time
- **Personalization Engine**: Build comprehensive user profiles

## Conclusion

The Conversation Context Tool provides a sophisticated yet efficient way to include historical context in tool calls. By generating intelligent summaries rather than passing raw conversation history, it enables more context-aware and personalized interactions while maintaining performance and cost efficiency.

The tool's modular design allows for easy integration with existing tools and workflows, making it a powerful addition to any conversational AI system that needs to maintain context across turns.
