# System Prompt Design Pattern

This document explains the new system prompt design pattern that maintains consistent persona while enabling tool differentiation.

## Overview

The new `SystemPromptManager` class provides a centralized way to manage system prompts with intelligent caching and context-aware generation. This ensures:

1. **Consistent Persona**: Core personality traits remain stable across all interactions
2. **Tool Differentiation**: Different contexts can have specialized instructions
3. **Performance**: Intelligent caching reduces redundant prompt generation
4. **Maintainability**: Centralized prompt management

## Architecture

### Core Components

1. **SystemPromptManager**: Singleton class that manages prompt generation and caching
2. **Core Persona**: Static prompt that defines the assistant's fundamental personality
3. **Context Instructions**: Specialized instructions for different use cases
4. **Dynamic Components**: Date and tools list that change over time

### Design Principles

- **Separation of Concerns**: Core persona is separate from context-specific instructions
- **Intelligent Caching**: Cache is refreshed only when necessary (tools change, date changes, etc.)
- **Context Awareness**: Different contexts can have specialized instructions while maintaining core persona
- **Backward Compatibility**: Existing `get_system_prompt()` function continues to work

## Usage Examples

### Basic Usage

```python
from utils.system_prompt import get_system_prompt, get_context_system_prompt

# Get the standard system prompt
standard_prompt = get_system_prompt()

# Get a context-specific prompt
translation_prompt = get_context_system_prompt(
    context='translation',
    target_language='Spanish'
)
```

### Service Integration

```python
# In a service class
def _get_system_prompt(self, task_type: str) -> str:
    from utils.system_prompt import get_context_system_prompt

    return get_context_system_prompt(
        context='text_processing',
        task_type=task_type
    )
```

### Tool-Specific Prompts

```python
# For image generation
image_prompt = get_context_system_prompt(
    context='image_generation'
)

# For PDF analysis
pdf_prompt = get_context_system_prompt(
    context='pdf_analysis'
)

# For code generation
code_prompt = get_context_system_prompt(
    context='code_generation'
)
```

## Available Contexts

The system supports the following contexts:

1. **translation**: For language translation tasks
2. **text_processing**: For text analysis, summarization, etc.
3. **image_generation**: For image creation requests
4. **pdf_analysis**: For document analysis
5. **code_generation**: For programming tasks
6. **research**: For information gathering

## Caching Behavior

The system uses intelligent caching with the following refresh triggers:

- **Cache TTL**: 5 minutes (configurable)
- **Tools List Changes**: When new tools are registered
- **Date Changes**: When the date changes (for date-sensitive prompts)
- **Force Refresh**: When explicitly requested

## Benefits

### 1. Consistent Persona

The core persona prompt remains static, ensuring the assistant maintains consistent personality traits across all interactions.

### 2. Tool Differentiation

Different contexts can have specialized instructions while maintaining the core persona:

```python
# Translation context
translation_prompt = get_context_system_prompt('translation', target_language='French')

# Code generation context
code_prompt = get_context_system_prompt('code_generation')
```

### 3. Performance Optimization

- **Reduced Redundancy**: Caching prevents regenerating identical prompts
- **Efficient Updates**: Only refreshes when necessary
- **Memory Efficient**: Single cached instance shared across the application

### 4. Maintainability

- **Centralized Management**: All prompts managed in one place
- **Easy Updates**: Core persona changes affect all contexts
- **Clear Separation**: Context-specific instructions are clearly separated

## Migration Guide

### For Existing Services

1. **Replace direct prompt generation**:

   ```python
   # Old way
   prompt = f"""You are a translator..."""

   # New way
   from utils.system_prompt import get_context_system_prompt
   prompt = get_context_system_prompt('translation', target_language='Spanish')
   ```

2. **Update service methods**:

   ```python
   # Old way
   def _get_system_prompt(self, task_type):
       base_prompts = {...}
       return base_prompts.get(task_type)

   # New way
   def _get_system_prompt(self, task_type):
       return get_context_system_prompt('text_processing', task_type=task_type)
   ```

### For New Services

1. **Choose appropriate context** from the available options
2. **Use `get_context_system_prompt()`** with relevant parameters
3. **Maintain core persona** by not overriding fundamental personality traits

## Configuration

### Cache TTL

The cache TTL can be adjusted in the `SystemPromptManager`:

```python
system_prompt_manager._cache_ttl_seconds = 600  # 10 minutes
```

### Adding New Contexts

To add a new context:

1. **Add context instructions** in `_get_context_instructions()`:

   ```python
   'new_context': f"""You are performing {kwargs.get('task', 'specialized')} tasks.

   Your goal is to... while maintaining your core personality.

   Remember to maintain your core personality and conversational style."""
   ```

2. **Use the new context** in your service:
   ```python
   prompt = get_context_system_prompt('new_context', task='specific_task')
   ```

## Best Practices

1. **Always use contexts** instead of creating custom prompts
2. **Maintain persona consistency** by not overriding core personality traits
3. **Use appropriate context** for each service type
4. **Leverage caching** by reusing the same context across similar requests
5. **Clear cache** when needed using `system_prompt_manager.clear_cache()`

## Troubleshooting

### Cache Issues

If prompts seem stale:

```python
from utils.system_prompt import system_prompt_manager
system_prompt_manager.clear_cache()
```

### Context Not Found

If a context doesn't exist, the system falls back to the core persona prompt. Add the missing context to `_get_context_instructions()`.

### Performance Issues

Monitor cache hit rates and adjust TTL if needed:

```python
# Increase cache duration for better performance
system_prompt_manager._cache_ttl_seconds = 900  # 15 minutes
```
