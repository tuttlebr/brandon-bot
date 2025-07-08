# PDF Context Switching Improvements

## Problem Statement

After a PDF is uploaded, the chatbot was getting "stuck" in PDF context mode, interpreting all subsequent messages (even simple ones like "thanks!" or "what's the weather?") as being related to the PDF. This made it difficult for users to switch topics or have natural conversations.

## Root Cause

The original `PDFContextService.should_inject_pdf_context` method had problematic logic:

```python
# If the message is short and doesn't specify what to do,
# assume it's about the PDF if one was recently uploaded
if len(message_lower.split()) < 10:
    return True
```

This meant ANY message with less than 10 words would trigger PDF context injection.

## Solution

We've implemented a smarter context detection system that:

### 1. **Explicit Non-PDF Indicators**

Messages containing these keywords in short messages (≤5 words) will NOT trigger PDF context:

- Greetings: "hello", "hi", "thanks", "thank you", "goodbye", "bye"
- Topic switches: "weather", "news", "image", "picture", "generate", "create"
- Casual conversation: "how are you", "what's up"

### 2. **Strong PDF Keywords**

These keywords ALWAYS trigger PDF context:

- Direct references: "pdf", "document", "file", "paper", "uploaded"
- Specific requests: "page", "summary", "summarize"

### 3. **Contextual Keywords**

These keywords only trigger PDF context when combined with questions (3+ words):

- Content queries: "text", "content", "says", "mentions", "according to"
- Analysis requests: "analyze", "explain", "describe", "find", "search"
- References: "section", "chapter", "paragraph", "quote"

### 4. **Smart Reference Detection**

For medium-length messages (3-10 words) with pronouns like "it", "this", "that", we check if it's a question before assuming PDF context.

### 5. **Topic Switch Detection**

Messages that contain both PDF references AND non-PDF indicators prioritize the non-PDF intent (e.g., "I'm done with the document, what's the weather?").

## Tool Description Updates

We've also updated tool descriptions to be more specific:

- **PDF Summary Tool**: "ONLY use this when explicitly asked to summarize a PDF document..."
- **PDF Text Processor**: "ONLY use this when explicitly asked to perform text processing on PDF pages..."
- **Assistant Tool**: "...Can work with regular text or PDF content when specifically requested."

## System Prompt Enhancement

Added guidance to the system prompt:

```
Remember to be context-aware - not every message after a PDF upload is about the PDF. Users may want to:
- Thank you or express gratitude
- Change topics entirely
- Ask unrelated questions
- Have casual conversation

Only use PDF-related tools when the user's message clearly indicates they want to work with the uploaded document.
```

## Testing Results

Our test suite shows the improved logic correctly handles:

- ✅ Simple thanks/greetings (no PDF context)
- ✅ Weather/news queries (no PDF context)
- ✅ Direct PDF questions (with PDF context)
- ✅ Document-specific requests (with PDF context)
- ✅ Topic switches (no PDF context for new topic)
- ✅ Contextual references with clear intent

## Usage Examples

### Messages that WON'T trigger PDF context:

- "Thanks!"
- "What's the weather?"
- "Hello, how are you?"
- "Generate an image of a sunset"
- "Show me the latest news"

### Messages that WILL trigger PDF context:

- "What does the PDF say about climate change?"
- "Summarize page 5"
- "Find information about the methodology"
- "What's in the uploaded document?"
- "Can you analyze the text on page 3?"

### Smart context switching:

- "Thanks for the summary! Now, what's the weather?" → Weather query (no PDF)
- "I'm done with the document. Tell me a joke." → Joke request (no PDF)

## Implementation Files

The improvements are implemented in:

- `services/pdf_context_service.py` - Core context detection logic
- `tools/pdf_summary.py` - Updated tool descriptions
- `tools/pdf_text_processor.py` - Updated tool descriptions
- `tools/assistant.py` - Updated to clarify PDF handling
- `utils/system_prompt.py` - Enhanced with context awareness guidance

## Future Improvements

1. **Conversation Memory**: Track the last N messages to better understand when users are continuing PDF discussion vs changing topics
2. **Explicit Commands**: Support commands like "stop PDF mode" or "new topic"
3. **User Preferences**: Allow users to set their preferred context switching behavior
4. **Smart Timeouts**: Reduce PDF context weight after X minutes of non-PDF questions
