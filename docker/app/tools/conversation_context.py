import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from openai import OpenAI
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class ContextType(str, Enum):
    """Types of context that can be generated"""

    CONVERSATION_SUMMARY = "conversation_summary"
    RECENT_TOPICS = "recent_topics"
    USER_PREFERENCES = "user_preferences"
    TASK_CONTINUITY = "task_continuity"


class ConversationContextResponse(BaseModel):
    """Response from the conversation context tool"""

    context_type: ContextType = Field(description="The type of context generated")
    summary: str = Field(description="The generated context summary")
    relevant_messages_count: int = Field(description="Number of messages analyzed")
    key_topics: List[str] = Field(description="Key topics extracted from conversation")
    user_intent: Optional[str] = Field(None, description="Current user intent if identifiable")
    direct_response: bool = Field(default=False, description="This tool provides context, not direct responses")


class ConversationContextTool:
    """Tool for generating conversation context summaries for other tools"""

    def __init__(self):
        self.name = "conversation_context"
        self.description = "Analyzes recent conversation history to provide relevant context summaries for tool operations. Use when tools need historical context about the user's requests, preferences, or ongoing tasks."

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert the tool to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context_type": {
                            "type": "string",
                            "enum": ["conversation_summary", "recent_topics", "user_preferences", "task_continuity"],
                            "description": "Type of context to generate: 'conversation_summary' for general summary, 'recent_topics' for topic extraction, 'user_preferences' for user preference analysis, 'task_continuity' for understanding ongoing tasks",
                        },
                        "message_count": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 20,
                            "default": 6,
                            "description": "Number of recent messages to analyze (default: 6, which covers ~3 conversation turns)",
                        },
                        "focus_query": {
                            "type": "string",
                            "description": "Optional specific query or topic to focus the context analysis on",
                        },
                    },
                    "required": ["context_type"],
                },
            },
        }

    def _create_llm_client(self, config: ChatConfig) -> OpenAI:
        """Create an OpenAI client for context analysis"""
        try:
            return OpenAI(api_key=config.api_key, base_url=config.llm_endpoint)
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            raise

    def _get_system_prompt(self, context_type: ContextType, focus_query: Optional[str] = None) -> str:
        """Get the appropriate system prompt for context analysis"""

        base_prompts = {
            ContextType.CONVERSATION_SUMMARY: """You are an expert conversation analyst. Create a concise, relevant summary of the conversation history that captures:
- Key topics discussed
- User's main requests or goals
- Important context for understanding current conversation
- Any ongoing tasks or problems being solved
Keep the summary focused and under 200 words.""",
            ContextType.RECENT_TOPICS: """You are an expert topic extractor. Analyze the conversation and identify:
- Main topics discussed (as a list)
- Recurring themes or interests
- Current focus areas
- Related concepts or domains
Focus on topics that might be relevant for tool operations.""",
            ContextType.USER_PREFERENCES: """You are an expert user behavior analyst. Identify patterns in the user's:
- Communication style and preferences
- Types of requests they make
- Preferred response formats
- Any expressed preferences or constraints
- Technical level or expertise indicators""",
            ContextType.TASK_CONTINUITY: """You are an expert task analyzer. Focus on understanding:
- What task or goal the user is working on
- Steps already completed or discussed
- Current stage of the task
- What information or help they might need next
- Context needed to continue the task effectively""",
        }

        prompt = base_prompts.get(context_type, base_prompts[ContextType.CONVERSATION_SUMMARY])

        if focus_query:
            prompt += f"\n\nSpecial focus: Pay particular attention to anything related to: {focus_query}"

        return prompt

    def _analyze_conversation_context(
        self,
        context_type: ContextType,
        messages: List[Dict[str, Any]],
        config: ChatConfig,
        focus_query: Optional[str] = None,
    ) -> ConversationContextResponse:
        """Analyze conversation messages to generate context"""

        logger.info(f"Analyzing {len(messages)} messages for context type: {context_type}")

        try:
            client = self._create_llm_client(config)
            system_prompt = self._get_system_prompt(context_type, focus_query)

            # Prepare conversation history for analysis
            conversation_text = self._format_messages_for_analysis(messages)

            if context_type == ContextType.RECENT_TOPICS:
                user_message = f"Extract and list the main topics from this conversation:\n\n{conversation_text}"
            elif context_type == ContextType.USER_PREFERENCES:
                user_message = f"Analyze user preferences and patterns from this conversation:\n\n{conversation_text}"
            elif context_type == ContextType.TASK_CONTINUITY:
                user_message = f"Analyze the ongoing task or goal in this conversation:\n\n{conversation_text}"
            else:  # CONVERSATION_SUMMARY
                user_message = f"Provide a concise summary of this conversation:\n\n{conversation_text}"

            if focus_query:
                user_message += f"\n\nFocus particularly on: {focus_query}"

            analysis_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(f"Making context analysis request with model: {config.llm_model_name}")

            response = client.chat.completions.create(
                model=config.llm_model_name,
                messages=analysis_messages,
                temperature=0.3,  # Lower temperature for more consistent analysis
                top_p=0.9,
                max_tokens=500,  # Limit response length for efficiency
            )

            result = response.choices[0].message.content.strip()

            # Extract key topics from the result
            key_topics = self._extract_key_topics(result, context_type)

            # Try to identify user intent
            user_intent = self._extract_user_intent(result, messages)

            logger.info(f"Successfully generated {context_type} context")

            return ConversationContextResponse(
                context_type=context_type,
                summary=result,
                relevant_messages_count=len(messages),
                key_topics=key_topics,
                user_intent=user_intent,
            )

        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            raise

    def _format_messages_for_analysis(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for LLM analysis"""
        formatted = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Skip system messages
            if role == "system":
                continue

            # Handle different content types
            if isinstance(content, dict):
                if content.get("type") == "image":
                    text_content = content.get("text", "[Image shared]")
                    formatted.append(f"{role.upper()}: {text_content}")
                else:
                    formatted.append(f"{role.upper()}: [Structured content]")
            elif isinstance(content, str):
                # Clean any existing context markers
                cleaned_content = self._clean_content(content)
                formatted.append(f"{role.upper()}: {cleaned_content}")

        return "\n\n".join(formatted)

    def _clean_content(self, content: str) -> str:
        """Clean content of metadata and markers"""
        import re

        # Remove context markers
        content = re.sub(r'<START_CONTEXT>.*?<END_CONTEXT>', '', content, flags=re.DOTALL)
        # Remove thinking tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Remove tool call instructions
        content = re.sub(r'<TOOLCALL.*?</TOOLCALL>', '', content, flags=re.DOTALL | re.IGNORECASE)

        return content.strip()

    def _extract_key_topics(self, result: str, context_type: ContextType) -> List[str]:
        """Extract key topics from the analysis result"""
        topics = []

        if context_type == ContextType.RECENT_TOPICS:
            # Try to extract list items or numbered items
            import re

            # Look for bullet points or numbered lists
            list_items = re.findall(r'(?:[-*•]\s*|^\d+\.\s*)(.+)', result, re.MULTILINE)
            topics.extend([item.strip() for item in list_items if item.strip()])

            # If no list found, try to extract topics from sentences
            if not topics:
                sentences = result.split('.')
                for sentence in sentences[:5]:  # Limit to first 5 sentences
                    if len(sentence.strip()) > 10:  # Skip very short fragments
                        topics.append(sentence.strip())
        else:
            # For other context types, extract key phrases
            import re

            # Look for quoted phrases or emphasized text
            quoted = re.findall(r'"([^"]+)"', result)
            topics.extend(quoted)

            # Extract capitalized phrases that might be topics
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', result)
            topics.extend(capitalized[:3])  # Limit to avoid noise

        # Clean and deduplicate
        topics = [t.strip() for t in topics if t.strip() and len(t.strip()) > 3]
        return list(dict.fromkeys(topics))[:5]  # Remove duplicates, limit to 5

    def _extract_user_intent(self, result: str, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Try to extract current user intent from analysis and recent messages"""

        # Look at the most recent user message for clues
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))

                # Simple intent detection patterns
                if any(word in content.lower() for word in ["help", "how", "can you", "what is"]):
                    return "seeking_information"
                elif any(word in content.lower() for word in ["create", "generate", "make", "write"]):
                    return "creation_request"
                elif any(word in content.lower() for word in ["fix", "error", "problem", "issue"]):
                    return "problem_solving"
                elif any(word in content.lower() for word in ["find", "search", "look for"]):
                    return "information_search"
                break

        return None

    def _run(
        self,
        context_type: str = None,
        message_count: int = 6,
        focus_query: str = None,
        messages: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> ConversationContextResponse:
        """Execute conversation context analysis"""

        # Support both direct parameters and dictionary input
        if context_type is None and "context_type" in kwargs:
            context_type = kwargs["context_type"]
        if message_count is None and "message_count" in kwargs:
            message_count = kwargs["message_count"]
        if focus_query is None and "focus_query" in kwargs:
            focus_query = kwargs["focus_query"]
        if messages is None and "messages" in kwargs:
            messages = kwargs["messages"]

        if context_type is None:
            raise ValueError("context_type parameter is required")
        if messages is None:
            raise ValueError("messages parameter is required for context analysis")

        # Validate context type
        try:
            context_enum = ContextType(context_type.lower())
        except ValueError:
            raise ValueError(f"Invalid context_type: {context_type}. Must be one of: {[t.value for t in ContextType]}")

        # Limit messages to the requested count, excluding system messages
        filtered_messages = []
        for msg in reversed(messages):
            if msg.get("role") != "system":
                filtered_messages.append(msg)
                if len(filtered_messages) >= message_count:
                    break

        # Reverse back to chronological order
        filtered_messages.reverse()

        logger.debug(f"Context analysis: {context_type}, {len(filtered_messages)} messages")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self._analyze_conversation_context(context_enum, filtered_messages, config, focus_query)

    def run_with_dict(self, params: Dict[str, Any]) -> ConversationContextResponse:
        """Execute context analysis with parameters provided as a dictionary"""

        if "context_type" not in params:
            raise ValueError("'context_type' key is required in parameters dictionary")
        if "messages" not in params:
            raise ValueError("'messages' key is required in parameters dictionary")

        return self._run(**params)


# Create a global instance and helper functions
conversation_context_tool = ConversationContextTool()


def get_conversation_context_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for conversation context"""
    return conversation_context_tool.to_openai_format()


def execute_conversation_context_with_dict(params: Dict[str, Any]) -> ConversationContextResponse:
    """Execute conversation context analysis with parameters as dictionary"""
    return conversation_context_tool.run_with_dict(params)
