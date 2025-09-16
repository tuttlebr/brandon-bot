"""
Conversation Context Tool - MVC Pattern Implementation

This tool analyzes conversation history to extract specific types of context
following the Model-View-Controller pattern.
"""

from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from models.chat_config import ChatConfig
from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import (
    BaseTool,
    BaseToolResponse,
    StreamingToolResponse,
    ToolController,
    ToolView,
)
from utils.config import config as app_config

# Configure logger
from utils.logging_config import get_logger
from utils.pdf_extractor import PDFDataExtractor
from utils.text_processing import strip_all_thinking_formats

logger = get_logger(__name__)


class ContextType(str, Enum):
    """Types of context that can be generated"""

    CONVERSATION_SUMMARY = "conversation_summary"
    RECENT_TOPICS = "recent_topics"
    USER_PREFERENCES = "user_preferences"
    TASK_CONTINUITY = "task_continuity"
    CREATIVE_DIRECTOR = "creative_director"
    DOCUMENT_ANALYSIS = "document_analysis"


class ConversationContextResponse(BaseToolResponse):
    """Response from conversation context analysis"""

    analysis: str = Field(description="The context analysis result")
    user_intent: Optional[str] = Field(
        None, description="Identified user intent"
    )
    conversation_type: Optional[str] = Field(
        None, description="Type of conversation"
    )
    key_topics: Optional[List[str]] = Field(
        None, description="Key topics identified"
    )
    has_document: bool = Field(
        default=False, description="Whether a document is being discussed"
    )
    document_info: Optional[str] = Field(
        None, description="Information about the document if present"
    )
    direct_response: bool = Field(
        default=True,
        description=(
            "Flag indicating this response should be returned directly to user"
        ),
    )
    success: bool = Field(default=True, description="Success status")
    error_message: Optional[str] = Field(
        None, description="Error message if failed"
    )

    @property
    def result(self) -> str:
        """Get the analysis result for direct responses"""
        return self.analysis


class StreamingConversationContextResponse(StreamingToolResponse):
    """Streaming response from conversation context analysis"""

    user_intent: Optional[str] = Field(
        None, description="Identified user intent"
    )
    conversation_type: Optional[str] = Field(
        None, description="Type of conversation"
    )
    key_topics: Optional[List[str]] = Field(
        None, description="Key topics identified"
    )
    has_document: bool = Field(
        default=False, description="Whether a document is being discussed"
    )
    document_info: Optional[str] = Field(
        None, description="Information about the document if present"
    )
    direct_response: bool = Field(
        default=True,
        description=(
            "Flag indicating this response should be returned directly to user"
        ),
    )


class ConversationContextController(ToolController):
    """Controller handling conversation context analysis logic"""

    def __init__(self, llm_type: str):
        self.llm_type = llm_type

    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the conversation context analysis request"""
        if "query" not in params:
            raise ValueError(
                "'query' key is required in parameters dictionary"
            )
        if "max_messages" not in params:
            raise ValueError(
                "'max_messages' key is required in parameters dictionary"
            )

        # Validate context type
        try:
            context_enum = ContextType(params["query"].lower())
        except ValueError:
            raise ValueError(
                f"Invalid query: {params['query']}. Must be one of:"
                f" {[t.value for t in ContextType]}"
            )

        # Limit messages to the requested count, excluding system messages
        messages = params.get("messages", [])
        filtered_messages = []
        for msg in reversed(messages):
            if msg.get("role") != "system":
                filtered_messages.append(msg)
                if len(filtered_messages) >= params["max_messages"]:
                    break

        # Reverse back to chronological order
        filtered_messages.reverse()

        logger.debug(
            f"Context analysis: {params['query']},"
            f" {len(filtered_messages)} messages"
        )

        # Create config from environment
        config = ChatConfig.from_environment()

        # Analyze conversation
        return self._analyze_conversation_context(
            context_enum,
            filtered_messages,
            config,
            params.get("focus_query"),
            params.get("pdf_data"),
        )

    async def process_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the conversation context analysis request asynchronously"""
        if "query" not in params:
            raise ValueError(
                "'query' key is required in parameters dictionary"
            )
        if "max_messages" not in params:
            raise ValueError(
                "'max_messages' key is required in parameters dictionary"
            )

        # Validate context type
        try:
            context_enum = ContextType(params["query"].lower())
        except ValueError:
            raise ValueError(
                f"Invalid query: {params['query']}. Must be one of:"
                f" {[t.value for t in ContextType]}"
            )

        # Limit messages to the requested count, excluding system messages
        messages = params.get("messages", [])
        filtered_messages = []
        for msg in reversed(messages):
            if msg.get("role") != "system":
                filtered_messages.append(msg)
                if len(filtered_messages) >= params["max_messages"]:
                    break

        # Reverse back to chronological order
        filtered_messages.reverse()

        logger.debug(
            f"Context analysis: {params['query']},"
            f" {len(filtered_messages)} messages"
        )

        # Create config from environment
        config = ChatConfig.from_environment()

        # Create streaming generator
        content_generator = self._analyze_conversation_context_streaming(
            context_enum,
            filtered_messages,
            config,
            params.get("focus_query"),
            params.get("pdf_data"),
        )

        # Return streaming response data
        return {
            "content_generator": content_generator,
            "conversation_type": context_enum.value,
            "has_document": params.get("pdf_data") is not None,
            "document_info": (
                params.get("pdf_data", {}).get(
                    "info", "No additional document information"
                )
                if params.get("pdf_data")
                else "No additional document information"
            ),
            "direct_response": True,
            "is_streaming": True,
        }

    def _analyze_conversation_context(
        self,
        context_type: ContextType,
        messages: List[Dict[str, Any]],
        config: ChatConfig,
        focus_query: Optional[str] = None,
        pdf_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze conversation context using LLM (non-streaming)

        Args:
            context_type: Type of context analysis
            messages: Conversation messages
            config: Chat configuration
            focus_query: Optional specific aspect to focus on
            pdf_data: Optional PDF data being discussed

        Returns:
            Analysis results dictionary
        """
        try:
            # Get LLM client and model
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            logger.info(f"Analyzing {context_type.value} context")

            # Get system prompt for the specific context type
            system_prompt = self._get_system_prompt(context_type, focus_query)

            # Prepare conversation history for analysis
            conversation_text = self._format_messages_for_analysis(messages)

            if context_type == ContextType.RECENT_TOPICS:
                user_message = (
                    "Extract and list the main topics from this"
                    f" conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.USER_PREFERENCES:
                user_message = (
                    "Analyze the user's preferences and interaction patterns"
                    f" from this conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.TASK_CONTINUITY:
                user_message = (
                    "Track the task progression and identify next steps from"
                    f" this conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.CREATIVE_DIRECTOR:
                user_message = (
                    "Maintain creative continuity and track project evolution"
                    f" from this conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.DOCUMENT_ANALYSIS:
                # Include document content if available
                doc_content = self._get_document_content(messages, pdf_data)
                if doc_content:
                    user_message = (
                        "Analyze this document in relation to the"
                        " conversation:\n\nDocument"
                        f" Content:\n{doc_content}\n\nConversation:\n{conversation_text}"
                    )
                else:
                    user_message = (
                        "Analyze document-related aspects of this"
                        f" conversation:\n\n{conversation_text}"
                    )
            else:  # CONVERSATION_SUMMARY
                user_message = (
                    "Summarize this conversation and identify if the latest"
                    f" message requires action:\n\n{conversation_text}"
                )

            if focus_query:
                user_message += f"\n\nFocus particularly on: {focus_query}"

            analysis_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(
                "Making context analysis request with model:"
                f" {model_name} (type: {self.llm_type})"
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=analysis_messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            # Strip think tags from result
            result = strip_all_thinking_formats(result)

            # Extract key topics from the result
            key_topics = self._extract_key_topics(result, context_type)

            # Try to identify user intent
            user_intent = self._extract_user_intent(result, messages)

            logger.info(
                f"Successfully generated {context_type} context analysis"
            )

            return {
                "analysis": result,
                "user_intent": user_intent,
                "conversation_type": context_type.value,
                "key_topics": key_topics,
                "has_document": pdf_data is not None,
                "document_info": (
                    pdf_data.get("info", "No additional document information")
                    if pdf_data
                    else "No additional document information"
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            raise

    async def _analyze_conversation_context_streaming(
        self,
        context_type: ContextType,
        messages: List[Dict[str, Any]],
        config: ChatConfig,
        focus_query: Optional[str] = None,
        pdf_data: Dict[str, Any] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream conversation context analysis using LLM

        Args:
            context_type: Type of context analysis
            messages: Conversation messages
            config: Chat configuration
            focus_query: Optional specific aspect to focus on
            pdf_data: Optional PDF data being discussed

        Yields:
            Analysis response chunks
        """
        from utils.text_processing import StreamingCombinedThinkingFilter

        try:
            # Get async LLM client and model
            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            logger.info(
                f"Analyzing {context_type.value} context with streaming"
            )

            # Get system prompt for the specific context type
            system_prompt = self._get_system_prompt(context_type, focus_query)

            # Prepare conversation history for analysis
            conversation_text = self._format_messages_for_analysis(messages)

            if context_type == ContextType.RECENT_TOPICS:
                user_message = (
                    "Extract and list the main topics from this"
                    f" conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.USER_PREFERENCES:
                user_message = (
                    "Analyze the user's preferences and interaction patterns"
                    f" from this conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.TASK_CONTINUITY:
                user_message = (
                    "Track the task progression and identify next steps from"
                    f" this conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.CREATIVE_DIRECTOR:
                user_message = (
                    "Maintain creative continuity and track project evolution"
                    f" from this conversation:\n\n{conversation_text}"
                )
            elif context_type == ContextType.DOCUMENT_ANALYSIS:
                # Include document content if available
                doc_content = self._get_document_content(messages, pdf_data)
                if doc_content:
                    user_message = (
                        "Analyze this document in relation to the"
                        " conversation:\n\nDocument"
                        f" Content:\n{doc_content}\n\nConversation:\n{conversation_text}"
                    )
                else:
                    user_message = (
                        "Analyze document-related aspects of this"
                        f" conversation:\n\n{conversation_text}"
                    )
            else:  # CONVERSATION_SUMMARY
                user_message = (
                    "Summarize this conversation and identify if the latest"
                    f" message requires action:\n\n{conversation_text}"
                )

            if focus_query:
                user_message += f"\n\nFocus particularly on: {focus_query}"

            analysis_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(
                "Making streaming context analysis request with model:"
                f" {model_name} (type: {self.llm_type})"
            )

            response = await client.chat.completions.create(
                model=model_name,
                messages=analysis_messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                stream=True,  # Enable streaming
            )

            # Create think tag filter for streaming
            think_filter = StreamingCombinedThinkingFilter()

            # Process stream with think tag filtering and yield chunks
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    # Filter think tags from the chunk
                    filtered_content = think_filter.process_chunk(
                        chunk_content
                    )
                    if filtered_content:
                        yield filtered_content

            # Yield any remaining content from the filter
            final_content = think_filter.flush()
            if final_content:
                yield final_content

        except Exception as e:
            logger.error(
                f"Error in streaming conversation context analysis: {e}"
            )
            yield f"Error analyzing context: {str(e)}"

    def _get_system_prompt(
        self, context_type: ContextType, focus_query: Optional[str] = None
    ) -> str:
        """Get the appropriate system prompt for context analysis"""
        from tools.tool_llm_config import get_tool_system_prompt

        # Map context types to their prompt keys
        context_prompt_map = {
            ContextType.CONVERSATION_SUMMARY: "conversation_context_summary",
            ContextType.RECENT_TOPICS: "conversation_context_recent_topics",
            ContextType.USER_PREFERENCES: (
                "conversation_context_user_preferences"
            ),
            ContextType.TASK_CONTINUITY: (
                "conversation_context_task_continuity"
            ),
            ContextType.CREATIVE_DIRECTOR: (
                "conversation_context_creative_director"
            ),
            ContextType.DOCUMENT_ANALYSIS: (
                "conversation_context_document_analysis"
            ),
        }

        prompt_key = context_prompt_map.get(
            context_type, "conversation_context_summary"
        )

        # Get the prompt from centralized configuration
        prompt = get_tool_system_prompt(prompt_key, "")

        if focus_query:
            prompt += (
                "\n\nSpecial focus: Pay particular attention to anything"
                f" related to: {focus_query}"
            )

        return prompt

    def _format_messages_for_analysis(
        self, messages: List[Dict[str, Any]]
    ) -> str:
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
        content = re.sub(
            r"<START_CONTEXT>.*?<END_CONTEXT>", "", content, flags=re.DOTALL
        )
        # Remove thinking tags
        content = strip_all_thinking_formats(content)
        # Remove tool call instructions
        content = re.sub(
            r"<TOOLCALL.*?</TOOLCALL>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        return content.strip()

    def _extract_key_topics(
        self, result: str, context_type: ContextType
    ) -> List[str]:
        """Extract key topics from the analysis result"""
        topics = []

        if context_type == ContextType.RECENT_TOPICS:
            # Try to extract list items or numbered items
            import re

            # Look for bullet points or numbered lists
            list_items = re.findall(
                r"(?:[-*•]\s*|^\d+\.\s*)(.+)", result, re.MULTILINE
            )
            topics.extend(
                [item.strip() for item in list_items if item.strip()]
            )

            # If no list found, try to extract topics from sentences
            if not topics:
                sentences = result.split(".")
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
            capitalized = re.findall(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", result
            )
            topics.extend(capitalized[:3])  # Limit to avoid noise

        # Clean and deduplicate
        topics = [
            t.strip() for t in topics if t.strip() and len(t.strip()) > 3
        ]
        return list(dict.fromkeys(topics))[:5]  # Remove duplicates, limit to 5

    def _extract_user_intent(
        self, result: str, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Try to extract current user intent from analysis and recent messages"""

        # Look at the most recent user message for clues
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))

                # Simple intent detection patterns
                if any(
                    word in content.lower()
                    for word in ["help", "how", "can you", "what is"]
                ):
                    return "seeking_information"
                elif any(
                    word in content.lower()
                    for word in ["create", "generate", "make", "write"]
                ):
                    return "creation_request"
                elif any(
                    word in content.lower()
                    for word in ["fix", "error", "problem", "issue"]
                ):
                    return "problem_solving"
                elif any(
                    word in content.lower()
                    for word in ["find", "search", "look for"]
                ):
                    return "information_search"
                break

        return None

    def _get_document_content(
        self, messages: List[Dict[str, Any]], pdf_data: Dict[str, Any] = None
    ) -> Optional[str]:
        """Extract document content from provided PDF data or messages as fallback"""

        # Use provided PDF data if available
        if pdf_data is not None:
            return self._get_pdf_content_from_pdf_data(pdf_data)

        # Fallback to scanning messages for PDF data
        return self._get_pdf_content_from_messages(messages)

    def _get_pdf_content_from_pdf_data(
        self, pdf_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract PDF content from explicitly passed PDF data

        Args:
            pdf_data: PDF data dictionary passed from LLM service

        Returns:
            Formatted PDF content string or None if not available
        """
        if pdf_data:
            # Limit to first 5 pages for context analysis
            limited_data = {
                "pages": pdf_data.get("pages", [])[:5],
                "filename": pdf_data.get("filename"),
            }
            # Get text but limit each page to 1000 chars
            pages_text = []
            for page in limited_data["pages"]:
                page_text = page.get("text", "")
                if page_text:
                    pages_text.append(
                        f"Page {page.get('page', '?')}: {page_text[:1000]}..."
                    )
            return "\n\n".join(pages_text) if pages_text else None
        return None

    def _get_pdf_content_from_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract PDF content from messages (including injected system messages)

        Args:
            messages: List of conversation messages

        Returns:
            Formatted PDF content string or None if not available
        """
        pdf_data = PDFDataExtractor.extract_from_messages(messages)
        if pdf_data:
            # Limit to first 5 pages for context analysis
            limited_data = {
                "pages": pdf_data.get("pages", [])[:5],
                "filename": pdf_data.get("filename"),
            }
            # Get text but limit each page to 1000 chars
            pages_text = []
            for page in limited_data["pages"]:
                page_text = page.get("text", "")
                if page_text:
                    pages_text.append(
                        f"Page {page.get('page', '?')}: {page_text[:1000]}..."
                    )
            return "\n\n".join(pages_text) if pages_text else None
        return None


class ConversationContextView(ToolView):
    """View for formatting conversation context responses"""

    def format_response(
        self, data: Dict[str, Any], response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format raw data into ConversationContextResponse"""
        try:
            # Check if this is a streaming response
            if data.get("is_streaming") and data.get("content_generator"):
                return StreamingConversationContextResponse(**data)
            else:
                return ConversationContextResponse(**data)
        except Exception as e:
            logger.error(f"Error formatting context response: {e}")
            return ConversationContextResponse(
                analysis="",
                success=False,
                error_message=f"Response formatting error: {str(e)}",
                error_code="FORMAT_ERROR",
            )

    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error into ConversationContextResponse"""
        error_code = "UNKNOWN_ERROR"
        if isinstance(error, ValueError):
            error_code = "VALIDATION_ERROR"
        elif isinstance(error, TimeoutError):
            error_code = "TIMEOUT_ERROR"

        return ConversationContextResponse(
            analysis="",
            success=False,
            error_message=str(error),
            error_code=error_code,
        )


class ConversationContextTool(BaseTool):
    """Tool for analyzing conversation context and user intent following MVC pattern"""

    def __init__(self):
        super().__init__()
        self.name = "conversation_context"
        self.description = (
            "INTERNAL SYSTEM TOOL: Analyze conversation history for context."
            " Never select for user queries."
        )

    def _initialize_mvc(self):
        """Initialize MVC components"""
        self._controller = ConversationContextController(self.llm_type)
        self._view = ConversationContextView()

    def get_definition(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "enum": [
                                "conversation_summary",
                                "recent_topics",
                                "user_preferences",
                                "task_continuity",
                                "creative_director",
                                "document_analysis",
                            ],
                            "description": (
                                "Type of context analysis to perform. Choose"
                                " 'conversation_summary' for overall summary,"
                                " 'recent_topics' to list discussion topics,"
                                " 'user_preferences' for user patterns,"
                                " 'task_continuity' for tracking tasks,"
                                " 'creative_director' for creative projects,"
                                " or 'document_analysis' for document-related"
                                " context."
                            ),
                        },
                        "max_messages": {
                            "type": "integer",
                            "description": (
                                "Maximum number of messages to analyze"
                                " (default: 20)"
                            ),
                            "default": 20,
                        },
                        "include_document_content": {
                            "type": "boolean",
                            "description": (
                                "Whether to include full document content in"
                                " analysis if documents are present"
                            ),
                            "default": True,
                        },
                        "but_why": {
                            "type": "string",
                            "description": (
                                "An integer from 1-5 where a larger number"
                                " indicates confidence this is the right tool"
                                " to help the user."
                            ),
                        },
                    },
                    "required": ["query", "max_messages", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[BaseToolResponse]:
        """Get the response type for this tool"""
        return ConversationContextResponse


# Helper functions for backward compatibility
def get_conversation_context_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for conversation context"""
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("conversation_context", ConversationContextTool)

    # Get the tool instance and return its definition
    tool = get_tool("conversation_context")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError(
            "Failed to get conversation context tool definition"
        )
