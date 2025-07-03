import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import BaseTool, BaseToolResponse
from utils.config import config as app_config
from utils.pdf_extractor import PDFDataExtractor
from utils.text_processing import strip_think_tags

# Configure logger
logger = logging.getLogger(__name__)


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
    user_intent: Optional[str] = Field(None, description="Identified user intent")
    conversation_type: Optional[str] = Field(None, description="Type of conversation")
    key_topics: Optional[List[str]] = Field(None, description="Key topics identified")
    has_document: bool = Field(default=False, description="Whether a document is being discussed")
    document_info: Optional[str] = Field(None, description="Information about the document if present")
    direct_response: bool = Field(
        default=True, description="Flag indicating this response should be returned directly to user",
    )
    success: bool = Field(default=True, description="Success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    @property
    def result(self) -> str:
        """Get the analysis result for direct responses"""
        return self.analysis


class ConversationContextTool(BaseTool):
    """Tool for analyzing conversation context and user intent"""

    def __init__(self):
        super().__init__()
        self.name = "conversation_context"
        self.description = "Analyzes conversation history to extract context, identify user intent, summarize topics, and track ongoing tasks or creative projects."
        # Use fast model for quick context analysis
        self.llm_type = "fast"

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI function calling format

        Returns:
            Dict containing the OpenAI-compatible tool definition
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The current user query to analyze in context",},
                        "max_messages": {
                            "type": "integer",
                            "description": "Maximum number of messages to analyze (default: 20)",
                            "default": 20,
                        },
                        "include_document_content": {
                            "type": "boolean",
                            "description": "Whether to include full document content in analysis if documents are present",
                            "default": True,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def _get_system_prompt(self, context_type: ContextType, focus_query: Optional[str] = None) -> str:
        """Get the appropriate system prompt for context analysis"""

        base_prompts = {
            ContextType.CONVERSATION_SUMMARY: """detailed thinking on
            You are summarizing the conversation history to provide context.

Create a concise overview that captures the main themes, user objectives, and ongoing tasks. Include critical background information needed for immediate context. Keep the summary focused and within 200 words.""",
            ContextType.RECENT_TOPICS: """detailed thinking on
            You are identifying and listing the main topics discussed in the conversation.

Extract and enumerate the primary discussion threads. Note recurring themes and the current focus of engagement. Include relevant related topics that may be important. Prioritize topics that are most relevant to ongoing tasks or questions.""",
            ContextType.USER_PREFERENCES: """detailed thinking on
            You are analyzing user interaction patterns and preferences.

Identify the user's communication style, typical request types, and preferred response formats. Note any stated constraints or preferences. Assess the user's apparent expertise level and information needs based on their interactions.""",
            ContextType.TASK_CONTINUITY: """detailed thinking on
            You are tracking task progression and continuity.

Identify the main task or objective being pursued. Document completed steps and current progress. Determine what stage the user is at in their task. Anticipate likely next steps and information needs. Provide essential context for seamless task continuation.""",
            ContextType.CREATIVE_DIRECTOR: """detailed thinking on
            You are maintaining creative project continuity and coherence.

Track the creative project's vision, scope, and goals. Document the evolution of core ideas and concepts. Ensure consistency in tone, style, and narrative direction. Identify opportunities for enhancement or new perspectives. Maintain an awareness of referenced materials and inspirations.""",
            ContextType.DOCUMENT_ANALYSIS: """detailed thinking on
            You are analyzing document content in relation to the conversation.

Summarize key points, themes, and main arguments from the document. Identify the document's structure and organization. Extract critical information, facts, and conclusions. Relate content to user queries and conversation context. Note any action items or recommendations. Identify connections between document content and conversation topics.""",
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
        pdf_data: Dict[str, Any] = None,
    ) -> ConversationContextResponse:
        """Analyze conversation messages to generate context"""

        logger.info(f"Analyzing {len(messages)} messages for context type: {context_type}")

        try:
            # Get the appropriate client and model based on this tool's LLM type
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)
            system_prompt = self._get_system_prompt(context_type, focus_query)

            # Prepare conversation history for analysis
            conversation_text = self._format_messages_for_analysis(messages)

            if context_type == ContextType.RECENT_TOPICS:
                user_message = f"Extract and list the main topics from this conversation:\n\n{conversation_text}"
            elif context_type == ContextType.USER_PREFERENCES:
                user_message = f"Analyze user preferences and patterns from this conversation:\n\n{conversation_text}"
            elif context_type == ContextType.TASK_CONTINUITY:
                user_message = f"Analyze the ongoing task or goal in this conversation:\n\n{conversation_text}"
            elif context_type == ContextType.DOCUMENT_ANALYSIS:
                # For document analysis, we need to get the document content and analyze it
                document_content = self._get_document_content(messages, pdf_data)
                if document_content:
                    if focus_query:
                        user_message = f"Analyze this document content focusing specifically on '{focus_query}' and relate it to the conversation context:\n\nDocument Content:\n{document_content}\n\nConversation Context:\n{conversation_text}"
                    else:
                        user_message = f"Analyze this document content comprehensively in relation to the user's conversation context:\n\nDocument Content:\n{document_content}\n\nConversation Context:\n{conversation_text}"
                else:
                    user_message = f"Analyze the conversation for document-related queries and provide guidance (no document content found):\n\n{conversation_text}"
            elif context_type == ContextType.CREATIVE_DIRECTOR:
                user_message = (
                    f"Analyze this creative project conversation for continuity and guidance:\n\n{conversation_text}"
                )
            else:  # CONVERSATION_SUMMARY
                user_message = f"Provide a concise summary of this conversation:\n\n{conversation_text}"

            if focus_query:
                user_message += f"\n\nFocus particularly on: {focus_query}"

            analysis_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(f"Making context analysis request with model: {model_name} (type: {self.llm_type})")

            response = client.chat.completions.create(
                model=model_name,
                messages=analysis_messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            # Extract key topics from the result
            key_topics = self._extract_key_topics(result, context_type)

            # Try to identify user intent
            user_intent = self._extract_user_intent(result, messages)

            logger.info(f"Successfully generated {context_type} context")

            return ConversationContextResponse(
                analysis=result,
                user_intent=user_intent,
                conversation_type=context_type,
                key_topics=key_topics,
                has_document=pdf_data is not None,
                document_info=pdf_data.get("info", "No additional document information")
                if pdf_data
                else "No additional document information",
            )

        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            return ConversationContextResponse(
                analysis="",
                user_intent=None,
                conversation_type=context_type,
                key_topics=None,
                has_document=False,
                document_info=None,
                success=False,
                error_message=str(e),
            )

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
        content = re.sub(r"<START_CONTEXT>.*?<END_CONTEXT>", "", content, flags=re.DOTALL)
        # Remove thinking tags
        content = strip_think_tags(content)
        # Remove tool call instructions
        content = re.sub(r"<TOOLCALL.*?</TOOLCALL>", "", content, flags=re.DOTALL | re.IGNORECASE)

        return content.strip()

    def _extract_key_topics(self, result: str, context_type: ContextType) -> List[str]:
        """Extract key topics from the analysis result"""
        topics = []

        if context_type == ContextType.RECENT_TOPICS:
            # Try to extract list items or numbered items
            import re

            # Look for bullet points or numbered lists
            list_items = re.findall(r"(?:[-*•]\s*|^\d+\.\s*)(.+)", result, re.MULTILINE)
            topics.extend([item.strip() for item in list_items if item.strip()])

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
            capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", result)
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

    def _get_document_content(self, messages: List[Dict[str, Any]], pdf_data: Dict[str, Any] = None) -> Optional[str]:
        """Extract document content from provided PDF data or messages as fallback"""

        # Use provided PDF data if available
        if pdf_data is not None:
            return self._get_pdf_content_from_pdf_data(pdf_data)

        # Fallback to scanning messages for PDF data
        return self._get_pdf_content_from_messages(messages)

    def _get_pdf_content_from_pdf_data(self, pdf_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract PDF content from explicitly passed PDF data

        Args:
            pdf_data: PDF data dictionary passed from LLM service

        Returns:
            Formatted PDF content string or None if not available
        """
        if pdf_data:
            # Limit to first 5 pages for context analysis
            limited_data = {'pages': pdf_data.get('pages', [])[:5], 'filename': pdf_data.get('filename')}
            # Get text but limit each page to 1000 chars
            pages_text = []
            for page in limited_data['pages']:
                page_text = page.get('text', '')
                if page_text:
                    pages_text.append(f"Page {page.get('page', '?')}: {page_text[:1000]}...")
            return "\n\n".join(pages_text) if pages_text else None
        return None

    def _get_pdf_content_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
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
            limited_data = {'pages': pdf_data.get('pages', [])[:5], 'filename': pdf_data.get('filename')}
            # Get text but limit each page to 1000 chars
            pages_text = []
            for page in limited_data['pages']:
                page_text = page.get('text', '')
                if page_text:
                    pages_text.append(f"Page {page.get('page', '?')}: {page_text[:1000]}...")
            return "\n\n".join(pages_text) if pages_text else None
        return None

    def execute(self, params: Dict[str, Any]) -> ConversationContextResponse:
        """
        Execute the tool with given parameters

        Args:
            params: Dictionary containing the required parameters

        Returns:
            ConversationContextResponse
        """
        return self.run_with_dict(params)

    def run_with_dict(self, params: Dict[str, Any]) -> ConversationContextResponse:
        """Execute context analysis with parameters provided as a dictionary"""

        if "query" not in params:
            raise ValueError("'query' key is required in parameters dictionary")
        if "max_messages" not in params:
            raise ValueError("'max_messages' key is required in parameters dictionary")

        # Validate context type
        try:
            context_enum = ContextType(params["query"].lower())
        except ValueError:
            raise ValueError(f"Invalid query: {params['query']}. Must be one of: {[t.value for t in ContextType]}")

        # Limit messages to the requested count, excluding system messages
        filtered_messages = []
        for msg in reversed(params["messages"]):
            if msg.get("role") != "system":
                filtered_messages.append(msg)
                if len(filtered_messages) >= params["max_messages"]:
                    break

        # Reverse back to chronological order
        filtered_messages.reverse()

        logger.debug(f"Context analysis: {params['query']}, {len(filtered_messages)} messages")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self._analyze_conversation_context(
            context_enum, filtered_messages, config, params.get("focus_query"), params.get("pdf_data")
        )


# Create a global instance and helper functions
conversation_context_tool = ConversationContextTool()


def get_conversation_context_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for conversation context"""
    return conversation_context_tool.to_openai_format()


def execute_conversation_context_with_dict(params: Dict[str, Any],) -> ConversationContextResponse:
    """Execute conversation context analysis with parameters as dictionary"""
    return conversation_context_tool.run_with_dict(params)
