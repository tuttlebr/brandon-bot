import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from pydantic import Field
from services.llm_client_service import llm_client_service
from services.pdf_analysis_service import PDFAnalysisService
from tools.base import BaseTool, BaseToolResponse
from utils.config import config as app_config

# Configure logger
logger = logging.getLogger(__name__)

# Supported languages for translation
SUPPORTED_LANGUAGES = [
    "English",
    "German",
    "French",
    "Italian",
    "Portuguese",
    "Hindi",
    "Spanish",
    "Thai",
]


class AssistantTaskType(str, Enum):
    """Enumeration of assistant task types"""

    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    PROOFREAD = "proofread"
    REWRITE = "rewrite"
    CRITIC = "critic"
    TRANSLATE = "translate"


class AssistantResponse(BaseToolResponse):
    """Response from the assistant tool"""

    original_text: str = Field(description="The original input text")
    task_type: AssistantTaskType = Field(description="The type of task performed")
    result: str = Field(description="The processed result")
    improvements: Optional[List[str]] = Field(None, description="List of improvements made (for proofreading)")
    summary_length: Optional[int] = Field(None, description="Length of summary in words (for summarizing)")
    source_language: Optional[str] = Field(None, description="Source language for translation")
    target_language: Optional[str] = Field(None, description="Target language for translation")
    processing_notes: Optional[str] = Field(None, description="Additional notes about the processing")
    direct_response: bool = Field(
        default=True, description="Flag indicating this response should be returned directly to user",
    )


class AssistantTool(BaseTool):
    """Tool for text processing tasks including summarizing, proofreading, rewriting, and translation"""

    def __init__(self):
        super().__init__()
        self.name = "text_assistant"
        self.description = "Performs advanced text processing tasks including document analysis, summarization, proofreading, rewriting, critiquing,  writing, and translation."
        # Use intelligent model for high-quality text processing
        self.llm_type = "intelligent"
        # Initialize PDF analysis service for intelligent document processing
        self.pdf_analysis_service = None

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
                        "task_type": {
                            "type": "string",
                            "enum": ["analyze", "summarize", "proofread", "rewrite", "critic", "translate",],
                            "description": "Performs advanced text processing tasks including document analysis, summarization, proofreading, rewriting, critiquing, writing, and translation.",
                        },
                        "text": {
                            "type": "string",
                            "description": "The text content to be processed. Use 'the PDF' or 'the document' when referring to uploaded PDF content, or provide specific text to process.",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "REQUIRED when analyzing PDF content: The specific question or task about the document (e.g., 'Who are the authors?', 'What is the main topic?', 'Summarize the key findings'). Optional for other text processing tasks.",
                        },
                        "source_language": {
                            "type": "string",
                            "enum": SUPPORTED_LANGUAGES,
                            "description": "The source language of the text to be translated. Optional - if not provided, the system will auto-detect from the supported languages.",
                        },
                        "target_language": {
                            "type": "string",
                            "enum": SUPPORTED_LANGUAGES,
                            "description": "The target language to translate the text into. Required for translation tasks.",
                        },
                    },
                    "required": ["task_type", "text"],
                },
            },
        }

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for BaseTool interface"""
        return self.to_openai_format()

    def _get_system_prompt(self, task_type: AssistantTaskType, instructions: Optional[str] = None) -> str:
        """
        Get the appropriate system prompt for the task type

        Args:
            task_type: The type of assistant task
            instructions: Optional user instructions

        Returns:
            Formatted system prompt
        """
        base_prompts = {
            AssistantTaskType.ANALYZE: """detailed thinking on
            You are analyzing a document to provide comprehensive insights and answer specific questions.

When analyzing documents, thoroughly understand the content, context, and purpose. Extract key insights, identify main themes and arguments, and provide accurate answers supported by evidence from the text. Be specific and cite relevant sections when answering questions. Consider the document's audience and broader implications in your analysis.""",
            AssistantTaskType.SUMMARIZE: """detailed thinking on
            You are creating a concise summary that captures the essential information.

Focus on extracting the most important points that deliver the core insights. Maintain the logical flow and causal relationships from the original. Eliminate redundancy while preserving all critical information. Tailor the summary density to the intended audience, avoiding unnecessary jargon when appropriate.""",
            AssistantTaskType.PROOFREAD: """detailed thinking on
            You are proofreading text to identify and correct errors.

Check for grammar, punctuation, and spelling mistakes. Improve awkward phrasing and ensure consistency in style. Mark significant changes with [**] brackets and provide brief explanations for important corrections. Focus on clarity and readability while maintaining the author's voice.""",
            AssistantTaskType.REWRITE: """detailed thinking on
            You are rewriting text to improve its effectiveness and impact.

Enhance clarity, flow, and engagement while preserving the core message. Adjust tone, formality, and style as needed for the target audience. Use stronger verbs and more precise language. Reorganize content for better logical flow when beneficial. For code, provide the rewritten version in appropriate code blocks.""",
            AssistantTaskType.CRITIC: """detailed thinking on
            You are providing constructive critique and actionable feedback.

Evaluate the text's strengths and weaknesses objectively. Identify areas for improvement in structure, argumentation, and execution. Consider the intended audience and purpose. Provide specific, actionable suggestions for enhancement. Be honest but constructive in your assessment.""",
            AssistantTaskType.TRANSLATE: """detailed thinking on
            You are translating text between English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

Provide accurate translation that preserves meaning and cultural context. Maintain appropriate tone and formality level. Handle idioms and cultural references appropriately. Ensure natural flow in the target language. If source language is not specified, auto-detect from supported languages.""",
        }

        prompt = base_prompts.get(task_type, base_prompts[AssistantTaskType.ANALYZE])

        if instructions:
            prompt += f"\n\nAdditional instructions: {instructions}"

        return prompt

    def _process_text(
        self,
        task_type: AssistantTaskType,
        text: str,
        config: ChatConfig,
        instructions: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> AssistantResponse:
        """
        Process text using the appropriate LLM task

        Args:
            task_type: The type of processing task
            text: The text to process
            config: ChatConfig instance with LLM configuration
            instructions: Optional user instructions
            source_language: Source language for translation (optional)
            target_language: Target language for translation (required for translation)

        Returns:
            AssistantResponse with the processed result
        """
        logger.info(f"Processing text with task type: {task_type}")

        try:
            # Check if this is an ANALYZE task with PDF content and specific instructions
            if (
                task_type == AssistantTaskType.ANALYZE
                and instructions
                and self._is_pdf_content(text)
                and len(text) > 5000
            ):  # Only use intelligent analysis for substantial content

                logger.info("Using intelligent PDF analysis for large document with specific query")
                return self._process_pdf_analysis(text, instructions, config)

            # Regular processing for all other cases
            # Get the appropriate client based on this tool's LLM type
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)
            system_prompt = self._get_system_prompt(task_type, instructions)

            # Prepare the user message based on task type
            if task_type == AssistantTaskType.ANALYZE:
                if instructions:
                    user_message = f"Please analyze the following document and answer this question: {instructions}\n\nDocument:\n{text}"
                else:
                    user_message = f"Please analyze the following document and provide key insights, main themes, and be ready to answer questions about its content:\n\n{text}"
            elif task_type == AssistantTaskType.SUMMARIZE:
                user_message = f"Please summarize the following text:\n\n{text}"
            elif task_type == AssistantTaskType.PROOFREAD:
                user_message = (
                    f"Please proofread the following text and provide corrections with explanations:\n\n{text}"
                )
            elif task_type == AssistantTaskType.REWRITE:
                user_message = f"Please rewrite and improve the following text:\n\n{text}"
            elif task_type == AssistantTaskType.CRITIC:
                user_message = f"Please critique the following text and provide feedback on the text:\n\n{text}"
            elif task_type == AssistantTaskType.WRITER:
                user_message = f"Please write a story based on the following prompt:\n\n{text}"
            elif task_type == AssistantTaskType.TRANSLATE:
                if not target_language:
                    raise ValueError("target_language is required for translation tasks")

                if source_language:
                    user_message = (
                        f"Please translate the following text from {source_language} to {target_language}:\n\n{text}"
                    )
                else:
                    user_message = f"Please translate the following text to {target_language} (auto-detect source language):\n\n{text}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(f"Making LLM request with model: {model_name} (type: {self.llm_type})")

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            # Process the response based on task type
            improvements = None
            summary_length = None
            processing_notes = None
            response_source_language = None
            response_target_language = None

            if task_type == AssistantTaskType.ANALYZE:
                processing_notes = f"Document analysis completed for {len(text.split())} words of content"
                if instructions:
                    processing_notes += (
                        f" with specific query: {instructions[:100]}{'...' if len(instructions) > 100 else ''}"
                    )
            elif task_type == AssistantTaskType.SUMMARIZE:
                summary_length = len(result.split())
                processing_notes = f"Original text: {len(text.split())} words, Summary: {summary_length} words"
            elif task_type == AssistantTaskType.PROOFREAD:
                # Try to extract improvements from the response
                if "improvements" in result.lower() or "corrections" in result.lower():
                    improvements = ["See detailed feedback in the result"]
                processing_notes = "Proofreading completed with suggestions for improvement"
            elif task_type == AssistantTaskType.REWRITE:
                processing_notes = "Text has been rewritten for improved clarity and flow"
            elif task_type == AssistantTaskType.CRITIC:
                processing_notes = "Text has been critiqued and feedback provided"
            elif task_type == AssistantTaskType.WRITER:
                processing_notes = "Text has been written"
            elif task_type == AssistantTaskType.TRANSLATE:
                response_source_language = source_language
                response_target_language = target_language
                if source_language:
                    processing_notes = f"Translation completed from {source_language} to {target_language}"
                else:
                    processing_notes = f"Translation completed (auto-detected source) to {target_language}"

            logger.info(f"Successfully processed text with {task_type}")

            return AssistantResponse(
                original_text=text,  # Truncate for storage
                task_type=task_type,
                result=result,
                improvements=improvements,
                summary_length=summary_length,
                source_language=response_source_language,
                target_language=response_target_language,
                processing_notes=processing_notes,
            )

        except Exception as e:
            logger.error(f"Error processing text with {task_type}: {e}")
            raise

    def _run(
        self,
        task_type: str = None,
        text: str = None,
        instructions: str = None,
        source_language: str = None,
        target_language: str = None,
        **kwargs,
    ) -> AssistantResponse:
        """
        Execute an assistant task with the given parameters.

        Args:
            task_type: The type of task to perform
            text: The text to process
            instructions: Optional instructions
            source_language: Source language for translation (optional)
            target_language: Target language for translation (required for translation)
            **kwargs: Can accept a dictionary with parameters

        Returns:
            AssistantResponse: The processed result
        """
        # Support both direct parameters and dictionary input
        if task_type is None and "task_type" in kwargs:
            task_type = kwargs["task_type"]
        if text is None and "text" in kwargs:
            text = kwargs["text"]
        if instructions is None and "instructions" in kwargs:
            instructions = kwargs["instructions"]
        if source_language is None and "source_language" in kwargs:
            source_language = kwargs["source_language"]
        if target_language is None and "target_language" in kwargs:
            target_language = kwargs["target_language"]

        if task_type is None:
            raise ValueError("task_type parameter is required")
        if text is None:
            raise ValueError("text parameter is required")

        # Extract messages from kwargs if available
        messages = kwargs.get("messages", [])

        # Check if we need to extract PDF content
        # This happens when:
        # 1. Text is empty or very short
        # 2. Text mentions "pdf" or "document"
        # 3. Messages contain PDF data OR PDF exists in session state
        text_lower = text.lower() if text else ""
        logger.debug(
            f"PDF extraction check - text: '{text}', text_length: {len(text) if text else 0}, messages_count: {len(messages) if messages else 0}"
        )

        if not text or len(text) < 50 or "pdf" in text_lower or "document" in text_lower:
            logger.info(f"Attempting to extract PDF content - first trying messages, then session state")
            pdf_content = None

            # First try to get PDF content from messages (injected context)
            if messages:
                pdf_content = self._get_pdf_content_from_messages(messages)
                if pdf_content:
                    logger.info("Found PDF content in messages")

            # If not found in messages, try to get directly from session state
            if not pdf_content:
                logger.info("No PDF content in messages, trying session state")
                pdf_content = self._get_pdf_content_from_session()
                if pdf_content:
                    logger.info("Found PDF content in session state")

            if pdf_content:
                # If text was asking about the PDF, replace it with the PDF content
                # Otherwise, append the PDF content to the existing text
                if not text or "pdf" in text_lower or "document" in text_lower:
                    logger.info(
                        f"Using PDF content as text for {task_type} task (PDF content length: {len(pdf_content)} chars)"
                    )
                    text = pdf_content
                else:
                    # Append PDF content to existing text
                    logger.info(
                        f"Appending PDF content to existing text (PDF content length: {len(pdf_content)} chars)"
                    )
                    text = f"{text}\n\n{pdf_content}"
            else:
                logger.warning("No PDF content found in messages or session state despite conditions being met")

        # Validate task type
        try:
            task_enum = AssistantTaskType(task_type.lower())
        except ValueError:
            raise ValueError(f"Invalid task_type: {task_type}. Must be one of: {[t.value for t in AssistantTaskType]}")

        # Validate translation parameters
        if task_enum == AssistantTaskType.TRANSLATE:
            if not target_language:
                raise ValueError("target_language parameter is required for translation tasks")
            if target_language not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"target_language '{target_language}' is not supported. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
                )
            if source_language and source_language not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"source_language '{source_language}' is not supported. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
                )

        logger.debug(f"_run method called with task_type: '{task_type}', text length: {len(text)}")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self._process_text(task_enum, text, config, instructions, source_language, target_language)

    def _get_pdf_content_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract PDF content from messages when available

        Args:
            messages: List of conversation messages

        Returns:
            Extracted PDF text content or None
        """
        if not messages:
            logger.debug("No messages provided for PDF extraction")
            return None

        logger.debug(f"Searching for PDF content in {len(messages)} messages")

        # Look for injected PDF data in system messages
        for i, message in enumerate(messages):
            logger.debug(f"Message {i}: role={message.get('role')}, content_type={type(message.get('content'))}")

            if message.get("role") == "system":
                try:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        logger.debug(f"System message content length: {len(content)}")
                        # Check if this looks like JSON
                        if content.strip().startswith("{") and content.strip().endswith("}"):
                            logger.debug("Found JSON-like system message, attempting to parse")
                            data = json.loads(content)
                            logger.debug(
                                f"Parsed JSON data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}"
                            )

                            if (
                                isinstance(data, dict)
                                and data.get("type") == "pdf_data"
                                and data.get("tool_name") == "process_pdf_document"
                            ):
                                logger.info("Found matching PDF data in system message")
                                # Found injected PDF data
                                pages = data.get("pages", [])
                                filename = data.get("filename", "Unknown")

                                if pages:
                                    logger.info(f"Extracting text from {len(pages)} PDF pages")
                                    # Extract all text from PDF pages
                                    document_text = []
                                    for page in pages:
                                        page_text = page.get("text", "")
                                        if page_text:
                                            document_text.append(f"[Page {page.get('page', '?')}]\n{page_text}")

                                    if document_text:
                                        full_text = "\n\n".join(document_text)
                                        logger.info(
                                            f"Extracted PDF content from '{filename}' with {len(pages)} pages, total text length: {len(full_text)}"
                                        )
                                        return full_text
                                else:
                                    logger.warning("PDF data found but no pages available")
                            else:
                                logger.debug("System message JSON doesn't match PDF data criteria")
                        else:
                            logger.debug("System message content doesn't look like JSON")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Failed to parse system message as JSON: {e}")
                    continue

        logger.debug("No PDF content found in any messages")
        return None

    def _is_pdf_content(self, text: str) -> bool:
        """
        Check if the text contains PDF page markers indicating it's from a PDF document

        Args:
            text: Text to check

        Returns:
            True if text appears to be from a PDF document
        """
        if not text:
            return False

        # Look for page markers that indicate PDF content
        pdf_indicators = [
            "[Page ",
            "Page 1:",
            "Page 2:",
            "\n\nPage ",
        ]

        for indicator in pdf_indicators:
            if indicator in text:
                return True

        return False

    def _process_pdf_analysis(self, text: str, instructions: str, config: ChatConfig) -> AssistantResponse:
        """
        Process PDF content using intelligent analysis service

        Args:
            text: PDF content with page markers
            instructions: User's specific question
            config: ChatConfig instance

        Returns:
            AssistantResponse with intelligent analysis result
        """
        try:
            # Import here to avoid circular imports
            import streamlit as st

            # Initialize progress tracking
            st.session_state.pdf_analysis_progress = {
                "status": "starting",
                "message": "Initializing intelligent PDF analysis...",
                "progress": 0,
            }

            # Initialize PDF analysis service if not already done
            if self.pdf_analysis_service is None:
                self.pdf_analysis_service = PDFAnalysisService(config)

            # Parse PDF content back into page structure
            pdf_data = self._parse_pdf_content(text)

            # Create a simplified analysis to avoid recursion
            # Use direct analysis instead of going through the full assistant tool
            result = self._analyze_pdf_directly(pdf_data, instructions)

            processing_notes = f"Intelligent PDF analysis completed for query: {instructions[:100]}{'...' if len(instructions) > 100 else ''}"

            return AssistantResponse(
                original_text=(text[:500] + "..." if len(text) > 500 else text),
                task_type=AssistantTaskType.ANALYZE,
                result=result,
                processing_notes=processing_notes,
            )

        except Exception as e:
            logger.error(f"Error in PDF analysis: {e}")
            # Fallback to regular processing
            logger.info("Falling back to regular text processing due to PDF analysis error")
            return self._fallback_regular_processing(text, instructions, config)

    def _analyze_pdf_directly(self, pdf_data: Dict[str, Any], instructions: str) -> str:
        """
        Analyze PDF directly without recursion through the assistant tool

        Args:
            pdf_data: PDF data dictionary
            instructions: User's question

        Returns:
            Analysis result
        """
        try:
            import streamlit as st

            pages = pdf_data.get("pages", [])
            total_pages = len(pages)
            filename = pdf_data.get("filename", "Document")

            logger.info(f"Starting direct PDF analysis for query '{instructions}' on {filename} ({total_pages} pages)")

            if total_pages == 0:
                return "The document appears to be empty or contains no extractable text."

            # Update progress
            st.session_state.pdf_analysis_progress = {
                "status": "analyzing",
                "message": f"Analyzing {total_pages} pages...",
                "progress": 10,
            }
            time.sleep(0.1)  # Brief pause for UI update

            # For large documents, use intelligent search approach
            if total_pages > 15:
                return self._analyze_large_document_direct(pages, instructions, filename)
            elif total_pages > 5:
                return self._analyze_medium_document_direct(pages, instructions, filename)
            else:
                return self._analyze_small_document_direct(pages, instructions, filename)

        except Exception as e:
            logger.error(f"Error in direct PDF analysis: {e}")
            return f"I encountered an error while analyzing the document: {str(e)}"

    def _analyze_small_document_direct(self, pages: List[Dict], instructions: str, filename: str) -> str:
        """Directly analyze small documents (≤5 pages) using regular LLM processing"""
        try:
            import streamlit as st

            # Update progress
            st.session_state.pdf_analysis_progress = {
                "status": "analyzing",
                "message": "Processing all pages together...",
                "progress": 30,
            }
            time.sleep(0.1)  # Brief pause for UI update

            # Combine all pages
            full_text = "\n\n".join(
                [f"Page {page.get('page', i+1)}:\n{page.get('text', '')}" for i, page in enumerate(pages)]
            )

            # Use regular LLM processing
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)
            system_prompt = self._get_system_prompt(AssistantTaskType.ANALYZE, instructions)

            user_message = f"Based on the document '{filename}', please answer this question: {instructions}\n\nDocument:\n{full_text}"

            # Update progress
            st.session_state.pdf_analysis_progress = {
                "status": "analyzing",
                "message": "Generating analysis...",
                "progress": 70,
            }
            time.sleep(0.1)  # Brief pause for UI update

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            # Update progress - completed
            st.session_state.pdf_analysis_progress = {
                "status": "completed",
                "message": "Analysis completed!",
                "progress": 100,
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing small document: {e}")
            return f"Error analyzing document: {str(e)}"

    def _analyze_medium_document_direct(self, pages: List[Dict], instructions: str, filename: str) -> str:
        """Directly analyze medium documents (6-15 pages) in batches"""
        try:
            import streamlit as st

            batch_size = max(3, len(pages) // 3)  # 3-5 pages per batch
            batch_results = []

            # Update progress
            st.session_state.pdf_analysis_progress = {
                "status": "analyzing",
                "message": f"Processing {len(pages)} pages in batches...",
                "progress": 20,
            }
            time.sleep(0.1)  # Brief pause for UI update

            # Process in batches
            for i in range(0, len(pages), batch_size):
                batch_end = min(i + batch_size, len(pages))
                batch_pages = pages[i:batch_end]

                # Update progress
                progress = 20 + (i / len(pages)) * 60
                st.session_state.pdf_analysis_progress = {
                    "status": "analyzing",
                    "message": f"Analyzing pages {i+1}-{batch_end}...",
                    "progress": int(progress),
                }
                time.sleep(0.1)  # Brief pause for UI update

                batch_text = "\n\n".join(
                    [f"Page {page.get('page', i+j+1)}:\n{page.get('text', '')}" for j, page in enumerate(batch_pages)]
                )

                # Use regular LLM processing
                client = llm_client_service.get_client(self.llm_type)
                model_name = llm_client_service.get_model_name(self.llm_type)
                system_prompt = self._get_system_prompt(AssistantTaskType.ANALYZE, instructions)

                user_message = f"Analyze pages {i+1}-{batch_end} of '{filename}' for this question: {instructions}. If relevant information is found, provide it with page numbers. If not relevant, say 'No relevant information found in these pages.'\n\nDocument:\n{batch_text}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    top_p=app_config.llm.DEFAULT_TOP_P,
                    frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                    presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                )

                batch_results.append(
                    {"page_range": f"{i+1}-{batch_end}", "analysis": response.choices[0].message.content.strip(),}
                )

            # Update progress
            st.session_state.pdf_analysis_progress = {
                "status": "analyzing",
                "message": "Combining results...",
                "progress": 85,
            }
            time.sleep(0.1)  # Brief pause for UI update

            # Combine batch results
            combined_findings = "\n\n".join(
                [f"Analysis of pages {result['page_range']}:\n{result['analysis']}" for result in batch_results]
            )

            # Final synthesis
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)
            system_prompt = self._get_system_prompt(AssistantTaskType.ANALYZE, instructions)

            user_message = f"Based on these analyses of different sections of '{filename}', provide a comprehensive answer to: {instructions}. Combine relevant information and provide a cohesive response.\n\nAnalyses:\n{combined_findings}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            # Update progress - completed
            st.session_state.pdf_analysis_progress = {
                "status": "completed",
                "message": "Analysis completed!",
                "progress": 100,
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing medium document: {e}")
            return f"Error analyzing document: {str(e)}"

    def _analyze_large_document_direct(self, pages: List[Dict], instructions: str, filename: str) -> str:
        """Directly analyze large documents (>15 pages) using intelligent search"""
        try:
            import streamlit as st

            # Update progress
            st.session_state.pdf_analysis_progress = {
                "status": "analyzing",
                "message": "Scanning pages for relevant content...",
                "progress": 20,
            }
            time.sleep(0.1)  # Brief pause for UI update

            # Step 1: Find relevant pages using quick scans
            relevant_pages = []
            batch_size = 5  # Scan 5 pages at a time

            for i in range(0, len(pages), batch_size):
                batch_end = min(i + batch_size, len(pages))
                batch_pages = pages[i:batch_end]

                # Update progress
                progress = 20 + (i / len(pages)) * 40
                st.session_state.pdf_analysis_progress = {
                    "status": "analyzing",
                    "message": f"Scanning pages {i+1}-{batch_end} for relevance...",
                    "progress": int(progress),
                }
                time.sleep(0.1)  # Brief pause for UI update

                # Create summary of each page for relevance checking
                page_summaries = []
                for j, page in enumerate(batch_pages):
                    page_text = page.get("text", "")[:1000]  # First 1000 chars
                    page_summaries.append(f"Page {page.get('page', i+j+1)}: {page_text}...")

                batch_text = "\n\n".join(page_summaries)

                # Use regular LLM to check relevance
                client = llm_client_service.get_client(self.llm_type)
                model_name = llm_client_service.get_model_name(self.llm_type)

                user_message = f"Given this query: '{instructions}', which of these pages (if any) contain relevant information? List ONLY the page numbers that are relevant, or say 'None' if no pages are relevant. Be specific about page numbers.\n\n{batch_text}"

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert at identifying relevant content. Only list page numbers that directly relate to the query.",
                    },
                    {"role": "user", "content": user_message},
                ]

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,  # Lower temperature for accuracy
                    top_p=app_config.llm.DEFAULT_TOP_P,
                    frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                    presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                )

                # Parse relevant page numbers
                relevant_in_batch = self._extract_page_numbers_simple(
                    response.choices[0].message.content, i + 1, batch_end
                )

                # Add full page data for relevant pages
                for page_num in relevant_in_batch:
                    for page in batch_pages:
                        if page.get("page") == page_num:
                            relevant_pages.append(page)
                            break

            logger.info(f"Found {len(relevant_pages)} relevant pages out of {len(pages)} total pages")

            if not relevant_pages:
                return f"I searched through all {len(pages)} pages of '{filename}' but couldn't find information directly related to your question: '{instructions}'. The document may not contain relevant information, or the question might need to be rephrased."

            # Update progress
            st.session_state.pdf_analysis_progress = {
                "status": "analyzing",
                "message": f"Analyzing {len(relevant_pages)} relevant pages...",
                "progress": 70,
            }
            time.sleep(0.1)  # Brief pause for UI update

            # Step 2: Deep analysis of relevant pages
            if len(relevant_pages) <= 5:
                # Analyze all relevant pages together
                full_text = "\n\n".join(
                    [f"Page {page.get('page')}:\n{page.get('text', '')}" for page in relevant_pages]
                )

                client = llm_client_service.get_client(self.llm_type)
                model_name = llm_client_service.get_model_name(self.llm_type)
                system_prompt = self._get_system_prompt(AssistantTaskType.ANALYZE, instructions)

                user_message = f"Based on these relevant pages from '{filename}', provide a comprehensive answer to: {instructions}. Include specific details and page references.\n\nDocument:\n{full_text}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    top_p=app_config.llm.DEFAULT_TOP_P,
                    frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                    presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                )

                result = response.choices[0].message.content.strip()
            else:
                # Process relevant pages in smaller batches
                batch_results = []
                batch_size = 3

                for i in range(0, len(relevant_pages), batch_size):
                    batch = relevant_pages[i : i + batch_size]
                    batch_text = "\n\n".join([f"Page {page.get('page')}:\n{page.get('text', '')}" for page in batch])

                    client = llm_client_service.get_client(self.llm_type)
                    model_name = llm_client_service.get_model_name(self.llm_type)
                    system_prompt = self._get_system_prompt(AssistantTaskType.ANALYZE, instructions)

                    user_message = f"Analyze these pages from '{filename}' for: {instructions}. Extract any relevant information with page numbers.\n\nDocument:\n{batch_text}"

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.3,
                        top_p=app_config.llm.DEFAULT_TOP_P,
                        frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                        presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                    )

                    batch_results.append(
                        {
                            "pages": [p.get("page") for p in batch],
                            "analysis": response.choices[0].message.content.strip(),
                        }
                    )

                # Synthesize final answer
                combined_findings = "\n\n".join(
                    [
                        f"Pages {', '.join(map(str, result['pages']))}:\n{result['analysis']}"
                        for result in batch_results
                    ]
                )

                client = llm_client_service.get_client(self.llm_type)
                model_name = llm_client_service.get_model_name(self.llm_type)
                system_prompt = self._get_system_prompt(AssistantTaskType.ANALYZE, instructions)

                user_message = f"Based on these detailed analyses from '{filename}', provide a final comprehensive answer to: {instructions}. Synthesize all relevant information into a cohesive response.\n\nAnalyses:\n{combined_findings}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    top_p=app_config.llm.DEFAULT_TOP_P,
                    frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                    presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                )

                result = response.choices[0].message.content.strip()

            # Update progress - completed
            st.session_state.pdf_analysis_progress = {
                "status": "completed",
                "message": "Analysis completed!",
                "progress": 100,
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing large document: {e}")
            return f"Error analyzing document: {str(e)}"

    def _extract_page_numbers_simple(self, text: str, start_page: int, end_page: int) -> List[int]:
        """Extract page numbers from LLM response (simple version)"""
        import re

        if "none" in text.lower() or "no pages" in text.lower():
            return []

        # Find all numbers that could be page numbers
        numbers = re.findall(r"\b(\d+)\b", text)
        page_numbers = []

        for num_str in numbers:
            try:
                num = int(num_str)
                if start_page <= num <= end_page:
                    page_numbers.append(num)
            except ValueError:
                continue

        return list(set(page_numbers))  # Remove duplicates

    def _parse_pdf_content(self, text: str) -> Dict[str, Any]:
        """
        Parse PDF content back into page structure for analysis

        Args:
            text: PDF content with page markers

        Returns:
            PDF data dictionary with pages
        """
        try:
            pages = []
            current_page = None
            current_text = []

            lines = text.split("\n")

            for line in lines:
                # Check for page marker
                if line.startswith("[Page ") and line.endswith("]"):
                    # Save previous page if exists
                    if current_page is not None and current_text:
                        pages.append(
                            {"page": current_page, "text": "\n".join(current_text).strip(),}
                        )

                    # Start new page
                    try:
                        page_num = int(line.replace("[Page ", "").replace("]", ""))
                        current_page = page_num
                        current_text = []
                    except ValueError:
                        # If can't parse page number, continue with current page
                        if current_text or not line.strip():
                            current_text.append(line)
                else:
                    # Regular content line
                    if current_text or line.strip():  # Don't start with empty lines
                        current_text.append(line)

            # Save final page
            if current_page is not None and current_text:
                pages.append({"page": current_page, "text": "\n".join(current_text).strip()})

            # If no pages were parsed (content doesn't have page markers), treat as single page
            if not pages:
                pages = [{"page": 1, "text": text}]

            return {"pages": pages, "filename": "Document"}

        except Exception as e:
            logger.error(f"Error parsing PDF content: {e}")
            # Fallback: treat entire text as single page
            return {"pages": [{"page": 1, "text": text}], "filename": "Document"}

    def _fallback_regular_processing(self, text: str, instructions: str, config: ChatConfig) -> AssistantResponse:
        """
        Fallback to regular text processing when PDF analysis fails

        Args:
            text: Text to process
            instructions: User instructions
            config: ChatConfig instance

        Returns:
            AssistantResponse from regular processing
        """
        try:
            # Use regular LLM processing as fallback
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)
            system_prompt = self._get_system_prompt(AssistantTaskType.ANALYZE, instructions)

            user_message = (
                f"Please analyze the following document and answer this question: {instructions}\n\nDocument:\n{text}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()
            processing_notes = f"Fallback analysis completed for {len(text.split())} words of content"

            return AssistantResponse(
                original_text=(text[:500] + "..." if len(text) > 500 else text),
                task_type=AssistantTaskType.ANALYZE,
                result=result,
                processing_notes=processing_notes,
            )

        except Exception as e:
            logger.error(f"Error in fallback processing: {e}")
            raise

    def _get_pdf_content_from_session(self) -> Optional[str]:
        """
        Extract PDF content directly from session state

        Returns:
            Extracted PDF text content or None
        """
        try:
            # Import here to avoid circular imports
            import streamlit as st
            from services.file_storage_service import FileStorageService

            logger.debug("Attempting to get PDF content from session state")

            # Check if there are stored PDFs in session
            if not hasattr(st.session_state, "stored_pdfs") or not st.session_state.stored_pdfs:
                logger.debug("No stored PDFs in session state")
                return None

            # Get the latest PDF ID
            latest_pdf_id = st.session_state.stored_pdfs[-1]
            logger.debug(f"Latest PDF ID in session: {latest_pdf_id}")

            # Get PDF data from file storage
            file_storage = FileStorageService()
            pdf_data = file_storage.get_pdf(latest_pdf_id)

            if not pdf_data:
                logger.warning(f"Failed to retrieve PDF data for ID: {latest_pdf_id}")
                return None

            pages = pdf_data.get("pages", [])
            filename = pdf_data.get("filename", "Unknown")

            if not pages:
                logger.warning("PDF data found but no pages available")
                return None

            logger.info(f"Extracting text from {len(pages)} PDF pages from session state")

            # Extract all text from PDF pages
            document_text = []
            for page in pages:
                page_text = page.get("text", "")
                if page_text:
                    document_text.append(f"[Page {page.get('page', '?')}]\n{page_text}")

            if document_text:
                full_text = "\n\n".join(document_text)
                logger.info(
                    f"Extracted PDF content from session state for '{filename}' with {len(pages)} pages, total text length: {len(full_text)}"
                )
                return full_text
            else:
                logger.warning("PDF pages found but no text content extracted")
                return None

        except ImportError as e:
            logger.debug(f"Cannot access session state (expected in some contexts): {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting PDF content from session state: {e}")
            return None

    def run_with_dict(self, params: Dict[str, Any]) -> AssistantResponse:
        """
        Execute an assistant task with parameters provided as a dictionary.

        Args:
                    params: Dictionary containing the required parameters
               Expected keys: 'task_type', 'text', and optionally 'instructions', 'source_language', 'target_language'
               For translation: source_language and target_language must be one of: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai

        Returns:
            AssistantResponse: The processed result
        """
        if "task_type" not in params:
            raise ValueError("'task_type' key is required in parameters dictionary")
        if "text" not in params:
            raise ValueError("'text' key is required in parameters dictionary")

        task_type = params["task_type"]
        text = params["text"]
        instructions = params.get("instructions")
        source_language = params.get("source_language")
        target_language = params.get("target_language")
        messages = params.get("messages", [])

        # Check if we need to extract PDF content
        # This happens when:
        # 1. Text is empty or very short
        # 2. Text mentions "pdf" or "document"
        # 3. Messages contain PDF data OR PDF exists in session state
        text_lower = text.lower() if text else ""
        if not text or len(text) < 50 or "pdf" in text_lower or "document" in text_lower:
            logger.info(f"Attempting to extract PDF content - first trying messages, then session state")
            pdf_content = None

            # First try to get PDF content from messages (injected context)
            if messages:
                pdf_content = self._get_pdf_content_from_messages(messages)
                if pdf_content:
                    logger.info("Found PDF content in messages")

            # If not found in messages, try to get directly from session state
            if not pdf_content:
                logger.info("No PDF content in messages, trying session state")
                pdf_content = self._get_pdf_content_from_session()
                if pdf_content:
                    logger.info("Found PDF content in session state")

            if pdf_content:
                # If text was asking about the PDF, replace it with the PDF content
                # Otherwise, append the PDF content to the existing text
                if not text or "pdf" in text_lower or "document" in text_lower:
                    logger.info(f"Using PDF content as text for {task_type} task")
                    text = pdf_content
                else:
                    # Append PDF content to existing text
                    text = f"{text}\n\n{pdf_content}"

        # Validate translation parameters
        if task_type.lower() == "translate":
            if not target_language:
                raise ValueError("'target_language' key is required for translation tasks")
            if target_language not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"target_language '{target_language}' is not supported. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
                )
            if source_language and source_language not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"source_language '{source_language}' is not supported. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
                )

        logger.debug(f"run_with_dict method called with task_type: '{task_type}', text length: {len(text)}")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self._process_text(
            AssistantTaskType(task_type.lower()), text, config, instructions, source_language, target_language,
        )

    def execute(self, params: Dict[str, Any]) -> AssistantResponse:
        """
        Execute the assistant tool with given parameters

        Args:
            params: Dictionary containing the required parameters

        Returns:
            AssistantResponse with the processed result
        """
        return self.run_with_dict(params)


# Create a global instance and helper functions for easy access
assistant_tool = AssistantTool()


def get_assistant_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for text assistant

    Returns:
        Dict containing the OpenAI tool definition
    """
    return assistant_tool.to_openai_format()


def execute_assistant_task(
    task_type: str,
    text: str,
    instructions: Optional[str] = None,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
) -> AssistantResponse:
    """
    Execute an assistant task with the given parameters

    Args:
        task_type: The type of task to perform ('summarize', 'proofread', 'rewrite', 'critic', 'writer', 'translate')
        text: The text to process
        instructions: Optional specific instructions
        source_language: Source language for translation (optional). Must be one of: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai
        target_language: Target language for translation (required for translation). Must be one of: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai

    Returns:
        AssistantResponse: The processed result
    """
    config = ChatConfig.from_environment()
    return assistant_tool._process_text(
        AssistantTaskType(task_type.lower()), text, config, instructions, source_language, target_language,
    )


def execute_assistant_with_dict(params: Dict[str, Any]) -> AssistantResponse:
    """
    Execute an assistant task with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters
               Expected keys: 'task_type', 'text', and optionally 'instructions', 'source_language', 'target_language'
               For translation: source_language and target_language must be one of: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai

    Returns:
        AssistantResponse: The processed result
    """
    return assistant_tool.run_with_dict(params)
