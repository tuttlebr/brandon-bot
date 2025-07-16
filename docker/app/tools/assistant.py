"""
Assistant Tool - Simplified Implementation

This tool delegates to specialized services for text processing tasks
instead of implementing everything in one monolithic class.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from pydantic import Field
from services.document_analyzer_service import DocumentAnalyzerService
from services.text_processor_service import TextProcessorService, TextTaskType
from services.translation_service import TranslationService
from tools.base import BaseTool, BaseToolResponse
from utils.pdf_extractor import PDFDataExtractor
from utils.text_processing import strip_think_tags

logger = logging.getLogger(__name__)


class AssistantTaskType(str, Enum):
    """Enumeration of assistant task types"""

    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    PROOFREAD = "proofread"
    REWRITE = "rewrite"
    CRITIC = "critic"
    TRANSLATE = "translate"
    DEVELOP = "develop"


class AssistantResponse(BaseToolResponse):
    """Response from the assistant tool"""

    original_text: str = Field(description="The original input text")
    task_type: AssistantTaskType = Field(description="The type of task performed")
    result: str = Field(description="The processed result")
    improvements: Optional[List[str]] = Field(
        None, description="List of improvements made (for proofreading)"
    )
    summary_length: Optional[int] = Field(
        None, description="Length of summary in words (for summarizing)"
    )
    source_language: Optional[str] = Field(
        None, description="Source language for translation"
    )
    target_language: Optional[str] = Field(
        None, description="Target language for translation"
    )
    processing_notes: Optional[str] = Field(
        None, description="Additional notes about the processing"
    )
    direct_response: bool = Field(
        default=True,
        description="Flag indicating this response should be returned directly to user",
    )


class AssistantTool(BaseTool):
    """
    Simplified Assistant Tool that delegates to specialized services

    This facade coordinates between different text processing services
    to provide a unified interface for all assistant tasks.
    """

    def __init__(self):
        super().__init__()
        self.name = "text_assistant"
        self.description = "ONLY use for text and document processing tasks when explicitly requested. Use 'analyze' for document insights and PDF analysis, 'summarize' to condense long content into key points, 'proofread' to correct errors and improve writing quality, 'rewrite' to enhance clarity and impact while preserving meaning, 'critic' for constructive feedback and improvement suggestions, 'translate' to convert text between languages, or 'develop' for programming assistance and code writing. DO NOT use for image analysis, general questions, web searches, or information lookup - use appropriate specialized tools for those tasks."
        self.supported_contexts = ['translation', 'text_processing', 'code_generation']

        # Initialize services
        config = ChatConfig.from_environment()
        self.text_processor = TextProcessorService(config, self.llm_type)
        self.translator = TranslationService(config, self.llm_type)
        self.document_analyzer = DocumentAnalyzerService(config, self.llm_type)

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
                        "task_type": {
                            "type": "string",
                            "enum": [
                                "analyze",
                                "summarize",
                                "proofread",
                                "rewrite",
                                "critic",
                                "translate",
                                "develop",
                            ],
                            "description": "The type of text processing task to perform. Choose 'analyze' for document analysis and insights, 'summarize' to condense long text into key points, 'proofread' to correct errors and improve style, 'rewrite' to enhance clarity and impact, 'critic' for constructive feedback and improvement suggestions, 'translate' to convert between languages, or 'develop' for programming and code assistance.",
                        },
                        "text": {
                            "type": "string",
                            "description": "The text content to be processed. Use 'the PDF' or 'the document' when referring to uploaded PDF content.",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "REQUIRED when analyzing PDF content: The specific task or analysis request about the document. For other tasks, use to provide specific guidance (e.g., 'focus on technical accuracy' for proofreading, 'make it more formal' for rewriting, 'target audience: executives' for summarize).",
                        },
                        "source_language": {
                            "type": "string",
                            "enum": self.translator.get_supported_languages(),
                            "description": "The source language for translation (optional - will auto-detect if not provided)",
                        },
                        "target_language": {
                            "type": "string",
                            "enum": self.translator.get_supported_languages(),
                            "description": "The target language for translation (required for translation tasks)",
                        },
                        "but_why": {
                            "type": "string",
                            "description": "A single sentence explaining why this tool was selected for the query.",
                        },
                    },
                    "required": ["task_type", "text", "but_why"],
                },
            },
        }

    def execute(self, params: Dict[str, Any]) -> AssistantResponse:
        """Execute the assistant tool with given parameters"""
        # Since this is called from tools that may not be in async context,
        # we need to handle the async execution
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._run_async(**params))

    async def _run_async(
        self,
        task_type: str,
        text: str,
        instructions: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AssistantResponse:
        """
        Execute an assistant task with the given parameters (async version)

        Args:
            task_type: The type of task to perform
            text: The text to process
            instructions: Optional instructions
            source_language: Source language for translation
            target_language: Target language for translation
            messages: Optional conversation messages
            **kwargs: Additional parameters

        Returns:
            AssistantResponse with the processed result
        """
        # Validate task type
        try:
            task_enum = AssistantTaskType(task_type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid task_type: {task_type}. Must be one of: {[t.value for t in AssistantTaskType]}"
            )

        # Extract PDF content if needed
        text = self._prepare_text_with_context(text, messages)

        # Route to appropriate service
        if task_enum == AssistantTaskType.TRANSLATE:
            return self._handle_translation(
                text, source_language, target_language, messages
            )
        elif task_enum == AssistantTaskType.ANALYZE:
            return await self._handle_analysis(text, instructions, messages)
        else:
            return await self._handle_text_processing(
                task_enum, text, instructions, messages
            )

    def _prepare_text_with_context(
        self, text: str, messages: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Prepare text with any injected context from messages or session"""

        # Check for PDF content in messages
        if messages:
            logger.debug(f"Checking {len(messages)} messages for PDF content")
            pdf_data = PDFDataExtractor.extract_from_messages(messages)
            if pdf_data:
                logger.info(
                    f"Found PDF data: {pdf_data.get('filename')} with {len(pdf_data.get('pages', []))} pages"
                )
                pdf_text = PDFDataExtractor.extract_text_from_pdf_data(pdf_data)
                if pdf_text:
                    logger.info(f"Extracted {len(pdf_text)} characters of PDF text")
                    # Replace generic references with actual content
                    text_lower = text.lower()
                    if (
                        text_lower in ["the pdf", "the document"]
                        or "pdf" in text_lower
                        or "document" in text_lower
                    ):
                        logger.info(
                            "Replacing generic PDF reference with actual content"
                        )
                        return pdf_text
                    else:
                        # Append as additional context
                        return f"{text}\n\n--- Additional Context ---\n\n{pdf_text}"
                else:
                    logger.warning("PDF data found but no text could be extracted")
            else:
                logger.debug("No PDF data found in messages")

        return text

    def _handle_translation(
        self,
        text: str,
        source_language: Optional[str],
        target_language: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
    ) -> AssistantResponse:
        """Handle translation tasks"""

        if not target_language:
            raise ValueError("target_language is required for translation tasks")

        result = self.translator.translate_text(
            text, target_language, source_language, messages
        )

        if result["success"]:
            return AssistantResponse(
                original_text=text,
                task_type=AssistantTaskType.TRANSLATE,
                result=result["result"],
                source_language=result.get("source_language"),
                target_language=target_language,
                processing_notes=result.get("processing_notes"),
            )
        else:
            raise Exception(result.get("error", "Translation failed"))

    async def _handle_analysis(
        self,
        text: str,
        instructions: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
    ) -> AssistantResponse:
        """Handle document analysis tasks"""

        # Check if this is PDF content
        if self._is_pdf_content(text) and instructions:
            # Parse PDF pages from context
            pages = self._parse_pdf_content(text)
            if pages:
                # For analysis, try to load the complete document
                complete_pages = self._load_complete_document_for_analysis(pages)
                if complete_pages:
                    logger.info(
                        f"Loaded complete document with {len(complete_pages)} pages for analysis"
                    )
                    pdf_data = {"pages": complete_pages, "filename": "Document"}
                    result = await self.document_analyzer.analyze_pdf_pages(
                        complete_pages, instructions, "Document"
                    )
                else:
                    # Fallback to context pages
                    pdf_data = {"pages": pages, "filename": "Document"}
                    result = await self.document_analyzer.analyze_pdf_pages(
                        pages, instructions, "Document"
                    )
            else:
                # Fallback to regular document analysis
                result = await self.document_analyzer.analyze_document(
                    text, instructions or "Analyze this document"
                )
        else:
            # Regular document analysis
            result = await self.document_analyzer.analyze_document(
                text, instructions or "Analyze this text"
            )

        if result["success"]:
            return AssistantResponse(
                original_text=text,
                task_type=AssistantTaskType.ANALYZE,
                result=strip_think_tags(result["result"]),
                processing_notes=result.get("processing_notes"),
            )
        else:
            raise Exception(result.get("error", "Analysis failed"))

    async def _handle_text_processing(
        self,
        task_type: AssistantTaskType,
        text: str,
        instructions: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
    ) -> AssistantResponse:
        """Handle text processing tasks (summarize, proofread, rewrite, critic, etc.)"""

        # Map AssistantTaskType to TextTaskType
        text_task_map = {
            AssistantTaskType.SUMMARIZE: TextTaskType.SUMMARIZE,
            AssistantTaskType.PROOFREAD: TextTaskType.PROOFREAD,
            AssistantTaskType.REWRITE: TextTaskType.REWRITE,
            AssistantTaskType.CRITIC: TextTaskType.CRITIC,
            AssistantTaskType.DEVELOP: TextTaskType.DEVELOP,
        }

        text_task_type = text_task_map.get(task_type)
        if not text_task_type:
            raise ValueError(f"Unsupported text processing task: {task_type}")

        result = await self.text_processor.process_text(
            text_task_type, text, instructions, messages
        )

        if result["success"]:
            # Extract task-specific metadata
            improvements = None
            summary_length = None

            if task_type == AssistantTaskType.SUMMARIZE:
                summary_length = len(result["result"].split())
            elif task_type == AssistantTaskType.PROOFREAD:
                if "improvements" in result["result"].lower():
                    improvements = ["See detailed feedback in the result"]

            return AssistantResponse(
                original_text=text,
                task_type=task_type,
                result=strip_think_tags(result["result"]),
                improvements=improvements,
                summary_length=summary_length,
                processing_notes=result.get("processing_notes"),
            )
        else:
            raise Exception(result.get("error", "Text processing failed"))

    def _is_pdf_content(self, text: str) -> bool:
        """Check if text contains PDF page markers"""
        pdf_indicators = ["[Page ", "Page 1:", "Page 2:", "\n\nPage "]
        return any(indicator in text for indicator in pdf_indicators)

    def _parse_pdf_content(self, text: str) -> List[Dict[str, Any]]:
        """Parse PDF content back into page structure"""
        pages = []
        current_page = None
        current_text = []

        lines = text.split("\n")

        for line in lines:
            if line.startswith("[Page ") and line.endswith("]"):
                # Save previous page
                if current_page is not None and current_text:
                    pages.append(
                        {"page": current_page, "text": "\n".join(current_text).strip()}
                    )

                # Start new page
                try:
                    page_num = int(line.replace("[Page ", "").replace("]", ""))
                    current_page = page_num
                    current_text = []
                except ValueError:
                    if current_text:
                        current_text.append(line)
            else:
                if current_text or line.strip():
                    current_text.append(line)

        # Save final page
        if current_page is not None and current_text:
            pages.append(
                {"page": current_page, "text": "\n".join(current_text).strip()}
            )

        return pages

    def _load_complete_document_for_analysis(
        self, context_pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Load the complete document from batch files for analysis

        Args:
            context_pages: Pages from context injection

        Returns:
            Complete list of all pages from the document
        """
        try:
            # Direct approach: Load all batch files for the current PDF
            from models.chat_config import ChatConfig
            from services.file_storage_service import FileStorageService

            ChatConfig.from_environment()
            file_storage = FileStorageService()

            # Get all PDF files and find the most recent one
            pdf_files = list(file_storage.pdfs_dir.glob("*.json"))
            if not pdf_files:
                logger.warning("No PDF files found")
                return context_pages

            # Filter out batch files and get main PDF files
            main_pdf_files = [f for f in pdf_files if "_batch_" not in f.stem]

            if main_pdf_files:
                # Sort by modification time to get most recent main PDF
                main_pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_pdf_file = main_pdf_files[0]
                pdf_id = latest_pdf_file.stem  # e.g., 'pdf_5990e3a1ea80'
            else:
                # No standalone PDF files found – fall back to using batch filename to derive base ID
                batch_files = [f for f in pdf_files if "_batch_" in f.stem]
                if not batch_files:
                    logger.warning("No PDF or batch files found in storage directory")
                    return context_pages

                # Use the most recent batch file to derive the base PDF ID
                batch_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_batch_file = batch_files[0]
                # Extract base ID before '_batch_' suffix
                pdf_id = latest_batch_file.stem.split("_batch_")[0]
                logger.info(
                    f"Derived base PDF ID '{pdf_id}' from batch file '{latest_batch_file.name}' because no standalone PDF file found."
                )

            logger.info(f"Loading complete document for PDF: {pdf_id}")

            # Get all batches for this PDF
            batches = file_storage.get_pdf_batches(pdf_id)
            logger.info(f"Attempting to load batches for PDF ID: {pdf_id}")
            logger.info(
                f"Available batch files: {[f.name for f in pdf_files if 'batch' in f.name]}"
            )

            if batches:
                logger.info(f"Found {len(batches)} batches for analysis")

                # Load all pages from all batches
                all_pages = []
                for i, batch in enumerate(batches):
                    batch_pages = batch.get("pages", [])
                    logger.info(f"Batch {i}: {len(batch_pages)} pages")
                    all_pages.extend(batch_pages)

                logger.info(
                    f"Successfully loaded {len(all_pages)} pages from batches for analysis"
                )

                return all_pages
            else:
                logger.warning(f"No batches found for PDF {pdf_id}")
                logger.warning(f"Available files: {[f.name for f in pdf_files]}")

        except Exception as e:
            logger.error(f"Error loading complete document for analysis: {e}")

        # Fallback to context pages
        logger.warning("Falling back to context pages for analysis")
        return context_pages


# Create a global instance for backward compatibility
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
        task_type: The type of task to perform
        text: The text to process
        instructions: Optional specific instructions
        source_language: Source language for translation (optional)
        target_language: Target language for translation (required for translation)

    Returns:
        AssistantResponse: The processed result
    """
    return assistant_tool.execute(
        {
            "task_type": task_type,
            "text": text,
            "instructions": instructions,
            "source_language": source_language,
            "target_language": target_language,
        }
    )


def execute_assistant_with_dict(params: Dict[str, Any]) -> AssistantResponse:
    """
    Execute an assistant task with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters
               Expected keys: 'task_type', 'text', and optionally 'instructions', 'source_language', 'target_language'

    Returns:
        AssistantResponse: The processed result
    """
    return assistant_tool.execute(params)
