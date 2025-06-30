"""
PDF Parser Tool - Retrieves PDF content when relevant to user queries

This tool allows the LLM to selectively retrieve PDF document content
only when it determines the user is asking questions related to uploaded PDFs.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import streamlit as st
from models.chat_config import ChatConfig
from pydantic import BaseModel, Field
from tools.base import BaseTool, BaseToolResponse
from utils.config import config

# Configure logger
logger = logging.getLogger(__name__)


class PDFContentResult(BaseModel):
    """Individual PDF page content"""

    page_number: int = Field(description="Page number")
    text: str = Field(description="Extracted text content from the page")


class PDFParseResponse(BaseToolResponse):
    """Response from PDF parsing tool"""

    filename: str = Field(description="Name of the PDF file")
    total_pages: int = Field(description="Total number of pages in the PDF")
    pages_requested: List[int] = Field(description="Page numbers that were requested")
    content: List[PDFContentResult] = Field(description="PDF page content")
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Status or error message")
    formatted_results: Optional[str] = Field(None, description="Formatted text for easy reading")


class PDFParserTool(BaseTool):
    """Tool for parsing and retrieving content from uploaded PDF documents"""

    def __init__(self):
        super().__init__()
        self.name = "retrieve_pdf_content"
        self.description = "Use this tool as the FIRST STEP when a user asks ANY question about a PDF they've uploaded. This retrieves the PDF content for analysis. Use this for: questions about PDF content, finding information in PDFs, or any PDF-related query. After retrieving content, you may need other tools for processing."

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
                "parameters": {"type": "object", "properties": {}, "required": [],},
            },
        }

    def _get_pdf_data_from_messages(self, messages: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve PDF data from messages list (including injected system messages)

        Args:
            messages: List of conversation messages to search in

        Returns:
            PDF data dictionary if available, None otherwise
        """
        if not messages:
            logger.error("No messages provided to search for PDF data")
            return None

        logger.info(f"Searching for PDF data in {len(messages)} messages")

        # First look for injected PDF data in system messages
        for message in messages:
            if message.get("role") == "system":
                try:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        data = json.loads(content)
                        if (
                            isinstance(data, dict)
                            and data.get("type") == "pdf_data"
                            and data.get("tool_name") == "process_pdf_document"
                        ):
                            logger.info(f"✅ Found injected PDF data for: {data.get('filename', 'Unknown')}")
                            return data
                except (json.JSONDecodeError, TypeError):
                    continue

        # Fallback: Look for tool messages containing PDF data
        for message in reversed(messages):
            if message.get("role") == "tool":
                try:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        tool_data = json.loads(content)
                        if (
                            isinstance(tool_data, dict)
                            and tool_data.get("tool_name") == "process_pdf_document"
                            and tool_data.get("status") == "success"
                        ):
                            logger.info(
                                f"✅ Found PDF data in tool message for: {tool_data.get('filename', 'Unknown')}"
                            )
                            return tool_data
                except (json.JSONDecodeError, TypeError):
                    continue

        logger.error("❌ No PDF data found in messages")
        return None

    def _format_content_for_display(self, content: List[PDFContentResult], filename: str) -> str:
        """
        Format PDF content for easy reading

        Args:
            content: List of PDF page content
            filename: Name of the PDF file

        Returns:
            Formatted string for display
        """
        if not content:
            return f"No content found in {filename}"

        formatted_parts = [f"Content from PDF: {filename}\n" + "=" * 50]

        for page_content in content:
            formatted_parts.append(f"\n📄 Page {page_content.page_number}:")
            formatted_parts.append("-" * 30)
            # Limit content length for readability
            text = page_content.text
            formatted_parts.append(text)

        return "\n".join(formatted_parts)

    def run_with_dict(self, params: Dict[str, Any]) -> PDFParseResponse:
        """
        Execute PDF content retrieval - returns all pages from the uploaded PDF

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'pdf_data' (optional), 'messages'

        Returns:
            PDFParseResponse: PDF content retrieval response
        """
        pdf_data = params.get("pdf_data", None)
        messages = params.get("messages", None)

        logger.info(f"PDF content retrieval called, pdf_data_provided: {pdf_data is not None}")

        # Use provided PDF data or look in messages
        if pdf_data is None:
            logger.info("No PDF data provided directly, looking in messages")
            pdf_data = self._get_pdf_data_from_messages(messages)

        if not pdf_data:
            return PDFParseResponse(
                filename="No PDF found",
                total_pages=0,
                pages_requested=[],
                content=[],
                success=False,
                message="No PDF document found in the current session. Please upload a PDF document first.",
            )

        filename = pdf_data.get("filename", "Unknown PDF")
        pages = pdf_data.get("pages", [])
        total_pages = len(pages)

        if not pages:
            return PDFParseResponse(
                filename=filename,
                total_pages=0,
                pages_requested=[],
                content=[],
                success=False,
                message="PDF document found but contains no readable pages.",
            )

        # Check if document has been summarized (for large documents)
        if (
            pdf_data.get('summarization_complete', False)
            and total_pages > config.file_processing.PDF_SUMMARIZATION_THRESHOLD
        ):
            logger.info(f"Using document summary for large PDF: {filename}")

            # Create a summary-based response
            doc_summary = pdf_data.get('document_summary', '')
            page_summaries = pdf_data.get('page_summaries', [])

            # Build content response with summary
            content_results = []

            # Add document summary as first "page"
            if doc_summary:
                content_results.append(PDFContentResult(page_number=0, text=f"DOCUMENT SUMMARY:\n{doc_summary}"))

            # Add page summaries for more detail if needed
            for summary in page_summaries[:5]:  # Limit to first 5 summaries
                content_results.append(
                    PDFContentResult(
                        page_number=-1,  # Special marker for summaries
                        text=f"Pages {summary['page_range']} Summary:\n{summary['summary']}",
                    )
                )

            # Add note about full content availability
            availability_note = (
                f"\n\nNote: This is a summarized view of a {total_pages}-page document. "
                f"The full text of all pages is available if you need specific details from particular pages."
            )

            if content_results:
                content_results[0].text += availability_note

            formatted_results = self._format_content_for_display(content_results, filename)

            return PDFParseResponse(
                filename=filename,
                total_pages=total_pages,
                pages_requested=[0],  # Summary is treated as page 0
                content=content_results,
                success=True,
                message=f"Retrieved document summary and key sections from '{filename}' ({total_pages} pages)",
                formatted_results=formatted_results,
            )

        # For smaller documents or if summarization not complete, return full content
        content_results = []
        pages_requested = []

        for i in range(total_pages):
            page_data = pages[i]
            page_num = page_data.get("page", i + 1)
            content_results.append(PDFContentResult(page_number=page_num, text=page_data.get("text", "")))
            pages_requested.append(page_num)

        # Format results for display
        formatted_results = self._format_content_for_display(content_results, filename)

        success_message = f"Retrieved complete content from all {len(content_results)} pages of '{filename}'"

        return PDFParseResponse(
            filename=filename,
            total_pages=total_pages,
            pages_requested=pages_requested,
            content=content_results,
            success=True,
            message=success_message,
            formatted_results=formatted_results,
        )

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for BaseTool interface"""
        return self.to_openai_format()

    def execute(self, params: Dict[str, Any]) -> PDFParseResponse:
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)


# Create a global instance and helper functions for easy access
pdf_parser_tool = PDFParserTool()


def get_pdf_parser_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for PDF content retrieval

    Returns:
        Dict containing the OpenAI tool definition
    """
    return pdf_parser_tool.to_openai_format()


def execute_pdf_parse_with_dict(params: Dict[str, Any]) -> PDFParseResponse:
    """
    Execute PDF content retrieval with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters

    Returns:
        PDFParseResponse: PDF content retrieval response
    """
    return pdf_parser_tool.run_with_dict(params)
