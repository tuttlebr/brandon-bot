"""
PDF Full Text Retrieval Tool - Forces retrieval of complete PDF content

This tool allows the LLM to retrieve the full text of PDF documents,
bypassing any summaries. Useful when specific details are needed.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from tools.base import BaseTool, BaseToolResponse

# Configure logger
logger = logging.getLogger(__name__)


class PDFFullTextResponse(BaseToolResponse):
    """Response from PDF full text retrieval tool"""

    success: bool = Field(description="Whether the retrieval was successful")
    filename: str = Field(description="Name of the PDF file")
    total_pages: int = Field(description="Total number of pages in the PDF")
    pages_retrieved: int = Field(description="Number of pages retrieved")
    content: List[Dict[str, Any]] = Field(description="Retrieved page content")
    message: str = Field(description="Status message")


class PDFFullTextTool(BaseTool):
    """
    PDF Full Text Retrieval Tool

    This tool retrieves the complete text content from PDF documents,
    bypassing summaries even for large documents.
    """

    def __init__(self):
        super().__init__()
        self.name = "retrieve_pdf_full_text"
        self.description = "Use this tool ONLY when: (1) User explicitly asks for 'full text' or 'complete text' from specific pages; (2) You need exact quotes or verbatim content; (3) The regular PDF content tool returned summaries but you need the raw text. This bypasses any summarization. Do NOT use as the first choice for PDF questions."

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
                        "page_numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of specific page numbers to retrieve (1-indexed). Leave empty to retrieve all pages.",
                        },
                        "max_pages": {
                            "type": "integer",
                            "default": 20,
                            "description": "Maximum number of pages to retrieve if page_numbers is empty (default: 20)",
                        },
                    },
                    "required": [],
                },
            },
        }

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for BaseTool interface"""
        return self.to_openai_format()

    def execute(self, params: Dict[str, Any]) -> PDFFullTextResponse:
        """Execute the tool with given parameters"""
        result = self.run_with_dict(params)
        return PDFFullTextResponse(**result)

    def run_with_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full text retrieval

        Args:
            params: Dictionary containing parameters

        Returns:
            Dictionary with full text content
        """
        page_numbers = params.get("page_numbers", [])
        max_pages = params.get("max_pages", 20)
        messages = params.get("messages", [])
        pdf_data = params.get("pdf_data", None)

        # Get PDF data from messages if not provided
        if pdf_data is None:
            pdf_data = self._get_pdf_data_from_messages(messages)

        if not pdf_data:
            return {"success": False, "message": "No PDF document found in session", "content": []}

        pages = pdf_data.get("pages", [])
        filename = pdf_data.get("filename", "Unknown")
        total_pages = len(pages)

        # Determine which pages to retrieve
        if page_numbers:
            # Retrieve specific pages
            pages_to_retrieve = [p - 1 for p in page_numbers if 0 < p <= total_pages]
        else:
            # Retrieve up to max_pages
            pages_to_retrieve = list(range(min(max_pages, total_pages)))

        # Build content
        content = []
        for idx in pages_to_retrieve:
            if 0 <= idx < total_pages:
                page_data = pages[idx]
                content.append({"page": page_data.get("page", idx + 1), "text": page_data.get("text", "")})

        return {
            "success": True,
            "filename": filename,
            "total_pages": total_pages,
            "pages_retrieved": len(content),
            "content": content,
            "message": f"Retrieved {len(content)} pages of full text from '{filename}'",
        }

    def _get_pdf_data_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get PDF data from messages (same implementation as pdf_parser)"""
        if not messages:
            return None

        # Look for injected PDF data in system messages
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
                            return data
                except (json.JSONDecodeError, TypeError):
                    continue

        return None


# Create global instance
pdf_full_text_tool = PDFFullTextTool()


def get_pdf_full_text_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    return pdf_full_text_tool.to_openai_format()


def execute_pdf_full_text_with_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute PDF full text retrieval with parameters as dictionary"""
    return pdf_full_text_tool.run_with_dict(params)
