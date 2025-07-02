"""
PDF Text Processor Tool

This tool enables text processing tasks (summarize, proofread, rewrite, etc.)
on PDF content with automatic chunking for large documents.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field
from tools.assistant import assistant_tool
from tools.base import BaseTool, BaseToolResponse

# Configure logger
logger = logging.getLogger(__name__)

# Constants for chunking
MAX_CHARS_PER_CHUNK = 10000  # Approximately 2000-2500 words
MAX_PAGES_PER_CHUNK = 5


class PDFTextProcessorResponse(BaseToolResponse):
    """Response from PDF text processor tool"""

    success: bool = Field(description="Whether the processing was successful")
    filename: str = Field(description="Name of the PDF file")
    task_type: str = Field(description="Type of text processing performed")
    pages_processed: List[int] = Field(description="Page numbers that were processed")
    result: str = Field(description="Processed result")
    chunks_processed: int = Field(default=1, description="Number of chunks processed")
    message: str = Field(description="Status message")
    direct_response: bool = Field(default=True, description="This provides a direct response to the user")


class PDFTextProcessorTool(BaseTool):
    """
    PDF Text Processor Tool

    This tool performs text processing tasks on PDF content,
    automatically handling chunking for large documents.
    """

    def __init__(self):
        super().__init__()
        self.name = "process_pdf_text"
        self.description = "Performs text processing operations (summarize, proofread, rewrite, translate) on specific pages or sections of PDF documents."

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
                            "enum": ["summarize", "proofread", "rewrite", "critic", "writer", "translate"],
                            "description": "The type of text processing task to perform",
                        },
                        "page_numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of specific page numbers to process (1-indexed). Leave empty to process all pages.",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Optional specific instructions for the task (e.g., 'make it more formal', 'focus on methodology')",
                        },
                        "source_language": {
                            "type": "string",
                            "description": "Source language for translation (optional)",
                        },
                        "target_language": {
                            "type": "string",
                            "description": "Target language for translation (required for translation tasks)",
                        },
                    },
                    "required": ["task_type"],
                },
            },
        }

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for BaseTool interface"""
        return self.to_openai_format()

    def execute(self, params: Dict[str, Any]) -> PDFTextProcessorResponse:
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def run_with_dict(self, params: Dict[str, Any]) -> PDFTextProcessorResponse:
        """
        Execute PDF text processing

        Args:
            params: Dictionary containing parameters

        Returns:
            PDFTextProcessorResponse
        """
        task_type = params.get("task_type")
        page_numbers = params.get("page_numbers", [])
        instructions = params.get("instructions", "")
        source_language = params.get("source_language")
        target_language = params.get("target_language")
        messages = params.get("messages", [])

        # Get PDF data from messages
        pdf_data = self._get_pdf_data_from_messages(messages)

        if not pdf_data:
            return PDFTextProcessorResponse(
                success=False,
                filename="Unknown",
                task_type=task_type,
                pages_processed=[],
                result="",
                message="No PDF document found. Please upload a PDF first.",
                direct_response=True,
            )

        filename = pdf_data.get("filename", "Unknown")
        pages = pdf_data.get("pages", [])
        total_pages = len(pages)

        # Determine which pages to process
        if page_numbers:
            pages_to_process = [p for p in page_numbers if 0 < p <= total_pages]
        else:
            # Process all pages
            pages_to_process = list(range(1, total_pages + 1))

        if not pages_to_process:
            return PDFTextProcessorResponse(
                success=False,
                filename=filename,
                task_type=task_type,
                pages_processed=[],
                result="",
                message="No valid pages to process.",
                direct_response=True,
            )

        # Extract text from selected pages
        extracted_text = self._extract_pages_text(pages, pages_to_process)

        # Check if we need to chunk
        if len(extracted_text) > MAX_CHARS_PER_CHUNK or len(pages_to_process) > MAX_PAGES_PER_CHUNK:
            # Process in chunks
            result = self._process_in_chunks(
                task_type, extracted_text, pages_to_process, instructions, source_language, target_language
            )
            chunks_processed = len(self._create_chunks(extracted_text, pages_to_process))
        else:
            # Process all at once
            result = self._process_text(task_type, extracted_text, instructions, source_language, target_language)
            chunks_processed = 1

        # Format the response
        pages_desc = self._format_page_range(pages_to_process)
        task_desc = self._get_task_description(task_type)

        return PDFTextProcessorResponse(
            success=True,
            filename=filename,
            task_type=task_type,
            pages_processed=pages_to_process,
            result=result,
            chunks_processed=chunks_processed,
            message=f"## {task_desc} of {filename} ({pages_desc})\n\n{result}",
            direct_response=True,
        )

    def _extract_pages_text(self, pages: List[Dict[str, Any]], page_numbers: List[int]) -> str:
        """Extract text from specified pages"""
        text_parts = []
        for page_num in page_numbers:
            if 0 < page_num <= len(pages):
                page_text = pages[page_num - 1].get("text", "")
                if page_text:
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
        return "\n\n".join(text_parts)

    def _create_chunks(self, text: str, page_numbers: List[int]) -> List[Dict[str, Any]]:
        """Create chunks from text, respecting page boundaries when possible"""
        chunks = []
        current_chunk = ""
        current_pages = []

        # Split by pages
        page_texts = text.split("[Page ")
        page_texts = [pt for pt in page_texts if pt]  # Remove empty strings

        for i, page_text in enumerate(page_texts):
            # Extract page number
            if "]" in page_text:
                page_num_str, content = page_text.split("]", 1)
                try:
                    page_num = int(page_num_str)
                except ValueError:
                    page_num = page_numbers[i] if i < len(page_numbers) else i + 1
            else:
                page_num = page_numbers[i] if i < len(page_numbers) else i + 1
                content = page_text

            # Check if adding this page would exceed chunk size
            if current_chunk and (
                len(current_chunk) + len(content) > MAX_CHARS_PER_CHUNK or len(current_pages) >= MAX_PAGES_PER_CHUNK
            ):
                # Save current chunk
                chunks.append({"text": current_chunk, "pages": current_pages.copy()})
                current_chunk = f"[Page {page_num}]{content}"
                current_pages = [page_num]
            else:
                if current_chunk:
                    current_chunk += f"\n\n[Page {page_num}]{content}"
                else:
                    current_chunk = f"[Page {page_num}]{content}"
                current_pages.append(page_num)

        # Add final chunk
        if current_chunk:
            chunks.append({"text": current_chunk, "pages": current_pages})

        return chunks

    def _process_in_chunks(
        self,
        task_type: str,
        text: str,
        page_numbers: List[int],
        instructions: str,
        source_language: Optional[str],
        target_language: Optional[str],
    ) -> str:
        """Process text in chunks and combine results"""
        chunks = self._create_chunks(text, page_numbers)

        logger.info(f"Processing {len(chunks)} chunks for {task_type} task")

        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_instruction = instructions
            if len(chunks) > 1:
                pages_desc = self._format_page_range(chunk["pages"])
                chunk_instruction = f"Process this chunk (pages {pages_desc}). {instructions}"

            result = self._process_text(task_type, chunk["text"], chunk_instruction, source_language, target_language)

            # For chunked processing, add chunk header
            if len(chunks) > 1:
                chunk_results.append(f"### Pages {self._format_page_range(chunk['pages'])}\n{result}")
            else:
                chunk_results.append(result)

        # Combine results
        if task_type == "summarize" and len(chunks) > 1:
            # For summaries, combine and create final summary
            combined = "\n\n".join(chunk_results)
            final_summary = self._process_text(
                "summarize", combined, "Create a cohesive summary from these section summaries", None, None
            )
            return final_summary
        else:
            # For other tasks, just combine with clear separation
            return "\n\n---\n\n".join(chunk_results)

    def _process_text(
        self,
        task_type: str,
        text: str,
        instructions: str,
        source_language: Optional[str],
        target_language: Optional[str],
    ) -> str:
        """Process text using the assistant tool"""
        try:
            # Prepare parameters for assistant tool
            assistant_params = {"task_type": task_type, "text": text, "instructions": instructions}

            if source_language:
                assistant_params["source_language"] = source_language
            if target_language:
                assistant_params["target_language"] = target_language

            # Execute assistant tool
            result = assistant_tool.execute(assistant_params)

            if hasattr(result, 'result'):
                return result.result
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return f"Error processing text: {str(e)}"

    def _format_page_range(self, page_numbers: List[int]) -> str:
        """Format page numbers into a readable range"""
        if not page_numbers:
            return "no pages"

        page_numbers = sorted(page_numbers)
        if len(page_numbers) == 1:
            return f"page {page_numbers[0]}"
        elif len(page_numbers) == 2:
            return f"pages {page_numbers[0]} and {page_numbers[1]}"
        elif page_numbers[-1] - page_numbers[0] == len(page_numbers) - 1:
            # Consecutive range
            return f"pages {page_numbers[0]}-{page_numbers[-1]}"
        else:
            # Non-consecutive
            return f"pages {', '.join(map(str, page_numbers[:-1]))} and {page_numbers[-1]}"

    def _get_task_description(self, task_type: str) -> str:
        """Get human-readable task description"""
        descriptions = {
            "summarize": "Summary",
            "proofread": "Proofreading",
            "rewrite": "Rewritten Version",
            "critic": "Critical Analysis",
            "writer": "Written Content",
            "translate": "Translation",
        }
        return descriptions.get(task_type, "Processed Text")

    def _get_pdf_data_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract PDF data from injected system messages"""
        if not messages:
            logger.debug("No messages provided to extract PDF data from")
            return None

        # Look for PDF content in system messages
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, str) and "## PDF Document Context" in content:
                    # This is an injected PDF context message
                    # Extract the PDF data from the content
                    try:
                        # Parse the PDF content from the system message
                        lines = content.split('\n')
                        filename = None
                        pages = []
                        in_pdf_content = False
                        current_page = None
                        current_page_text = []

                        for line in lines:
                            if "The user has uploaded a PDF document:" in line:
                                # Extract filename
                                import re

                                match = re.search(r"'([^']+)'", line)
                                if match:
                                    filename = match.group(1)
                            elif line.strip() == "---BEGIN PDF CONTENT---":
                                in_pdf_content = True
                            elif line.strip() == "---END PDF CONTENT---":
                                in_pdf_content = False
                                # Save last page if any
                                if current_page is not None and current_page_text:
                                    pages.append({"page": current_page, "text": '\n'.join(current_page_text).strip()})
                            elif in_pdf_content:
                                if line.startswith("### Page "):
                                    # Save previous page if any
                                    if current_page is not None and current_page_text:
                                        pages.append(
                                            {"page": current_page, "text": '\n'.join(current_page_text).strip()}
                                        )
                                    # Start new page
                                    try:
                                        current_page = int(line.replace("### Page ", "").strip())
                                        current_page_text = []
                                    except ValueError:
                                        pass
                                elif current_page is not None:
                                    current_page_text.append(line)

                        if pages:
                            logger.info(f"Extracted PDF data from system message: {filename} with {len(pages)} pages")
                            return {"filename": filename or "Unknown", "pages": pages}
                    except Exception as e:
                        logger.error(f"Error parsing PDF data from system message: {e}")

        logger.debug("No PDF data found in system messages")
        return None


# Create global instance
pdf_text_processor_tool = PDFTextProcessorTool()


def get_pdf_text_processor_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    return pdf_text_processor_tool.to_openai_format()


def execute_pdf_text_processor_with_dict(params: Dict[str, Any]) -> PDFTextProcessorResponse:
    """Execute PDF text processor with parameters as dictionary"""
    return pdf_text_processor_tool.run_with_dict(params)
