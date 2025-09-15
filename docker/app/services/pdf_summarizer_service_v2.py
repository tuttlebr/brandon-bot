"""
PDFSummarizerService V2
-----------------------
Provides `summarize_pdf` which selects strategy based on token/char count:
    • SMALL  (<= ~10k chars): single-pass summarization.
    • LARGE  (> 10k  chars): map-reduce summarization over chunks produced by PDFChunkingService.

Uses TextProcessorService (summarize task) for chunk-level and final reduction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from services.pdf_chunking_service import PDFChunkingService
from services.text_processor_service import TextProcessorService, TextTaskType
from tools.tool_llm_config import get_tool_llm_type

from utils.logging_config import get_logger

logger = get_logger(__name__)

_SMALL_CHAR_LIMIT = 32000
_CHUNK_SUMMARY_TOKENS = 400  # heuristic


class PDFSummarizerServiceV2:
    def __init__(self, config: ChatConfig | None = None):
        self.config = config or ChatConfig.from_environment()
        # Use the LLM type configured for pdf_assistant tool
        llm_type = get_tool_llm_type("pdf_assistant")
        logger.info(f"PDFSummarizerServiceV2 using LLM type: {llm_type}")
        self.text_processor = TextProcessorService(
            self.config, llm_type=llm_type
        )
        self.chunker = PDFChunkingService(self.config)

    # ------------------------------------------------------------
    async def summarize_pdf(
        self, pdf_data: Dict[str, Any], user_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return {summary: str, strategy: str}."""
        logger.info(
            "Starting PDF summarization. PDF data keys:"
            f" {list(pdf_data.keys())}"
        )

        # Check if pages exist
        if "pages" not in pdf_data:
            logger.error(
                "PDF data missing 'pages' key. Available keys:"
                f" {list(pdf_data.keys())}"
            )
            return {
                "summary": "Error: PDF data is missing page content",
                "strategy": "error",
            }

        pages = pdf_data.get("pages", [])
        if not pages:
            logger.error("PDF has no pages")
            return {
                "summary": "Error: PDF has no pages to summarize",
                "strategy": "error",
            }

        char_count: int = pdf_data.get("char_count") or sum(
            len(p.get("text", "")) for p in pages
        )
        logger.info(f"PDF has {len(pages)} pages and {char_count} characters")

        if char_count <= _SMALL_CHAR_LIMIT:
            return await self._summarize_small(pdf_data, user_instruction)
        return await self._summarize_large(pdf_data, user_instruction)

    # ------------------------------------------------------------
    async def _summarize_small(
        self, pdf_data: Dict[str, Any], user_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info("🔍 Small PDF summarization strategy selected")
        logger.info(
            f"   Character count: {pdf_data.get('char_count', 'unknown')}"
        )
        logger.info(f"   Page count: {len(pdf_data.get('pages', []))}")

        # Extract text from pages with error handling
        page_texts = []
        for i, page in enumerate(pdf_data["pages"]):
            if isinstance(page, dict) and "text" in page:
                page_texts.append(page["text"])
            else:
                logger.warning(
                    f"Page {i} missing text field. Page data: {page}"
                )

        if not page_texts:
            logger.error("No text found in any pages")
            return {
                "summary": "Error: No text content found in PDF pages",
                "strategy": "error",
            }

        full_text = "\n\n".join(page_texts)
        logger.info(
            f"📝 Concatenated text: {len(full_text)} characters from"
            f" {len(page_texts)} pages"
        )

        try:
            logger.info(
                "🚀 Sending text to TextProcessorService for summarization..."
            )
            # Use user instruction or default
            instructions = (
                user_instruction
                or "Provide a concise summary of the document."
            )
            result = await self.text_processor.process_text(
                TextTaskType.SUMMARIZE,
                text=full_text,
                instructions=instructions,
            )

            if not result.get("success", True):
                logger.error(
                    "Text processing failed:"
                    f" {result.get('error', 'Unknown error')}"
                )
                return {
                    "summary": (
                        "Error:"
                        f" {result.get('error', 'Failed to summarize document')}"
                    ),
                    "strategy": "error",
                }

            return {"summary": result["result"], "strategy": "small"}
        except Exception as e:
            logger.error(f"Exception during summarization: {e}", exc_info=True)
            return {"summary": f"Error: {str(e)}", "strategy": "error"}

    # ------------------------------------------------------------
    async def _summarize_large(
        self, pdf_data: Dict[str, Any], user_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(
            "🔍 Large PDF summarization strategy selected (map-reduce)"
        )
        logger.info(
            f"   Character count: {pdf_data.get('char_count', 'unknown')}"
        )
        logger.info(f"   Page count: {len(pdf_data.get('pages', []))}")

        # Step 1: ensure chunks exist
        logger.info("📄 Creating chunks for large document...")
        chunks = self.chunker.chunk_pdf_document(pdf_data)
        if not chunks:
            raise RuntimeError("Chunking failed for large PDF summarization")

        logger.info(f"✅ Created {len(chunks)} chunks for processing")

        # Step 2: map – summarize each chunk in parallel
        import asyncio

        logger.info(
            f"🗺️  Starting MAP phase: Summarizing {len(chunks)} chunks in"
            " parallel..."
        )

        async def summarize_chunk(chunk_idx, chunk):
            try:
                logger.info(
                    "   Processing chunk"
                    f" {chunk_idx + 1}/{len(chunks)} ({len(chunk['text'])} chars)..."
                )
                result = await self.text_processor.process_text(
                    TextTaskType.SUMMARIZE,
                    text=chunk["text"],
                    instructions="Summarize this part of the document.",
                )
                if not result.get("success", True):
                    logger.error(
                        f"Chunk {chunk_idx + 1} summarization failed:"
                        f" {result.get('error')}"
                    )
                    return {
                        "result": (
                            f"[Error summarizing chunk {chunk_idx + 1}:"
                            f" {result.get('error')}]"
                        )
                    }
                logger.info(
                    f"   ✓ Chunk {chunk_idx + 1} summarized successfully"
                )
                return result
            except Exception as e:
                logger.error(
                    f"Exception summarizing chunk {chunk_idx + 1}: {e}"
                )
                return {
                    "result": f"[Error in chunk {chunk_idx + 1}: {str(e)}]"
                }

        chunk_results = await asyncio.gather(
            *(summarize_chunk(i, c) for i, c in enumerate(chunks)),
            return_exceptions=False,
        )
        partial_summaries = [r["result"] for r in chunk_results]

        logger.info(
            f"✅ MAP phase complete: {len(partial_summaries)} chunk summaries"
            " generated"
        )

        # Step 3: reduce – recursively combine summaries into <= _CHUNK_SUMMARY_TOKENS
        iteration = 0
        while len(partial_summaries) > 1:
            iteration += 1
            logger.info(
                f"🔄 REDUCE phase iteration {iteration}: Combining"
                f" {len(partial_summaries)} summaries..."
            )

            combined_batches: List[str] = []
            for i in range(0, len(partial_summaries), 5):
                batch_text = "\n\n".join(partial_summaries[i : i + 5])
                combined_batches.append(batch_text)

            logger.info(
                f"   Created {len(combined_batches)} batches for reduction"
            )

            # summarize each combined batch
            async def summarize_batch(batch_idx, batch_text):
                logger.info(
                    "   Reducing batch"
                    f" {batch_idx + 1}/{len(combined_batches)} ({len(batch_text)} chars)..."
                )
                result = await self.text_processor.process_text(
                    TextTaskType.SUMMARIZE,
                    text=batch_text,
                    instructions=(
                        "Condense the following summary segments into a"
                        " shorter combined summary."
                    ),
                )
                logger.info(f"   ✓ Batch {batch_idx + 1} reduced successfully")
                return result["result"]

            partial_summaries = await asyncio.gather(
                *(
                    summarize_batch(i, b)
                    for i, b in enumerate(combined_batches)
                )
            )
            logger.info(
                f"✅ REDUCE iteration {iteration} complete:"
                f" {len(partial_summaries)} summaries remaining"
            )

        logger.info("🎉 Map-reduce summarization complete!")
        return {"summary": partial_summaries[0], "strategy": "large"}
