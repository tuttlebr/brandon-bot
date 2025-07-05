# PDF Batch Processing Implementation

## Overview

The batch processing feature addresses memory issues when users upload large PDF documents. Instead of loading the entire document into memory, the system now intelligently splits and stores large PDFs in manageable chunks.

## Architecture

### Components

1. **PDFBatchProcessor** (`services/pdf_batch_processor.py`)
   - Determines when batch processing is needed
   - Splits PDFs into configurable batch sizes
   - Manages batch creation and merging

2. **FileStorageService** (enhanced)
   - `store_pdf_batch()`: Stores individual batches
   - `get_pdf_batches()`: Retrieves all batches for a PDF
   - `merge_pdf_batches()`: Combines batches when needed

3. **PDFContextService** (enhanced)
   - Smart context injection based on user queries
   - Loads only relevant pages from batches
   - Extracts page numbers from queries

### Configuration

Key settings in `utils/config.py`:

```python
PDF_BATCH_PROCESSING_THRESHOLD = 50  # Trigger batch processing for PDFs > 50 pages
PDF_PAGES_PER_BATCH = 100            # Pages per batch
PDF_CONTEXT_MAX_PAGES = 30          # Max pages in context
PDF_CONTEXT_MAX_CHARS = 100000      # Max characters in context
```

## Implementation Details

### Batch Processing Flow

1. **Detection**: When a PDF is uploaded, `FileController` checks page count
2. **Splitting**: If pages > threshold, `PDFBatchProcessor` creates batches
3. **Storage**: Each batch is stored separately with metadata
4. **Reference**: Session state stores PDF ID and batch info

### Smart Context Loading

When a user asks about a batch-processed PDF:

1. **Query Analysis**: Extract page numbers from the query
2. **Batch Selection**: Determine which batches contain relevant pages
3. **Lazy Loading**: Load only the required batches
4. **Context Injection**: Include only relevant pages in LLM context

### Memory Optimization

- Regular PDFs: Stored as single JSON file
- Batch PDFs: Split into multiple files (20 pages each)
- Context loading: Limited to 100KB or 30 pages
- No full document loading for large PDFs

## API Changes

### FileController

```python
def _handle_batch_processing(self, filename: str, pdf_data: dict):
    """Handle batch processing for large PDFs"""
```

### PDFContextService

```python
def _get_batch_processed_pdf(self, pdf_id: str, batch_info: Dict[str, Any]):
    """Get batch-processed PDF metadata"""

def _load_pages_from_batches(self, pdf_id: str, page_numbers: List[int]):
    """Load specific pages from PDF batches"""
```

## Usage Examples

### Uploading a Large PDF

```python
# Automatic detection and batch processing
# No code changes needed - handled internally
uploaded_file = st.file_uploader("Choose PDF")
file_controller.process_pdf_upload(uploaded_file)
```

### Querying Batch-Processed PDFs

Users can query normally, but for best results:

```
"What does page 125 say about revenue?"  # Loads only batch containing page 125
"Summarize pages 50-75"                   # Loads specific batches
"What are the main conclusions?"          # Loads first few batches
```

## Performance Considerations

### Memory Usage

- Before: 500MB PDF = 500MB in memory
- After: 500MB PDF = 20 pages in memory (~ 2MB)

### Trade-offs

- **Pros**: Handles massive PDFs, prevents OOM errors, faster initial processing
- **Cons**: Slightly slower page access, requires batch lookup

## Future Enhancements

1. **Intelligent Batching**: Group related content together
2. **Embedding Search**: Use embeddings to find relevant batches
3. **Caching**: Cache frequently accessed batches
4. **Async Loading**: Pre-load likely next batches

## Testing

To test batch processing:

1. Upload a PDF with > 50 pages
2. Verify batch creation in logs
3. Ask questions about specific pages
4. Monitor memory usage

## Troubleshooting

### Common Issues

1. **Pages not found**: Check batch boundaries
2. **Slow queries**: Optimize batch size
3. **Missing content**: Verify batch storage

### Debug Logging

Enable debug logs to trace batch processing:

```python
logger.debug(f"Processing batch: pages {start_idx + 1} to {end_idx}")
logger.info(f"Created {len(batches)} batches for {total_pages} pages")
```
