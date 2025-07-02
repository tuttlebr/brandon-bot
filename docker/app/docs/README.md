# Streamlit Chat Application Documentation

This directory contains comprehensive documentation for the Streamlit Chat Application that can be served via API endpoints.

## Quick Start - Simplified Server (Recommended)

We provide a lightweight documentation server that requires minimal dependencies:

### 1. Install Minimal Dependencies

```bash
pip install -r requirements_simple.txt
# Only requires: fastapi, uvicorn, markdown
```

### 2. Run the Simple Documentation Server

```bash
python simple_docs_server.py
```

### 3. Access Documentation

- Home: `http://localhost:8001/docs`
- Table of Contents: `http://localhost:8001/docs/toc`
- Search: `http://localhost:8001/docs/search`
- API endpoints: `http://localhost:8001/`

## Features of Simple Server

- ✅ **Minimal Dependencies** - Only 3 Python packages needed
- ✅ **Fast Loading** - No heavy build process
- ✅ **Clean UI** - Professional styling with no external assets
- ✅ **Search** - Built-in search functionality
- ✅ **API Access** - JSON endpoints for integration
- ✅ **No 404 Errors** - All assets embedded in HTML

## Docker Deployment

### Using the Simplified Dockerfile

```bash
# Build
docker build -f Dockerfile.docs -t docs-server ./docker

# Run
docker run -p 8001:8001 docs-server
```

### Docker Compose Integration

```yaml
services:
  docs:
    build:
      context: ./docker
      dockerfile: Dockerfile.docs
    ports:
      - "8001:8001"
    restart: unless-stopped
```

## API Endpoints

The simple server provides these endpoints:

- `GET /docs` - Documentation home page
- `GET /docs/toc` - Table of contents
- `GET /docs/page/{path}` - View specific page
- `GET /docs/search` - Search interface
- `GET /docs/api/pages` - List all pages (JSON)
- `GET /docs/api/content/{path}` - Get page content (JSON)
- `GET /docs/api/search?q={query}` - Search API (JSON)

## Integration with Main App

Add to your Streamlit application:

```python
import streamlit as st
import requests

# Add documentation link
if st.sidebar.button("📚 View Documentation"):
    st.sidebar.markdown("[Open Docs](http://localhost:8001/docs)")

# Or embed directly
def show_doc_page(page_path):
    response = requests.get(f"http://localhost:8001/docs/api/content/{page_path}")
    if response.status_code == 200:
        data = response.json()
        st.markdown(data["content"])
```

## Alternative: Full MkDocs Setup

If you prefer the full MkDocs setup with Material theme:

### 1. Install Full Dependencies

```bash
pip install -r requirements.txt
# Includes: mkdocs, mkdocs-material, and plugins
```

### 2. Build Documentation

```bash
# First, create a mkdocs.yml with proper config:
cat > mkdocs.yml << 'EOF'
site_name: Streamlit Chat Documentation
docs_dir: .
site_dir: site
# ... rest of config
EOF

# Build
mkdocs build
```

### 3. Serve with MkDocs

```bash
mkdocs serve
# Or use serve_docs.py for API access
```

## Advantages of Each Approach

### Simple Server (simple_docs_server.py)
- ✅ Minimal dependencies (3 packages)
- ✅ No build step required
- ✅ Fast startup
- ✅ No external asset issues
- ✅ Embedded styling
- ✅ Works immediately

### Full MkDocs
- ✅ More features and themes
- ✅ Better for large documentation
- ✅ Plugin ecosystem
- ❌ Heavy dependencies
- ❌ Build step required
- ❌ Asset path issues

## Adding New Documentation

1. Create new `.md` files in appropriate directories
2. No configuration needed for simple server
3. Files are automatically discovered and served

## File Structure

```
docs/
├── README.md                    # This file
├── requirements_simple.txt      # Minimal deps (3 packages)
├── requirements.txt            # Full deps (MkDocs)
├── simple_docs_server.py       # Lightweight server
├── serve_docs.py              # Full-featured server
├── integration_example.py      # Streamlit integration
├── index.md                   # Home page
├── getting-started/           # Getting started guides
├── user-guide/               # User documentation
├── architecture/             # Technical docs
├── api/                      # API reference
├── configuration/            # Config guides
└── troubleshooting.md        # Troubleshooting
```

## Production Deployment

For production, we recommend:

1. Use the simple server for lower resource usage
2. Put it behind a reverse proxy (nginx)
3. Enable caching headers
4. Consider CDN for static assets

Example nginx config:

```nginx
location /docs {
    proxy_pass http://localhost:8001;
    proxy_set_header Host $host;
    proxy_cache_valid 200 1h;
}
```

## Support

For issues or questions:
1. Check the troubleshooting guide
2. Review the architecture documentation
3. Submit an issue on GitHub
