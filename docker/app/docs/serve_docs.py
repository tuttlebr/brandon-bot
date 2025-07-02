#!/usr/bin/env python3
"""
Documentation API Server - Simplified and robust version
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# Initialize FastAPI app
app = FastAPI(title="Streamlit Chat Documentation", docs_url=None, redoc_url=None, openapi_url=None)

# Documentation root directory
DOCS_ROOT = Path(__file__).parent

# CSS for documentation pages
DOCS_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
    color: #333;
}
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}
h1 { border-bottom: 2px solid #eee; padding-bottom: 0.3em; }
code {
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
}
pre {
    background: #f4f4f4;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
}
pre code {
    background: none;
    padding: 0;
}
blockquote {
    border-left: 4px solid #ddd;
    margin: 0;
    padding-left: 20px;
    color: #666;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background: #f4f4f4;
    font-weight: bold;
}
a { color: #3498db; text-decoration: none; }
a:hover { text-decoration: underline; }
.nav { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
.nav ul { list-style: none; margin: 0; padding: 0; }
.nav li { display: inline; margin-right: 20px; }
.section { margin-bottom: 30px; }
.note {
    background: #e3f2fd;
    border-left: 4px solid #2196F3;
    padding: 10px 15px;
    margin: 20px 0;
}
.warning {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 10px 15px;
    margin: 20px 0;
}
"""


def render_markdown_simple(content: str) -> str:
    """Simple markdown to HTML conversion"""
    lines = content.split('\n')
    html_lines = []
    in_code_block = False
    in_list = False

    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                html_lines.append('</pre>')
                in_code_block = False
            else:
                html_lines.append('<pre>')
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(line)
            continue

        # Headers
        if line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('#### '):
            html_lines.append(f'<h4>{line[5:]}</h4>')
        # Lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{line.strip()[2:]}</li>')
        # Empty line
        elif not line.strip():
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<br>')
        # Regular paragraph
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            # Basic inline formatting
            text = line
            # Bold
            while '**' in text:
                text = text.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
            # Italic
            while '*' in text and '<strong>' not in text and '</strong>' not in text:
                text = text.replace('*', '<em>', 1).replace('*', '</em>', 1)
            # Inline code
            while '`' in text:
                text = text.replace('`', '<code>', 1).replace('`', '</code>', 1)
            html_lines.append(f'<p>{text}</p>')

    if in_list:
        html_lines.append('</ul>')
    if in_code_block:
        html_lines.append('</pre>')

    return '\n'.join(html_lines)


def create_page(title: str, content: str, nav_html: str = "") -> str:
    """Create a complete HTML page"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} - Streamlit Chat Documentation</title>
        <style>{DOCS_CSS}</style>
    </head>
    <body>
        {nav_html}
        {content}
    </body>
    </html>
    """


def get_nav_html() -> str:
    """Get navigation HTML"""
    return """
    <div class="nav">
        <ul>
            <li><a href="/docs/">Home</a></li>
            <li><a href="/docs/quickstart">Quick Start</a></li>
            <li><a href="/docs/user-guide">User Guide</a></li>
            <li><a href="/docs/architecture">Architecture</a></li>
            <li><a href="/docs/api">API Reference</a></li>
        </ul>
    </div>
    """


def get_home_content():
    """Get home page content"""
    # Try to load index.md
    index_path = DOCS_ROOT / "index.md"
    if index_path.exists():
        content = index_path.read_text()
        return render_markdown_simple(content)
    else:
        return """
        <h1>Streamlit Chat Documentation</h1>
        <p>Welcome to the Streamlit Chat Application documentation.</p>

        <div class="section">
            <h2>Getting Started</h2>
            <ul>
                <li><a href="/docs/quickstart">Quick Start Guide</a> - Get up and running in 5 minutes</li>
                <li><a href="/docs/installation">Installation Guide</a> - Detailed setup instructions</li>
                <li><a href="/docs/first-steps">First Steps</a> - Your first chat session</li>
            </ul>
        </div>

        <div class="section">
            <h2>Features</h2>
            <ul>
                <li><a href="/docs/chat-interface">Chat Interface</a> - Using the chat features</li>
                <li><a href="/docs/pdf-analysis">PDF Analysis</a> - Document processing capabilities</li>
                <li><a href="/docs/image-generation">Image Generation</a> - Creating images with AI</li>
            </ul>
        </div>

        <div class="section">
            <h2>Technical Documentation</h2>
            <ul>
                <li><a href="/docs/architecture">Architecture Overview</a> - System design and components</li>
                <li><a href="/docs/api">API Reference</a> - Service and controller APIs</li>
                <li><a href="/docs/configuration">Configuration Guide</a> - Environment setup</li>
            </ul>
        </div>
        """


# Root redirect
@app.get("/")
async def root():
    """Redirect to /docs/"""
    return RedirectResponse(url="/docs/", status_code=302)


# Documentation home
@app.get("/docs")
async def docs_redirect():
    """Redirect /docs to /docs/"""
    return RedirectResponse(url="/docs/", status_code=302)


@app.get("/docs/")
async def docs_home():
    """Documentation home page"""
    html_content = get_home_content()
    return HTMLResponse(content=create_page("Home", html_content, get_nav_html()))


@app.get("/docs/quickstart")
async def quickstart():
    """Quick start page"""
    file_path = DOCS_ROOT / "getting-started" / "quickstart.md"
    if file_path.exists():
        content = render_markdown_simple(file_path.read_text())
        return HTMLResponse(content=create_page("Quick Start", content, get_nav_html()))

    # Fallback content
    content = """
    <h1>Quick Start Guide</h1>
    <p>Get the Streamlit Chat Application running quickly.</p>

    <h2>Prerequisites</h2>
    <ul>
        <li>Docker and Docker Compose installed</li>
        <li>NVIDIA API key</li>
    </ul>

    <h2>Setup</h2>
    <ol>
        <li>Clone the repository</li>
        <li>Create a .env file with your API keys</li>
        <li>Run: <code>docker compose up -d</code></li>
        <li>Access the app at <code>http://localhost:8080</code></li>
    </ol>
    """
    return HTMLResponse(content=create_page("Quick Start", content, get_nav_html()))


@app.get("/docs/user-guide")
async def user_guide():
    """User guide page"""
    content = """
    <h1>User Guide</h1>

    <div class="section">
        <h2>Chat Interface</h2>
        <p>The chat interface provides natural conversation with AI models.</p>
        <ul>
            <li><a href="/docs/chat-interface">Chat Interface Guide</a></li>
            <li><a href="/docs/pdf-analysis">PDF Analysis Guide</a></li>
            <li><a href="/docs/image-generation">Image Generation Guide</a></li>
        </ul>
    </div>
    """
    return HTMLResponse(content=create_page("User Guide", content, get_nav_html()))


@app.get("/docs/architecture")
async def architecture():
    """Architecture page"""
    file_path = DOCS_ROOT / "architecture" / "overview.md"
    if file_path.exists():
        content = render_markdown_simple(file_path.read_text())
        return HTMLResponse(content=create_page("Architecture", content, get_nav_html()))

    content = """
    <h1>Architecture Overview</h1>
    <p>The application follows an MVC architecture with clear separation of concerns.</p>

    <h2>Components</h2>
    <ul>
        <li><strong>Controllers</strong> - Handle user interactions and orchestrate services</li>
        <li><strong>Services</strong> - Core business logic and external integrations</li>
        <li><strong>Tools</strong> - Specialized capabilities for the AI models</li>
    </ul>
    """
    return HTMLResponse(content=create_page("Architecture", content, get_nav_html()))


@app.get("/docs/api")
async def api_reference():
    """API reference page"""
    content = """
    <h1>API Reference</h1>

    <div class="section">
        <h2>Services</h2>
        <ul>
            <li>LLMService - Language model interactions</li>
            <li>ChatService - Chat management</li>
            <li>PDFService - Document processing</li>
        </ul>
    </div>

    <div class="section">
        <h2>Controllers</h2>
        <ul>
            <li>SessionController - Session management</li>
            <li>MessageController - Message handling</li>
            <li>ResponseController - Response generation</li>
        </ul>
    </div>
    """
    return HTMLResponse(content=create_page("API Reference", content, get_nav_html()))


@app.get("/docs/{page_name}")
async def get_page(page_name: str):
    """Get a specific documentation page"""
    # Try different paths
    paths_to_try = [
        DOCS_ROOT / f"{page_name}.md",
        DOCS_ROOT / "getting-started" / f"{page_name}.md",
        DOCS_ROOT / "user-guide" / f"{page_name}.md",
        DOCS_ROOT / "architecture" / f"{page_name}.md",
        DOCS_ROOT / "configuration" / f"{page_name}.md",
    ]

    for path in paths_to_try:
        if path.exists():
            content = render_markdown_simple(path.read_text())
            title = page_name.replace('-', ' ').title()
            return HTMLResponse(content=create_page(title, content, get_nav_html()))

    # 404 page
    content = f"""
    <h1>Page Not Found</h1>
    <p>The page "{page_name}" was not found.</p>
    <p><a href="/docs/">Return to documentation home</a></p>
    """
    return HTMLResponse(content=create_page("Not Found", content, get_nav_html()), status_code=404)


@app.get("/docs/api/toc")
async def api_toc():
    """Get table of contents as JSON"""
    return JSONResponse(
        {
            "sections": [
                {"title": "Getting Started", "pages": ["quickstart", "installation", "first-steps"]},
                {"title": "User Guide", "pages": ["chat-interface", "pdf-analysis", "image-generation"]},
                {"title": "Technical", "pages": ["architecture", "api", "configuration", "troubleshooting"]},
            ]
        }
    )


@app.get("/docs/api/search")
async def api_search(q: str = ""):
    """Simple search functionality"""
    if not q:
        return JSONResponse({"results": []})

    results = []
    query = q.lower()

    # Search through markdown files
    for md_file in DOCS_ROOT.rglob("*.md"):
        try:
            content = md_file.read_text()
            if query in content.lower():
                # Get the line containing the match
                for i, line in enumerate(content.split('\n')):
                    if query in line.lower():
                        results.append(
                            {
                                "file": str(md_file.relative_to(DOCS_ROOT)),
                                "line": i + 1,
                                "text": line.strip()[:100] + "..." if len(line) > 100 else line.strip(),
                            }
                        )
                        if len(results) >= 10:
                            break
        except:
            continue

    return JSONResponse({"query": q, "results": results})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
