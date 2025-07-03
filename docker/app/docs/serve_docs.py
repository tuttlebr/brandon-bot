#!/usr/bin/env python3
"""
Documentation API Server - Simplified and robust version
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# Initialize FastAPI app
app = FastAPI(title="Nano Chat Documentation", docs_url=None, redoc_url=None, openapi_url=None)

# Documentation root directory
DOCS_ROOT = Path(__file__).parent

# CSS for documentation pages - Matching Nano chat interface
DOCS_CSS = """
:root {
    --brand-color: #76b900;
    --bg-primary: #0e1117;
    --bg-secondary: #1e2329;
    --bg-tertiary: #282c34;
    --text-primary: #ffffff;
    --text-secondary: #b8bcc8;
    --text-muted: #8b92a4;
    --border-color: #3d4147;
    --code-bg: #1a1d23;
    --link-color: #76b900;
    --link-hover: #8fd11f;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0;
    line-height: 1.6;
    color: var(--text-primary);
    background: var(--bg-primary);
    min-height: 100vh;
}

/* Header styling similar to chat interface */
.header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 2rem;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--brand-color);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.header h1::before {
    content: "🤖";
    font-size: 1.75rem;
}

/* Navigation styling */
.nav {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    margin: 2rem;
}

.nav ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.nav li {
    margin: 0;
}

.nav a {
    color: var(--text-secondary);
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    transition: all 0.2s ease;
    display: inline-block;
}

.nav a:hover {
    background: var(--bg-tertiary);
    color: var(--brand-color);
    text-decoration: none;
}

/* Content area */
.content {
    padding: 2rem;
    max-width: 900px;
    margin: 0 auto;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary);
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: 600;
}

h1 {
    font-size: 2.5rem;
    border-bottom: 2px solid var(--brand-color);
    padding-bottom: 0.5rem;
}

h2 {
    font-size: 2rem;
    color: var(--brand-color);
}

h3 {
    font-size: 1.5rem;
}

p {
    color: var(--text-secondary);
    margin: 1rem 0;
}

/* Code styling */
code {
    background: var(--code-bg);
    color: var(--brand-color);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 0.9em;
}

pre {
    background: var(--code-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    margin: 1rem 0;
}

pre code {
    background: none;
    padding: 0;
    color: var(--text-secondary);
}

/* Blockquotes */
blockquote {
    border-left: 4px solid var(--brand-color);
    margin: 1rem 0;
    padding-left: 1rem;
    color: var(--text-secondary);
    font-style: italic;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1.5rem 0;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
}

th, td {
    border: 1px solid var(--border-color);
    padding: 0.75rem;
    text-align: left;
}

th {
    background: var(--bg-secondary);
    font-weight: 600;
    color: var(--brand-color);
}

td {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
}

tr:hover td {
    background: var(--bg-secondary);
}

/* Links */
a {
    color: var(--link-color);
    text-decoration: none;
    transition: color 0.2s ease;
}

a:hover {
    color: var(--link-hover);
    text-decoration: underline;
}

/* Lists */
ul, ol {
    color: var(--text-secondary);
    margin: 1rem 0;
    padding-left: 2rem;
}

li {
    margin: 0.5rem 0;
}

/* Sections */
.section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

/* Notes and warnings - similar to chat messages */
.note, .warning, .success {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid;
}

.note {
    background: rgba(118, 185, 0, 0.1);
    border-color: var(--brand-color);
    color: var(--text-secondary);
}

.warning {
    background: rgba(255, 193, 7, 0.1);
    border-color: #ffc107;
    color: var(--text-secondary);
}

.success {
    background: rgba(40, 167, 69, 0.1);
    border-color: #28a745;
    color: var(--text-secondary);
}

/* Feature cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.feature-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

.feature-card:hover {
    border-color: var(--brand-color);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(118, 185, 0, 0.2);
}

.feature-card h3 {
    color: var(--brand-color);
    margin-top: 0;
}

/* Responsive */
@media (max-width: 768px) {
    body {
        font-size: 0.95rem;
    }

    .nav ul {
        flex-direction: column;
        gap: 0.5rem;
    }

    .content {
        padding: 1rem;
    }

    h1 {
        font-size: 2rem;
    }

    h2 {
        font-size: 1.5rem;
    }
}

/* Animations */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.content > * {
    animation: fadeIn 0.3s ease-out;
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
        <title>{title} - Nano Chat Documentation</title>
        <style>{DOCS_CSS}</style>
    </head>
    <body>
        <div class="header">
            <h1>Nano Chat Documentation</h1>
        </div>
        {nav_html}
        <div class="content">
            {content}
        </div>
    </body>
    </html>
    """


def get_nav_html() -> str:
    """Get navigation HTML"""
    return """
    <div class="nav">
        <ul>
            <li><a href="/docs/">🏠 Home</a></li>
            <li><a href="/docs/quickstart">🚀 Quick Start</a></li>
            <li><a href="/docs/user-guide">📖 User Guide</a></li>
            <li><a href="/docs/architecture">🏗️ Architecture</a></li>
            <li><a href="/docs/api">🔧 API Reference</a></li>
            <li><a href="/docs/troubleshooting">🐛 Troubleshooting</a></li>
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
        <h1>Welcome to Nano Chat</h1>
        <p class="lead">A production-ready conversational AI platform powered by NVIDIA's advanced language models.</p>

        <div class="feature-grid">
            <div class="feature-card">
                <h3>🚀 Getting Started</h3>
                <ul>
                    <li><a href="/docs/quickstart">Quick Start Guide</a> - Get up and running in 5 minutes</li>
                    <li><a href="/docs/installation">Installation Guide</a> - Detailed setup instructions</li>
                    <li><a href="/docs/first-steps">First Steps</a> - Your first chat session</li>
                </ul>
            </div>

            <div class="feature-card">
                <h3>💡 Key Features</h3>
                <ul>
                    <li><a href="/docs/chat-interface">Chat Interface</a> - Natural conversations with AI</li>
                    <li><a href="/docs/pdf-analysis">PDF Analysis</a> - Intelligent document processing</li>
                    <li><a href="/docs/image-generation">Image Generation</a> - Create images with AI</li>
                    <li><a href="/docs/search-features">Search Tools</a> - Web and knowledge search</li>
                </ul>
            </div>

            <div class="feature-card">
                <h3>🔧 Technical Docs</h3>
                <ul>
                    <li><a href="/docs/architecture">Architecture Overview</a> - System design</li>
                    <li><a href="/docs/api">API Reference</a> - Service and controller APIs</li>
                    <li><a href="/docs/configuration">Configuration</a> - Environment setup</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>🤖 About Nano</h2>
            <p>Nano is powered by NVIDIA's state-of-the-art language models, offering three specialized models for different use cases:</p>
            <ul>
                <li><strong>Fast Model</strong> - Quick responses for simple queries</li>
                <li><strong>Standard Model</strong> - Balanced performance for general use</li>
                <li><strong>Intelligent Model</strong> - Advanced reasoning for complex tasks</li>
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
