#!/usr/bin/env python3
"""
Documentation API Server - Enhanced version with better navigation and content handling
"""

import re
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# Initialize FastAPI app
app = FastAPI(
    title="Nemotron Chat Documentation", docs_url=None, redoc_url=None, openapi_url=None
)

# Documentation root directory
DOCS_ROOT = Path(__file__).parent

# Enhanced CSS for documentation pages
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

/* Mermaid diagram support */
.mermaid {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    text-align: center;
}
"""


def render_markdown_simple(content: str) -> str:
    """Enhanced markdown to HTML conversion with better code block handling"""
    lines = content.split('\n')
    html_lines = []
    in_code_block = False
    in_list = False
    in_blockquote = False
    code_language = ""
    code_content = []

    for i, line in enumerate(lines):
        # Code blocks with language support
        if line.strip().startswith('```'):
            if in_code_block:
                # Ending a code block
                if code_language == 'mermaid':
                    # Render mermaid diagram
                    html_lines.append('<div class="mermaid">')
                    html_lines.extend(code_content)
                    html_lines.append('</div>')
                else:
                    # Regular code block
                    html_lines.append(f'<pre class="language-{code_language}"><code>')
                    for code_line in code_content:
                        escaped_line = (
                            code_line.replace('&', '&amp;')
                            .replace('<', '&lt;')
                            .replace('>', '&gt;')
                        )
                        html_lines.append(escaped_line)
                    html_lines.append('</code></pre>')

                in_code_block = False
                code_language = ""
                code_content = []
            else:
                # Starting a code block
                # Extract language if specified
                lang_match = re.match(r'^```(\w+)', line.strip())
                if lang_match:
                    code_language = lang_match.group(1)
                else:
                    code_language = ""
                in_code_block = True
                code_content = []
            continue

        if in_code_block:
            # Collect code block content
            code_content.append(line)
            continue

        # Blockquotes
        if line.strip().startswith('>'):
            if not in_blockquote:
                html_lines.append('<blockquote>')
                in_blockquote = True
            html_lines.append(line.strip()[1:].strip())
            # Check if next line is not a blockquote
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith('>'):
                html_lines.append('</blockquote>')
                in_blockquote = False
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
        elif line.startswith('##### '):
            html_lines.append(f'<h5>{line[6:]}</h5>')
        # Lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{process_inline_formatting(line.strip()[2:])}</li>')
        # Empty line
        elif not line.strip():
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            if in_blockquote:
                html_lines.append('</blockquote>')
                in_blockquote = False
            html_lines.append('<br>')
        # Regular paragraph
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            if in_blockquote:
                html_lines.append('</blockquote>')
                in_blockquote = False

            # Process the line for inline formatting
            processed_line = process_inline_formatting(line)

            # Regular paragraph
            html_lines.append(f'<p>{processed_line}</p>')

    # Close any open tags
    if in_list:
        html_lines.append('</ul>')
    if in_code_block:
        # Handle unclosed code blocks
        if code_language == 'mermaid':
            html_lines.append('<div class="mermaid">')
            html_lines.extend(code_content)
            html_lines.append('</div>')
        else:
            html_lines.append(f'<pre class="language-{code_language}"><code>')
            for code_line in code_content:
                escaped_line = (
                    code_line.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                )
                html_lines.append(escaped_line)
            html_lines.append('</code></pre>')
    if in_blockquote:
        html_lines.append('</blockquote>')

    return '\n'.join(html_lines)


def process_inline_formatting(text: str) -> str:
    """Process inline markdown formatting"""
    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # Bold **text**
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)

    # Italic *text* (but not **text**)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)

    # Inline code `text`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    return text


def create_page(title: str, content: str, nav_html: str = "") -> str:
    """Create a complete HTML page with mermaid support"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} - Nemotron Chat Documentation</title>
        <style>{DOCS_CSS}</style>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    primaryColor: '#76b900',
                    primaryTextColor: '#fff',
                    primaryBorderColor: '#3d4147',
                    lineColor: '#76b900',
                    secondaryColor: '#1e2329',
                    tertiaryColor: '#282c34',
                    background: '#0e1117',
                    mainBkg: '#1e2329',
                    secondBkg: '#282c34',
                    tertiaryBkg: '#3d4147',
                    textColor: '#b8bcc8',
                    lineColor: '#76b900',
                    borderColor: '#3d4147'
                }}
            }});
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Nemotron Chat Documentation</h1>
        </div>
        {nav_html}
        <div class="content">
            {content}
        </div>
    </body>
    </html>
    """


def get_nav_html() -> str:
    """Get enhanced navigation HTML"""
    return """
    <div class="nav">
        <ul>
            <li><a href="/docs/">🏠 Home</a></li>
            <li><a href="/docs/getting-started">🚀 Getting Started</a></li>
            <li><a href="/docs/user-guide">📖 User Guide</a></li>
            <li><a href="/docs/architecture">🏗️ Architecture</a></li>
            <li><a href="/docs/api">🔧 API Reference</a></li>
            <li><a href="/docs/configuration">⚙️ Configuration</a></li>
            <li><a href="/docs/deployment">🚢 Deployment</a></li>
            <li><a href="/docs/developer">👨‍💻 Developer</a></li>
            <li><a href="/docs/faq">❓ FAQ</a></li>
            <li><a href="/docs/troubleshooting">🐛 Troubleshooting</a></li>
        </ul>
    </div>
    """


def get_section_pages() -> Dict[str, List[str]]:
    """Get all pages organized by section"""
    sections = {
        "getting-started": ["quickstart", "installation", "first-steps"],
        "user-guide": [
            "chat-interface",
            "pdf-analysis",
            "pdf_context_switching",
            "image-generation",
            "image-upload-vlm",
            "search-features",
            "system_prompt_pattern",
        ],
        "architecture": ["overview", "services"],
        "api": ["services", "controllers", "tools", "streaming"],
        "configuration": ["environment", "models"],
        "deployment": ["docker"],
        "developer": ["batch-processing"],
    }
    return sections


def get_section_content(section: str) -> str:
    """Get section overview content"""
    section_overviews = {
        "getting-started": """
        <h1>Getting Started</h1>
        <p>Welcome to Nemotron Chat! Get up and running quickly with these guides.</p>
        <div class="feature-grid">
            <div class="feature-card">
                <h3>🚀 Quick Start</h3>
                <p>Get Nemotron running in 5 minutes with Docker.</p>
                <a href="/docs/getting-started/quickstart">View Guide →</a>
            </div>
            <div class="feature-card">
                <h3>📦 Installation</h3>
                <p>Detailed installation instructions for all platforms.</p>
                <a href="/docs/getting-started/installation">View Guide →</a>
            </div>
            <div class="feature-card">
                <h3>👋 First Steps</h3>
                <p>Your first conversation with Nemotron.</p>
                <a href="/docs/getting-started/first-steps">View Guide →</a>
            </div>
        </div>
        """,
        "user-guide": """
        <h1>User Guide</h1>
        <p>Learn how to use all of Nemotron's features effectively.</p>
        <div class="feature-grid">
            <div class="feature-card">
                <h3>💬 Chat Interface</h3>
                <p>Master the conversation features.</p>
                <a href="/docs/user-guide/chat-interface">Learn More →</a>
            </div>
            <div class="feature-card">
                <h3>📄 PDF Analysis</h3>
                <p>Analyze documents intelligently.</p>
                <a href="/docs/user-guide/pdf-analysis">Learn More →</a>
            </div>
            <div class="feature-card">
                <h3>🎨 Image Generation</h3>
                <p>Create images with AI.</p>
                <a href="/docs/user-guide/image-generation">Learn More →</a>
            </div>
            <div class="feature-card">
                <h3>📷 Image Analysis</h3>
                <p>Analyze uploaded images with vision models.</p>
                <a href="/docs/user-guide/image-upload-vlm">Learn More →</a>
            </div>
            <div class="feature-card">
                <h3>🔍 Search Features</h3>
                <p>Web search and information retrieval.</p>
                <a href="/docs/user-guide/search-features">Learn More →</a>
            </div>
            <div class="feature-card">
                <h3>⚙️ System Prompts</h3>
                <p>Advanced prompt management patterns.</p>
                <a href="/docs/user-guide/system_prompt_pattern">Learn More →</a>
            </div>
        </div>
        """,
        "architecture": """
        <h1>Architecture</h1>
        <p>Understand Nemotron's technical architecture and design patterns.</p>
        <ul>
            <li><a href="/docs/architecture/overview">Architecture Overview</a> - System design and patterns</li>
            <li><a href="/docs/architecture/services">Services Architecture</a> - Service layer details</li>
        </ul>
        """,
        "api": """
        <h1>API Reference</h1>
        <p>Complete API documentation for developers.</p>
        <ul>
            <li><a href="/docs/api/services">Services API</a> - Service layer reference</li>
            <li><a href="/docs/api/controllers">Controllers API</a> - Controller reference</li>
            <li><a href="/docs/api/tools">Tools API</a> - Tool system documentation</li>
            <li><a href="/docs/api/streaming">Streaming API</a> - Response streaming</li>
        </ul>
        """,
        "configuration": """
        <h1>Configuration</h1>
        <p>Configure Nemotron for your environment.</p>
        <ul>
            <li><a href="/docs/configuration/environment">Environment Variables</a> - All configuration options</li>
            <li><a href="/docs/configuration/models">Model Configuration</a> - LLM model setup</li>
        </ul>
        """,
        "deployment": """
        <h1>Deployment</h1>
        <p>Deploy Nemotron to production.</p>
        <ul>
            <li><a href="/docs/deployment/docker">Docker Deployment</a> - Container deployment guide</li>
        </ul>
        """,
        "developer": """
        <h1>Developer Guide</h1>
        <p>Advanced development topics and implementation details.</p>
        <ul>
            <li><a href="/docs/developer/batch-processing">Batch Processing</a> - PDF batch processing implementation</li>
        </ul>
        """,
    }
    return section_overviews.get(
        section, f"<h1>{section.title()}</h1><p>Documentation section.</p>"
    )


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
    index_path = DOCS_ROOT / "index.md"
    if index_path.exists():
        content = render_markdown_simple(index_path.read_text())
    else:
        content = get_home_content()

    return HTMLResponse(content=create_page("Home", content, get_nav_html()))


def get_home_content():
    """Get default home page content"""
    return """
    <h1>Welcome to Nemotron Chat</h1>
    <p class="lead">A production-ready conversational AI platform powered by NVIDIA's advanced language models.</p>

    <div class="feature-grid">
        <div class="feature-card">
            <h3>🚀 Getting Started</h3>
            <ul>
                <li><a href="/docs/getting-started/quickstart">Quick Start Guide</a> - Get up and running in 5 minutes</li>
                <li><a href="/docs/getting-started/installation">Installation Guide</a> - Detailed setup instructions</li>
                <li><a href="/docs/getting-started/first-steps">First Steps</a> - Your first chat session</li>
            </ul>
        </div>

        <div class="feature-card">
            <h3>💡 Key Features</h3>
            <ul>
                <li><a href="/docs/user-guide/chat-interface">Chat Interface</a> - Natural conversations with AI</li>
                <li><a href="/docs/user-guide/pdf-analysis">PDF Analysis</a> - Intelligent document processing</li>
                <li><a href="/docs/user-guide/image-generation">Image Generation</a> - Create images with AI</li>
                <li><a href="/docs/user-guide/image-upload-vlm">Image Analysis</a> - Analyze uploaded images</li>
                <li><a href="/docs/user-guide/search-features">Search Tools</a> - Web and knowledge search</li>
            </ul>
        </div>

        <div class="feature-card">
            <h3>🔧 Technical Docs</h3>
            <ul>
                <li><a href="/docs/architecture/overview">Architecture Overview</a> - System design</li>
                <li><a href="/docs/api/services">API Reference</a> - Service and controller APIs</li>
                <li><a href="/docs/configuration/environment">Configuration</a> - Environment setup</li>
                <li><a href="/docs/developer/batch-processing">Batch Processing</a> - Advanced PDF handling</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>🤖 NVIDIA Nemotron</h2>
        <p>Build enterprise agentic AI with benchmark-winning open reasoning and multimodal foundation models. This app is powered by NVIDIA's state-of-the-art language models, offering specialized models for different use cases:</p>
        <ul>
            <li><strong>llama-3.1-nemotron-nano-8b-v1 (fast)</strong> - Leading reasoning and agentic AI accuracy model for PC and edge.</li>
            <li><strong>llama-3.3-nemotron-super-49b-v1 (llm)</strong> - High efficiency model with leading accuracy for reasoning, tool calling, chat, and instruction following.</li>
            <li><strong>llama-3.1-nemotron-ultra-253b-v1 (ultra)</strong> - Superior inference efficiency with highest accuracy for scientific and complex math reasoning, coding, tool calling, and instruction following.</li>
            <li><strong>llama-3.1-nemotron-nano-vl-8b-v1 (vlm)</strong> - Multi-modal vision-language model that understands text/img and creates informative responses</li>
        </ul>
    </div>
    """


# Section landing pages
@app.get("/docs/{section}")
async def section_page(section: str):
    """Handle section landing pages"""
    sections = get_section_pages()

    if section in sections:
        content = get_section_content(section)
        return HTMLResponse(
            content=create_page(section.title(), content, get_nav_html())
        )
    else:
        # Try to find as a standalone page
        return await get_page(section)


# Specific documentation pages
@app.get("/docs/{section}/{page_name}")
async def get_section_page(section: str, page_name: str):
    """Get a specific documentation page within a section"""
    file_path = DOCS_ROOT / section / f"{page_name}.md"

    if file_path.exists():
        content = render_markdown_simple(file_path.read_text())
        title = page_name.replace('-', ' ').replace('_', ' ').title()
        return HTMLResponse(content=create_page(title, content, get_nav_html()))

    # 404 page
    content = f"""
    <h1>Page Not Found</h1>
    <p>The page "{section}/{page_name}" was not found.</p>
    <p><a href="/docs/">Return to documentation home</a></p>
    """
    return HTMLResponse(
        content=create_page("Not Found", content, get_nav_html()), status_code=404
    )


@app.get("/docs/{page_name}")
async def get_page(page_name: str):
    """Get a standalone documentation page"""
    # First try direct file in docs root
    file_path = DOCS_ROOT / f"{page_name}.md"
    if file_path.exists():
        content = render_markdown_simple(file_path.read_text())
        title = page_name.replace('-', ' ').replace('_', ' ').title()
        return HTMLResponse(content=create_page(title, content, get_nav_html()))

    # If not found in root, search through all subdirectories
    for subdir in DOCS_ROOT.iterdir():
        if subdir.is_dir():
            potential_file = subdir / f"{page_name}.md"
            if potential_file.exists():
                content = render_markdown_simple(potential_file.read_text())
                title = page_name.replace('-', ' ').replace('_', ' ').title()
                return HTMLResponse(content=create_page(title, content, get_nav_html()))

    # 404 page
    content = f"""
    <h1>Page Not Found</h1>
    <p>The page "{page_name}" was not found.</p>
    <p><a href="/docs/">Return to documentation home</a></p>
    """
    return HTMLResponse(
        content=create_page("Not Found", content, get_nav_html()), status_code=404
    )


# API endpoints
@app.get("/docs/api/toc")
async def api_toc():
    """Get table of contents as JSON"""
    sections = get_section_pages()
    toc = []

    for section, pages in sections.items():
        toc.append(
            {
                "title": section.replace('-', ' ').title(),
                "path": section,
                "pages": [
                    {
                        "title": page.replace('-', ' ').replace('_', ' ').title(),
                        "path": page,
                    }
                    for page in pages
                ],
            }
        )

    return JSONResponse({"sections": toc})


@app.get("/docs/api/search")
async def api_search(q: str = ""):
    """Enhanced search functionality"""
    if not q:
        return JSONResponse({"results": []})

    results = []
    query = q.lower()

    # Search through all markdown files
    for md_file in DOCS_ROOT.rglob("*.md"):
        if md_file.name == "README.md":
            continue

        try:
            content = md_file.read_text()
            if query in content.lower():
                # Get context around matches
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if query in line.lower():
                        # Get surrounding context
                        start = max(0, i - 1)
                        end = min(len(lines), i + 2)
                        context = '\n'.join(lines[start:end])

                        results.append(
                            {
                                "file": str(md_file.relative_to(DOCS_ROOT)),
                                "title": md_file.stem.replace('-', ' ')
                                .replace('_', ' ')
                                .title(),
                                "line": i + 1,
                                "context": (
                                    context[:200] + "..."
                                    if len(context) > 200
                                    else context
                                ),
                                "url": f"/docs/{str(md_file.relative_to(DOCS_ROOT)).replace('.md', '')}",
                            }
                        )

                        if len(results) >= 20:  # Limit results
                            break

                if len(results) >= 20:
                    break
        except Exception:
            continue

    return JSONResponse({"query": q, "results": results})


@app.get("/docs/api/all-pages")
async def api_all_pages():
    """Get all available pages"""
    pages = []

    for md_file in DOCS_ROOT.rglob("*.md"):
        if md_file.name == "README.md":
            continue

        rel_path = md_file.relative_to(DOCS_ROOT)
        pages.append(
            {
                "path": str(rel_path).replace('.md', ''),
                "title": md_file.stem.replace('-', ' ').replace('_', ' ').title(),
                "section": (
                    rel_path.parent.name if rel_path.parent.name != '.' else 'root'
                ),
            }
        )

    return JSONResponse({"pages": sorted(pages, key=lambda x: x["path"])})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
