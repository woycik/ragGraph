"""Convert README.md to a styled HTML file and open it in the browser."""

import re
import webbrowser
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).parent
README = ROOT / "README.md"
OUT = ROOT / "README.html"

md = MarkdownIt("gfm-like", {"html": True, "linkify": False, "typographer": True})

raw = README.read_text(encoding="utf-8")

# Fix image paths — make them relative to the HTML file location (same dir as README)
body = md.render(raw)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Multi-Agent Graph-RAG Explorer</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}

  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.7;
    margin: 0;
    padding: 2rem 1rem;
  }}

  .page {{
    max-width: 960px;
    margin: 0 auto;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 2.5rem 3rem;
  }}

  h1 {{ font-size: 2rem; border-bottom: 1px solid #30363d; padding-bottom: .5rem; color: #e6edf3; }}
  h2 {{ font-size: 1.4rem; border-bottom: 1px solid #21262d; padding-bottom: .3rem; margin-top: 2rem; color: #e6edf3; }}
  h3 {{ font-size: 1.1rem; color: #c9d1d9; }}

  a {{ color: #58a6ff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  p {{ margin: .6rem 0 1rem; }}

  code {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: .875em;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: .15em .4em;
    color: #e6edf3;
  }}

  pre {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    overflow-x: auto;
    font-size: .875em;
    line-height: 1.55;
  }}
  pre code {{
    background: none;
    border: none;
    padding: 0;
    font-size: 1em;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0 1.5rem;
    font-size: .9em;
  }}
  th {{
    background: #21262d;
    color: #c9d1d9;
    font-weight: 600;
    padding: .55rem .85rem;
    border: 1px solid #30363d;
    text-align: left;
  }}
  td {{
    padding: .5rem .85rem;
    border: 1px solid #30363d;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{ background: #1c2128; }}

  img {{
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid #30363d;
    margin: 1rem 0;
    display: block;
  }}

  hr {{
    border: none;
    border-top: 1px solid #30363d;
    margin: 2rem 0;
  }}

  ul, ol {{ padding-left: 1.6rem; }}
  li {{ margin: .25rem 0; }}

  blockquote {{
    border-left: 4px solid #3d88f5;
    padding: .3rem 1rem;
    margin: 1rem 0;
    background: #1c2128;
    border-radius: 0 6px 6px 0;
    color: #8b949e;
  }}
</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Rendered: {OUT}")
webbrowser.open(OUT.as_uri())
