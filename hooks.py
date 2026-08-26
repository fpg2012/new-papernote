"""MkDocs hook：将页面标题设为正文第一个 H1（front matter 无 title 时）。"""

import re


def on_page_markdown(markdown, page, config, files):
    if "title" not in page.meta:
        m = re.search(r"^\s*#\s+(.+?)\s*$", markdown, re.MULTILINE)
        if m:
            page.title = m.group(1).strip()
    return markdown
