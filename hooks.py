"""MkDocs hooks：
1. 页面标题取正文第一个 H1（front matter 无标准 title 字段时）；
2. 将 front matter 中的论文元信息（领域/发表时间/链接等）注入为页面顶部美观的信息条。
"""

import html as _html
import re

# front matter 中要展示的字段（按此顺序）
META_KEYS = ("领域", "发表时间", "读文章时间", "链接", "Zotero", "页码说明")


def _escape(value):
    return _html.escape(str(value))


def _render_value(key, value):
    """链接字段渲染为可点击链接，其余字段转义为纯文本。"""
    text = _escape(value)
    if key == "链接" and str(value).startswith("http"):
        return f'<a href="{text}" target="_blank" rel="noopener">{text}</a>'
    return text


def on_page_markdown(markdown, page, config, files):
    # 1) 标题：优先取第一个 H1
    if "title" not in page.meta:
        m = re.search(r"^\s*#\s+(.+?)\s*$", markdown, re.MULTILINE)
        if m:
            page.title = m.group(1).strip()

    # 2) 元信息条：注入到第一个 H1 之后
    items = []
    for key in META_KEYS:
        value = page.meta.get(key)
        if value in (None, ""):
            continue
        items.append(
            f'<span class="pm-item"><span class="pm-key">{_escape(key)}</span>'
            f"{_render_value(key, value)}</span>"
        )
    if items:
        info = '<div class="paper-meta">' + "".join(items) + "</div>"
        m = re.search(r"^#\s+.+$", markdown, re.MULTILINE)
        if m:
            markdown = markdown[: m.end()] + "\n\n" + info + "\n\n" + markdown[m.end():]

    return markdown
