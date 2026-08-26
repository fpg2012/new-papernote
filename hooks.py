"""MkDocs hooks：
1. 页面标题取正文第一个 H1（front matter 无标准 title 字段时）；
2. 将 front matter 中的论文元信息（领域/发表时间/链接等）注入为页面顶部美观的信息条；
3. 在首页（index.md）的 <!-- INDEX-BEGIN/END --> 占位处自动生成笔记索引表格。
"""

import glob
import html as _html
import os
import re

import yaml

# front matter 中要展示的字段（按此顺序）
META_KEYS = ("领域", "发表时间", "读文章时间", "链接", "Zotero", "页码说明")


def _escape(value):
    return _html.escape(str(value))


def _render_value(key, value, meta):
    """链接字段渲染为可点击链接（显示为 分类-ID 简洁格式），其余字段转义。"""
    if key == "链接" and str(value).startswith("http"):
        href = _escape(value)
        label = href
        m = re.search(r"abs/([\d.]+)", str(value))
        if m:
            cat = meta.get("arXiv分类")
            label = f"{cat}-{m.group(1)}" if cat else m.group(1)
        return f'<a href="{href}" target="_blank" rel="noopener">{_escape(label)}</a>'
    return _escape(value)


def _inject_meta_bar(markdown, page):
    """把 front matter 元信息作为胶囊条注入到第一个 H1 之后。"""
    items = []
    for key in META_KEYS:
        value = page.meta.get(key)
        if value in (None, ""):
            continue
        items.append(
            f'<span class="pm-item"><span class="pm-key">{_escape(key)}</span>'
            f"{_render_value(key, value, page.meta)}</span>"
        )
    if not items:
        return markdown
    info = '<div class="paper-meta">' + "".join(items) + "</div>"
    m = re.search(r"^#\s+.+$", markdown, re.MULTILINE)
    if m:
        markdown = markdown[: m.end()] + "\n\n" + info + "\n\n" + markdown[m.end():]
    return markdown


def _build_index_table(docs_dir):
    """从各笔记 front matter 生成索引表格 markdown。"""
    rows = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "2*.md"))):
        text = open(path, encoding="utf-8").read()
        if not text.startswith("---"):
            continue
        try:
            meta = yaml.safe_load(text.split("---", 2)[1])
        except Exception:
            continue
        link = str(meta.get("链接", ""))
        m = re.search(r"abs/([\d.]+)", link)
        if not m:
            continue
        aid = m.group(1)
        pub = str(meta.get("发表时间", ""))
        d = re.search(r"\d{4}-\d{2}-\d{2}", pub)
        rows.append(
            {
                "title": str(meta.get("标题", "")).replace("|", "\\|"),
                "read": str(meta.get("读文章时间", ""))[:10],
                "date": d.group(0) if d else "",
                "aid": aid,
                "cat": str(meta.get("arXiv分类", "?")),
                "file": os.path.basename(path),
            }
        )
    # 按生成日期（读文章时间）倒序；同日内的按文件名保持稳定顺序
    rows.sort(key=lambda r: r["file"])
    rows.sort(key=lambda r: r["read"], reverse=True)

    lines = ["| 标题 | 生成日期 | 文章日期 | arXiv |", "| --- | --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| [{r['title']}]({r['file']}) | {r['read']} | {r['date']} | "
            f"[{r['cat']}-{r['aid']}](https://arxiv.org/abs/{r['aid']}) |"
        )
    return "\n".join(lines)


def on_page_markdown(markdown, page, config, files):
    # 1) 标题：优先取第一个 H1
    if "title" not in page.meta:
        m = re.search(r"^\s*#\s+(.+?)\s*$", markdown, re.MULTILINE)
        if m:
            page.title = m.group(1).strip()

    # 2) 元信息条
    markdown = _inject_meta_bar(markdown, page)

    # 3) 首页索引表格
    if page.file.name == "index":
        table = _build_index_table(config["docs_dir"])
        markdown = re.sub(
            r"<!-- INDEX-BEGIN -->.*?<!-- INDEX-END -->",
            f"<!-- INDEX-BEGIN -->\n{table}\n<!-- INDEX-END -->",
            markdown,
            flags=re.S,
        )

    return markdown
