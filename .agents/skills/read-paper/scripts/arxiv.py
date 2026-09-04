#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arxiv 元信息小工具：输入 arxiv ID / abs URL / pdf URL，输出 JSON 元信息，可选下载 PDF。

用法:
    python3 arxiv.py 2401.00001                          # 元信息 JSON
    python3 arxiv.py https://arxiv.org/abs/2401.00001    # 同上，自动提取 ID
    python3 arxiv.py https://arxiv.org/pdf/2401.00001    # 同上
    python3 arxiv.py 2401.00001 --pdf /tmp/paper.pdf     # 顺便下载 PDF
    python3 arxiv.py --search "attention is all you need" # 只有标题时搜 arxiv（取第一条）

依赖: 仅标准库 (urllib + xml.etree)，无第三方包。
"""
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
ARX = "{http://arxiv.org/schemas/atom}"
UA = {"User-Agent": "read-paper-skill/1.0 (paper reading helper)"}


def extract_id(s):
    """从 arxiv ID / URL 中提取纯 ID，如 2401.00001。"""
    s = s.strip()
    if s.startswith("http"):
        s = s.split("?")[0]
        for part in s.split("/"):
            p = part.replace("arXiv:", "")
            if p and (p[0].isdigit() or p.lower().startswith("hep")):
                return p
        raise SystemExit(f"无法从 URL 中提取 arxiv ID: {s}")
    return s.replace("arXiv:", "")


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch(arxiv_id=None, search=None):
    params = {"max_results": 1}
    if search:
        params["search_query"] = 'ti:"%s"' % search
    else:
        params["id_list"] = arxiv_id
    url = API + "?" + urllib.parse.urlencode(params)
    root = ET.fromstring(_get(url))
    entry = root.find("a:entry", NS)
    if entry is None:
        what = search if search else arxiv_id
        raise SystemExit(f"arxiv 上没有找到: {what}")

    def txt(tag):
        el = entry.find("a:" + tag, NS)
        return el.text.strip() if el is not None and el.text else ""

    def arx(tag):
        el = entry.find(ARX + tag)
        return el.text.strip() if el is not None and el.text else ""

    authors = []
    for a in entry.findall("a:author", NS):
        name = a.find("a:name", NS)
        if name is not None and name.text:
            authors.append(name.text.strip())

    pdf_url, abs_url = "", ""
    for l in entry.findall("a:link", NS):
        if l.get("title") == "pdf":
            pdf_url = l.get("href", "")
        elif l.get("rel") == "alternate":
            abs_url = l.get("href", "")

    cats = [c.get("term") for c in entry.findall("a:category", NS) if c.get("term")]

    # 从条目自身的 <id>（如 http://arxiv.org/abs/1706.03762v7）反推纯 ID，去掉版本号
    entry_id = txt("id")
    real_id = ""
    if "/abs/" in entry_id:
        real_id = entry_id.split("/abs/")[-1]
        if "v" in real_id and real_id.rsplit("v", 1)[-1].isdigit():
            real_id = real_id.rsplit("v", 1)[0]
    if arxiv_id:
        real_id = arxiv_id

    return {
        "arxiv_id": real_id,
        "title": txt("title").replace("\n", " ").strip(),
        "authors": authors,
        "published": txt("published"),
        "updated": txt("updated"),
        "summary": txt("summary").replace("\n", " ").strip(),
        "primary_category": cats[0] if cats else "",
        "categories": cats,
        "pdf_url": pdf_url,
        "abs_url": abs_url,
        "doi": arx("doi"),
        "journal_ref": arx("journal_ref"),
    }


def download(url, path):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--search":
        meta = fetch(search=" ".join(args[1:]))
    else:
        aid = extract_id(args[0])
        meta = fetch(arxiv_id=aid)
    if "--pdf" in args:
        i = args.index("--pdf")
        out = args[i + 1] if i + 1 < len(args) else "paper.pdf"
        meta["downloaded_pdf"] = download(meta["pdf_url"], out)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
