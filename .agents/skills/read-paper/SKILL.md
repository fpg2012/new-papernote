---
name: read-paper
description: 读论文并产出结构化中文笔记（重点：AI 领域）。给定 arxiv 链接 / PDF 链接 / DOI / 本地 PDF / 标题，先查本机 Zotero（没有则添加条目与 PDF），再用 PaddleOCR 把论文转成 Markdown 通读，按固定结构写笔记（TL;DR、作者与单位、紧密相关工作、问题与方法、实验、图表清单、快速总结），正文用简明白话、关键公式完整带符号说明、所有引用标注原始页码 (P.页码)，最后把完整笔记输出给用户并归档到笔记文件夹。Use when the user asks to read a paper / 读文章 / 读论文 / 整理论文笔记 / 精读 / 论文讲解, especially AI papers.
user-invocable: true
metadata:
  notes-dir: "paper-notes"              # 笔记归档目录，相对会话工作目录
  language: "zh-CN"                     # 笔记正文语言
  filename-format: "日期_首作者_短标题.md"  # 例：2025-06-01_vaswani_attention-is-all-you-need.md
  depends-on:
    - paddleocr-vl                      # OCR：PDF -> Markdown
    - zotero-local-api                  # 查/写本机 Zotero 库
---

# 读文章（read-paper）技能

把一篇论文读透，产出一份**结构化中文笔记**：先管好文献（Zotero），再把论文 OCR 成 Markdown 认真读，最后按固定结构写笔记、带 `(P.页码)` 标注，输出给用户并归档。

## 何时使用

- 用户给出一篇论文（arxiv 链接、PDF 链接、DOI、本地 PDF 路径，或仅标题）要求「读一下 / 读文章 / 整理论文笔记 / 讲讲这篇论文」
- 重点场景：AI 领域论文（但也适用其他领域）
- 用户没有其他更具体的诉求时，默认按本技能的流程与结构执行

## 输入形式与准备

| 输入 | 说明 |
|---|---|
| arxiv 链接 | `https://arxiv.org/abs/2401.00001` 或 `.../pdf/2401.00001` |
| PDF 链接 | 任意可下载的 PDF URL |
| DOI | 用 DOI 反查元信息并定位 PDF |
| 本地 PDF | 直接使用；同时可作为 Zotero 入库依据 |
| 标题 | 用 arxiv API 搜标题定位 |

**每次执行前**：确认 Zotero 桌面客户端在运行、Local API 可用（见 `zotero-local-api` 技能的前置条件；curl 一律加 `--noproxy '*'`）。

## 工作流程

### 1. 解析输入，取元信息

- arxiv ID / URL → 运行 `scripts/arxiv.py` 拿 title、authors、published、primary_category、pdf_url、abstract（元信息也用于 Zotero 入库和 frontmatter）。
  ```bash
  PY=<skill_base>/scripts/arxiv.py
  python3 $PY 2401.00001                     # 元信息 JSON
  python3 $PY 2401.00001 --pdf /tmp/paper.pdf  # 顺便下载 PDF
  python3 $PY --search "attention is all you need"  # 只有标题时搜 arxiv
  ```
- DOI / 本地 PDF → 从论文首页（或 crossref）提取标题、作者、发表时间；PDF 用 OCR 前的临时路径。

### 2. 查 Zotero 里有没有这篇

用 `zotero-local-api` 技能搜索，**按标题词 + 作者**匹配（不止搜 URL）：

```bash
B="curl -sS -m 10 --noproxy '*' -H 'Zotero-API-Version: 3'"
# 按标题关键词搜（qmode=titleCreatorYear 已够；必要时 qmode=everything 全文搜）
$B "http://127.0.0.1:23119/api/users/0/items?q=<标题关键词>&qmode=titleCreatorYear&limit=10" \
  | jq -r '.[] | "\(.key)  [\(.data.itemType)] \(.data.title)  \(.data.date // "")"'
```

- **命中**（标题/作者明显匹配）→ 直接进入第 4 步，用库里的附件 PDF；若条目没有 PDF 附件，按第 3 步补一个附件。
- **未命中** → 第 3 步入库。

### 3. Zotero 里没有 → 添加条目与 PDF

写入需要**运行时授权**（会弹窗）：先 `POST /api/local/authorize`（appName 如 "DSH paper reader"），弹窗出现时**请用户点 Allow**（remember 用 true 可免重复授权）。授权 key 与 `Zotero-Server-ID` 头必须带上，详细规则见 `zotero-local-api` 技能「写入」一节。

流程：

1. **创建条目**（arxiv 用 `preprint` 类型，填 `title`、`creators`（作者）、`date`（如 2024-01-02）、`url`、`archiveID: arXiv:<id>`；非 arxiv 用对应类型）：
   ```bash
   $B -H "Zotero-API-Key: <key>" -H 'Content-Type: application/json' \
      -X POST http://127.0.0.1:23119/api/users/0/items \
      -d '[{"itemType":"preprint","title":"...","creators":[{"creatorType":"author","firstName":"...","lastName":"..."}],"date":"2024-01-02","url":"https://arxiv.org/abs/...","archiveID":"arXiv:..."}]'
   ```
2. **下载 PDF**（arxiv 直接下 `pdf_url`；其他先下载或复用输入文件）到临时目录。
3. **挂附件**（本地三段式：init → 上传 → register，只支持 `imported_file`；端点细节见 zotero-local-api 技能）：创建 `imported_file` 子附件条目，拿到它的 key 后按三段式把 PDF 传上去。
4. 若授权弹窗被拒绝或写入失败 → **跳过入库**，继续用下载的 PDF 做阅读，并在笔记 frontmatter 注明 `Zotero: 未入库`。

### 4. 定位本地 PDF

- 库里已有时：`GET /items/<key>/file`（302 → `file:///home/<user>/Zotero/storage/<key>/....pdf`），用 `curl -I -L --noproxy '*'` 取最终 `file://` 路径。
- 刚下载时：用第 3 步的临时 PDF 路径。

### 5. PaddleOCR 转 Markdown

用 `paddleocr-vl` 技能，**合并输出**方便通读：

```bash
PY=<skill_base>/scripts/paddleocr.py   # paddleocr-vl 的脚本
python3 $PY /path/to/paper.pdf --merge -o /path/to/ocr_out
```

- 结果：`ocr_out/combined.md`（全文）+ `ocr_out/doc_N.md`（逐页）+ `ocr_out/imgs/`（图表切图）。
- **缓存会自动命中**：同一份 PDF 之前 OCR 过就不重复调 API，直接复用缓存结果。
- 输出里 `<!-- page N -->` 是**页码标记（0 起算）**：`<!-- page 3 -->` = PDF 物理第 4 页。

### 6. 通读并核对

- 通读 `combined.md`，理解问题、方法、实验。**不要只扫摘要**；方法、公式、实验表、结论都要过。
- **公式务必与原文一致**：OCR 常把数学符号识别错。关键公式对照 `imgs/` 里的页面切图（可用 read_image 看）或原文 PDF 修正；符号的说明（含义、维度、下标）由你补全。
- 图表看不清时，用 `read_image` 看 `imgs/` 里对应的切图。
- 页面对不上时，`grep -n "<!-- page" combined.md` 定位分页边界。

### 7. 写笔记并归档

- 按「语言风格要求」和「笔记结构」写。
- **归档**：`mkdir -p <会话工作目录>/paper-notes/`，保存为 `<日期>_<首作者>_<短标题>.md`（日期=今天，首作者=姓氏小写，短标题=标题去停用词后的 3~5 个英文词、小写连字符；中文标题则取前几个字）。
- **输出**：把完整笔记正文（含 frontmatter）直接贴给用户，并给出归档路径。

## 语言风格要求（写作时逐条自查）

1. **人话**：简明、易懂的白话；不过度口语化（避免「贼快」「一堆」这类）。
2. **打比方要慎重**：能直接说清楚的东西就不要打比方；确有必要时比喻必须贴切且简短。
3. **公式原则**：
   - 一条公式能说清楚的事，不必用 10 句话绕。
   - 公式**必须完整**，且**每个符号都要有说明**（含义、类型/维度、下标含义）。
   - 不适合用公式表达的内容（动机、工程细节、直觉）不要硬塞公式。
   - **行间公式必须适配 Typora 规则**：`$$` 定界符各占一行，公式内容独立成行，即 `$$` 后必须换行、公式后换行再接 `$$`；**禁止写成单行 `$$...$$`**（Typora 无法正确渲染）。正确写法：
     ```
     $$
     E = mc^2
     $$
     ```
     行内公式（`$...$`）不受此限制，保持单行即可。
4. **页码**：所有引用原文内容/图表的地方，用 `(P.页码)` 标注。
5. **列表前必须留空行**（CommonMark/MkDocs 都要求）：凡是「段落 / 加粗的标题行（如 `**核心贡献：**`）」后面**直接**跟列表项、中间没有空行，列表符会被当成前一段落的 lazy-continuation 文本，渲染成一个 `<p>` 里的大段字面 `1.`/`-`，而不是真正的 `<ol>`/`<ul>`。因此**在任何一个列表块（`-`/`*`/`1.`）之前都要有独立空行**。正确写法：
   ```
   **核心贡献：**

   1. 结论一
   2. 结论二

   - 局限一
   ```
6. **front matter 的 YAML 必须合法**（MkDocs 只在 YAML 能解析时才剥离它；解析失败会把原文当正文渲染在页面顶部）：所有字段值**一律加英文双引号**，尤其值是标题、含 `:`（如英文副标题）、含 `（）`、`；`、`/` 等时。含 `:` 的标题不引号必炸。写完后自查首行 `---` 之间的 YAML 能否 `yaml.safe_load`。

## 笔记结构（模板，逐节产出）

```markdown
---
领域: "<所属领域，如 NLP / 大语言模型、3D 重建 / 神经渲染>"
发表时间: "<arxiv v1 日期 或 会议/期刊正式时间>"
读文章时间: "<今天，YYYY-MM-DD>"
标题: "<论文标题，含冒号也必须加双引号>"
链接: "<arxiv abs 或 DOI>"
Zotero: "<已入库 / 未入库>"
页码说明: "<页码约定，如 PDF 页码>"（可选）
---
（字段值一律加英文双引号；若笔记站点需要 `arXiv分类`，再加 `arXiv分类: "cs.CV"` 等）

# <论文标题>

## 1. TL;DR

一两句话：这篇文章**解决什么问题**、**核心贡献**是什么。

## 2. 作者与单位

- 作者1（单位 A）
- 作者2（单位 A；单位 B）

## 3. 紧密相关的工作与交叉点

直接相关的工作有哪些；这篇站在什么**交叉点**上——承接了谁、融合了哪几条线、与最近的关键工作差异在哪。

## 4. 问题与核心方法

### 4.1 要解决的问题

### 4.2 核心方法

全篇最详细的部分。按步骤写清楚方法；关键公式完整给出、逐个符号说明。先讲思路（一两句直觉，不绕弯），再给形式化定义。

## 5. 实验

- 实验设置：数据集、基线、指标。
- 每类实验分开写：**为了说明什么问题 → 结果如何 → 有什么有趣/反常的情况**（如某数据集上出乎意料、消融里某模块几乎没贡献、与直觉相反的现象）。

## 6. 图表清单

- Fig.1 <图的内容> (P.2)
- Table 1 <表的内容> (P.4)
（正文出现的图、表**全部**列出，带页码；包括附录里的 Fig.S1、Table S1 等。）

## 7. 快速总结

3~6 条：核心贡献、主要局限、值得借鉴/可延伸的点。
```

## 页码标注规则

- **优先用页面上印刷的页码**（正文、图注里可见的数字），写成 `(P.4)`。
- 页面没有印刷页码时，用 **PDF 物理页码**（从 1 开始），并在笔记 frontmatter 注明「页码为 PDF 页码」。
- OCR 的 `<!-- page N -->` 是 0 起算的 PDF 页索引：标注时换算成物理页码 N+1，再对印刷页码（通常一致，但带 supplementary 的论文要小心偏移）。

## 常见坑

- **Zotero 未运行 / 未开启 Local API**：端口无监听或 403 → 请用户启动 Zotero / 在 Settings→Advanced 勾选允许本地通信。
- **代理环境变量**：curl 访问 localhost 一律 `--noproxy '*'`。
- **OCR 公式失真**：关键公式必须对照原文修正，不能直接抄 OCR 输出。
- **行间公式写成单行 `$$...$$`**：Typora 无法渲染；必须 `$$` 各占一行、公式独立成行（见「语言风格要求」第 3 条）。写完笔记后用 `grep -c '\$\$[^$].*[^$]\$\$'` 自查残留。
- **列表前缺空行**：段落/加粗行后直接跟 `-`/`1.` 而无空行 → 渲染成一段文字而非列表（见「语言风格要求」第 5 条）。自查：任何非空非列表行后紧跟列表行，中间必须有空行。
- **front matter 值未加引号 / YAML 非法**：含 `:` 的标题不引号会让整个 YAML 解析失败，MkDocs 不剥离它、原文泄漏在页面顶部。所有值加双引号，写完用 `yaml.safe_load` 校验（见「语言风格要求」第 6 条）。
- **OCR 页码偏移**：0 起算 vs 物理页码 vs 印刷页码，写 (P.x) 前先换算确认。
- **重复 OCR**：同一 PDF 优先吃缓存，别重复调 API。
- **写入授权弹窗**：别绕过；用户拒绝就降级为「不入库 + 继续阅读」。

## 完成检查

- [ ] TL;DR 一两句话，含问题 + 贡献
- [ ] 作者、单位齐全
- [ ] 方法部分详细、公式完整且符号有说明
- [ ] 行间公式均为 Typora 兼容格式（`$$` 独占一行，无单行 `$$...$$`）
- [ ] 图表清单覆盖全部 Fig/Table，均带 (P.页码)
- [ ] 正文引用处有 (P.页码)
- [ ] front matter YAML 合法（所有值加引号，含 `:` 标题必引号，`yaml.safe_load` 通过）
- [ ] 所有列表块前都有空行（无「段落行紧跟列表行」的情况）
- [ ] 已归档到笔记文件夹 + 完整笔记已输出给用户
