---
name: arxiv-query
description: 通过 arXiv 官方 API 与 RSS 源查询论文：扫当天/近期新论文、按关键字/分类/作者/日期精确检索、按 arXiv ID 拉取元数据，内置官方 3 秒限流规范避免被封。Use when the user asks to scan arXiv for today's or recent papers, search arXiv by keyword/category/author/date, fetch paper metadata by arXiv ID, or list new papers in a category.
user-invocable: true
metadata:
  api-base: "https://export.arxiv.org/api/query"
  rss-base: "https://rss.arxiv.org/rss"
  rate-limit: "官方建议连续请求间隔 3 秒；单次 max_results ≤ 30000（每次切片 ≤ 2000）"
  auth: "无需 API key，完全免费"
  requires: "bash + curl + python3（解析 Atom/RSS，仅标准库）"
---

# arXiv 查询技能

arXiv 官方接口是**免费、稳定、无需 API key** 的。本技能提供两个入口：**RSS 源**（每日扫描，零限流风险）和 **query API**（精确检索），并附一个封装好的辅助脚本 `scripts/arxiv.sh`，自动遵守官方限流规范。

## 何时使用

- 用户要求「看看今天/最近有什么论文」「扫一下 xx 分类」
- 按关键字、分类、作者、日期、arXiv ID 检索论文
- 需要批量拉一批论文的标题/摘要/元数据（低频、克制地）

## 选型：RSS vs API（先想清楚再动手）

| 需求 | 用哪个 | 为什么 |
|---|---|---|
| 扫「今天有什么新论文」（按分类） | **RSS**：`https://rss.arxiv.org/rss/<分类>` | 一次 GET 拿全部分类最新约 300 条，官方为订阅设计，不存在限流问题 |
| 按关键字/作者/日期/多条件组合检索 | **API**：`https://export.arxiv.org/api/query` | 支持 `search_query` 布尔语法与日期过滤 |
| 按已知 arXiv ID 查元数据 | **API** 的 `id_list` 参数 | 一个请求可带多个 ID（逗号分隔） |
| 大批量抓全量元数据 | **不要用 query API**，用 [OAI-PMH](https://info.arxiv.org/help/bulk_data.html) | 官方明确指定，query API 会被限流/封 IP |

## 前置条件

1. 网络可达 `export.arxiv.org`、`rss.arxiv.org`（国内网络一般直连可用；如被墙需代理，注意 curl 代理环境变量）
2. `python3` 可用（解析 Atom/RSS 只用标准库 `xml.etree`；辅助脚本已内置解析，无需装 feedparser/jq）
3. 大量使用前读一遍官方文档：[API 用户手册](https://info.arxiv.org/help/api/user-manual.html)（本地副本 `references/api_user_manual.txt`）、[RSS 说明](https://info.arxiv.org/help/rss.html)

## 快速开始（辅助脚本）

脚本位置：`<skill_base>/scripts/arxiv.sh`（skill 的 base 目录即 `~/.dsh/skills/arxiv-query/`）。加执行权限后可直接用；抓取内置自动重试（网络瞬时故障最多重试 3 次）：

```bash
S=~/.dsh/skills/arxiv-query/scripts/arxiv.sh

# 1) 扫今天 cs.AI 和 cs.LG 的新论文（RSS，零风险）—— 推荐日常用
$S today cs.AI cs.LG

# 2) 关键字检索（API；search_query 语法见下）
$S search 'cat:cs.AI AND ti:"large language model"' 10

# 3) 最近 3 天 cs.LG 提交的论文
$S since cs.LG 3 50

# 4) 某个分类最新的 20 篇
$S recent cs.CL 20

# 5) 按 ID 查元数据
$S id 2608.12325 2401.00001

# 6) 带摘要（追加 --abstract）
$S search 'abs:retrieval augmented generation' 5 --abstract
```

输出格式：`[日期] arXivID | 标题 | 分类`；原始 XML 存 `/tmp/arxiv_out.xml`（RSS 为 `/tmp/arxiv_rss_<cat>.xml`），需要摘要/作者等完整字段时用脚本参数或直接解析该文件。

## API 参考（export.arxiv.org/api/query）

### 请求参数

| 参数 | 说明 |
|---|---|
| `search_query` | 检索表达式（字段+布尔语法，见下）；留空则返回全部（配合 id_list） |
| `id_list` | 逗号分隔的 arXiv ID，直接拉取指定论文（按 ID 查时优先用它，能正确处理多版本） |
| `start` / `max_results` | 分页：0 起始；单次最大 30000，**每次切片 ≤ 2000** |
| `sortBy` | `relevance`（默认）\| `lastUpdatedDate` \| `submittedDate` |
| `sortOrder` | `ascending` \| `descending` |

### search_query 字段前缀与布尔语法

- 字段：`ti:`（标题）`au:`（作者）`abs:`（摘要）`cat:`（分类，如 `cat:cs.AI`）`all:`（全字段）
- 组合：`AND`、`OR`、`ANDNOT`；括号分组；短语用双引号，如 `ti:"large language model"`
- 例：`cat:cs.AI AND au:lecun AND abs:world model`
- 注意：`+` 在 URL 里代表空格，`:` `[` `]` `"` 需 URL 编码（辅助脚本已处理；手写 curl 时用 `--data-urlencode` 或编码）

### 按日期过滤（submittedDate）

格式：`[YYYYMMDDHHMM TO YYYYMMDDHHMM]`，**UTC 时间、精确到分钟**：

```
https://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+submittedDate:[202608130000+TO+202608152359]&sortBy=submittedDate&sortOrder=descending
```

**重要**：arXiv 按批次公告（美东时间周日~周四晚 ~20:00，即 UTC 周一~周五 00:00 前后）。「今天的新论文」通常落在最近 24–48 小时的窗口里，扫 `since <cat> 2` 或 `submittedDate` 用 48 小时窗口最稳妥。

## 限流与礼仪（务必遵守，避免被封 IP）

- **连续请求间隔 ≥ 3 秒**（官方原文："incorporate a 3 second delay"）；辅助脚本已内置，可用 `ARXIV_SLEEP` 调大
- 单次 `max_results` 上限 30000、**切片上限 2000**；超过 30000 直接 HTTP 400
- 命中 > 1000 的结果：官方建议细化查询或分小页取，别一次拉大结果集
- 大批量/全量元数据走 OAI-PMH（`https://export.arxiv.org/oai2`）而不是 query API
- 日常「扫今天」场景用 RSS 即可，几乎不占 API 配额
- 没有硬性请求配额，但暴力调用会被限流/封 IP；被封后过一段时间自动解封

## RSS 常用分类（`https://rss.arxiv.org/rss/<分类>`）

| 分类 | 含义 |
|---|---|
| `cs.AI` | 人工智能 |
| `cs.LG` | 机器学习（含 deep learning） |
| `cs.CL` | 计算语言学 / NLP |
| `cs.CV` | 计算机视觉 |
| `cs.NE` | 神经与进化计算 |
| `stat.ML` | 统计机器学习 |
| `cs.CR` | 密码学与安全 |
| `cs.IR` | 信息检索 |

完整分类表见 <https://arxiv.org/category_taxonomy>。RSS 每条含标题、arXiv ID（在 `<link>` 里）、摘要、分类、公告日期。

## 常见工作流示例

```bash
# 每日例行：扫关注的几个分类（一条命令，共 N 次请求）
$S today cs.AI cs.LG stat.ML

# 找最近一周某主题（先 RSS 扫一遍，再对感兴趣的 API 细查）
$S search 'cat:cs.CL AND abs:multimodal' 30 --abstract

# 追踪某位作者近一个月的工作
$S search 'au:"geoffrey hinton" AND submittedDate:[202607010000 TO 202608152359]' 20

# 拿到 ID 后读全文/做笔记：配合 read-paper 技能
#   read-paper https://arxiv.org/abs/2608.12325
```

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 响应是 HTML/超时 | 网络不通或被墙 | 检查代理；`curl -v` 看实际连接 |
| HTTP 400 | `max_results` > 30000，或参数非法 | 减小切片（≤2000） |
| 0 结果 | 日期窗口内无公告批次 | 放宽到 48 小时窗口（arXiv 按批次公告） |
| 被封（连串 403/拒绝） | 请求太频繁 | 停一段时间（分钟~小时级）再恢复，加大间隔 |
| 想拿作者/摘要完整字段 | 脚本只打印标题 | 解析 `/tmp/arxiv_out.xml`（Atom），或给脚本加 `--abstract` |
| RSS 里没有想要的关键词过滤 | RSS 只按分类 | 改用 API `search_query` |

## 已验证记录（2026-08 实测）

- `rss.cs.AI` 返回 HTTP 200、约 300 条 `<item>`，标题/链接/摘要齐全
- API `submittedDate` 日期过滤、`sortBy=submittedDate&sortOrder=descending` 均可用（cs.AI 五天窗口返回 700+ 条）
- 最新提交批次以 `published` 时间戳呈现，与 RSS 公告日期对应
- 参考文档：`references/api_user_manual.txt`（官方用户手册全文）、`references/rss.txt`（官方 RSS 说明）
