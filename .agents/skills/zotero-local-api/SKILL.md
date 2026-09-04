---
name: zotero-local-api
description: 通过 Zotero 桌面客户端的 Local API（http://localhost:23119/api/）读取、搜索、导出和写入本机 Zotero 论文库。Use when the user asks to query, search, export, or modify their local Zotero library, locate attachment PDF file paths, run saved searches, count items, or list collections/tags via the local HTTP API.
user-invocable: true
metadata:
  zotero-api-version: "3"
  min-zotero-version: "7 (read) / 10 (write)"
  base-url: "http://localhost:23119/api/"
---

# Zotero Local API 技能

读取/写入**本机 Zotero 论文库**的官方接口：Zotero 桌面客户端内置的 Local HTTP API（Web API v3 的本地实现，数据直接来自本地 `zotero.sqlite`）。**离线可用、无速率限制、读取无需认证、比 Web API 快**。

## 何时使用

- 用户要求读取/搜索/统计本机 Zotero 库（条目、集合、标签、已保存搜索、附件）
- 需要定位附件 PDF 在磁盘上的路径（`file://` 重定向）
- 需要把 Zotero 数据导出为 BibTeX/RIS/CSL JSON 等格式
- 需要程序化写入/更新条目（Zotero 10+，需弹窗授权）

## 前置条件（先检查再动手）

1. **Zotero 桌面客户端必须正在运行**（Local API 是客户端内置服务）。未运行 → 端口无监听，`Connection refused`。
2. **必须开启本地服务开关**：Settings（设置）→ Advanced（高级）→ 勾选 *"Allow other applications on this computer to communicate with Zotero"*（允许其他应用程序与此计算机上的 Zotero 通信）。未开启 → `403 Forbidden: Local API is not enabled`。
   - 如果 Zotero 在运行但返回 403，请用户到 GUI 里勾选（立即生效，无需重启），不要直接改 prefs.js。
3. **注意本机代理环境变量**（常见于开发机）：若设置了 `ALL_PROXY`/`http_proxy`，curl 访问 localhost 会走代理导致 502。**所有 curl 必须加 `--noproxy '*'`**。

## 快速开始

```bash
B="curl -sS -m 10 --noproxy '*' -H 'Zotero-API-Version: 3'"

# 健康检查（200 + 版本头即正常；根路径 body 为 "Nothing to see here."）
$B -D - http://127.0.0.1:23119/api/ -o /dev/null

# 条目总数（format=versions 返回每行一个 key）
$B http://127.0.0.1:23119/api/users/0/items?format=versions | wc -l

# 最近加入的 5 条顶层条目
$B "http://127.0.0.1:23119/api/users/0/items/top?limit=5&sort=dateAdded&direction=desc" | jq -r '.[] | "[\(.data.itemType)] \(.data.title // .data.name)"'

# 全文搜索
$B "http://127.0.0.1:23119/api/users/0/items?q=segmentation&qmode=everything&limit=10" | jq -r '.[].data.title'

# 集合列表
$B "http://127.0.0.1:23119/api/users/0/collections?limit=50" | jq -r '.[] | "\(.key)  \(.data.name)"'

# 导出 BibTeX
$B "http://127.0.0.1:23119/api/users/0/items?format=bibtex&limit=50"
```

用户 ID 用 `0` 或真实数字 ID；组库用 `/groups/<groupID>`。

## 端点速查（前缀 `http://localhost:23119/api/users/0/`）

| 资源 | 端点 |
|---|---|
| 集合 | `/collections`、`/collections/top`、`/collections/<key>`、`/collections/<key>/collections` |
| 集合内条目 | `/collections/<key>/items`、`/collections/<key>/items/top` |
| 条目 | `/items`、`/items/top`、`/items/trash`、`/items/<key>`、`/items/<key>/children` |
| 我的发表 | `/publications/items` |
| 已保存搜索 | `/searches`、`/searches/<key>`、**`/searches/<key>/items`（本地独有：真正执行搜索）** |
| 标签 | `/tags`、`/tags/<name>`、`/items/<key>/tags`、`/items/tags`、`/collections/<key>/tags` |
| 分组库 | `/groups`、`/groups/<id>`（元数据受限、只读） |
| 附件文件 | `/items/<key>/file`（302→`file://`）、`/file/view`、`/file/view/url`（纯文本 URL） |

## 查询参数

- `q=` + `qmode=titleCreatorYear|everything`（`everything` 含全文索引；默认 titleCreatorYear）
- `itemType=`、`tag=`：布尔语法 `||`(OR) `-`(NOT) `&`(AND)、`tag=-foo` 排除；值需 URL 编码
- `sort=`（title/creator/dateAdded/dateModified/...）、`direction=asc|desc`、`limit=`、`start=`、`since=<version>` 增量拉取
- **本地 API 不分页**：不传 limit 时一次返回全部匹配（Web API 默认 25/上限 100）
- 返回格式 `format=json|keys|versions|bib|bibtex|biblatex|ris|csljson|csv|mods|rdf_*|tei|wikipedia|coins|bookmarks`
- `include=data,bib,citation`（json 模式下附带格式化参考文献）；bib 可配 `style=`（如 `apa`）、`linkwrap=1`、`locale=`
- 注意：**不支持 Atom**（`format=atom` → 501）

## 与 Web API 的差异（本地独有/不同）

- 读取零认证（不要转发端口）；无分页上限；无 Atom
- `/searches/<key>/items` 真正执行已保存搜索（Web API 只给元数据）
- `/items/<key>/file` 返回 302 → `file://` 本地路径（Web API 是授权下载）
- item type/field 端点返回**本地语言**名称；`/api/creatorFields` 例外恒为英文
- 无二进制差分上传：`PATCH .../file` → 405，整文件上传
- 响应带 `Zotero-Schema-Version` 头（本地 schema 版本）

## 写入（Zotero 10+）

写入需**运行时授权**的本地 key（与 zotero.org 的 key 无关、不能预创建）：

```bash
# 1) 申请授权（会弹窗；需先取 Server ID）
$B -H 'Content-Type: application/json' -X POST http://127.0.0.1:23119/api/local/authorize \
  -d '{"appName":"My Tool"}'          # 响应: {"key":"<32位>","remember":bool}
# 2) 携带 key 写请求
$B -H "Zotero-API-Key: <key>" -X POST http://127.0.0.1:23119/api/users/0/items -d '[...]'
```

规则要点：

- 弹窗选项 Allow / Always Allow / Deny；`remember:false` 时 key **一次性**（首写即失效），客户端要准备好 401 后重新授权；429（每分钟最多 5 次弹窗）+ `Retry-After`
- 写请求**必须**带 `Zotero-Server-ID` 头，否则 428；ID 不符 412（丢弃缓存重来）
- key 无范围限制（可写用户能编辑的所有库）；用户可在 Settings → Advanced 一键"Clear Write Authorizations"
- 支持 POST/PUT/PATCH/DELETE（items/collections/searches）、标签删除、全文写入、文件上传
- 文件上传为本地三段式：init（md5 去重，`{"exists":1}` 表示本地已有）→ POST 到 `/api/local/uploads/<uploadKey>`（1 小时有效，≤4GB，仅 `imported_file`/`imported_url`）→ register（`upload=<key>`，204）
- 全文：`PUT /items/<key>/fulltext`；批量 `POST /fulltext`（≤10 条）
- `Zotero-Write-Token` 支持但重启即忘
- 写入是普通本地修改：UI 立即可见，同步库下次同步上传

## Server ID 与版本（Zotero 10+）

- 每个响应带 `Zotero-Server-ID` 头，标识"哪份数据库"（存在数据库里，跨重启/升级不变）→ **按 ID 分区缓存**
- 本地对象版本由本地维护，与云端/其他实例**无任何关系**，勿跨实例比较；群组元数据除外（同步版本）
- 旧版 Zotero 返回同步版本（未同步为 0），10+ 为本地版本，客户端应丢弃历史存储的版本

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| Connection refused | Zotero 未运行 | 请用户启动 Zotero |
| 502 / 走代理 | 环境有 `ALL_PROXY` 等 | curl 加 `--noproxy '*'` |
| 403 "Local API is not enabled" | 开关未开 | 请用户在 Settings → Advanced 勾选（无需重启） |
| 400 | 用了其他用户 ID | 用 `0` 或真实 ID |
| 501 | format=atom | 用 json/keys/versions |
| 405 | PATCH file | 整文件上传 |
| 401 写入 | key 失效/一次性 | 重新 POST /api/local/authorize |

## 选型对比

- **Local API（首选）**：本机程序读写、离线、快、无需 key。缺点：需 Zotero 运行、仅本机。
- **Web API**（`api.zotero.org`）：远程访问/服务器端程序，需 zotero.org API key，走网络、有配额。
- **直读 SQLite**（`zotero.sqlite`）：不推荐，官方不保证结构稳定，需先关闭 Zotero 防损坏。

## 已验证记录（2026-08，Zotero 9.0.6）

- 根端点 200 + `Zotero-API-Version: 3`、`Zotero-Schema-Version: 42`
- 条目总数、集合列表、已保存搜索执行（无命中返回 `[]`）、附件 302 → `file:///home/<user>/Zotero/storage/<key>/....pdf` 均实测通过
- 参考原始官方文档文本：`references/local_api.txt`；辅助脚本：`scripts/zotero.sh`
