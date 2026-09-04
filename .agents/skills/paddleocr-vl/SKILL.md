---
name: paddleocr-vl
description: PaddleOCR-VL 在线 OCR：把 PDF/图片整篇识别为 Markdown（逐页或合并），带本地结果缓存，避免重复 OCR。Use when the user asks to OCR a PDF or image into markdown/text, extract text from scanned or image-based documents, convert papers to markdown, or reuse previously cached OCR results.
user-invocable: true
metadata:
  model: "PaddleOCR-VL-1.6"
  api: "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
  default-cache-dir: "./.cache/paddleocr-vl（相对当前工作目录；勿用 ~/.cache）"
  requires: "python3 + requests"
---

# PaddleOCR-VL 在线 OCR 技能

单文件工具：把 **PDF / 图片 → Markdown**（官方 PaddleOCR-VL 在线 API），带**结果缓存**（按文件内容哈希，命中后不再调 API）。

## 脚本位置

`scripts/paddleocr.py` —— 单文件、零依赖（仅 requests）。可当命令行用，也可 `import` 当库用。

## 何时使用

- 用户要求把 PDF / 扫描件 / 图片 OCR 成 Markdown 或文本
- 需要把论文（尤其扫描版）转成可读文本 / 喂给 LLM
- 同一文件需要反复取 OCR 结果（缓存避免重复计费）

## 前置条件

1. `pip install requests`
2. 网络可达 `https://paddleocr.aistudio-app.com`（在线 API）
3. **API token 不内嵌在脚本里**。读取顺序：`--token` 参数 > 环境变量 `PADDLEOCR_TOKEN` > 脚本同目录 `.env`。首次使用前：

   ```bash
   cd <skill_base>/scripts
   cp .env.example .env     # 然后编辑 .env 填入真实 PADDLEOCR_TOKEN
   ```

   `.env` 含密钥，已加入 `.gitignore`，切勿提交。

## 命令行用法

```bash
PY=<skill_base>/scripts/paddleocr.py

python $PY paper.pdf --cache-dir ./.cache/paddleocr-vl      # OCR 整份 PDF -> ./output/
python $PY paper.pdf --cache-dir ./.cache/paddleocr-vl --merge   # 合并为单个 combined.md
python $PY paper.pdf -o out --no-cache                      # 指定输出目录、绕过缓存强制重跑
python $PY https://example.com/doc.pdf --cache-dir ./.cache/paddleocr-vl  # 直接传 URL
python $PY --list-models                                    # 查看已知模型
```

> **缓存目录**：每次调用都显式带 `--cache-dir`，并放在**当前工作目录**下（如 `./.cache/paddleocr-vl`）。
> 不要使用 `~/.cache` —— 该目录在很多环境下不可写，会导致缓存保存失败。

常用参数：`-m/--model`、`-o/--output-dir`、`--cache-dir`、`--no-cache`、`--merge`、`--token`、`--poll-interval`、`--timeout`、`--doc-unwarping`、`--chart-recognition`、`-v`、`--list-models`、`--version`。

## 作为库使用

```python
import sys
sys.path.insert(0, "<skill_base>/scripts")
import paddleocr  # 单文件模块

cache = paddleocr.ResultCache("./.cache/paddleocr-vl")  # 放当前工作目录下，勿用 ~/.cache
key = cache.key_for("paper.pdf", "PaddleOCR-VL-1.6", {})
records = cache.load_raw(key)                 # 1) 先查缓存
if records is None:
    records = paddleocr.PaddleOCRClient().ocr("paper.pdf")  # 2) 没有就调 API
    cache.save_raw(key, records)
manifest = paddleocr.write_results(records, "output", merge=True)  # 3) 导出
print(manifest["files"])
```

## 缓存机制

- 缓存 key = `sha256(输入内容哈希 + 模型/参数指纹)`；本地文件按内容分块哈希（适合大 PDF）
- 命中缓存：不调 API；导出文件也完整时连图片下载都跳过
- URL 输入按 URL 字符串哈希 —— **URL 内容变化时记得 `--no-cache`**
- 默认缓存目录为**当前工作目录**下的 `.cache/paddleocr-vl`（相对路径，不写死 `~/.cache`）
- **调用时始终显式传 `--cache-dir`**，放在当前工作目录下（如 `./.cache/paddleocr-vl`）；不要使用 `~/.cache`（通常不可写）
- 运行时会在开头打印 `[cache] 缓存目录: <path>`，方便确认实际缓存位置

## 注意事项

- **缓存目录放当前工作目录下**（如 `./.cache/paddleocr-vl`），不要写入 `~/.cache`（通常不可写）
- token 属敏感信息：脚本不内嵌 token，一律从环境变量 `PADDLEOCR_TOKEN` 或脚本同目录 `.env` 读取；`.env` 已 gitignore，切勿提交
- 在线 API，需要网络；失败时抛 `OCRJobError`（HTTP 状态 + 服务端错误信息）
- 大批量任务建议保持默认 5s 轮询间隔，避免请求过密
