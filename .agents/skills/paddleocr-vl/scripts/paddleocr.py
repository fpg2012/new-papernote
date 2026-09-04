#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PaddleOCR-VL 在线 OCR —— 单文件版：PDF/图片 → Markdown，带结果缓存。

============================================================
命令行用法
============================================================
    python paddleocr.py paper.pdf                  # OCR 整份 PDF -> output/
    python paddleocr.py paper.pdf --merge          # 合并为单个 combined.md
    python paddleocr.py paper.pdf -o out --no-cache   # 指定目录、强制重跑
    python paddleocr.py https://example.com/x.pdf  # 直接传 URL
    python paddleocr.py --list-models              # 查看已知模型

============================================================
作为库使用
============================================================
    from paddleocr import PaddleOCRClient, ResultCache, write_results

    cache = ResultCache(".cache/paddleocr-vl")  # 放当前工作目录下，勿用 ~/.cache
    key = cache.key_for("paper.pdf", "PaddleOCR-VL-1.6", {})
    records = cache.load_raw(key)                 # 1) 先查缓存
    if records is None:
        records = PaddleOCRClient().ocr("paper.pdf")   # 2) 没有就调 API
        cache.save_raw(key, records)
    write_results(records, "output", merge=True)  # 3) 导出 markdown

依赖: pip install requests
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import requests

# =====================================================================
# 配置
# =====================================================================
# 不内置任何 token。token 从环境变量 PADDLEOCR_TOKEN，或脚本所在目录的 .env 文件中读取
# （见 _load_dotenv）。可在同目录放 .env：PADDLEOCR_TOKEN=xxx ；不要提交到 git。


def _load_dotenv() -> None:
    """从脚本所在目录的 .env 文件读取配置进环境变量（不覆盖已存在的环境变量）。"""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

DEFAULT_MODEL = "PaddleOCR-VL-1.6"
KNOWN_MODELS = ("PaddleOCR-VL-1.6",)  # 仅展示用，不强制校验，可传新模型

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
# 默认缓存目录：相对当前工作目录（cwd），不写死 ~/.cache（该目录通常不可写）。
# 使用时建议始终显式传 --cache-dir 指定位置。
DEFAULT_CACHE_DIR = ".cache/paddleocr-vl"
DEFAULT_OUTPUT_DIR = "output"

__version__ = "0.1.0"

PathLike = Union[str, os.PathLike]
ProgressCallback = Callable[[str, dict], None]  # (state, data)


# =====================================================================
# 异常
# =====================================================================
class OCRJobError(RuntimeError):
    """任务提交失败或 OCR 任务本身失败。"""


# =====================================================================
# API 客户端：提交（本地文件 / URL）-> 轮询 -> 拉取 jsonl
# =====================================================================
class PaddleOCRClient:
    """PaddleOCR-VL 在线任务客户端。

    token 优先级: 构造参数 > 环境变量 PADDLEOCR_TOKEN > 脚本同目录 .env 文件。
    取不到时抛出 OCRJobError，提醒配置 token（不要硬编码进脚本）。
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = JOB_URL,
        model: str = DEFAULT_MODEL,
        optional_payload: Optional[dict] = None,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.token = token or os.environ.get("PADDLEOCR_TOKEN")
        if not self.token:
            raise OCRJobError(
                "未提供 PaddleOCR token：请设置环境变量 PADDLEOCR_TOKEN，"
                "或在脚本目录放一个 .env（PADDLEOCR_TOKEN=xxx）。不要把 token 硬编码进脚本。"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.optional_payload = optional_payload or {}
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    def _headers(self, json_mode: bool = False) -> dict:
        headers = {"Authorization": f"bearer {self.token}"}
        if json_mode:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _raise_for_status(resp: requests.Response, action: str) -> None:
        if resp.status_code != 200:
            raise OCRJobError(f"{action}失败 (HTTP {resp.status_code}): {resp.text[:500]}")

    # ------------------------------------------------------------------
    def submit(self, input_path_or_url: str) -> str:
        """提交任务，返回 jobId。input 为本地路径或 http(s) URL。"""
        if input_path_or_url.startswith("http"):
            payload = {
                "fileUrl": input_path_or_url,
                "model": self.model,
                "optionalPayload": self.optional_payload,
            }
            resp = self.session.post(
                self.base_url,
                json=payload,
                headers=self._headers(json_mode=True),
                timeout=self.timeout,
            )
        else:
            if not os.path.exists(input_path_or_url):
                raise FileNotFoundError(f"文件不存在: {input_path_or_url}")
            data = {
                "model": self.model,
                "optionalPayload": json.dumps(self.optional_payload),
            }
            with open(input_path_or_url, "rb") as f:
                resp = self.session.post(
                    self.base_url,
                    headers=self._headers(),
                    data=data,
                    files={"file": f},
                    timeout=self.timeout,
                )
        self._raise_for_status(resp, "提交任务")
        return resp.json()["data"]["jobId"]

    # ------------------------------------------------------------------
    def poll(
        self, job_id: str, on_progress: Optional[ProgressCallback] = None
    ) -> dict:
        """轮询直到 done / failed，返回最终结果的 data 字段（含 resultUrl）。"""
        while True:
            resp = self.session.get(
                f"{self.base_url}/{job_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            self._raise_for_status(resp, "查询任务")
            data = resp.json()["data"]
            state = data["state"]

            if state == "failed":
                raise OCRJobError(data.get("errorMsg") or "任务失败，无错误信息")
            if on_progress:
                on_progress(state, data)
            if state == "done":
                return data

            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    def fetch_jsonl(self, jsonl_url: str) -> list:
        """下载结果 jsonl，返回逐行解析后的记录列表。"""
        resp = self.session.get(jsonl_url, timeout=self.timeout)
        resp.raise_for_status()
        records = []
        for line in resp.text.strip().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------
    def ocr(
        self,
        input_path_or_url: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> list:
        """完整流程: 提交 -> 轮询 -> 拉取 jsonl，返回逐行记录。"""
        job_id = self.submit(input_path_or_url)
        data = self.poll(job_id, on_progress=on_progress)
        return self.fetch_jsonl(data["resultUrl"]["jsonUrl"])


# =====================================================================
# 缓存：按「输入内容哈希 + 模型参数」缓存原始结果，避免反复 OCR
# =====================================================================
class ResultCache:
    """按 key 存取 OCR 原始结果与导出清单。

    缓存 key = sha256(输入内容哈希 + 模型与 optionalPayload 指纹)
    - 本地文件按内容哈希（分块读取，适合大 PDF）
    - URL 按 URL 字符串哈希（URL 内容变化时请用 --no-cache 强制重跑）

    目录结构:
        <cache_dir>/<key>/raw.jsonl      原始 jsonl 记录（OCR 的唯一事实来源）
        <cache_dir>/<key>/manifest.json  上次导出清单（相对 output_dir 的路径）
    """

    def __init__(self, cache_dir: PathLike, enabled: bool = True) -> None:
        self.root = Path(cache_dir).expanduser()
        self.enabled = enabled

    # ------------------------------------------------------------------
    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def key_for_input(self, input_path_or_url: str) -> str:
        """输入内容哈希：本地文件按内容，URL 按字符串。"""
        if input_path_or_url.startswith("http"):
            return self._sha256(input_path_or_url.encode("utf-8"))
        h = hashlib.sha256()
        with open(input_path_or_url, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def key_for(
        self, input_path_or_url: str, model: str, optional_payload: Optional[dict]
    ) -> str:
        """完整缓存 key = 输入哈希 + (模型 + 参数) 指纹。"""
        payload_json = json.dumps(optional_payload or {}, sort_keys=True)
        fingerprint = self._sha256(f"{model}|{payload_json}".encode("utf-8"))
        return self._sha256(
            (self.key_for_input(input_path_or_url) + fingerprint).encode("utf-8")
        )

    # ------------------------------------------------------------------
    def entry_dir(self, key: str) -> Path:
        return self.root / key

    def raw_path(self, key: str) -> Path:
        return self.entry_dir(key) / "raw.jsonl"

    def manifest_path(self, key: str) -> Path:
        return self.entry_dir(key) / "manifest.json"

    # ------------------------------------------------------------------
    def load_raw(self, key: str) -> Optional[list]:
        """读取缓存的原始 jsonl 记录；无缓存返回 None。"""
        path = self.raw_path(key)
        if not path.exists():
            return None
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records or None

    def save_raw(self, key: str, records: list) -> Path:
        path = self.raw_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    # ------------------------------------------------------------------
    def load_manifest(self, key: str) -> Optional[dict]:
        path = self.manifest_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_manifest(self, key: str, manifest: dict) -> Path:
        path = self.manifest_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


# =====================================================================
# 导出：原始结果 -> 逐页 markdown + 图片（可合并 combined.md）
# =====================================================================
def _download(url: str, dest: Path, timeout: Optional[float]) -> None:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(resp.content)


def _rel(base: Path, path: Path) -> str:
    """返回相对 base 的 POSIX 路径，便于写进缓存清单。"""
    return str(path.relative_to(base)).replace(os.sep, "/")


def write_results(
    records: list,
    output_dir: PathLike,
    merge: bool = False,
    timeout: Optional[float] = None,
) -> dict:
    """把逐行记录导出为 markdown 与图片（纯函数，幂等）。

    records:    fetch_jsonl / cache.load_raw 返回的列表
    merge:      True 时把所有页合并为一个 combined.md
    返回:       {"files": [相对 output_dir 的路径...], "pages": n}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    merged_parts: list[str] = []
    page_num = 0

    for record in records:
        result = record["result"]
        for res in result.get("layoutParsingResults", []):
            markdown = res.get("markdown", {}) or {}

            # 1) 单页 markdown
            md_filename = output_dir / f"doc_{page_num}.md"
            md_filename.write_text(markdown.get("text", ""), encoding="utf-8")
            files.append(_rel(output_dir, md_filename))

            if merge:
                merged_parts.append(
                    f"<!-- page {page_num} -->\n\n{markdown.get('text', '')}"
                )

            # 2) markdown 内引用的图片
            for img_rel_path, img_url in (markdown.get("images") or {}).items():
                full = output_dir / img_rel_path
                _download(img_url, full, timeout)
                files.append(_rel(output_dir, full))

            # 3) 额外的输出图片
            for img_name, img_url in (res.get("outputImages") or {}).items():
                full = output_dir / f"{img_name}_{page_num}.jpg"
                _download(img_url, full, timeout)
                files.append(_rel(output_dir, full))

            page_num += 1

    if merge and merged_parts:
        combined = output_dir / "combined.md"
        combined.write_text("\n\n".join(merged_parts), encoding="utf-8")
        files.append(_rel(output_dir, combined))

    return {"files": files, "pages": page_num}


# =====================================================================
# 命令行
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddleocr",
        description="PaddleOCR-VL 在线 OCR：把 PDF/图片识别为 markdown，带结果缓存。",
        epilog="示例: python paddleocr.py paper.pdf --merge",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="本地文件路径或 http(s) URL（支持 PDF、图片等）",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"模型名（默认 {DEFAULT_MODEL}；"
            f"已知: {', '.join(KNOWN_MODELS)}，可传新模型）"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=(
            f"缓存目录（默认 {DEFAULT_CACHE_DIR}，相对当前工作目录；"
            "建议显式指定，不要依赖 ~/.cache）"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用缓存，强制重新 OCR",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="把所有页合并为一个 combined.md",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="API token（默认: 环境变量 PADDLEOCR_TOKEN，或脚本同目录 .env）",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="轮询任务状态的间隔秒数（默认 5）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="每个 HTTP 请求的超时秒数（默认不设超时）",
    )
    # optionalPayload 开关（默认全关，与官方示例一致）
    parser.add_argument(
        "--doc-orientation-classify",
        action="store_true",
        help="启用文档方向分类",
    )
    parser.add_argument(
        "--doc-unwarping",
        action="store_true",
        help="启用文档畸变矫正",
    )
    parser.add_argument(
        "--chart-recognition",
        action="store_true",
        help="启用图表识别",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="显示更详细进度（每页提取进度）",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出已知模型后退出",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _on_progress(state: str, data: dict, verbose: bool) -> None:
    if state == "pending":
        print("[ocr] 任务排队中 (pending)...")
    elif state == "running":
        if verbose:
            prog = data.get("extractProgress") or {}
            print(
                f"[ocr] 处理中: {prog.get('extractedPages')}/{prog.get('totalPages')} 页"
            )
        else:
            print("[ocr] 处理中 (running)...")
    elif state == "done":
        prog = data.get("extractProgress") or {}
        try:
            cost = int(prog.get("endTime", 0) or 0) - int(
                prog.get("startTime", 0) or 0
            )
        except (TypeError, ValueError):
            cost = None
        pages = prog.get("extractedPages", "?")
        print(f"[ocr] 完成: {pages} 页" + (f"，耗时 {cost}s" if cost is not None else ""))


def _print_files(files: Sequence[str], output_dir: Path) -> None:
    for rel in files:
        print(f"  {output_dir / rel}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        print("已知模型:")
        for name in KNOWN_MODELS:
            print(f"  {name}")
        return 0

    if not args.input:
        build_parser().print_usage(sys.stderr)
        print("paddleocr: error: 需要提供 input（本地文件路径或 URL）", file=sys.stderr)
        return 2

    if not args.input.startswith("http") and not os.path.exists(args.input):
        print(f"[错误] 文件不存在: {args.input}", file=sys.stderr)
        return 1

    optional_payload = {
        "useDocOrientationClassify": args.doc_orientation_classify,
        "useDocUnwarping": args.doc_unwarping,
        "useChartRecognition": args.chart_recognition,
    }

    output_dir = Path(args.output_dir)
    cache = ResultCache(args.cache_dir, enabled=not args.no_cache)
    if cache.enabled:
        print(f"[cache] 缓存目录: {cache.root.resolve()}")
    key = cache.key_for(args.input, args.model, optional_payload)

    # ---- 1. 尝试命中缓存 ---- #
    records = None
    if cache.enabled:
        records = cache.load_raw(key)
        if records is not None:
            print(f"[cache] 命中原始结果: {cache.entry_dir(key)}")
            manifest = cache.load_manifest(key)
            if manifest and all((output_dir / f).exists() for f in manifest["files"]):
                print(f"[cache] 导出文件已存在，直接复用（{len(manifest['files'])} 个文件）")
                _print_files(manifest["files"], output_dir)
                return 0

    # ---- 2. 没有缓存则调用 API ---- #
    if records is None:
        client = PaddleOCRClient(
            token=args.token,
            model=args.model,
            optional_payload=optional_payload,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
        print(f"[ocr] 提交任务: {args.input}（模型 {args.model}）")
        records = client.ocr(
            args.input, on_progress=lambda s, d: _on_progress(s, d, args.verbose)
        )
        if cache.enabled:
            saved = cache.save_raw(key, records)
            print(f"[cache] 已保存原始结果: {saved}")

    # ---- 3. 导出（幂等；缓存命中但导出缺失时也会走到这里） ---- #
    manifest = write_results(
        records, output_dir, merge=args.merge, timeout=args.timeout
    )
    if cache.enabled:
        cache.save_manifest(key, manifest)

    print(
        f"[ok] 共 {manifest['pages']} 页，导出 {len(manifest['files'])} 个文件 -> "
        f"{output_dir.resolve()}"
    )
    _print_files(manifest["files"], output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
