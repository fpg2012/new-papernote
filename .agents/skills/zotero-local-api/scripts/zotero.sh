#!/usr/bin/env bash
# Zotero Local API 辅助脚本（配合 zotero-local-api skill 使用）
# 用法: ./zotero.sh <command> [args...]
#   status          健康检查（打印版本头）
#   count           条目总数
#   search <q>      全文/标题快速搜索（默认 qmode=everything）
#   recent [n]      最近加入的顶层条目（默认 5）
#   collections     集合列表
#   tags            标签列表
#   searches        已保存搜索列表
#   run-search <k>  执行已保存搜索（本地独有）
#   file <key>      附件磁盘路径（file://）
#   export <fmt>    导出全部条目（bibtex/ris/csljson/csv/...）
set -euo pipefail

BASE="${ZOTERO_API:-http://127.0.0.1:23119/api}"
USER_ID="${ZOTERO_USER_ID:-0}"
# 关键：绕过本机代理（ALL_PROXY 等），否则 localhost 请求会 502
CURL=(curl -sS -m 15 --noproxy '*' -H 'Zotero-API-Version: 3')
P="$BASE/users/$USER_ID"

die() { echo "error: $*" >&2; exit 1; }
need_jq() { command -v jq >/dev/null || die "需要 jq"; }

case "${1:-}" in
  status)
    "${CURL[@]}" -D - "$BASE/" -o /dev/null
    ;;
  count)
    "${CURL[@]}" "$P/items?format=versions" | wc -l
    ;;
  search)
    need_jq
    q="${2:?用法: zotero.sh search <q>}"
    "${CURL[@]}" "$P/items?q=$(printf %s "$q" | sed 's/ /%20/g')&qmode=everything&limit=${3:-10}" \
      | jq -r '.[] | "[\(.data.itemType)] \(.data.title // .data.name // .key)"'
    ;;
  recent)
    need_jq
    n="${2:-5}"
    "${CURL[@]}" "$P/items/top?limit=$n&sort=dateAdded&direction=desc" \
      | jq -r '.[] | "[\(.data.itemType)] \(.data.title // .data.name // .key)"'
    ;;
  collections)
    need_jq
    "${CURL[@]}" "$P/collections?limit=200" | jq -r '.[] | "\(.key)  \(.data.name)"'
    ;;
  tags)
    need_jq
    "${CURL[@]}" "$P/tags?limit=200" | jq -r '.[] | "\(.tag)  (\(.meta.numItems))"'
    ;;
  searches)
    need_jq
    "${CURL[@]}" "$P/searches" | jq -r '.[] | "\(.key)  \(.data.name)"'
    ;;
  run-search)
    need_jq
    k="${2:?用法: zotero.sh run-search <searchKey>}"
    "${CURL[@]}" "$P/searches/$k/items?limit=${3:-50}" \
      | jq -r '.[] | "[\(.data.itemType)] \(.data.title // .data.name // .key)"'
    ;;
  file)
    "${CURL[@]}" -D - "$P/items/${2:?用法: zotero.sh file <itemKey>}/file" -o /dev/null | grep -i '^Location:' || die "无 Location 头（可能不是附件或已被删除）"
    ;;
  export)
    fmt="${2:?用法: zotero.sh export <format>}"
    "${CURL[@]}" "$P/items?format=$fmt&limit=${3:-100}"
    ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
