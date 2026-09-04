#!/usr/bin/env bash
# arXiv 官方 API / RSS 辅助脚本（配合 arxiv-query skill 使用）
# 依赖: bash + curl + python3（解析 Atom/RSS 用标准库，无需第三方包）
#
# 用法: ./arxiv.sh <command> [args...]
#   today <cat> [cat...]      RSS 扫各分类最新一批论文（零限流风险，推荐每日用）
#   search '<query>' [n] [--abstract]
#                             API 精确检索（search_query 语法见 SKILL.md）
#   recent <cat> [n]          分类下按提交时间倒序最新 n 篇
#   since <cat> <days> [n]    分类下最近 N 天内提交的论文
#   id <arxiv-id> [id...]     按 arXiv ID 拉取元数据（id_list）
#
# 每次 API 请求会遵守官方 3 秒间隔建议（环境变量 ARXIV_SLEEP 可调）。
# 原始响应存 /tmp/arxiv_out.xml（Atom）或 /tmp/arxiv_rss_<cat>.xml，可自行再解析。
set -euo pipefail

API="${ARXIV_API:-https://export.arxiv.org/api/query}"
RSS="https://rss.arxiv.org/rss"
SLEEP="${ARXIV_SLEEP:-3}"          # 官方建议连续请求间隔 3 秒
OUT="/tmp/arxiv_out.xml"
CURL=(curl -sS -m 30)

die() { echo "error: $*" >&2; exit 1; }
urlencode() { python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

# 带重试的抓取：网络瞬时故障（TLS 中断/超时）自动重试最多 3 次
fetch() {
  local url="$1" out="$2" i
  for i in 1 2 3; do
    if "${CURL[@]}" "$url" > "$out" && [ -s "$out" ]; then
      return 0
    fi
    echo "# 请求失败，重试 $i/3: $url" >&2
    sleep 2
  done
  die "请求失败（已重试 3 次）: $url"
}

# 解析 API 返回的 Atom feed（参数: XML 文件路径 [--abstract]），输出: [日期] id | 标题 | 分类
parse_atom() {
  python3 - "$@" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET
xmlfile = sys.argv[1]
show_abs = '--abstract' in sys.argv[2:]
ns = {'a': 'http://www.w3.org/2005/Atom', 'os': 'http://a9.com/-/spec/opensearch/1.1/'}
try:
    root = ET.parse(xmlfile).getroot()
except ET.ParseError as e:
    print(f'# 解析失败（响应可能不是 Atom XML）: {e}', file=sys.stderr)
    sys.exit(1)
t = root.find('os:totalResults', ns)
if t is not None and t.text:
    print(f'# totalResults: {t.text}', file=sys.stderr)
n = 0
for e in root.findall('a:entry', ns):
    n += 1
    aid = (e.findtext('a:id', default='', namespaces=ns) or '').strip()
    title = ' '.join((e.findtext('a:title', default='', namespaces=ns) or '').split())
    pub = (e.findtext('a:published', default='', namespaces=ns) or '').strip()
    cats = [c.get('term') for c in e.findall('a:category', ns)]
    print(f'[{pub[:10]}] {aid} | {title} | {",".join(cats)}')
    if show_abs:
        ab = ' '.join((e.findtext('a:summary', default='', namespaces=ns) or '').split())
        print(f'    {ab[:300]}')
print(f'# entries: {n}', file=sys.stderr)
PYEOF
}

# 解析 RSS feed（参数: XML 文件路径），输出: [时间] arXivID | 标题
parse_rss() {
  python3 - "$1" <<'PYEOF'
import sys, re
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
n = 0
for item in root.findall('.//item'):
    n += 1
    title = ' '.join((item.findtext('title') or '').split())
    link = (item.findtext('link') or '').strip()
    m = re.search(r'arxiv\.org/abs/([^\s/]+)', link)
    aid = m.group(1) if m else '?'
    pub = (item.findtext('pubDate') or '').strip()
    print(f'[{pub[:16]}] {aid} | {title}')
print(f'# entries: {n}', file=sys.stderr)
PYEOF
}

# 通用 API 查询：q=search_query, n=max_results
api_query() {
  local q="$1" n="$2" extra="${3:-}"
  fetch "$API?search_query=$(urlencode "$q")&start=0&max_results=$n&sortBy=submittedDate&sortOrder=descending" "$OUT"
  parse_atom "$OUT" $extra
}

case "${1:-}" in
  today)
    [ $# -ge 2 ] || die "用法: arxiv.sh today <cat> [cat...]（如 cs.AI cs.LG stat.ML）"
    shift
    for c in "$@"; do
      echo "== $c =="
      fetch "$RSS/$c" "/tmp/arxiv_rss_$c.xml"
      parse_rss "/tmp/arxiv_rss_$c.xml"
      sleep 1   # 多个分类间也轻量间隔，避免扎堆
    done
    ;;
  search)
    [ $# -ge 2 ] || die "用法: arxiv.sh search '<query>' [n] [--abstract]"
    q="$2"; n="${3:-20}"; extra="${4:-}"
    api_query "$q" "$n" "$extra"
    ;;
  recent)
    [ $# -ge 2 ] || die "用法: arxiv.sh recent <cat> [n]"
    c="$2"; n="${3:-20}"
    api_query "cat:$c" "$n"
    ;;
  since)
    [ $# -ge 3 ] || die "用法: arxiv.sh since <cat> <days> [n]"
    c="$2"; days="$3"; n="${4:-50}"
    from=$(date -u -d "$days days ago" +%Y%m%d%H%M)
    to=$(date -u +%Y%m%d%H%M)
    api_query "cat:$c AND submittedDate:[$from TO $to]" "$n"
    ;;
  id)
    [ $# -ge 2 ] || die "用法: arxiv.sh id <arxiv-id> [id...]"
    shift
    idlist=$(printf '%s' "$*" | tr ' ' ',')
    fetch "$API?id_list=$idlist&max_results=$#" "$OUT"
    parse_atom "$OUT"
    ;;
  *)
    sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
