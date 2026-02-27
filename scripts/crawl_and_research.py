#!/usr/bin/env python3
"""
Crawl a documentation site, compile help docs into markdown, and produce a capability report.

Default outputs (timestamped run folder):
- help_docs_full.md
- capability_report.md
- coverage.json
- qa_report.json
- run_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

DEFAULT_CONTENT_SELECTOR = "#content"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Codex Web Docs Capability Research Skill)"
SKILL_OUTPUT_DIRNAME = "web-docs-capability-research"

RESTRICTED_PHRASES = (
    "verify you are human",
    "checking if the site connection is secure",
    "captcha",
    "cf-chl",
    "access denied",
    "please log in",
    "sign in to continue",
    "authentication required",
)

CAPABILITY_KEYWORDS = (
    "support",
    "integrat",
    "configure",
    "manage",
    "autom",
    "analy",
    "track",
    "monitor",
    "optimiz",
    "order",
    "booking",
    "payment",
    "faq",
    "port",
    "sms",
    "voice",
    "dashboard",
    "report",
)


@dataclass
class FetchResult:
    path: str
    url: str
    status: int | None
    final_url: str | None
    content_type: str
    text: str | None
    error: str | None
    restriction_type: str | None


def strip_invisible(text: str) -> str:
    return text.replace("\u200b", "").replace("\ufeff", "").strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if not parsed.scheme:
        base_url = "https://" + base_url.strip()
        parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("base_url must be http/https")
    path = parsed.path.rstrip("/")
    if path:
        # Keep only origin; this crawler assumes whole-site docs crawl.
        return f"{parsed.scheme}://{parsed.netloc}"
    return f"{parsed.scheme}://{parsed.netloc}"


def normalize_path(raw: str, allowed_hosts: set[str]) -> str | None:
    if not raw:
        return None

    v = raw.strip()
    if not v or v.startswith("#"):
        return None
    if v.startswith(("mailto:", "tel:", "sms:", "javascript:")):
        return None

    if v.startswith("http://") or v.startswith("https://"):
        parts = urlsplit(v)
        if parts.netloc not in allowed_hosts:
            return None
        path = parts.path or "/"
    else:
        if v.startswith("//"):
            return None
        path = v if v.startswith("/") else "/" + v

    path = unquote(path).split("#", 1)[0].split("?", 1)[0]
    if not path:
        path = "/"

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    static_prefixes = (
        "/mintlify-assets/",
        "/_next/",
        "/cdn-cgi/",
        "/assets/",
        "/images/",
        "/fonts/",
        "/static/",
    )
    if path.startswith(static_prefixes):
        return None

    if re.search(r"\.[a-zA-Z0-9]{1,8}$", path):
        return None

    return path


def parse_bracketed_array(src: str, start_idx: int) -> str | None:
    i = src.find("[", start_idx)
    if i == -1:
        return None

    depth = 0
    in_str = False
    escaped = False
    j = i

    while j < len(src):
        ch = src[j]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    j += 1
                    return src[i:j]
        j += 1

    return None


def parse_tabs_from_html(html: str) -> list[dict[str, Any]] | None:
    patterns = [r'\\"tabs\\":', r'"tabs":']
    for pat in patterns:
        m = re.search(pat, html)
        if not m:
            continue

        arr = parse_bracketed_array(html, m.end())
        if not arr:
            continue

        candidates = [arr]
        # For escaped embedded JSON in script payloads.
        try:
            decoded = arr.encode("utf-8").decode("unicode_escape")
            candidates.append(decoded)
        except Exception:
            pass

        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except Exception:
                continue
            if isinstance(parsed, list):
                # tabs structure: [{tab, groups:[...]}]
                if all(isinstance(x, dict) for x in parsed):
                    return parsed

    return None


def walk_tabs_paths(node: Any, out: set[str], allowed_hosts: set[str]) -> None:
    if isinstance(node, str):
        slug = node.strip()
        if not slug:
            return
        path = "/" if slug == "index" else normalize_path("/" + slug, allowed_hosts)
        if path:
            out.add(path)
        return

    if isinstance(node, dict):
        for p in node.get("pages", []):
            walk_tabs_paths(p, out, allowed_hosts)
        for g in node.get("groups", []):
            walk_tabs_paths(g, out, allowed_hosts)
        return

    if isinstance(node, list):
        for item in node:
            walk_tabs_paths(item, out, allowed_hosts)


def fetch_sitemap_paths(session: requests.Session, base_url: str, allowed_hosts: set[str]) -> tuple[set[str], dict[str, str]]:
    paths: set[str] = set()
    lastmod: dict[str, str] = {}

    sitemap_url = urljoin(base_url, "/sitemap.xml")
    try:
        res = session.get(sitemap_url, timeout=30)
    except Exception:
        return paths, lastmod

    if res.status_code != 200:
        return paths, lastmod

    try:
        root = ET.fromstring(res.text)
    except Exception:
        return paths, lastmod

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    for url_node in root.findall("sm:url", ns):
        loc = url_node.find("sm:loc", ns)
        if loc is None or not (loc.text or "").strip():
            continue
        path = normalize_path(loc.text.strip(), allowed_hosts)
        if not path:
            continue
        paths.add(path)

        lm = url_node.find("sm:lastmod", ns)
        if lm is not None and (lm.text or "").strip():
            lastmod[path] = lm.text.strip()

    return paths, lastmod


def detect_restriction(status: int | None, final_url: str | None, base_path: str, text: str | None) -> str | None:
    if status in (401, 403, 407):
        return "http_auth_or_forbidden"

    body = (text or "").lower()
    if any(p in body for p in RESTRICTED_PHRASES):
        return "verification_or_login_gate"

    if final_url:
        fpath = urlsplit(final_url).path.lower()
        if fpath and fpath != base_path.lower() and any(x in fpath for x in ("/login", "/signin", "/auth")):
            return "redirect_to_login"

    return None


def fetch_path(
    session: requests.Session,
    base_url: str,
    path: str,
    retry: int,
) -> FetchResult:
    url = urljoin(base_url, path)
    err = None

    for _ in range(max(1, retry)):
        try:
            res = session.get(url, timeout=30, allow_redirects=True)
            ctype = res.headers.get("content-type", "")
            text = res.text if "text/html" in ctype else None
            restriction = detect_restriction(res.status_code, res.url, path, text)
            return FetchResult(
                path=path,
                url=url,
                status=res.status_code,
                final_url=res.url,
                content_type=ctype,
                text=text,
                error=None,
                restriction_type=restriction,
            )
        except Exception as e:
            err = str(e)
            time.sleep(0.2)

    return FetchResult(
        path=path,
        url=url,
        status=None,
        final_url=None,
        content_type="",
        text=None,
        error=err or "unknown_error",
        restriction_type=None,
    )


def discover_links(html: str, allowed_hosts: set[str]) -> set[str]:
    paths: set[str] = set()

    # Standard hrefs.
    for m in re.findall(r'href="([^"]+)"', html):
        p = normalize_path(m, allowed_hosts)
        if p:
            paths.add(p)

    # Escaped hrefs in serialized payloads.
    for m in re.findall(r'\\\\"href\\\\":\\\\"([^"\\\\]+)', html):
        p = normalize_path(m.replace("\\/", "/"), allowed_hosts)
        if p:
            paths.add(p)

    # Unescaped href JSON fragments.
    for m in re.findall(r'"href":"([^"]+)"', html):
        p = normalize_path(m, allowed_hosts)
        if p:
            paths.add(p)

    return paths


def clean_markdown(raw_md: str) -> str:
    text = strip_invisible(raw_md).replace("\xa0", " ")
    text = unescape(text)

    cleaned_lines = []
    for line in text.splitlines():
        # Remove standalone anchor-only artifacts like [​](#section)
        if re.match(r"^\s*\[[^\]]*\]\(#.+\)\s*$", line):
            continue

        # Convert headings like ## [​](#x) Title -> ## Title
        line = re.sub(r"^(#{1,6})\s+\[[^\]]*\]\(#.+\)\s*(.*)$", r"\1 \2", line)
        line = strip_invisible(line)

        # Normalize repeated whitespace.
        line = re.sub(r"[ \t]+", " ", line).rstrip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_page_content(
    path: str,
    html: str,
    content_selector: str,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one(content_selector)

    title_tag = soup.title.get_text(" ", strip=True) if soup.title else None
    desc_meta = soup.find("meta", attrs={"name": "description"})
    description = (desc_meta.get("content") or "").strip() if desc_meta else ""

    if content is None:
        return {
            "path": path,
            "has_content": False,
            "title_tag": title_tag,
            "h1": None,
            "description": description,
            "markdown": "",
            "headings": [],
            "list_items": [],
            "intro_paragraphs": [],
        }

    for bad in content.select("script,style,button"):
        bad.decompose()

    h1 = content.find("h1")
    h1_text = h1.get_text(" ", strip=True) if h1 else None

    headings = [
        h.get_text(" ", strip=True)
        for h in content.find_all(["h2", "h3", "h4"])
        if h.get_text(" ", strip=True)
    ]

    list_items = [
        li.get_text(" ", strip=True)
        for li in content.find_all("li")
        if li.get_text(" ", strip=True)
    ]

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in content.find_all("p")
        if p.get_text(" ", strip=True)
    ]

    raw_md = md(str(content), heading_style="ATX")
    markdown = clean_markdown(raw_md)
    if h1_text:
        markdown = re.sub(r"^#\s+.*?\n+", "", markdown, count=1, flags=re.S)

    return {
        "path": path,
        "has_content": True,
        "title_tag": title_tag,
        "h1": h1_text,
        "description": description,
        "markdown": markdown,
        "headings": headings,
        "list_items": list_items,
        "intro_paragraphs": paragraphs[:3],
    }


def normalize_capability_key(text: str) -> str:
    x = text.lower().strip()
    x = re.sub(r"[^a-z0-9\s\-/&]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x[:140]


def extract_capability_candidates(page_content: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    for h in page_content.get("headings", []):
        clean = strip_invisible(h)
        if len(clean) >= 4:
            candidates.append(clean)

    for li in page_content.get("list_items", []):
        li = strip_invisible(li)
        low = li.lower()
        if any(k in low for k in CAPABILITY_KEYWORDS):
            candidates.append(li)

    # Add intro statements likely describing capabilities.
    for p in page_content.get("intro_paragraphs", []):
        p = strip_invisible(p)
        low = p.lower()
        if any(k in low for k in CAPABILITY_KEYWORDS):
            candidates.append(p)

    dedup = []
    seen = set()
    for c in candidates:
        key = normalize_capability_key(c)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(c.strip())

    return dedup[:8]


def render_menu_tree(
    tabs: list[dict[str, Any]] | None,
    page_details: dict[str, dict[str, Any]],
    base_url: str,
    allowed_hosts: set[str],
) -> str:
    if not tabs:
        return "- Menu tree unavailable / 未解析到菜单树"

    lines: list[str] = []

    def page_label(path: str) -> str:
        info = page_details.get(path, {})
        return info.get("h1") or info.get("title_tag") or path

    def walk(item: Any, indent: int) -> None:
        prefix = "  " * indent
        if isinstance(item, str):
            p = "/" if item == "index" else normalize_path("/" + item, allowed_hosts)
            if not p:
                return
            status = page_details.get(p, {}).get("status")
            st = "OK" if status == 200 else (str(status) if status is not None else "UNKNOWN")
            lines.append(f"{prefix}- [{page_label(p)}]({base_url}{p}) ({st})")
            return

        if isinstance(item, dict):
            group = item.get("group") or item.get("tab") or "Untitled"
            hidden = " [hidden]" if item.get("hidden") else ""
            lines.append(f"{prefix}- {group}{hidden}")
            for p in item.get("pages", []):
                walk(p, indent + 1)
            for g in item.get("groups", []):
                walk(g, indent + 1)

    for tab in tabs:
        tab_name = tab.get("tab", "Untitled Tab")
        lines.append(f"- {tab_name}")
        for g in tab.get("groups", []):
            walk(g, 1)

    return "\n".join(lines)


def render_help_docs_markdown(
    base_url: str,
    mode: str,
    generated_at: str,
    sitemap_paths: set[str],
    crawled_paths: set[str],
    ordered_paths: list[str],
    page_details: dict[str, dict[str, Any]],
    tabs: list[dict[str, Any]] | None,
    failed_paths: list[dict[str, Any]],
    allowed_hosts: set[str],
) -> str:
    lines: list[str] = []

    lines.append("# Help Documentation Compilation / 帮助文档汇编")
    lines.append("")
    lines.append(f"- Generated At (UTC) / 生成时间: {generated_at}")
    lines.append(f"- Base URL / 站点: {base_url}")
    lines.append(f"- Mode / 模式: {mode}")
    lines.append(f"- Sitemap Paths / sitemap 路径数: {len(sitemap_paths)}")
    lines.append(f"- Crawled Paths / 抓取路径数: {len(crawled_paths)}")
    lines.append(f"- Exported Pages / 导出页面数: {len(ordered_paths)}")
    lines.append("")

    lines.append("## Menu Hierarchy / 菜单层级")
    lines.append("")
    lines.append(render_menu_tree(tabs, page_details, base_url, allowed_hosts))
    lines.append("")

    if failed_paths:
        lines.append("## Failed or Non-200 Paths / 失败或非 200 页面")
        lines.append("")
        for item in failed_paths:
            p = item["path"]
            st = item.get("status")
            rtype = item.get("restriction_type")
            reason = rtype or "non-200"
            lines.append(f"- `{p}`: HTTP {st} ({reason})")
        lines.append("")

    lines.append("## Documentation Content / 文档正文")
    lines.append("")

    for idx, p in enumerate(ordered_paths, 1):
        info = page_details[p]
        title = info.get("h1") or info.get("title_tag") or p
        lines.append(f"### {idx}. {title}")
        lines.append("")
        lines.append(f"- URL: {base_url}{p}")
        lines.append(f"- Last Modified (sitemap): {info.get('lastmod') or 'N/A'}")
        if info.get("description"):
            lines.append(f"- Description / 描述: {info['description']}")
        lines.append("")
        lines.append(info.get("markdown", "").strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_capability_report(
    base_url: str,
    generated_at: str,
    ordered_paths: list[str],
    page_details: dict[str, dict[str, Any]],
) -> str:
    cap_map: dict[str, dict[str, Any]] = {}
    page_caps: dict[str, list[str]] = {}

    for p in ordered_paths:
        info = page_details[p]
        cands = extract_capability_candidates(info)
        page_caps[p] = cands

        for c in cands:
            key = normalize_capability_key(c)
            if not key:
                continue
            if key not in cap_map:
                cap_map[key] = {"name": c, "evidence": []}
            if f"{base_url}{p}" not in cap_map[key]["evidence"]:
                cap_map[key]["evidence"].append(f"{base_url}{p}")

    ranked = sorted(cap_map.values(), key=lambda x: (-len(x["evidence"]), x["name"]))

    lines: list[str] = []
    lines.append("# Capability Research Report / 能力研究报告")
    lines.append("")
    lines.append(f"- Generated At (UTC) / 生成时间: {generated_at}")
    lines.append(f"- Base URL / 站点: {base_url}")
    lines.append(f"- Exported Pages / 导出页面数: {len(ordered_paths)}")
    lines.append(f"- Capability Themes / 能力主题数: {len(ranked)}")
    lines.append("")

    lines.append("## Top Capability Themes / 主要能力主题")
    lines.append("")
    if ranked:
        for i, item in enumerate(ranked[:40], 1):
            lines.append(f"{i}. **{item['name']}**")
            for ev in item["evidence"][:5]:
                lines.append(f"   - Evidence / 证据: {ev}")
    else:
        lines.append("- No capability themes extracted / 未提取到能力主题")
    lines.append("")

    lines.append("## Page-level Capability Notes / 页面级能力摘要")
    lines.append("")
    for p in ordered_paths:
        info = page_details[p]
        title = info.get("h1") or info.get("title_tag") or p
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- URL: {base_url}{p}")

        if page_caps[p]:
            lines.append("- Key Capabilities / 关键能力:")
            for cap in page_caps[p]:
                lines.append(f"  - {cap}")
        else:
            lines.append("- Key Capabilities / 关键能力: None detected / 未识别")

        intro = info.get("intro_paragraphs") or []
        if intro:
            lines.append("- Page Summary / 页面摘要:")
            for para in intro[:2]:
                lines.append(f"  - {para}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def ensure_output_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl docs and build capability/help-docs outputs")
    parser.add_argument("--base-url", required=True, help="Base site URL, e.g. https://docs.maple.inc")
    parser.add_argument("--mode", choices=("help-docs", "capability", "all"), default="all")
    parser.add_argument("--out-dir", default=None, help="Base output directory (timestamped run folder will be created)")
    parser.add_argument("--lang", choices=("bilingual", "zh", "en"), default="bilingual")
    parser.add_argument("--content-selector", default=DEFAULT_CONTENT_SELECTOR)
    parser.add_argument("--max-pages", type=int, default=800)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--include-query-links", action="store_true", help="Include links with query strings when discovering paths")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start_time = time.time()

    try:
        base_url = normalize_base_url(args.base_url)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    host = urlsplit(base_url).netloc
    allowed_hosts = {host, host.replace("www.", ""), f"www.{host.replace('www.', '')}"}

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    default_out_base = Path.cwd() / "output" / SKILL_OUTPUT_DIRNAME
    out_base = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_base.resolve()
    run_dir = ensure_output_dir(out_base)

    generated_at = utc_now_iso()

    sitemap_paths, sitemap_lastmod = fetch_sitemap_paths(session, base_url, allowed_hosts)

    sources: dict[str, set[str]] = defaultdict(set)
    queue: deque[str] = deque()
    seen: set[str] = set()

    for p in sorted(sitemap_paths):
        queue.append(p)
        sources[p].add("sitemap")

    queue.append("/")
    sources["/"].add("root")

    tabs: list[dict[str, Any]] | None = None
    tabs_paths: set[str] = set()
    fetched: dict[str, FetchResult] = {}

    while queue and len(seen) < args.max_pages:
        path = queue.popleft()
        if path in seen:
            continue
        seen.add(path)

        result = fetch_path(session, base_url, path, args.retry)
        fetched[path] = result

        if result.text is None:
            continue

        page_tabs = parse_tabs_from_html(result.text)
        if page_tabs:
            if tabs is None:
                tabs = page_tabs
            for tab in page_tabs:
                walk_tabs_paths(tab, tabs_paths, allowed_hosts)
            for p in tabs_paths:
                if p not in seen:
                    queue.append(p)
                sources[p].add("tabs")

        for p in discover_links(result.text, allowed_hosts):
            if not args.include_query_links:
                # normalize_path already drops query by design
                pass
            if p not in seen:
                queue.append(p)
            sources[p].add("href")

    page_details: dict[str, dict[str, Any]] = {}
    exportable: list[str] = []
    real_404: list[str] = []
    restricted: list[dict[str, Any]] = []
    no_content: list[str] = []
    failed_paths: list[dict[str, Any]] = []

    for p in sorted(fetched.keys()):
        res = fetched[p]
        info: dict[str, Any] = {
            "path": p,
            "status": res.status,
            "final_url": res.final_url,
            "restriction_type": res.restriction_type,
            "lastmod": sitemap_lastmod.get(p),
            "error": res.error,
        }

        if res.status == 404:
            real_404.append(p)
        if res.restriction_type:
            restricted.append(
                {
                    "path": p,
                    "status": res.status,
                    "restriction_type": res.restriction_type,
                    "final_url": res.final_url,
                }
            )

        if res.status != 200 or res.text is None:
            failed_paths.append(
                {
                    "path": p,
                    "status": res.status,
                    "restriction_type": res.restriction_type,
                    "error": res.error,
                }
            )
            page_details[p] = info
            continue

        content_info = extract_page_content(p, res.text, args.content_selector)
        info.update(content_info)

        if content_info["has_content"]:
            exportable.append(p)
        else:
            no_content.append(p)

        page_details[p] = info

    # Order exported pages: tabs first, then remaining sorted.
    exportable_set = set(exportable)
    ordered_paths: list[str] = []
    ordered_seen: set[str] = set()

    def add_order(path: str) -> None:
        if path in exportable_set and path not in ordered_seen:
            ordered_paths.append(path)
            ordered_seen.add(path)

    if tabs:
        def walk_order(item: Any) -> None:
            if isinstance(item, str):
                p = "/" if item == "index" else normalize_path("/" + item, allowed_hosts)
                if p:
                    add_order(p)
                return
            if isinstance(item, dict):
                for v in item.get("pages", []):
                    walk_order(v)
                for v in item.get("groups", []):
                    walk_order(v)
                return
            if isinstance(item, list):
                for v in item:
                    walk_order(v)

        walk_order(tabs)

    for p in sorted(exportable):
        add_order(p)

    coverage = {
        "generated_at_utc": generated_at,
        "base_url": base_url,
        "mode": args.mode,
        "sitemap_paths": sorted(sitemap_paths),
        "menu_paths": sorted(tabs_paths),
        "discovered_paths": sorted(sources.keys()),
        "crawled_paths": sorted(fetched.keys()),
        "exported_paths": ordered_paths,
        "failed_paths": sorted([f["path"] for f in failed_paths]),
        "source_map": {k: sorted(v) for k, v in sorted(sources.items())},
    }

    qa_report = {
        "generated_at_utc": generated_at,
        "base_url": base_url,
        "counts": {
            "discovered": len(sources),
            "crawled": len(fetched),
            "exportable": len(ordered_paths),
            "real_404": len(sorted(set(real_404))),
            "restricted": len(restricted),
            "no_content": len(no_content),
            "failed_non_200": len(failed_paths),
        },
        "exportable": ordered_paths,
        "real_404": sorted(set(real_404)),
        "restricted": restricted,
        "no_content": sorted(set(no_content)),
        "details": {
            p: {
                "status": page_details[p].get("status"),
                "restriction_type": page_details[p].get("restriction_type"),
                "has_content": page_details[p].get("has_content", False),
                "final_url": page_details[p].get("final_url"),
                "error": page_details[p].get("error"),
                "sources": sorted(sources.get(p, [])),
            }
            for p in sorted(page_details.keys())
        },
    }

    run_summary = {
        "generated_at_utc": generated_at,
        "base_url": base_url,
        "mode": args.mode,
        "lang": args.lang,
        "content_selector": args.content_selector,
        "max_pages": args.max_pages,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "outputs": {},
        "counts": qa_report["counts"],
    }

    # Always write coverage + QA + summary.
    coverage_path = run_dir / "coverage.json"
    qa_path = run_dir / "qa_report.json"
    summary_path = run_dir / "run_summary.json"

    coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qa_path.write_text(json.dumps(qa_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_summary["outputs"]["coverage"] = str(coverage_path)
    run_summary["outputs"]["qa_report"] = str(qa_path)

    if args.mode in ("help-docs", "all"):
        docs_md = render_help_docs_markdown(
            base_url=base_url,
            mode=args.mode,
            generated_at=generated_at,
            sitemap_paths=sitemap_paths,
            crawled_paths=set(fetched.keys()),
            ordered_paths=ordered_paths,
            page_details=page_details,
            tabs=tabs,
            failed_paths=failed_paths,
            allowed_hosts=allowed_hosts,
        )
        docs_path = run_dir / "help_docs_full.md"
        docs_path.write_text(docs_md, encoding="utf-8")
        run_summary["outputs"]["help_docs_full"] = str(docs_path)

    if args.mode in ("capability", "all"):
        cap_md = render_capability_report(
            base_url=base_url,
            generated_at=generated_at,
            ordered_paths=ordered_paths,
            page_details=page_details,
        )
        cap_path = run_dir / "capability_report.md"
        cap_path.write_text(cap_md, encoding="utf-8")
        run_summary["outputs"]["capability_report"] = str(cap_path)

    run_summary["outputs"]["run_summary"] = str(summary_path)
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] Base URL: {base_url}")
    print(f"[OK] Run directory: {run_dir}")
    print(f"[OK] Discovered: {qa_report['counts']['discovered']}")
    print(f"[OK] Exportable: {qa_report['counts']['exportable']}")
    print(f"[OK] 404: {qa_report['counts']['real_404']}")
    print(f"[OK] Restricted: {qa_report['counts']['restricted']}")
    print(f"[OK] No content: {qa_report['counts']['no_content']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
