---
name: web-docs-capability-research
description: Research a website or product's capabilities and compile web-based help documentation into a structured markdown handbook. Use when users ask to study what a site/product can do, extract docs/help-center content, build a readable documentation package, or summarize capability evidence from documentation pages.
---

# Web Docs Capability Research

## Overview

Use this skill to crawl a documentation/help site, export structured markdown documentation, and generate a capability research report with source evidence.

This skill does not require users to explicitly ask for QA; coverage and classification outputs are produced as built-in execution artifacts.

## Prerequisites

- Python 3.9+
- Python packages: `requests`, `beautifulsoup4`, `lxml`, `markdownify`

Quick check:

```bash
python3 - <<'PY'
import requests, bs4, lxml, markdownify
print('ok')
PY
```

## Workflow

1. Set the target docs base URL.
2. Run the crawler script in the mode requested by user intent.
3. Review generated artifacts and report counts/links back to the user.
4. If needed, rerun with a custom `--content-selector`.

### Command

```bash
python3 scripts/crawl_and_research.py \
  --base-url <url> \
  --mode all \
  --lang bilingual \
  --content-selector "#content"
```

### Modes

- `help-docs`: only generate the full markdown help handbook.
- `capability`: only generate capability research report.
- `all` (default): generate both.

## Output Artifacts

Each run writes into a timestamped folder:

`<cwd>/output/web-docs-capability-research/<YYYYMMDD-HHMMSS>/`

Files:

- `help_docs_full.md`: compiled help docs with menu-aware ordering.
- `capability_report.md`: capability themes and page-level capability notes.
- `coverage.json`: discovery and export coverage data.
- `qa_report.json`: internal quality classification (`exportable`, `real_404`, `restricted`, `no_content`).
- `run_summary.json`: run parameters, timing, output paths, high-level counts.

Schema reference: `references/output_schema.md`.

## Implementation Notes

- Discovery combines `sitemap.xml`, navigation config (`tabs/groups/pages`), and page `href` links.
- Main content extraction uses a configurable selector (default `#content`).
- Report framework is bilingual; page body text stays in source language.
- Restriction classification is automatic (HTTP auth/forbidden, login redirects, verification pages).

## Troubleshooting

- If many pages end up in `no_content`, rerun with the real content container selector, for example:

```bash
python3 scripts/crawl_and_research.py --base-url <url> --content-selector "main"
```

- If a site is very large, constrain crawl scope with `--max-pages`.
