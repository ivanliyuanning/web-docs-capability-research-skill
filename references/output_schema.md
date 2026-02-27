# Output Schema Reference

This skill always emits JSON artifacts that can be parsed by other scripts.

## `coverage.json`

Top-level keys:

- `generated_at_utc`: ISO UTC timestamp
- `base_url`: crawled base URL
- `mode`: `help-docs` | `capability` | `all`
- `sitemap_paths`: paths discovered from sitemap
- `menu_paths`: paths discovered from navigation tab config
- `discovered_paths`: all discovered candidate paths
- `crawled_paths`: paths that were fetched
- `exported_paths`: paths exported as documentation pages
- `failed_paths`: crawled paths with non-200 outcomes
- `source_map`: map of path -> discovery sources (`sitemap`, `tabs`, `href`, `root`)

## `qa_report.json`

Top-level keys:

- `generated_at_utc`
- `base_url`
- `counts`
  - `discovered`
  - `crawled`
  - `exportable`
  - `real_404`
  - `restricted`
  - `no_content`
  - `failed_non_200`
- `exportable`: list of exportable paths
- `real_404`: list of 404 paths
- `restricted`: list of objects
  - `path`
  - `status`
  - `restriction_type`
  - `final_url`
- `no_content`: list of 200 pages without content selector match
- `details`: per-path diagnostic map

## `run_summary.json`

Top-level keys:

- `generated_at_utc`
- `base_url`
- `mode`
- `lang`
- `content_selector`
- `max_pages`
- `elapsed_seconds`
- `outputs`: output file paths
- `counts`: copied from `qa_report.counts`
