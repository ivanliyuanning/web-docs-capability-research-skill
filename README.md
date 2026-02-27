# web-docs-capability-research-skill

A Codex skill for:
- researching a website/product's capabilities
- crawling help/doc sites
- compiling structured markdown documentation

## Main files
- `SKILL.md`
- `scripts/crawl_and_research.py`
- `references/output_schema.md`

## Quick run
```bash
python3 scripts/crawl_and_research.py --base-url https://docs.maple.inc --mode all --lang bilingual --content-selector "#content"
```
