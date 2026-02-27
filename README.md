# web-docs-capability-research-skill

## 主要用途

这个 Skill 主要用于研究一个网站或产品的功能。通常可以在官网帮助文档中看到：

- 产品能力介绍
- 产品包含的功能模块
- 具体操作方式与配置流程

## 为什么要用这个 Skill

网页版帮助文档通常分散在多个页面，阅读和检索都不方便。  
这个 Skill 会把帮助文档内容统一抓取并整理成结构化文档，方便：

- 一次性阅读完整内容
- 快速定位功能与操作说明
- 直接同步到 NotebookLM 等平台做总结和分析

## 典型产物

- `help_docs_full.md`：汇总后的完整帮助文档
- `capability_report.md`：能力研究报告（含证据页面）
- `coverage.json`：覆盖情况
- `qa_report.json`：页面质量分类（如 404、受限、无正文）
- `run_summary.json`：本次执行摘要

## 快速运行

```bash
python3 scripts/crawl_and_research.py \
  --base-url https://docs.maple.inc \
  --mode all \
  --lang bilingual \
  --content-selector "#content"
```
