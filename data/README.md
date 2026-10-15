# Data

**Source:** SEC EDGAR, latest annual 10-K of three semiconductor companies.
**Downloaded:** 2026-10-15 with `src/download_10k.py` (files in `data/raw/`, not committed).

| Ticker | CIK | Filed | Item 1 (Business) | Item 1A (Risk Factors) |
|---|---|---|---|---|
| NVDA | 1045810 | 2026-02-25 | 48,272 chars | 114,959 chars |
| AMD | 2488 | 2026-02-04 | 59,889 chars | 131,884 chars |
| INTC | 50863 | 2026-01-23 | 82,023 chars | 96,346 chars |

About 533k characters in total (roughly 130k tokens).

**Why only Items 1 and 1A:** they name suppliers, customers, competitors, products and risks, which is what the graph needs. The financial sections are long, numeric and expensive to extract.

**Intel note:** Intel files its 10-K as an annual report without "Item 1." headings in the body. Its Item 1 is the "Overview" through "Our Business" sections, and Item 1A is "Risk Factors" (see its Form 10-K Cross-Reference Index).

**Known noise:** page footers such as `Table of Contents` and page numbers remain in the text; the Day 3 chunker removes them.

Reproduce: set `SEC_USER_AGENT` in `.env`, then `uv run python src/download_10k.py`.
