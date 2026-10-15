"""Download the latest 10-K for each company from SEC EDGAR and keep Items 1 and 1A.

Run: uv run python src/download_10k.py
Output: data/raw/<ticker>.html (original filing) and data/raw/<ticker>.json (the two sections)
"""

import json
import os
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

COMPANIES = {"NVDA": 1045810, "AMD": 2488, "INTC": 50863}  # ticker -> SEC CIK
RAW_DIR = Path("data/raw")
MIN_SECTION_CHARS = 5000  # anything shorter is a table-of-contents entry, not the real section

# SEC blocks requests that don't identify themselves as "Name email".
HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}

# Headings must start a line, so a mention like 'see "Item 1A. Risk Factors"' mid-paragraph doesn't match.
ITEM_1 = re.compile(r"^\s*item\s*1\.?\s*business\b", re.I | re.M)
ITEM_1A = re.compile(r"^\s*item\s*1a\.?\s*risk\s+factors\b", re.I | re.M)
AFTER_1A = re.compile(r"^\s*item\s*(1b|1c|2)\b", re.I | re.M)


def heading(title: str) -> re.Pattern:
    """A whole-line heading, skipping lines followed by a page number (TOC entries and page footers)."""
    return re.compile(rf"^\s*{title}\s*$(?!\n\s*\d+\s*$)", re.I | re.M)


# (start, end) patterns per section. Intel files its 10-K as an annual report without
# "Item 1." headings in the body, so it uses its own section titles.
STANDARD = {"item_1": (ITEM_1, ITEM_1A), "item_1a": (ITEM_1A, AFTER_1A)}
SECTIONS = {
    "INTC": {
        "item_1": (heading("Overview"), heading("Management.s Discussion and Analysis")),
        "item_1a": (heading("Risk Factors"), heading("Other Key Information")),
    },
}


def get(url: str) -> httpx.Response:
    response = httpx.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response


def latest_10k(cik: int) -> dict:
    """Look up the newest 10-K in the company's filing index."""
    recent = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    i = recent["form"].index("10-K")  # the lists are newest first
    accession = recent["accessionNumber"][i]
    return {
        "cik": cik,
        "accession": accession,
        "filing_date": recent["filingDate"][i],
        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{recent['primaryDocument'][i]}",
    }


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["ix:header", "script", "style"]):  # ix:header holds hidden XBRL data, not prose
        tag.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text)


def find_section(text: str, start: re.Pattern, end: re.Pattern) -> str:
    """The first start->end span long enough to be the real section (the TOC repeats every heading)."""
    for s in start.finditer(text):
        e = end.search(text, s.end())
        if e and e.start() - s.start() >= MIN_SECTION_CHARS:
            return text[s.start() : e.start()].strip()
    return ""


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, cik in COMPANIES.items():
        filing = latest_10k(cik)
        html_path = RAW_DIR / f"{ticker}.html"
        if not html_path.exists():  # cache: re-runs don't download again
            html_path.write_text(get(filing["url"]).text)

        text = html_to_text(html_path.read_text())
        patterns = SECTIONS.get(ticker, STANDARD)
        sections = {name: find_section(text, start, end) for name, (start, end) in patterns.items()}
        record = {"ticker": ticker, **filing, "sections": sections}
        (RAW_DIR / f"{ticker}.json").write_text(json.dumps(record, indent=2))

        sizes = ", ".join(f"{name}: {len(body):,} chars" for name, body in sections.items())
        warning = "" if all(sections.values()) else "  ⚠ a section looks missing"
        print(f"{ticker} ({filing['filing_date']})  {sizes}{warning}")


if __name__ == "__main__":
    main()
