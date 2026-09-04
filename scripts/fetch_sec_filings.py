"""SEC EDGAR 10-K filing downloader for Veritas RAG.

IMPORTANT SEC ACCESS POLICY NOTE:
The U.S. Securities and Exchange Commission (SEC) requires all automated requests to declare
a specific User-Agent header in the format: "Sample Company Name AdminContact@domain.com".
Generic headers (such as python-requests, curl, or empty agents) will be blocked with HTTP 403.
Update the SEC_USER_AGENT constant below with your actual corporate identifier before running in production.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sec_downloader")

# ==============================================================================
# SEC EDGAR IDENTIFICATION (Update with your own organization & contact email)
# ==============================================================================
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT", "VeritasEnterpriseRAG research-admin@veritas-rag.internal"
)
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """Retrieve 10-digit CIK for a given stock ticker symbol from SEC company_tickers.json."""
    url = "https://www.sec.gov/files/company_tickers.json"
    logger.info(f"Fetching CIK mapping for ticker: {ticker.upper()}...")
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        target = ticker.upper()
        for _, item in data.items():
            if item.get("ticker", "").upper() == target:
                cik = str(item["cik_str"]).zfill(10)
                logger.info(f"Found CIK: {cik} for {target} ({item.get('title')})")
                return cik
    except Exception as e:
        logger.error(f"Failed to lookup CIK for {ticker}: {e}")
    return None


def fetch_latest_10k_filings(
    ticker: str,
    output_dir: Path = Path("data/raw"),
    max_count: int = 2,
) -> int:
    """Download the most recent 10-K filings for the specified ticker."""
    cik = get_cik_from_ticker(ticker)
    if not cik:
        logger.error(f"Could not resolve CIK for ticker '{ticker}'")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"Fetching submissions metadata from {submissions_url}...")

    # Respect SEC rate limit: max 10 requests per second
    time.sleep(0.2)

    try:
        resp = requests.get(submissions_url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        sub_data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
        return 0

    recent = sub_data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    downloaded = 0
    clean_ticker = ticker.lower()

    for idx, form in enumerate(forms):
        if form == "10-K":
            acc_num = accession_numbers[idx]
            acc_nodash = acc_num.replace("-", "")
            prim_doc = primary_docs[idx]
            fdate = filing_dates[idx]

            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{prim_doc}"
            file_name = f"{clean_ticker}_10k_{fdate}_{prim_doc}"
            dest_file = output_dir / file_name

            logger.info(f"Downloading 10-K ({fdate}) from {filing_url}...")
            time.sleep(0.2)
            try:
                f_resp = requests.get(filing_url, headers=SEC_HEADERS, timeout=30)
                f_resp.raise_for_status()
                dest_file.write_bytes(f_resp.content)
                logger.info(f"Saved filing to {dest_file} ({len(f_resp.content)} bytes)")
                downloaded += 1
                if downloaded >= max_count:
                    break
            except Exception as e:
                logger.error(f"Error downloading {filing_url}: {e}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Fetch Form 10-K filings from SEC EDGAR")
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, MSFT)"
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of recent 10-K filings to download"
    )
    parser.add_argument(
        "--out", type=str, default="data/raw", help="Target output directory"
    )
    args = parser.parse_args()

    count = fetch_latest_10k_filings(
        ticker=args.ticker,
        output_dir=Path(args.out),
        max_count=args.count,
    )
    print(f"Downloaded {count} 10-K filings for {args.ticker.upper()}.")


if __name__ == "__main__":
    main()
