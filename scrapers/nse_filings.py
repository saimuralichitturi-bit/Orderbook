"""
NSE Corporate Filings Scraper
Source: https://www.nseindia.com/companies-listing/corporate-filings-announcements
Fetches all announcement types for any company symbol, 5 years back.
Downloads PDFs. Stores as Parquet.
"""

import io
import time
import hashlib
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    NSE_BASE, NSE_HEADERS, FILING_CATEGORIES,
    REQUEST_TIMEOUT, REQUEST_DELAY,
    HISTORY_YEARS, DAILY_DAYS_BACK,
    PDF_DIR, PARQUET_DIR,
)


class NSEFilingsScraper:
    """Scrapes NSE corporate filings — all categories, any date range."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._warm_up()

    def _warm_up(self):
        """Hit NSE homepage to get session cookies (required for API access)."""
        try:
            self.session.get(NSE_BASE, timeout=REQUEST_TIMEOUT)
            logger.info("NSE session ready")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning(f"NSE warm-up failed (continuing anyway): {e}")

    # ─── Company Search ───────────────────────────────────────────

    def search_company(self, query: str) -> list[dict]:
        """
        Live search for company by name or symbol.
        Returns list of {symbol, name} dicts.
        """
        try:
            url = f"{NSE_BASE}/api/search/autocomplete?q={query}"
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("symbols", []):
                if item.get("result_sub_type") == "equity":
                    results.append({
                        "symbol": item["symbol"],
                        "name": item.get("symbol_info", item["symbol"]),
                    })
            return results
        except Exception as e:
            logger.error(f"Company search failed: {e}")
            return []

    # ─── Filings Fetch ────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def _fetch_csv(self, symbol: str, from_date: str, to_date: str, subject: str = "") -> pd.DataFrame:
        """
        Fetch filings CSV from NSE API.
        from_date / to_date: "DD-MM-YYYY"
        subject: NSE category string (empty = all)
        """
        params = {
            "index": "equities",
            "symbol": symbol,
            "from_date": from_date,
            "to_date": to_date,
            "csv": "true",
        }
        if subject:
            params["subject"] = subject

        url = f"{NSE_BASE}/api/corporate-announcements"
        resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        text = resp.text.lstrip("\ufeff")  # strip BOM
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().upper() for c in df.columns]
        return df

    def fetch_filings(
        self,
        symbol: str,
        category: str = "All",
        from_date: date = None,
        to_date: date = None,
    ) -> pd.DataFrame:
        """
        Fetch filings for a company across a date range.
        Chunks into 6-month windows to avoid NSE result limits.
        """
        to_dt   = to_date   or date.today()
        from_dt = from_date or (to_dt - timedelta(days=365 * HISTORY_YEARS))
        subject = FILING_CATEGORIES.get(category, "")

        logger.info(f"Fetching {category} filings for {symbol}: {from_dt} → {to_dt}")

        # Chunk into 6-month windows to avoid NSE pagination issues
        chunks = []
        cursor = from_dt
        while cursor < to_dt:
            end = min(cursor + timedelta(days=180), to_dt)
            try:
                df = self._fetch_csv(
                    symbol,
                    cursor.strftime("%d-%m-%Y"),
                    end.strftime("%d-%m-%Y"),
                    subject,
                )
                if not df.empty:
                    chunks.append(df)
                    logger.debug(f"  {cursor} → {end}: {len(df)} rows")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.warning(f"  Chunk {cursor}→{end} failed: {e}")
            cursor = end + timedelta(days=1)

        if not chunks:
            logger.warning(f"No filings found for {symbol} / {category}")
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True).drop_duplicates()
        df = self._clean(df, symbol, category)
        logger.info(f"Total: {len(df)} filings for {symbol} / {category}")
        return df

    def _clean(self, df: pd.DataFrame, symbol: str, category: str) -> pd.DataFrame:
        """Standardise columns and types."""
        rename = {
            "SYMBOL": "symbol",
            "COMPANY NAME": "company",
            "SUBJECT": "subject",
            "DETAILS": "details",
            "BROADCAST DATE/TIME": "broadcast_dt",
            "ATTACHMENT": "pdf_url",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["symbol"]    = symbol
        df["category"]  = category
        if "broadcast_dt" in df.columns:
            df["broadcast_dt"] = pd.to_datetime(df["broadcast_dt"], errors="coerce")
            df = df.sort_values("broadcast_dt", ascending=False)
        df["scraped_at"] = datetime.now().isoformat()

        # Generate stable ID for deduplication
        id_cols = ["symbol", "broadcast_dt", "subject", "pdf_url"]
        available = [c for c in id_cols if c in df.columns]
        df["filing_id"] = df[available].astype(str).agg("|".join, axis=1).apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
        )
        return df

    # ─── PDF Download ─────────────────────────────────────────────

    def download_pdf(self, pdf_url: str, filing_id: str) -> Path | None:
        """
        Download filing PDF from NSE archives.
        Skips if already downloaded. Returns local path.
        """
        if not pdf_url or not isinstance(pdf_url, str) or not pdf_url.startswith("http"):
            return None

        dest = PDF_DIR / f"{filing_id}.pdf"
        if dest.exists():
            return dest  # already have it

        try:
            resp = requests.get(pdf_url, timeout=30, headers=NSE_HEADERS, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"PDF downloaded: {dest.name}")
            return dest
        except Exception as e:
            logger.warning(f"PDF download failed ({pdf_url}): {e}")
            return None

    def download_pdfs_for_df(self, df: pd.DataFrame, max_pdfs: int = 50) -> pd.DataFrame:
        """Download PDFs for all filings in a DataFrame. Adds 'local_pdf' column."""
        if "pdf_url" not in df.columns or "filing_id" not in df.columns:
            return df

        paths = []
        downloaded = 0
        for _, row in df.iterrows():
            if downloaded >= max_pdfs:
                paths.append(None)
                continue
            path = self.download_pdf(row["pdf_url"], row["filing_id"])
            paths.append(str(path) if path else None)
            if path:
                downloaded += 1
                time.sleep(0.5)

        df = df.copy()
        df["local_pdf"] = paths
        logger.info(f"Downloaded {downloaded} PDFs")
        return df

    # ─── Parquet Storage ──────────────────────────────────────────

    def save_parquet(self, df: pd.DataFrame, symbol: str, category: str) -> Path:
        """
        Save filings to Parquet. Merges with existing data (no duplicates).
        Parquet is columnar, compressed — tiny footprint vs CSV.
        """
        safe_cat = category.replace(" ", "_").replace("/", "_")
        path = PARQUET_DIR / f"{symbol}_{safe_cat}.parquet"

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            if "filing_id" in df.columns:
                df = df.drop_duplicates(subset=["filing_id"])

        # Sort newest first
        if "broadcast_dt" in df.columns:
            df = df.sort_values("broadcast_dt", ascending=False)

        df.to_parquet(path, index=False, compression="snappy")
        logger.info(f"Parquet saved: {path.name} ({len(df)} rows, {path.stat().st_size//1024} KB)")
        return path

    def load_parquet(self, symbol: str, category: str = "All") -> pd.DataFrame:
        """Load filings from local Parquet cache."""
        safe_cat = category.replace(" ", "_").replace("/", "_")
        path = PARQUET_DIR / f"{symbol}_{safe_cat}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def list_cached_symbols(self) -> list[str]:
        """Return list of symbols we have Parquet data for."""
        files = list(PARQUET_DIR.glob("*.parquet"))
        symbols = sorted({f.stem.split("_")[0] for f in files})
        return symbols


# ─── Daily pipeline entry point ───────────────────────────────────

def run_daily_scrape(symbols: list[str], days_back: int = DAILY_DAYS_BACK):
    """
    Called by GitHub Actions every morning.
    Fetches last N days of filings for all watchlist symbols.
    """
    from config import WATCHLIST
    targets = symbols or WATCHLIST
    scraper = NSEFilingsScraper()

    to_dt   = date.today()
    from_dt = to_dt - timedelta(days=days_back)

    results = {}
    for symbol in targets:
        logger.info(f"Processing {symbol}...")
        frames = []
        for cat in FILING_CATEGORIES:
            df = scraper.fetch_filings(symbol, cat, from_dt, to_dt)
            if not df.empty:
                df = scraper.download_pdfs_for_df(df, max_pdfs=5)
                scraper.save_parquet(df, symbol, cat)
                frames.append(df)
        if frames:
            results[symbol] = pd.concat(frames, ignore_index=True)

    logger.info(f"Daily scrape complete. Processed {len(results)} symbols.")
    return results


def run_historical_scrape(symbol: str):
    """
    One-time full 5-year fetch for a new symbol.
    Downloads all categories + PDFs.
    """
    scraper = NSEFilingsScraper()
    to_dt   = date.today()
    from_dt = to_dt - timedelta(days=365 * HISTORY_YEARS)

    for cat in FILING_CATEGORIES:
        df = scraper.fetch_filings(symbol, cat, from_dt, to_dt)
        if not df.empty:
            df = scraper.download_pdfs_for_df(df, max_pdfs=100)
            scraper.save_parquet(df, symbol, cat)
        time.sleep(REQUEST_DELAY * 2)


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    run_historical_scrape(sym)
