"""
Main pipeline — scrapes ALL watchlist stocks.
- First run: fetches 5 years of history for every stock
- Daily runs (GitHub Actions): detects latest date in each parquet
  and only fetches new data from that date onwards
"""

import sys
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger

logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

sys.path.append(str(Path(__file__).parent))

from config import WATCHLIST, GDRIVE_CREDENTIALS, GDRIVE_FOLDER_ID, PARQUET_DIR, HISTORY_YEARS, FILING_CATEGORIES
from scrapers.nse_filings import NSEFilingsScraper
from processors.ai_analyzer import analyze_batch
from storage.drive_handler import DriveHandler


def get_last_scraped_date(symbol: str) -> date:
    """
    Check existing parquet for this symbol and return the latest broadcast date.
    If no parquet exists, return 5 years ago (full history fetch).
    """
    # Check the "All" category parquet — it has the full date range
    path = PARQUET_DIR / f"{symbol}_All.parquet"
    if not path.exists():
        return date.today() - timedelta(days=365 * HISTORY_YEARS)

    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["broadcast_dt"])
        df["broadcast_dt"] = pd.to_datetime(df["broadcast_dt"], errors="coerce")
        latest = df["broadcast_dt"].dropna().max()
        if pd.isna(latest):
            return date.today() - timedelta(days=365 * HISTORY_YEARS)
        # Go back 1 extra day to avoid missing same-day filings
        return latest.date() - timedelta(days=1)
    except Exception as e:
        logger.warning(f"Could not read parquet for {symbol}: {e}")
        return date.today() - timedelta(days=365 * HISTORY_YEARS)


def _pull_from_drive_first():
    """
    Before scraping NSE, pull any parquets from Drive that we don't have locally.
    This ensures get_last_scraped_date() sees the correct latest date and we
    only fetch genuinely new data rather than re-fetching everything.
    """
    if not GDRIVE_FOLDER_ID or not Path(GDRIVE_CREDENTIALS).exists():
        logger.info("Drive not configured — skipping pre-pull")
        return
    try:
        logger.info("Pre-pulling parquets from Drive (skip existing)...")
        drive = DriveHandler()
        pulled = drive.sync_parquets_from_drive()
        if pulled:
            logger.info(f"Pulled {len(pulled)} parquets from Drive: {pulled}")
        else:
            logger.info("All parquets already local (or Drive is empty)")
    except Exception as e:
        logger.warning(f"Drive pre-pull failed (continuing anyway): {e}")


def run(symbols: list[str] = None):
    start = datetime.now()
    symbols = symbols or WATCHLIST
    logger.info(f"=== Pipeline Start | {len(symbols)} stocks | {start.strftime('%Y-%m-%d %H:%M')} ===")

    # ── Pull missing parquets from Drive first ─────────────────────
    # This prevents re-scraping 5 years of data on a fresh machine
    _pull_from_drive_first()

    scraper = NSEFilingsScraper()
    total_rows = 0

    for symbol in symbols:
        from_dt = get_last_scraped_date(symbol)
        to_dt   = date.today()
        days_span = (to_dt - from_dt).days

        if days_span <= 0:
            logger.info(f"{symbol}: already up to date")
            continue

        logger.info(f"{symbol}: fetching {from_dt} → {to_dt} ({days_span} days)")

        for cat in FILING_CATEGORIES:
            try:
                df = scraper.fetch_filings(symbol, cat, from_dt, to_dt)
                if df.empty:
                    continue

                # Download PDFs (cap at 10 per category to stay within time budget)
                df = scraper.download_pdfs_for_df(df, max_pdfs=10)

                # AI analysis (Gemini) for filings that have PDFs
                df = analyze_batch(df, max_pdfs=10)

                # Merge + save parquet
                scraper.save_parquet(df, symbol, cat)
                total_rows += len(df)
                logger.info(f"  {symbol} / {cat}: {len(df)} rows saved")

            except Exception as e:
                logger.error(f"  {symbol} / {cat} failed: {e}")
                continue

    # ── Sync everything to Google Drive ───────────────────────────
    if GDRIVE_FOLDER_ID and Path(GDRIVE_CREDENTIALS).exists():
        logger.info("Syncing to Google Drive...")
        try:
            drive = DriveHandler()
            drive.sync_all()
        except Exception as e:
            logger.error(f"Drive sync failed: {e}")
    else:
        logger.warning("Drive not configured — skipping sync")

    elapsed = (datetime.now() - start).total_seconds()
    summary = {
        "run_at":       start.isoformat(),
        "symbols":      symbols,
        "total_rows":   total_rows,
        "elapsed_s":    round(elapsed, 1),
    }
    Path("logs/last_run.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"=== Done in {elapsed:.0f}s | {total_rows} total rows ===")
    return summary


if __name__ == "__main__":
    # python run_pipeline.py              → full watchlist
    # python run_pipeline.py RELIANCE TCS → specific symbols
    syms = sys.argv[1:] if len(sys.argv) > 1 else None
    run(syms)
