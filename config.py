"""Configuration — NSE Corporate Filings Pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY") or os.getenv("Claude", "")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")    or os.getenv("Google", "")
DEEPSEEK_API_KEY     = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
SUPERMEMORY_API_KEY  = os.getenv("SUPERMEMORY_API_KEY", "")
MEM0_API_KEY         = os.getenv("MEM0_API_KEY", "")

# ── AI Models ────────────────────────────────────────────────────
# Split: Gemini 40% | Groq 30% | OpenRouter 20% | DeepSeek 10%
DEEPSEEK_MODEL      = "deepseek-chat"
DEEPSEEK_API_URL    = "https://api.deepseek.com"
GROQ_MODEL          = "llama-3.3-70b-versatile"       # free, 14,400 req/day
GROQ_API_URL        = "https://api.groq.com/openai/v1"
GEMINI_MODEL        = "gemini-2.0-flash"               # free, 1,500 req/day
OPENROUTER_MODEL    = "meta-llama/llama-3.3-70b-instruct:free"  # free
OPENROUTER_API_URL  = "https://openrouter.ai/api/v1"

# ── Google Drive ─────────────────────────────────────────────────
GDRIVE_CREDENTIALS = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "credentials.json")
# Strip spaces around the folder ID (dotenv sometimes keeps them)
GDRIVE_FOLDER_ID   = (os.getenv("GOOGLE_DRIVE_FOLDER_ID") or "").strip()
IS_SHARED_DRIVE    = GDRIVE_FOLDER_ID.startswith("0A")  # Shared Drive IDs start with 0A

# ── Local Paths ──────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
PARQUET_DIR   = DATA_DIR / "parquet"
PDF_DIR       = DATA_DIR / "pdfs"
ANALYSIS_DIR  = DATA_DIR / "analysis"

for d in [PARQUET_DIR, PDF_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── NSE Endpoints ─────────────────────────────────────────────────
NSE_BASE     = "https://www.nseindia.com"
NSE_ARCHIVES = "https://nsearchives.nseindia.com"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}

# ── Filing Categories (match NSE subject filter) ──────────────────
FILING_CATEGORIES = {
    "All":                  "",
    "Financial Results":    "Financial Results",
    "Annual Report":        "Annual Report",
    "Dividend":             "Dividend",
    "Board Meeting":        "Board Meeting",
    "AGM / EGM":            "AGM/EGM",
    "Investor Presentation":"Investor Presentation",
    "Acquisition":          "Acquisition",
    "Insider Trading":      "Insider Trading / SAST",
}

# ── Scraping defaults ─────────────────────────────────────────────
HISTORY_YEARS   = 5          # How many years back to fetch on first run
DAILY_DAYS_BACK = 2          # How many days back in daily refresh
REQUEST_TIMEOUT = 20
REQUEST_DELAY   = 1.2        # seconds between NSE calls

# ── Default watchlist for daily pipeline ─────────────────────────
WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
    "NESTLEIND", "WIPRO", "SUNPHARMA", "ONGC", "TATAMOTORS",
]
