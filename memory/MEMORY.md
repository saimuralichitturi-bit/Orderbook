# Project Memory

## Active Project: project_orderbook (Hedge Fund Filing Intelligence)
Path: C:\Users\chpha\OneDrive\Desktop\workshop\project_orderbook\

### What it does
- Scrapes NSE corporate filings from: https://www.nseindia.com/companies-listing/corporate-filings-announcements
- 5 years of data, all filing types (Financial Results, Annual Report, Dividend, Board Meeting, AGM, etc.)
- Downloads PDFs, analyzes with DeepSeek (primary) → Groq (fallback 1) → Gemini (fallback 2)
- Stores in Parquet format (snappy compressed), syncs to Google Shared Drive
- GitHub Actions runs at 7 AM IST daily (weekdays)
- Streamlit UI with dark theme, company search, AI insights, charts

### File Structure
- `config.py` — all settings, API keys, paths
- `scrapers/nse_filings.py` — NSE filings scraper (session-based, chunks 6-month windows)
- `processors/ai_analyzer.py` — DeepSeek → Groq → Gemini fallback PDF analysis
- `storage/drive_handler.py` — Google Drive (Shared Drive) service account handler
- `run_pipeline.py` — GitHub Actions entry point (incremental per-stock date detection)
- `app/streamlit_app.py` — main Streamlit dashboard
- `.github/workflows/daily_scrape.yml` — 7 AM IST (1:30 AM UTC) Mon-Fri automation

### AI Provider Chain
- Primary: DeepSeek (`deepseek-chat` via OpenAI-compatible API at https://api.deepseek.com)
- Fallback 1: Groq (`llama-3.3-70b-versatile` at https://api.groq.com/openai/v1) — both use `openai` package
- Fallback 2: Gemini (`gemini-2.0-flash`) using NEW `google-genai` SDK (NOT deprecated `google.generativeai`)
- All use `response_format={"type": "json_object"}` for structured output

### Keys in .env (current state)
- ANTHROPIC_API_KEY — set
- GEMINI_API_KEY — set (see .env, never commit)
- DEEPSEEK_API_KEY — set (see .env, never commit)
- GROQ_API_KEY — add from https://console.groq.com/keys
- GOOGLE_DRIVE_FOLDER_ID=0AMEkWPMWm9SoUk9PVA (Shared Drive — starts with 0A)
- GOOGLE_DRIVE_CREDENTIALS_PATH=credentials.json

### Google Drive — Shared Drive (IMPORTANT)
- Folder ID starts with "0A" = Shared Drive → IS_SHARED_DRIVE=True
- ALL Drive API calls need `supportsAllDrives=True`, `includeItemsFromAllDrives=True`, `corpora="drive"`, `driveId=...`
- Service account: trading-data-service@trading-automation-479710.iam.gserviceaccount.com

### GitHub Actions secrets needed
- DEEPSEEK_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
- GOOGLE_DRIVE_FOLDER_ID, GDRIVE_CREDENTIALS_JSON (full JSON content of credentials.json)

### Pipeline Status (as of 2026-02-28)
- RELIANCE: 1,723 rows ✓, TCS: 1,267 rows ✓, HDFCBANK: 934 rows ✓, INFY: 1,119 rows ✓
- ICICIBANK: in progress (2,131 All rows done, now doing categories)
- 101 PDFs downloaded, 24 AI analyses cached in data/analysis/*.json
- Run: `python run_pipeline.py` (incremental — skips stocks already up to date)
- Launch UI: `streamlit run app/streamlit_app.py` (from project root)

### Key APIs confirmed working
- NSE filings CSV: `GET /api/corporate-announcements?index=equities&symbol=RELIANCE&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY&csv=true`
- NSE company search: `GET /api/search/autocomplete?q=RELIANCE`
- PDF attachments: `https://nsearchives.nseindia.com/corporate/...pdf`
- NSE requires homepage warm-up for session cookies first
- DeepSeek confirmed working (cheap, excellent JSON output)

### Watchlist (20 stocks)
RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK,
LT, AXISBANK, BAJFINANCE, MARUTI, TITAN, NESTLEIND, WIPRO, SUNPHARMA, ONGC, TATAMOTORS
