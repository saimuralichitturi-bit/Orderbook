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

### Orderbook Intelligence Engine (2026-03-01)
- **`processors/orderbook_engine.py`** (NEW) — Full orderbook extraction + ML pipeline
- **Algorithm**: Regex pre-filter (no API cost for irrelevant PDFs) → AI extraction (DeepSeek/Gemini/Groq) → K-Means clustering (silhouette score picks optimal K=2-5) → Time series (velocity, acceleration, z-score anomaly) → AI reasoning chain
- **Cache**: `data/orderbook/{filing_id}.json` (per-PDF) + `{symbol}_orderbook.parquet` + `{symbol}_analysis.json`
- **Key functions**: `extract_orderbook_from_pdf()`, `batch_extract_orderbook()`, `detect_trends()`, `ai_orderbook_reasoning()`, `save_orderbook()`, `load_orderbook()`
- **Streamlit tab**: "📦 Orderbook" added between AI Insights and Chat. Tab order: 📋 Filings | 📈 Charts | 🤖 AI Insights | 📦 Orderbook | 💬 Chat
- **Visualizations**: Stacked area (cumulative MW by energy type) + anomaly stars, Monthly bar + momentum line, INR Cr dual-axis, Quarterly activity bar, K-Means cluster scatter, Energy type pie+bar, Individual orders table with filters
- **AI card**: Investment Grade (A/B/C/D), Overall Assessment, Action signal, Reasoning chain (4 steps), Strengths/Concerns/Catalysts, 12-month outlook, Sector context
- **Extraction fields per entry**: type, description, value_numeric, value_unit, value_mw, value_inr_cr, counterparty, project_location, duration_years, energy_type, contract_type, is_positive_signal, reasoning, confidence
- **ENERGY_COLORS**: solar=#ffa726, wind=#58a6ff, thermal=#f85149, hydro=#3fb950, gas=#d29922
- **MW normalization**: GW×1000, uses INR_CONVERSION dict for financial normalization

### Drive Auto-Sync (2026-03-01)
- **Dashboard**: on startup with no local data, auto-triggers Drive sync once per session (`_drive_sync_tried` session state flag). Manual "☁️ Sync from Drive" button always visible top-right of overview.
- **DriveHandler new methods**: `list_files(subfolder)` (paginated), `sync_parquets_from_drive()` (skips existing), `sync_analysis_from_drive()`, `_find_file_id(filename, subfolder)`.
- **run_pipeline.py**: `_pull_from_drive_first()` called before NSE scraping so `get_last_scraped_date()` sees Drive data and only fetches genuinely new data.
- **Empty state**: shows helpful message about Drive config status instead of just "Run pipeline".

### Major Features Added (2026-03-01)
- **Category AI Fix**: `_NSE_SUBJECT_MAP` expanded to 100+ keywords. AI recategorize button in Charts tab saves to `data/category_cache.json`. `categorise_subject(subject, ai_cache)` now accepts cache dict.
- **Investment Thesis**: `ANALYSIS_PROMPT` updated with `investment_thesis`, `key_catalysts`, `time_horizon` fields. Rendered in `_render_analysis()` with green success box + rocket bullets.
- **No 50-filing cap**: Chat tab passes ALL filings. `_filter_relevant_filings()` in `chat_agent.py` uses Groq to score/rank filings per question (top 60 most relevant returned).
- **Auto-fetch PDF from Drive**: `fetch_pdf_for_analysis(filing_id, pdf_url, local_pdf)` tries local → Drive → NSE URL. "Analyze" button now auto-fetches instead of showing "download manually" message.
- **Pre/Live Analysis**: AI Insights tab has radio "📁 Saved Analysis | 🔴 Live Analysis". Saved shows cached JSONs (full investment thesis). Live: multi-select up to 10 filings, runs real-time analysis.
- **Drive handler**: Added `download_pdf_by_name(filename, subfolder)` to `DriveHandler`.
- **Stat card colors**: Changed from blue (#58a6ff) to amber (#ffa726) to match dashboard style.
- **DATA_DIR, PDF_DIR** now imported in streamlit_app.py from config.

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
