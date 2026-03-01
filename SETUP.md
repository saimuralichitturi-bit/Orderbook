# NSE Corporate Filings Intelligence — Setup Guide

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure .env

Your `.env` is already configured with Claude + Gemini keys.
Add your Google Drive folder ID:

```
GOOGLE_DRIVE_FOLDER_ID=<paste folder ID from Drive URL>
GOOGLE_DRIVE_CREDENTIALS_PATH=credentials.json
```

**Get the folder ID:** Open Google Drive → create a folder → copy the ID from the URL
(e.g., `https://drive.google.com/drive/folders/1ABC123xyz` → ID is `1ABC123xyz`)

## 3. Place Google Drive credentials

Drop your service account JSON file as `credentials.json` in this folder.

**Share your Drive folder with the service account email** (found in credentials.json as `client_email`).

## 4. Run historical scrape (first time)

```bash
# Fetch 5 years of filings for a company
python -m scrapers.nse_filings RELIANCE

# Or run full pipeline for watchlist
python run_pipeline.py
```

## 5. Launch Streamlit

```bash
streamlit run app/streamlit_app.py
```

## 6. GitHub Actions setup

Add these **secrets** in your GitHub repo (Settings → Secrets → Actions):

| Secret Name              | Value |
|--------------------------|-------|
| `ANTHROPIC_API_KEY`      | Your Claude API key |
| `GEMINI_API_KEY`         | Your Gemini API key |
| `GOOGLE_DRIVE_FOLDER_ID` | Drive folder ID |
| `GDRIVE_CREDENTIALS_JSON`| Full contents of credentials.json |

The workflow runs automatically at **7:00 AM IST** every weekday.
You can also trigger it manually from GitHub → Actions → Run workflow.

## How it works

```
Every morning (7 AM IST):
  GitHub Actions
    → scrapes NSE filings (last 2 days) for watchlist
    → downloads new PDFs
    → runs AI analysis (Claude Haiku → Gemini fallback)
    → saves to Parquet (compressed, columnar)
    → syncs to Google Drive

Streamlit app:
    → search any NSE company
    → filter by filing type
    → view filings + AI analysis + charts
    → reads from local Parquet cache or live from NSE
```

## Filing types available

| Type | What it contains |
|------|-----------------|
| Financial Results | Quarterly P&L, revenue, profit, EPS |
| Annual Report | Full year report PDF |
| Dividend | Dividend announcements + record dates |
| Board Meeting | Meeting notices (results, dividends, buybacks) |
| AGM / EGM | Shareholder meeting notices |
| Investor Presentation | Management presentations |
| Acquisition | M&A announcements |
| Insider Trading | SAST disclosures |
