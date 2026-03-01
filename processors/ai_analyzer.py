"""
AI Analyzer — 4-provider chain with weighted random routing.
Split: Gemini 40% | Groq 30% | OpenRouter 20% | DeepSeek 10%
All free providers are prioritised; DeepSeek (paid) is last resort.
"""

import json
import random
import fitz  # pymupdf
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_URL,
    GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL,
    GEMINI_API_KEY, GEMINI_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_API_URL,
    ANALYSIS_DIR,
)
import pandas as pd


# ─── PDF Text Extraction ──────────────────────────────────────────

def extract_pdf_text(pdf_path: str | Path, max_chars: int = 15000) -> str:
    """Extract readable text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars].strip()
    except Exception as e:
        logger.error(f"PDF extraction failed ({pdf_path}): {e}")
        return ""


# ─── Prompt ───────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a senior equity analyst for an Indian hedge fund analyzing an NSE corporate filing.

Company: {company}
Filing Type: {subject}
Date: {date}

PDF Content:
{text}

Return a JSON object with exactly these keys (no extra text, no markdown):
{{
  "summary": "2-3 sentence executive summary for fund managers",
  "sentiment": "bullish | bearish | neutral",
  "sentiment_score": <float -1.0 to 1.0>,
  "key_highlights": ["highlight 1", "highlight 2", "highlight 3"],
  "risk_factors": ["risk 1", "risk 2"],
  "financial_data": {{
    "revenue": <number in crores or null>,
    "net_profit": <number in crores or null>,
    "eps": <number or null>,
    "revenue_growth_pct": <number or null>,
    "profit_growth_pct": <number or null>,
    "dividend_per_share": <number or null>,
    "currency": "INR"
  }},
  "action_signal": "buy | sell | hold | watch",
  "confidence": <float 0.0 to 1.0>,
  "tags": ["tag1", "tag2"]
}}"""


# ─── Provider functions ────────────────────────────────────────────

def _analyze_deepseek(text: str, company: str, subject: str, date_str: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
    prompt = ANALYSIS_PROMPT.format(company=company, subject=subject, date=date_str, text=text)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1024,
    )
    result = json.loads(resp.choices[0].message.content.strip())
    result["model_used"] = DEEPSEEK_MODEL
    return result


def _analyze_groq(text: str, company: str, subject: str, date_str: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_URL)
    prompt = ANALYSIS_PROMPT.format(company=company, subject=subject, date=date_str, text=text)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1024,
    )
    result = json.loads(resp.choices[0].message.content.strip())
    result["model_used"] = GROQ_MODEL
    return result


def _analyze_openrouter(text: str, company: str, subject: str, date_str: str) -> dict:
    from openai import OpenAI
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_URL,
        default_headers={"HTTP-Referer": "https://github.com/nse-filings-pipeline"},
    )
    prompt = ANALYSIS_PROMPT.format(company=company, subject=subject, date=date_str, text=text)
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1024,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if model returns them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())
    result["model_used"] = OPENROUTER_MODEL
    return result


def _analyze_gemini(text: str, company: str, subject: str, date_str: str) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = ANALYSIS_PROMPT.format(company=company, subject=subject, date=date_str, text=text)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    raw = resp.text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1][4:].strip() if parts[1].startswith("json") else parts[1].strip()
    result = json.loads(raw)
    result["model_used"] = GEMINI_MODEL
    return result


# ─── Provider registry ────────────────────────────────────────────

_PROVIDERS = {
    "gemini":      (_analyze_gemini,      lambda: GEMINI_API_KEY),
    "groq":        (_analyze_groq,        lambda: GROQ_API_KEY),
    "openrouter":  (_analyze_openrouter,  lambda: OPENROUTER_API_KEY),
    "deepseek":    (_analyze_deepseek,    lambda: DEEPSEEK_API_KEY),
}

# Weighted split — free providers first, paid DeepSeek minimal
_PROVIDER_WEIGHTS = {
    "gemini":     40,   # free — 1,500 req/day
    "groq":       30,   # free — 14,400 req/day
    "openrouter": 20,   # free — rate limited but generous
    "deepseek":   10,   # paid — ~$0.002/filing
}


def _build_order() -> list[str]:
    """Pick a random primary provider (weighted), then fill fallbacks."""
    available = [p for p in _PROVIDER_WEIGHTS if _PROVIDERS[p][1]()]
    if not available:
        return []

    weights  = [_PROVIDER_WEIGHTS[p] for p in available]
    primary  = random.choices(available, weights=weights, k=1)[0]
    rest     = [p for p in available if p != primary]
    # Remaining order: free first (groq, openrouter, gemini), deepseek last
    free_rest = [p for p in ["gemini", "groq", "openrouter"] if p in rest]
    paid_rest = [p for p in ["deepseek"] if p in rest]
    return [primary] + free_rest + paid_rest


# ─── Main analyze function ────────────────────────────────────────

def analyze_filing(
    pdf_path: str | Path,
    company: str,
    subject: str,
    date_str: str,
    filing_id: str,
) -> dict:
    """
    Analyze a PDF filing with weighted random provider routing.
    Caches result to JSON — skips if already analyzed.
    """
    out_path = ANALYSIS_DIR / f"{filing_id}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    text = extract_pdf_text(pdf_path)
    if not text:
        logger.warning(f"Empty PDF text: {pdf_path}")
        return {}

    result = {}
    order  = _build_order()

    for provider in order:
        if result:
            break
        fn, _ = _PROVIDERS[provider]
        try:
            result = fn(text, company, subject, date_str)
            logger.info(
                f"{provider.upper()} ✓ {company}/{subject} "
                f"→ {result.get('sentiment','?')} / {result.get('action_signal','?')}"
            )
        except Exception as e:
            logger.warning(f"{provider} failed ({e}) — trying next...")

    if not result:
        logger.error(f"All providers failed for {company}/{subject}")

    if result:
        result["filing_id"]   = filing_id
        result["analyzed_at"] = datetime.now().isoformat()
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# ─── Batch helper ─────────────────────────────────────────────────

def analyze_batch(df: pd.DataFrame, max_pdfs: int = 20) -> pd.DataFrame:
    """Run AI analysis for all rows with downloaded PDFs. Adds ai_* columns."""
    if "local_pdf" not in df.columns or "filing_id" not in df.columns:
        return df

    rows = df[
        df["local_pdf"].notna() &
        df["local_pdf"].astype(str).ne("None") &
        df["local_pdf"].astype(str).ne("")
    ].head(max_pdfs)

    analyses = []
    for _, row in rows.iterrows():
        pdf_path = str(row.get("local_pdf", ""))
        if not pdf_path or not Path(pdf_path).exists():
            continue
        result = analyze_filing(
            pdf_path  = pdf_path,
            company   = str(row.get("company", row.get("symbol", ""))),
            subject   = str(row.get("subject", "")),
            date_str  = str(row.get("broadcast_dt", "")),
            filing_id = str(row.get("filing_id", "")),
        )
        if result:
            analyses.append({
                "filing_id":          row["filing_id"],
                "ai_summary":         result.get("summary", ""),
                "ai_sentiment":       result.get("sentiment", ""),
                "ai_sentiment_score": result.get("sentiment_score", 0),
                "ai_signal":          result.get("action_signal", ""),
                "ai_highlights":      json.dumps(result.get("key_highlights", [])),
                "ai_risks":           json.dumps(result.get("risk_factors", [])),
                "ai_financial":       json.dumps(result.get("financial_data", {})),
                "ai_confidence":      result.get("confidence", 0),
                "ai_tags":            json.dumps(result.get("tags", [])),
                "ai_model":           result.get("model_used", ""),
            })

    if not analyses:
        return df
    return df.merge(pd.DataFrame(analyses), on="filing_id", how="left")


def load_analysis(filing_id: str) -> dict:
    path = ANALYSIS_DIR / f"{filing_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_financial_timeseries(symbol: str = None) -> pd.DataFrame:
    """Aggregate AI-extracted financials across all analyzed filings for charting."""
    records = []
    for path in ANALYSIS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            fin = data.get("financial_data", {})
            if fin and any(fin.get(k) for k in ["revenue", "net_profit", "eps"]):
                records.append({
                    "filing_id":          data.get("filing_id", ""),
                    "analyzed_at":        data.get("analyzed_at", ""),
                    "revenue":            fin.get("revenue"),
                    "net_profit":         fin.get("net_profit"),
                    "eps":                fin.get("eps"),
                    "revenue_growth_pct": fin.get("revenue_growth_pct"),
                    "profit_growth_pct":  fin.get("profit_growth_pct"),
                    "dividend_per_share": fin.get("dividend_per_share"),
                    "sentiment_score":    data.get("sentiment_score", 0),
                })
        except Exception:
            pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["analyzed_at"] = pd.to_datetime(df["analyzed_at"], errors="coerce")
    return df.sort_values("analyzed_at")
