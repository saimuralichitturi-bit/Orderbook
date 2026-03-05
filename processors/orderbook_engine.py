"""
Orderbook Intelligence Engine
==============================
Extracts structured numerical data from ALL PDFs for a company,
detects trends using unsupervised learning, and reasons about contract quality.

Algorithm Pipeline:
  1. Regex pre-filter  → quickly check if PDF has orderbook-relevant numbers
  2. AI extraction     → DeepSeek/Gemini/Groq extracts structured entries
  3. K-Means clustering → group similar orders (silhouette score picks optimal K)
  4. Time series       → velocity, acceleration, z-score anomaly detection
  5. AI reasoning      → "Is this contract good or bad? Why?" with logical chain

References:
  - StandardScaler + KMeans: sklearn 1.5+
  - Silhouette score for optimal K selection
  - Z-score 3-sigma anomaly detection on cumulative series
"""

import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_URL,
    GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL,
    GEMINI_API_KEY, GEMINI_MODEL,
    PDF_DIR, DATA_DIR,
)

ORDERBOOK_DIR = DATA_DIR / "orderbook"
ORDERBOOK_DIR.mkdir(parents=True, exist_ok=True)


# ─── Regex pre-filter patterns ────────────────────────────────────

_CAPACITY_RE = re.compile(
    r"(\d{1,6}(?:[,\d]*)?(?:\.\d+)?)\s*"
    r"(MW|GW|MWAC|MWp|MWac|Megawatt|Gigawatt|KW|KWh|MWh|MMSCMD|MMTPA)",
    re.IGNORECASE,
)
_FINANCIAL_RE = re.compile(
    r"(?:Rs\.?|₹|INR|USD|\$)\s*(\d{1,6}(?:[,\d]*)?(?:\.\d+)?)\s*"
    r"(Cr|Crore|Crores|Lakh|L|Mn|Million|Billion|Bn|M)",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"(\d{1,2})\s*(?:year|yr|years|annum|annual)s?",
    re.IGNORECASE,
)
_ORDERBOOK_KEYWORDS = re.compile(
    r"\b(ppa|order|contract|capacity|expansion|supply|tender|bid|win|"
    r"commission|award|agreement|mou|epc|power purchase|megawatt|gigawatt|"
    r"thermal|solar|wind|hydro|gas|acquisition|joint venture|alliance|"
    r"deal|engagement|mandate|revenue|backlog|pipeline|renewal|ramp|"
    r"total contract value|tcv|acv|digital transformation|outsourc|"
    r"it services|bpo|consulting|implementation|platform|cloud|"
    r"billion|million|crore|usd|inr)\b",
    re.IGNORECASE,
)

INR_CONVERSION = {
    "cr": 1, "crore": 1, "crores": 1,
    "lakh": 0.01, "l": 0.01,
    "mn": 8.5, "million": 8.5,       # USD Mn → INR Cr (~85 INR/USD)
    "billion": 8500, "bn": 8500, "b": 8500,
    "m": 8.5,
    # IT sector USD units
    "usd_mn": 8.5, "usd mn": 8.5,
    "usd_bn": 8500, "usd bn": 8500,
    "usd_billion": 8500, "usd_million": 8.5,
    "inr_cr": 1, "inr cr": 1,
    # Plain USD (assume millions if > 1000, else raw)
    "usd": 8.5 / 100,  # 1 USD = ~0.085 INR Cr (i.e., raw USD ÷ 100 for Cr)
}
MW_CONVERSION = {"mw": 1, "mwac": 1, "mwp": 1, "gw": 1000, "gigawatt": 1000,
                 "megawatt": 1, "kwh": 0.001, "mwh": 0.001}


def _has_orderbook_content(text: str) -> bool:
    """Quick regex check — does this PDF likely have orderbook data?"""
    keyword_hits = len(_ORDERBOOK_KEYWORDS.findall(text))
    number_hits = len(_CAPACITY_RE.findall(text)) + len(_FINANCIAL_RE.findall(text))
    return keyword_hits >= 2 and number_hits >= 1


def _regex_extract_numbers(text: str) -> list[dict]:
    """Fast regex extraction as validation baseline."""
    found = []
    for match in _CAPACITY_RE.finditer(text):
        val_str = match.group(1).replace(",", "")
        unit = match.group(2).upper()
        try:
            val = float(val_str)
            found.append({"value": val, "unit": unit,
                          "context": text[max(0, match.start()-80):match.end()+80].strip()})
        except ValueError:
            pass
    for match in _FINANCIAL_RE.finditer(text):
        val_str = match.group(1).replace(",", "")
        unit = match.group(2)
        try:
            val = float(val_str)
            found.append({"value": val, "unit": unit,
                          "context": text[max(0, match.start()-80):match.end()+80].strip()})
        except ValueError:
            pass
    return found


# ─── AI Extraction Prompt ─────────────────────────────────────────

_EXTRACTION_PROMPT = """You are an expert at extracting orderbook and contract data from Indian NSE corporate filings.

Company: {company}
Filing Subject: {subject}
Filing Date: {date}

PDF Content (key excerpts):
{text}

Pre-detected numbers by regex: {regex_hits}

Extract ALL entries representing business orders, contracts, deals, wins, capacity additions, PPAs, supply agreements, IT contracts, service agreements, acquisitions, or significant financial commitments.

This company may be in ANY sector — energy, IT/software, banking, manufacturing, pharma, etc.
- Energy companies: extract MW/GW capacity, PPA values, EPC contracts
- IT/software companies (TCS, Infosys, Wipro): extract deal wins, client contracts, $ or ₹ TCV (total contract value), headcount-related numbers
- Banks/NBFC: loan book numbers, NPA figures, AUM
- Manufacturing: order backlog, supply contracts, capacity expansion (units/tonnes)
- Any sector: acquisitions, JVs, penalties, regulatory outcomes

Rules:
- Include EVERY contract value (MW, ₹ Cr, USD Mn, USD Bn, units, tonnes, etc.)
- For IT deals: USD Bn/Mn contract wins are the key metric — convert to INR Cr (1 USD = 85 INR)
- If the same deal appears in multiple units, create ONE entry with both
- Compliance/penalty amounts count as entries (type="compliance")
- Board meeting outcomes referencing contracts → extract those too

Return ONLY this JSON (no markdown, no explanation):
{{
  "has_orderbook_data": true,
  "entries": [
    {{
      "type": "order_win|ppa|epc|supply|capacity_addition|it_deal|acquisition|financial|compliance|other",
      "description": "concise 1-line description of the deal/order",
      "value_numeric": <primary number>,
      "value_unit": "MW|GW|INR_Cr|USD_Mn|USD_Bn|USD|units|tonnes|%|other",
      "value_inr_cr": <estimated INR crore equivalent or null>,
      "value_mw": <capacity in MW if energy sector, else null>,
      "counterparty": "client or partner name or null",
      "project_location": "country/state/city or null",
      "duration_years": <contract duration in years or null>,
      "sector": "energy|it_services|banking|manufacturing|pharma|telecom|other",
      "energy_type": "thermal|solar|wind|hydro|gas|oil|mixed|null",
      "contract_type": "PPA|EPC|O&M|IT_services|BPO|consulting|supply|loan|equity|penalty|acquisition|other|null",
      "is_positive_signal": true,
      "reasoning": "1-2 sentences: why this matters for investors",
      "confidence": 0.85
    }}
  ],
  "total_mw_this_filing": <null or total MW sum (energy only)>,
  "total_inr_cr_this_filing": <null or total INR Cr equivalent sum>
}}

If no orderbook data found, return: {{"has_orderbook_data": false, "entries": []}}"""


def _call_ai_extraction(prompt: str) -> dict:
    """Try DeepSeek → Gemini → Groq for structured extraction."""
    providers = []
    if DEEPSEEK_API_KEY:
        providers.append(("deepseek", DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL))
    if GEMINI_API_KEY:
        providers.append(("gemini", GEMINI_API_KEY, None, GEMINI_MODEL))
    if GROQ_API_KEY:
        providers.append(("groq", GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL))

    for name, key, url, model in providers:
        try:
            if name == "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=key)
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0.1
                    ),
                )
                raw = resp.text.strip()
            else:
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url=url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=2500,
                )
                raw = resp.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Orderbook extraction failed ({name}): {e}")

    return {"has_orderbook_data": False, "entries": []}


# ─── Main extraction function ─────────────────────────────────────

def extract_orderbook_from_pdf(
    pdf_path,
    company: str,
    subject: str,
    date_str: str,
    filing_id: str,
    force: bool = False,
) -> dict:
    """
    Extract orderbook entries from a single PDF.
    Caches result to ORDERBOOK_DIR/{filing_id}.json.
    Uses regex pre-filter to skip non-relevant PDFs cheaply.
    """
    cache_path = ORDERBOOK_DIR / f"{filing_id}.json"
    if cache_path.exists() and not force:
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    from processors.ai_analyzer import extract_pdf_text
    text = extract_pdf_text(str(pdf_path), max_chars=25000)

    if not text.strip():
        result = {"has_orderbook_data": False, "entries": [], "reason": "empty_pdf"}
        cache_path.write_text(json.dumps(result, indent=2))
        return result

    # Regex pre-filter — skip irrelevant PDFs without AI call
    regex_hits = _regex_extract_numbers(text)
    if not _has_orderbook_content(text):
        result = {"has_orderbook_data": False, "entries": [], "reason": "no_keywords"}
        cache_path.write_text(json.dumps(result, indent=2))
        return result

    # Build regex summary for AI context
    regex_summary = "; ".join(
        f"{h['value']} {h['unit']}" for h in regex_hits[:10]
    ) or "none detected"

    prompt = _EXTRACTION_PROMPT.format(
        company=company,
        subject=subject,
        date=date_str,
        text=text[:18000],
        regex_hits=regex_summary,
    )

    result = _call_ai_extraction(prompt)
    result["filing_id"] = filing_id
    result["company"] = company
    result["subject"] = subject
    result["date"] = date_str
    result["regex_hits_count"] = len(regex_hits)

    cache_path.write_text(json.dumps(result, indent=2))
    logger.info(
        f"Orderbook extracted: {filing_id} | "
        f"entries={len(result.get('entries', []))} | "
        f"has_data={result.get('has_orderbook_data')}"
    )
    return result


# ─── Batch extraction ─────────────────────────────────────────────

def batch_extract_orderbook(
    symbol: str,
    df: pd.DataFrame,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Process all PDFs for a company and return a flat DataFrame of orderbook entries.
    Skips already-cached extractions. Tries local → PDF_DIR → Drive for each PDF.
    """
    all_entries = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if progress_callback:
            progress_callback(i, total, f"Processing PDF {i+1}/{total}: {str(row.get('subject',''))[:40]}")

        filing_id = str(row.get("filing_id", ""))
        if not filing_id or filing_id == "nan":
            continue

        # Check cache first (avoids re-processing)
        cache_path = ORDERBOOK_DIR / f"{filing_id}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                _append_entries(all_entries, cached, row)
                continue
            except Exception:
                pass

        # Resolve PDF path
        pdf_path = _resolve_pdf(filing_id, str(row.get("local_pdf", "")),
                                str(row.get("pdf_url", "")))
        if not pdf_path:
            continue

        date_str = str(row.get("broadcast_dt", ""))[:10]
        result = extract_orderbook_from_pdf(
            pdf_path=pdf_path,
            company=str(row.get("company", symbol)),
            subject=str(row.get("subject", "")),
            date_str=date_str,
            filing_id=filing_id,
        )
        _append_entries(all_entries, result, row)

    if progress_callback:
        progress_callback(total, total, "Extraction complete")

    if not all_entries:
        return pd.DataFrame()

    ob_df = pd.DataFrame(all_entries)
    ob_df["date"] = pd.to_datetime(ob_df["date"], errors="coerce")
    ob_df = ob_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return ob_df


def _resolve_pdf(filing_id: str, local_pdf: str, pdf_url: str) -> Path | None:
    """Try local PDF_DIR → local_pdf path → Drive → NSE URL."""
    # 1. Standard local path
    local = PDF_DIR / f"{filing_id}.pdf"
    if local.exists():
        return local
    # 2. Column-stored path
    if local_pdf and local_pdf not in ["nan", "None", ""] and Path(local_pdf).exists():
        return Path(local_pdf)
    # 3. Google Drive
    try:
        from storage.drive_handler import DriveHandler
        dh = DriveHandler()
        buf = dh.download_pdf_by_name(f"{filing_id}.pdf")
        if buf:
            local.write_bytes(buf.read())
            return local
    except Exception:
        pass
    # 4. Direct NSE URL
    if pdf_url and pdf_url.startswith("http"):
        try:
            from scrapers.nse_filings import NSEFilingsScraper
            scraper = NSEFilingsScraper()
            path = scraper.download_pdf(pdf_url, filing_id)
            if path:
                return path
        except Exception:
            pass
    return None


def _append_entries(entries: list, result: dict, row) -> None:
    """Flatten extraction result into entry rows."""
    if not result.get("has_orderbook_data"):
        return
    date_str = str(row.get("broadcast_dt", ""))[:10]
    subject = str(row.get("subject", ""))
    filing_id = str(row.get("filing_id", ""))

    for entry in result.get("entries", []):
        # Normalize value_mw
        val = entry.get("value_numeric") or 0
        unit = str(entry.get("value_unit", "")).lower()
        val_mw = entry.get("value_mw")
        if val_mw is None:
            val_mw = val * MW_CONVERSION.get(unit, 0) if unit in MW_CONVERSION else None

        # Normalize value_inr_cr — handle USD Mn/Bn from IT deal filings
        val_inr = entry.get("value_inr_cr")
        if val_inr is None:
            unit_key = unit.replace(" ", "_").lower()
            if unit_key in INR_CONVERSION:
                val_inr = val * INR_CONVERSION[unit_key]
            elif unit in INR_CONVERSION:
                val_inr = val * INR_CONVERSION[unit]
        # If still None but value_numeric exists with a USD-like unit, try to convert
        if val_inr is None and val and any(u in unit for u in ["usd", "dollar", "bn", "mn", "billion", "million"]):
            if any(u in unit for u in ["bn", "billion"]):
                val_inr = val * 8500
            elif any(u in unit for u in ["mn", "million", "usd"]):
                val_inr = val * 8.5
        # Clamp unrealistic values (e.g. 0.001 Cr → skip)
        if val_inr is not None and val_inr < 0.01:
            val_inr = None

        entries.append({
            "filing_id":       filing_id,
            "date":            date_str,
            "filing_subject":  subject,
            "type":            entry.get("type", "other"),
            "description":     entry.get("description", ""),
            "value_numeric":   entry.get("value_numeric"),
            "value_unit":      entry.get("value_unit", ""),
            "value_mw":        val_mw,
            "value_inr_cr":    val_inr,
            "counterparty":    entry.get("counterparty", ""),
            "project_location":entry.get("project_location", ""),
            "duration_years":  entry.get("duration_years"),
            "energy_type":     entry.get("energy_type", ""),
            "contract_type":   entry.get("contract_type", ""),
            "is_positive":     bool(entry.get("is_positive_signal", True)),
            "reasoning":       entry.get("reasoning", ""),
            "confidence":      float(entry.get("confidence", 0.5)),
            "total_mw_filing": result.get("total_mw_this_filing"),
            "total_inr_filing":result.get("total_inr_cr_this_filing"),
        })


# ─── Trend Detection (Unsupervised Learning) ──────────────────────

def detect_trends(ob_df: pd.DataFrame) -> dict:
    """
    Multi-layer trend detection:
    1. Cumulative MW and INR Cr timeseries
    2. Velocity (first derivative) and acceleration (second derivative)
    3. Z-score anomaly detection (3-sigma) on rolling window
    4. K-Means clustering with silhouette score for optimal K
    5. Quarterly velocity to detect acceleration/deceleration
    """
    if ob_df.empty:
        return {}

    trends = {}

    # ── 1. Cumulative MW timeseries ───────────────────────────────
    mw_df = ob_df[ob_df["value_mw"].notna() & (ob_df["value_mw"] > 0)].copy()
    if not mw_df.empty:
        daily_mw = (
            mw_df.groupby("date")["value_mw"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        daily_mw["cumulative_mw"] = daily_mw["value_mw"].cumsum()

        # Monthly resampled
        monthly_mw = (
            daily_mw.set_index("date")["value_mw"]
            .resample("ME")
            .sum()
            .reset_index()
        )
        monthly_mw.columns = ["month", "mw_added"]
        monthly_mw["cumulative_mw"] = monthly_mw["mw_added"].cumsum()

        trends["total_mw"] = float(daily_mw["value_mw"].sum())
        trends["latest_cumulative_mw"] = float(daily_mw["cumulative_mw"].iloc[-1])
        trends["mw_timeseries"] = _df_to_records(daily_mw)
        trends["monthly_mw"] = _df_to_records(monthly_mw)

        # Velocity & acceleration on monthly series
        if len(monthly_mw) >= 3:
            monthly_mw["velocity"] = monthly_mw["mw_added"].diff().fillna(0)
            monthly_mw["acceleration"] = monthly_mw["velocity"].diff().fillna(0)
            trends["mw_velocity"] = _df_to_records(monthly_mw[["month", "velocity", "acceleration"]])

            # Z-score anomaly detection
            if len(monthly_mw) >= 4:
                roll_mean = monthly_mw["mw_added"].rolling(3, min_periods=1).mean()
                roll_std  = monthly_mw["mw_added"].rolling(3, min_periods=1).std().fillna(1)
                z_scores  = ((monthly_mw["mw_added"] - roll_mean) / roll_std).fillna(0)
                anomaly_mask = z_scores.abs() > 2.0
                anomalies = monthly_mw[anomaly_mask][["month", "mw_added"]].copy()
                anomalies["z_score"] = z_scores[anomaly_mask].values
                trends["mw_anomalies"] = _df_to_records(anomalies)

    # ── 2. Cumulative INR Cr timeseries ───────────────────────────
    inr_df = ob_df[ob_df["value_inr_cr"].notna() & (ob_df["value_inr_cr"] > 0)].copy()
    if not inr_df.empty:
        daily_inr = (
            inr_df.groupby("date")["value_inr_cr"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        daily_inr["cumulative_inr_cr"] = daily_inr["value_inr_cr"].cumsum()

        monthly_inr = (
            daily_inr.set_index("date")["value_inr_cr"]
            .resample("ME")
            .sum()
            .reset_index()
        )
        monthly_inr.columns = ["month", "inr_cr_added"]
        monthly_inr["cumulative_inr_cr"] = monthly_inr["inr_cr_added"].cumsum()

        trends["total_inr_cr"] = float(daily_inr["value_inr_cr"].sum())
        trends["inr_timeseries"] = _df_to_records(daily_inr)
        trends["monthly_inr"] = _df_to_records(monthly_inr)

    # ── 3. Energy type breakdown ──────────────────────────────────
    energy_breakdown = ob_df.groupby("energy_type").agg(
        count=("value_mw", "count"),
        total_mw=("value_mw", lambda x: x.dropna().sum()),
        total_inr_cr=("value_inr_cr", lambda x: x.dropna().sum()),
    ).reset_index()
    trends["energy_breakdown"] = _df_to_records(energy_breakdown)

    # ── 4. Contract type breakdown ────────────────────────────────
    type_breakdown = ob_df.groupby("type").agg(
        count=("value_mw", "count"),
        total_mw=("value_mw", lambda x: x.dropna().sum()),
    ).reset_index()
    trends["type_distribution"] = ob_df["type"].value_counts().to_dict()
    trends["type_breakdown"] = _df_to_records(type_breakdown)

    # ── 5. K-Means clustering ─────────────────────────────────────
    cluster_result = _cluster_orders(ob_df)
    if cluster_result:
        trends["clusters"] = cluster_result

    # ── 6. Quarterly order velocity ───────────────────────────────
    ob_q = ob_df.copy()
    ob_q["quarter"] = ob_q["date"].dt.to_period("Q").astype(str)
    quarterly = ob_q.groupby("quarter").agg(
        count=("type", "count"),
        total_mw=("value_mw", lambda x: x.dropna().sum()),
        total_inr_cr=("value_inr_cr", lambda x: x.dropna().sum()),
        bullish=("is_positive", "sum"),
    ).reset_index()
    trends["quarterly"] = _df_to_records(quarterly)

    if len(quarterly) >= 4:
        recent = quarterly["count"].iloc[-2:].mean()
        older  = quarterly["count"].iloc[-4:-2].mean()
        if older > 0:
            trends["velocity_change_pct"] = round((recent - older) / older * 100, 1)
        trends["growth_trajectory"] = (
            "accelerating" if trends.get("velocity_change_pct", 0) > 15
            else "decelerating" if trends.get("velocity_change_pct", 0) < -15
            else "steady"
        )

    # ── 7. Signal summary ─────────────────────────────────────────
    total = len(ob_df)
    # Ensure is_positive column exists
    if "is_positive" not in ob_df.columns:
        ob_df["is_positive"] = True
    pos = int(ob_df["is_positive"].fillna(True).astype(bool).sum())
    trends["total_entries"] = total
    trends["bullish_ratio"]  = round(pos / total, 3) if total > 0 else 0
    trends["bearish_count"]  = total - pos

    # Always set totals (default 0 if no data) so UI always has values
    if "total_mw" not in trends:
        trends["total_mw"] = 0.0
    if "total_inr_cr" not in trends:
        # Try summing value_inr_cr directly as fallback
        if "value_inr_cr" in ob_df.columns:
            trends["total_inr_cr"] = float(ob_df["value_inr_cr"].dropna().sum())
        else:
            trends["total_inr_cr"] = 0.0
    if "growth_trajectory" not in trends:
        trends["growth_trajectory"] = "steady"

    return trends


def _cluster_orders(ob_df: pd.DataFrame) -> list[dict] | None:
    """
    K-Means clustering on order features.
    Uses silhouette score to find optimal K (2–5).
    Returns cluster summaries or None if insufficient data.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        # Build feature matrix — normalize MW to INR Cr equivalent
        feature_rows = []
        valid_idx = []
        for i, (_, row) in enumerate(ob_df.iterrows()):
            val = row.get("value_numeric") or 0
            unit = str(row.get("value_unit", "")).lower()
            mw = row.get("value_mw") or 0
            inr = row.get("value_inr_cr") or (mw * 5 if mw else val)
            duration = row.get("duration_years") or 20
            is_pos = 1.0 if row.get("is_positive") else -1.0
            if inr > 0 or mw > 0:
                feature_rows.append([float(inr), float(mw), float(duration), is_pos])
                valid_idx.append(i)

        if len(feature_rows) < 4:
            return None

        X = np.array(feature_rows)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Find optimal K by silhouette score
        best_k, best_score, best_labels = 2, -1, None
        for k in range(2, min(6, len(feature_rows))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        if best_labels is None:
            return None

        # Build cluster summaries
        ob_valid = ob_df.iloc[valid_idx].copy().reset_index(drop=True)
        ob_valid["cluster"] = best_labels

        cluster_labels = {
            0: "Large Strategic", 1: "Mid-Size Growth",
            2: "Small Tactical", 3: "Mega Projects", 4: "Compliance/Other",
        }
        summaries = []
        for cid in sorted(ob_valid["cluster"].unique()):
            cdf = ob_valid[ob_valid["cluster"] == cid]
            avg_mw = cdf["value_mw"].dropna().mean()
            avg_inr = cdf["value_inr_cr"].dropna().mean()
            top_type = cdf["type"].value_counts().index[0] if not cdf.empty else "other"
            summaries.append({
                "cluster_id":   int(cid),
                "label":        cluster_labels.get(cid, f"Cluster {cid}"),
                "count":        len(cdf),
                "avg_mw":       round(avg_mw, 1) if not np.isnan(avg_mw) else None,
                "avg_inr_cr":   round(avg_inr, 1) if not np.isnan(avg_inr) else None,
                "dominant_type":top_type,
                "bullish_pct":  round(cdf["is_positive"].mean() * 100, 1),
                "silhouette":   round(best_score, 3),
                "entries":      cdf["description"].tolist()[:5],
            })

        return summaries
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        return None


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe list of dicts."""
    records = []
    for r in df.to_dict("records"):
        clean = {}
        for k, v in r.items():
            if hasattr(v, "item"):
                v = v.item()  # numpy scalar → python
            if isinstance(v, float) and (v != v):  # NaN
                v = None
            if hasattr(v, "isoformat"):
                v = str(v)[:10]
            clean[k] = v
        records.append(clean)
    return records


# ─── AI Reasoning ─────────────────────────────────────────────────

_REASONING_PROMPT = """You are a senior equity analyst specializing in Indian infrastructure, energy, and conglomerate companies.

Company: {symbol}
Orderbook Intelligence Report:

Total Orders Analyzed: {total_entries}
Total Capacity Won: {total_mw} MW
Total Contract Value: ₹{total_inr_cr} Cr
Bullish Signal Ratio: {bullish_pct}%
Growth Trajectory: {trajectory}
Order Velocity Change: {velocity_pct}% (vs prior period)

Energy Mix: {energy_mix}
Contract Type Mix: {type_mix}

Recent Orders (last 15):
{recent_orders}

Top Cluster Profiles:
{clusters}

Apply logical reasoning to answer: Is this company's orderbook healthy? Is it worth investing in?

Return ONLY this JSON:
{{
  "overall_assessment": "bullish|bearish|neutral",
  "investment_grade": "A|B|C|D",
  "order_quality_score": <0-100>,
  "growth_trajectory": "accelerating|steady|decelerating|volatile",
  "executive_summary": "3-4 sentences covering orderbook size, quality, momentum, and investment case",
  "key_strengths": ["strength 1", "strength 2", "strength 3"],
  "key_concerns": ["concern 1", "concern 2"],
  "reasoning_chain": [
    "Observation: [what the data shows]",
    "Implication: [what it means]",
    "Evidence: [specific orders that support this]",
    "Conclusion: [investment view]"
  ],
  "recommended_action": "strong_buy|buy|hold|reduce|sell",
  "12m_outlook": "concise forward view based on order pipeline",
  "catalysts_to_watch": ["catalyst 1", "catalyst 2"],
  "risks": ["risk 1", "risk 2"],
  "sector_context": "how this compares to sector peers and why it matters"
}}"""


def ai_orderbook_reasoning(symbol: str, ob_df: pd.DataFrame, trends: dict) -> dict:
    """
    Use DeepSeek/Gemini/Groq to reason about orderbook trends.
    Returns structured investment assessment with logical reasoning chain.
    """
    if ob_df.empty:
        return {}

    # Build recent orders text
    recent = ob_df.tail(15)
    recent_lines = []
    for _, row in recent.iterrows():
        line = (
            f"• {str(row.get('date',''))[:10]}: {row.get('description','?')} "
            f"[{row.get('value_numeric','')} {row.get('value_unit','')}]"
        )
        if row.get("reasoning"):
            line += f" → {row['reasoning']}"
        recent_lines.append(line)

    # Energy mix
    energy_mix = json.dumps(
        ob_df.groupby("energy_type")["value_mw"].sum().dropna().round(0).astype(int).to_dict()
    ) if "energy_type" in ob_df.columns else "{}"

    # Cluster summary
    cluster_text = ""
    if trends.get("clusters"):
        for c in trends["clusters"]:
            cluster_text += (
                f"  Cluster '{c['label']}': {c['count']} orders, "
                f"avg {c.get('avg_mw','?')} MW, "
                f"avg ₹{c.get('avg_inr_cr','?')} Cr, "
                f"{c['bullish_pct']}% bullish\n"
            )

    prompt = _REASONING_PROMPT.format(
        symbol=symbol,
        total_entries=trends.get("total_entries", len(ob_df)),
        total_mw=round(trends.get("total_mw", 0), 0),
        total_inr_cr=round(trends.get("total_inr_cr", 0), 0),
        bullish_pct=round(trends.get("bullish_ratio", 0) * 100, 1),
        trajectory=trends.get("growth_trajectory", "unknown"),
        velocity_pct=trends.get("velocity_change_pct", "N/A"),
        energy_mix=energy_mix,
        type_mix=json.dumps(trends.get("type_distribution", {})),
        recent_orders="\n".join(recent_lines) or "No recent orders",
        clusters=cluster_text or "Insufficient data for clustering",
    )

    providers = []
    if DEEPSEEK_API_KEY:
        providers.append(("deepseek", DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL))
    if GEMINI_API_KEY:
        providers.append(("gemini", GEMINI_API_KEY, None, GEMINI_MODEL))
    if GROQ_API_KEY:
        providers.append(("groq", GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL))

    for name, key, url, model in providers:
        try:
            if name == "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=key)
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0.2
                    ),
                )
                return json.loads(resp.text.strip())
            else:
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url=url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=2500,
                )
                return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"AI reasoning failed ({name}): {e}")

    return {}


# ─── Orderbook save/load ───────────────────────────────────────────

def save_orderbook(symbol: str, ob_df: pd.DataFrame, trends: dict, reasoning: dict):
    """Persist orderbook dataframe + analysis to disk AND Google Drive."""
    pq_path = ORDERBOOK_DIR / f"{symbol}_orderbook.parquet"
    js_path = ORDERBOOK_DIR / f"{symbol}_analysis.json"

    if not ob_df.empty:
        ob_df.to_parquet(pq_path, index=False)
    full = {"trends": trends, "reasoning": reasoning, "generated_at": datetime.now().isoformat()}
    js_path.write_text(json.dumps(full, indent=2, default=str))

    # Sync to Google Drive so data survives Streamlit Cloud reruns
    try:
        from storage.drive_handler import DriveHandler
        dh = DriveHandler()
        if not ob_df.empty:
            dh.upload_parquet(pq_path, subfolder="orderbook")
        dh.upload_json(js_path, subfolder="orderbook")
        logger.info(f"Orderbook synced to Drive for {symbol}")
    except Exception as e:
        logger.warning(f"Drive sync failed for orderbook ({symbol}): {e}")


def load_orderbook(symbol: str) -> tuple[pd.DataFrame, dict, dict]:
    """Load cached orderbook data — local first, then Google Drive fallback."""
    pq_path = ORDERBOOK_DIR / f"{symbol}_orderbook.parquet"
    js_path = ORDERBOOK_DIR / f"{symbol}_analysis.json"

    # Try Drive if local files missing (Streamlit Cloud ephemeral FS)
    if not pq_path.exists() or not js_path.exists():
        try:
            from storage.drive_handler import DriveHandler
            dh = DriveHandler()
            if not pq_path.exists():
                buf = dh.download_pdf_by_name(f"{symbol}_orderbook.parquet", subfolder="orderbook")
                if buf:
                    pq_path.write_bytes(buf.read())
            if not js_path.exists():
                buf = dh.download_pdf_by_name(f"{symbol}_analysis.json", subfolder="orderbook")
                if buf:
                    js_path.write_bytes(buf.read())
        except Exception as e:
            logger.warning(f"Drive load failed for orderbook ({symbol}): {e}")

    ob_df = pd.DataFrame()
    if pq_path.exists():
        try:
            ob_df = pd.read_parquet(pq_path)
            if "date" in ob_df.columns:
                ob_df["date"] = pd.to_datetime(ob_df["date"], errors="coerce")
        except Exception:
            pass

    trends, reasoning = {}, {}
    if js_path.exists():
        try:
            data = json.loads(js_path.read_text())
            trends    = data.get("trends", {})
            reasoning = data.get("reasoning", {})
        except Exception:
            pass

    return ob_df, trends, reasoning
